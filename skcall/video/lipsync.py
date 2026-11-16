"""Client for the lip-sync service (services/lipsync, sklipsync with MuseTalk).

Wire protocol (fixed; the service is built against the same text):

  WebSocket ws://<host>:<port>/v1/lipsync
  header    Authorization: Bearer <SKCALL_LIPSYNC_TOKEN>
  client -> {"type":"hello","avatar":"lumina","fps":25,"engine":"musetalk"}
  server -> {"type":"ready","fps":25,"crop":{"x":..,"y":..,"w":..,"h":..},"frame_size":[w,h]}
  per utterance:
    client -> {"type":"utterance","id":"<uuid>"}
    client -> binary 16 kHz mono s16le PCM chunks
    client -> {"type":"end","id":"<uuid>"}
    client -> {"type":"cancel","id":"<uuid>"}            (barge-in)
    server -> binary: u32 BE frame index, u16 BE uuid length, uuid ascii, JPEG mouth crop
    server -> {"type":"done","id":"<uuid>","frames":N}
  keepalive: {"type":"ping"} / {"type":"pong"}, either side

Frame i is the mouth at audio time i/fps from the start of the utterance.

The client never raises into the call: if the service is unreachable, slow
or drops, `available` goes False, utterances get no frames, and the
compositor keeps showing the idle loop. It reconnects in the background.
"""
from __future__ import annotations

import asyncio
import json
import logging
import struct
import time
import uuid
from dataclasses import dataclass, field

import aiohttp
import numpy as np

LOG = logging.getLogger("skcall.lipsync")

PCM_RATE = 16000
CHUNK_SAMPLES = PCM_RATE // 5     # 200 ms per binary message (6.4 KB)


def pack_frame(index: int, utt_id: str, jpeg: bytes) -> bytes:
    u = utt_id.encode("ascii")
    return struct.pack(">IH", index, len(u)) + u + jpeg


def unpack_frame(data: bytes) -> tuple[int, str, bytes]:
    if len(data) < 6:
        raise ValueError("short frame message")
    index, n = struct.unpack_from(">IH", data, 0)
    if len(data) < 6 + n:
        raise ValueError("truncated utterance id")
    return index, data[6:6 + n].decode("ascii"), data[6 + n:]


@dataclass
class Crop:
    x: int
    y: int
    w: int
    h: int


@dataclass
class LipsyncUtterance:
    """Frames for one utterance as they arrive. frames[i] is a JPEG."""
    id: str
    n_samples: int                    # 16 kHz samples sent
    fps: float
    frames: dict[int, bytes] = field(default_factory=dict)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    expected: int | None = None       # from "done"
    cancelled: bool = False
    t_sent: float = 0.0
    t_first_frame: float | None = None
    arrived: asyncio.Event = field(default_factory=asyncio.Event)   # set on every new frame

    @property
    def duration_s(self) -> float:
        return self.n_samples / PCM_RATE

    @property
    def n_frames(self) -> int:
        """Frames covering the audio (the service's count once it is done)."""
        if self.expected is not None:
            return self.expected
        return int(np.ceil(self.duration_s * self.fps))

    def ready(self, lead: int) -> bool:
        """The first `lead` frames are here (or the service has finished)."""
        if self.done.is_set():
            return True
        return all(i in self.frames for i in range(min(lead, self.n_frames)))

    async def wait_ready(self, lead: int, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while not self.ready(lead):
            left = deadline - time.monotonic()
            if left <= 0 or self.cancelled:
                return False
            self.arrived.clear()
            try:
                await asyncio.wait_for(self.arrived.wait(), left)
            except asyncio.TimeoutError:
                return self.ready(lead)
        return True


class LipsyncClient:
    def __init__(self, url: str, token: str = "", avatar: str = "lumina", fps: int = 25,
                 engine: str = "musetalk", connect_timeout: float = 5.0, ping_s: float = 10.0,
                 retry_s: tuple[float, float] = (1.0, 30.0), on_ready=None):
        self.url, self.token = url, token
        self.on_ready = on_ready          # called with the client after every (re)connect
        self.avatar, self.fps, self.engine = avatar, fps, engine
        self.connect_timeout = connect_timeout
        self.ping_s = ping_s
        self.retry_s = retry_s
        self.crop: Crop | None = None
        self.frame_size: tuple[int, int] | None = None
        self.utts: dict[str, LipsyncUtterance] = {}
        self._http: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._runner: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._closed = False
        self._send_lock = asyncio.Lock()
        self.stats = {"connects": 0, "frames": 0, "bad_frames": 0, "utterances": 0, "cancels": 0}

    # ----------------------------------------------------------- lifecycle
    @property
    def available(self) -> bool:
        return self._ws is not None and not self._ws.closed and self._ready.is_set()

    def start(self) -> None:
        """Connect (and keep reconnecting) in the background."""
        if self._runner is None:
            self._runner = asyncio.create_task(self._run())

    async def wait_ready(self, timeout: float) -> bool:
        try:
            await asyncio.wait_for(self._ready.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def aclose(self) -> None:
        self._closed = True
        if self._runner is not None:
            self._runner.cancel()
            await asyncio.gather(self._runner, return_exceptions=True)
        if self._ws is not None:
            await self._ws.close()
        if self._http is not None:
            await self._http.close()

    async def _run(self) -> None:
        delay = self.retry_s[0]
        while not self._closed:
            try:
                await self._session()
                delay = self.retry_s[0]
            except asyncio.CancelledError:
                raise
            except Exception as e:
                LOG.warning("lipsync service %s unavailable: %r (retry in %.0f s)", self.url, e, delay)
            self._ready.clear()
            self._ws = None
            for u in self.utts.values():            # nothing more will arrive for these
                u.done.set()
                u.arrived.set()
            self.utts.clear()
            if self._closed:
                return
            await asyncio.sleep(delay)
            delay = min(delay * 2, self.retry_s[1])

    async def _session(self) -> None:
        if self._http is None:
            self._http = aiohttp.ClientSession()
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        ws = await self._http.ws_connect(self.url, headers=headers,
                                         timeout=aiohttp.ClientWSTimeout(ws_close=self.connect_timeout),
                                         max_msg_size=8 * 1024 * 1024)
        try:
            await ws.send_json({"type": "hello", "avatar": self.avatar, "fps": self.fps,
                                "engine": self.engine})
            msg = await asyncio.wait_for(ws.receive(), self.connect_timeout)
            if msg.type != aiohttp.WSMsgType.TEXT:
                raise ConnectionError(f"expected ready, got {msg.type}")
            ready = json.loads(msg.data)
            if ready.get("type") != "ready":
                raise ConnectionError(f"expected ready, got {ready.get('type')}: {ready}")
            c = ready["crop"]
            self.crop = Crop(int(c["x"]), int(c["y"]), int(c["w"]), int(c["h"]))
            fs = ready.get("frame_size")
            self.frame_size = (int(fs[0]), int(fs[1])) if fs else None
            if int(ready.get("fps", self.fps)) != self.fps:
                LOG.warning("service runs %s fps, asked %s; using its rate", ready.get("fps"), self.fps)
                self.fps = int(ready["fps"])
            self._ws = ws
            self._ready.set()
            self.stats["connects"] += 1
            if self.on_ready is not None:
                try:
                    self.on_ready(self)
                except Exception:
                    LOG.warning("on_ready hook failed", exc_info=True)
            LOG.info("lipsync ready: avatar=%s fps=%s crop=%s frame_size=%s", self.avatar, self.fps,
                     self.crop, self.frame_size)
            pinger = asyncio.create_task(self._pinger(ws))
            try:
                await self._receive(ws)
            finally:
                pinger.cancel()
        finally:
            await ws.close()

    async def _pinger(self, ws) -> None:
        while not ws.closed:
            await asyncio.sleep(self.ping_s)
            try:
                async with self._send_lock:
                    await ws.send_json({"type": "ping"})
            except Exception:
                return

    async def _receive(self, ws) -> None:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.BINARY:
                try:
                    idx, uid, jpeg = unpack_frame(msg.data)
                except ValueError:
                    self.stats["bad_frames"] += 1
                    continue
                u = self.utts.get(uid)
                if u is None or u.cancelled:
                    continue           # a cancelled or unknown utterance: drop
                if u.t_first_frame is None:
                    u.t_first_frame = time.monotonic()
                u.frames[idx] = jpeg
                self.stats["frames"] += 1
                u.arrived.set()
            elif msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    m = json.loads(msg.data)
                except ValueError:
                    continue
                ty = m.get("type")
                if ty == "done":
                    u = self.utts.get(m.get("id", ""))
                    if u is not None:
                        u.expected = int(m.get("frames", len(u.frames)))
                        u.done.set()
                        u.arrived.set()
                elif ty == "ping":
                    async with self._send_lock:
                        await ws.send_json({"type": "pong"})
                elif ty == "error":
                    LOG.warning("lipsync service error: %s", m)
                    u = self.utts.get(m.get("id", ""))
                    if u is not None:
                        u.done.set()
                        u.arrived.set()
            elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
                break
        LOG.warning("lipsync connection closed")

    # ----------------------------------------------------------- utterances
    def submit(self, pcm16k: np.ndarray) -> LipsyncUtterance | None:
        """Start lip-syncing one utterance (int16 mono 16 kHz). Returns at once;
        frames fill in as the service renders them. None if the service is
        not connected (the caller just plays the audio over the idle loop)."""
        if not self.available:
            return None
        u = LipsyncUtterance(id=str(uuid.uuid4()), n_samples=len(pcm16k), fps=self.fps,
                             t_sent=time.monotonic())
        self.utts[u.id] = u
        self.stats["utterances"] += 1
        asyncio.ensure_future(self._send(u, pcm16k.astype("<i2", copy=False)))
        return u

    async def _send(self, u: LipsyncUtterance, pcm: np.ndarray) -> None:
        ws = self._ws
        try:
            async with self._send_lock:
                if ws is None or ws.closed or u.cancelled:
                    raise ConnectionError("not connected")
                await ws.send_json({"type": "utterance", "id": u.id})
                for i in range(0, len(pcm), CHUNK_SAMPLES):
                    if u.cancelled:
                        return
                    await ws.send_bytes(pcm[i:i + CHUNK_SAMPLES].tobytes())
                await ws.send_json({"type": "end", "id": u.id})
        except Exception as e:
            LOG.info("lipsync send for %s failed: %r", u.id[:8], e)
            u.done.set()
            u.arrived.set()

    def cancel(self, u: LipsyncUtterance) -> None:
        """Barge-in: stop rendering it and drop anything still coming."""
        if u.cancelled:
            return
        u.cancelled = True
        u.frames.clear()
        u.done.set()
        u.arrived.set()
        self.utts.pop(u.id, None)
        self.stats["cancels"] += 1
        ws = self._ws
        if ws is not None and not ws.closed:
            async def _c():
                try:
                    async with self._send_lock:
                        await ws.send_json({"type": "cancel", "id": u.id})
                except Exception:
                    pass
            asyncio.ensure_future(_c())

    def forget(self, u: LipsyncUtterance) -> None:
        """Done with it (played out, or cached elsewhere): stop routing frames."""
        self.utts.pop(u.id, None)

    async def render_all(self, pcm16k: np.ndarray, timeout: float = 20.0) -> dict[int, bytes] | None:
        """Lip-sync a whole clip and wait for every frame (filler pre-render)."""
        u = self.submit(pcm16k)
        if u is None:
            return None
        try:
            await asyncio.wait_for(u.done.wait(), timeout)
        except asyncio.TimeoutError:
            self.cancel(u)
            return None
        self.forget(u)
        return dict(u.frames) if u.frames else None
