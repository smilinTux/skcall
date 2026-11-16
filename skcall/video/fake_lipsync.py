"""A fake lip-sync service speaking the sklipsync wire protocol (see lipsync.py).

For tests and loopback self-tests: no GPU, no model. Each crop is a flat
JPEG whose colour encodes the frame index (crop_value(i)), so a receiver can
tell exactly which frame it is looking at. Frames are produced as the audio
arrives, at `rtf` times realtime after a `ttff` delay, like MuseTalk does.

  python -m skcall.video.fake_lipsync --port 18799 --token t
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

import numpy as np
from aiohttp import WSMsgType, web

from skcall.video.lipsync import PCM_RATE, pack_frame

LOG = logging.getLogger("skcall.fake_lipsync")


def crop_value(i: int) -> int:
    """Grey level of frame i's crop: distinct for consecutive frames, far from 0/255."""
    return 40 + (i * 23) % 180


def index_from_value(v: float, near: int = 0) -> int:
    """Inverse of crop_value, choosing the candidate closest to `near`."""
    best = None
    for i in range(max(0, near - 200), near + 200):
        d = abs(crop_value(i) - v)
        if best is None or d < best[0] or (d == best[0] and abs(i - near) < abs(best[1] - near)):
            best = (d, i)
    return best[1]


def flat_jpeg(w: int, h: int, value: int) -> bytes:
    import cv2
    img = np.full((h, w, 3), value, np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


@dataclass
class _Utt:
    id: str
    samples: int = 0
    ended: bool = False
    cancelled: bool = False
    t_start: float = 0.0
    task: asyncio.Task | None = None
    sent: int = 0


@dataclass
class FakeLipsyncServer:
    token: str = "test-token"
    crop: dict = field(default_factory=lambda: {"x": 200, "y": 330, "w": 160, "h": 120})
    frame_size: tuple[int, int] = (560, 720)
    crop_size: tuple[int, int] = (128, 96)        # JPEG size (resized by the client)
    fps: int = 25
    ttff: float = 0.08
    rtf: float = 2.0
    value_fn: object = None                       # frame index -> grey level
    host: str = "127.0.0.1"
    port: int = 0
    hellos: list = field(default_factory=list)
    utterances: list = field(default_factory=list)
    cancelled: list = field(default_factory=list)
    pings: int = 0

    def __post_init__(self):
        self._runner: web.AppRunner | None = None
        self._jpegs: dict[int, bytes] = {}
        self._sockets: set = set()

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}/v1/lipsync"

    def _jpeg(self, i: int) -> bytes:
        v = (self.value_fn or crop_value)(i)
        if v not in self._jpegs:
            self._jpegs[v] = flat_jpeg(*self.crop_size, v)
        return self._jpegs[v]

    async def start(self) -> "FakeLipsyncServer":
        app = web.Application()
        app.router.add_get("/v1/lipsync", self._handle)
        async def health(_req):
            return web.json_response({"ok": True})
        app.router.add_get("/health", health)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]
        return self

    async def stop(self) -> None:
        for ws in list(self._sockets):
            await ws.close()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def kill_connections(self) -> None:
        """Simulate the service dying mid-call (the listener stays up)."""
        for ws in list(self._sockets):
            await ws.close()

    async def _handle(self, req: web.Request) -> web.StreamResponse:
        if req.headers.get("Authorization") != f"Bearer {self.token}":
            return web.Response(status=401, text="bad token")
        ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024)
        await ws.prepare(req)
        self._sockets.add(ws)
        utts: dict[str, _Utt] = {}
        current: _Utt | None = None
        lock = asyncio.Lock()
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    m = json.loads(msg.data)
                    ty = m.get("type")
                    if ty == "hello":
                        self.hellos.append(m)
                        await ws.send_json({"type": "ready", "fps": self.fps, "crop": self.crop,
                                            "frame_size": list(self.frame_size)})
                    elif ty == "utterance":
                        current = _Utt(m["id"], t_start=time.monotonic())
                        utts[current.id] = current
                        self.utterances.append(current)
                        current.task = asyncio.create_task(self._render(ws, current, lock))
                    elif ty == "end":
                        u = utts.get(m.get("id"))
                        if u:
                            u.ended = True
                    elif ty == "cancel":
                        u = utts.get(m.get("id"))
                        self.cancelled.append(m.get("id"))
                        if u:
                            u.cancelled = True
                            if u.task:
                                u.task.cancel()
                    elif ty == "ping":
                        self.pings += 1
                        await ws.send_json({"type": "pong"})
                elif msg.type == WSMsgType.BINARY and current is not None:
                    current.samples += len(msg.data) // 2
        finally:
            for u in utts.values():
                if u.task:
                    u.task.cancel()
            self._sockets.discard(ws)
        return ws

    async def _render(self, ws, u: _Utt, lock: asyncio.Lock) -> None:
        try:
            await asyncio.sleep(self.ttff)
            i = 0
            while True:
                have = int(u.samples / PCM_RATE * self.fps)
                total = int(np.ceil(u.samples / PCM_RATE * self.fps)) if u.ended else None
                if total is not None and i >= total:
                    break
                if i >= have and not (total is not None and i < total):
                    await asyncio.sleep(0.005)
                    continue
                due = u.t_start + self.ttff + i / (self.fps * self.rtf)
                wait = due - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
                if u.cancelled or ws.closed:
                    return
                async with lock:
                    await ws.send_bytes(pack_frame(i, u.id, self._jpeg(i)))
                u.sent += 1
                i += 1
            async with lock:
                await ws.send_json({"type": "done", "id": u.id, "frames": i})
        except (asyncio.CancelledError, ConnectionError):
            pass


async def _main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18799)
    ap.add_argument("--token", default="test-token")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    s = await FakeLipsyncServer(token=a.token, host=a.host, port=a.port).start()
    LOG.info("fake lipsync on %s", s.url)
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(_main())
