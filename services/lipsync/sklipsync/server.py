"""sklipsync WebSocket server (aiohttp). Protocol: see protocol.py.

Threads: one GPU worker (every engine.next_batch call goes through it, so
utterances from several calls interleave batch by batch), one avatar-prep
worker (landmarks and face parsing on CPU, so a new avatar does not stall
live calls), and a small CPU pool for blending and JPEG encoding.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from aiohttp import WSMsgType, web

from . import __version__
from .auth import client_for
from .avatars import AvatarNotFound, list_avatars, load_avatar
from .engines import AvatarHandle, Engine, EngineUnavailable, RenderStream
from .protocol import PATH, PCM_RATE, pack_frame

LOG = logging.getLogger("sklipsync")


@dataclass
class Utt:
    id: str
    stream: RenderStream
    handle: AvatarHandle
    engine: Engine
    fps: int
    t_start: float
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    t_first_pcm: float | None = None
    t_first_frame: float | None = None
    t_last_frame: float | None = None
    samples: int = 0
    carry: bytes = b""
    sent: int = 0
    ended: bool = False
    cancelled: bool = False
    task: asyncio.Task | None = None


class LipsyncServer:
    def __init__(self, engines: dict[str, Engine], avatars_dir: Path, tokens: dict[str, str] | str,
                 max_fps: int = 25, jpeg_quality: int = 85, cpu_workers: int = 4):
        """tokens maps bearer token to client name; a bare string is the single client "legacy"."""
        if isinstance(tokens, str):
            tokens = {tokens: "legacy"} if tokens else {}
        if not tokens:
            raise ValueError("at least one bearer token is required")
        self.engines = engines
        self.avatars_dir = Path(avatars_dir)
        self._tokens = dict(tokens)
        self.max_fps = max_fps
        self.jpeg_quality = jpeg_quality
        self.gpu = ThreadPoolExecutor(1, "sklipsync-gpu")
        self.prep = ThreadPoolExecutor(1, "sklipsync-prep")
        self.cpu = ThreadPoolExecutor(cpu_workers, "sklipsync-cpu")
        self.handles: dict[tuple[str, str], tuple[str, AvatarHandle]] = {}
        self._prep_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self.t0 = time.time()
        self.connections = 0
        self.active: dict[str, Utt] = {}
        self.stats = {"utterances": 0, "cancelled": 0, "frames": 0, "errors": 0}
        self.last: dict | None = None

    # ------------------------------------------------------------ avatars
    def prepare_sync(self, engine_name: str, avatar_id: str) -> AvatarHandle:
        """Blocking prepare (startup warm-up and the CLI)."""
        eng = self.engines[engine_name]
        avatar = load_avatar(self.avatars_dir, avatar_id)
        key = avatar.cache_key(eng.name)
        h = eng.prepare(avatar)
        self.handles[(engine_name, avatar_id)] = (key, h)
        return h

    async def get_handle(self, engine_name: str, avatar_id: str) -> AvatarHandle:
        eng = self.engines[engine_name]
        avatar = load_avatar(self.avatars_dir, avatar_id)
        key = avatar.cache_key(eng.name)
        k = (engine_name, avatar_id)
        hit = self.handles.get(k)
        if hit and hit[0] == key:
            return hit[1]
        lock = self._prep_locks.setdefault(k, asyncio.Lock())
        async with lock:
            hit = self.handles.get(k)
            if hit and hit[0] == key:
                return hit[1]
            LOG.info("preparing avatar %s for %s", avatar_id, engine_name)
            h = await asyncio.get_running_loop().run_in_executor(self.prep, eng.prepare, avatar)
            self.handles[k] = (key, h)
            return h

    # ------------------------------------------------------------ http
    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get(PATH, self.ws_handler)
        app.router.add_get("/v1/info", self.info)
        app.router.add_get("/healthz", self.healthz)
        app.router.add_get("/health", self.healthz)
        app.on_shutdown.append(self._on_shutdown)
        return app

    async def _on_shutdown(self, _app):
        for u in list(self.active.values()):
            self._cancel(u)

    async def info(self, _req: web.Request) -> web.Response:
        """Unauthenticated: lets a client confirm what it is talking to."""
        return web.json_response({"service": "lipsync", "interface": 1, "version": __version__})

    async def healthz(self, _req: web.Request) -> web.Response:
        avatars = {a: "available" for a in list_avatars(self.avatars_dir)}
        for (_e, a) in self.handles:
            avatars[a] = "ready"
        return web.json_response({
            "ok": True, "service": "sklipsync", "version": __version__,
            "engines": {n: e.info() for n, e in self.engines.items()},
            "avatars": avatars, "max_fps": self.max_fps,
            "connections": self.connections, "active_utterances": len(self.active),
            "uptime_s": round(time.time() - self.t0), "stats": self.stats, "last_utterance": self.last,
        })

    async def ws_handler(self, req: web.Request) -> web.StreamResponse:
        client = client_for(req.headers.get("Authorization"), self._tokens)
        if client is None:
            LOG.warning("rejected connection from %s: client=unknown (bad or missing token)", req.remote)
            return web.Response(status=401, text="bad token")
        LOG.info("accepted connection from %s: client=%s", req.remote, client)
        ws = web.WebSocketResponse(max_msg_size=8 * 1024 * 1024, heartbeat=30)
        await ws.prepare(req)
        self.connections += 1
        peer = req.remote
        send_lock = asyncio.Lock()
        sess: dict = {}
        utts: dict[str, Utt] = {}
        current: Utt | None = None

        async def send_json(obj):
            async with send_lock:
                if not ws.closed:
                    await ws.send_json(obj)

        async def fail(code, message, utt_id=None, close=False):
            self.stats["errors"] += 1
            LOG.warning("%s: %s: %s", peer, code, message)
            m = {"type": "error", "code": code, "message": message}
            if utt_id:
                m["id"] = utt_id
            await send_json(m)
            if close:
                await ws.close()

        try:
            async for msg in ws:
                if msg.type == WSMsgType.BINARY:
                    u = current
                    if u is None or u.ended or u.cancelled:
                        continue
                    data = u.carry + msg.data
                    n = len(data) // 2 * 2
                    u.carry = data[n:]
                    if n:
                        if u.t_first_pcm is None:
                            u.t_first_pcm = time.monotonic()
                        u.stream.push(np.frombuffer(data[:n], dtype="<i2"))
                        u.samples += n // 2
                        u.wake.set()
                    continue
                if msg.type != WSMsgType.TEXT:
                    if msg.type == WSMsgType.ERROR:
                        LOG.warning("%s: websocket error %s", peer, ws.exception())
                    break
                try:
                    m = json.loads(msg.data)
                    ty = m.get("type")
                except (ValueError, AttributeError):
                    await fail("bad_message", "expected a JSON object")
                    continue
                if ty == "ping":
                    await send_json({"type": "pong"})
                elif ty == "pong":
                    pass
                elif ty == "hello":
                    name = m.get("engine") or "musetalk"
                    if name not in self.engines:
                        reason = f"engine {name!r} is not enabled on this server"
                        if name == "ditto":
                            from .engines.ditto import DittoEngine
                            try:
                                DittoEngine().load()
                            except EngineUnavailable as e:
                                reason = str(e)
                        await fail("engine_unavailable", reason, close=True)
                        break
                    try:
                        fps = int(m.get("fps") or 25)
                    except (TypeError, ValueError):
                        fps = 25
                    fps = max(5, min(fps, self.max_fps))
                    avatar_id = str(m.get("avatar") or "")
                    try:
                        handle = await self.get_handle(name, avatar_id)
                    except AvatarNotFound as e:
                        await fail("avatar_not_found", str(e), close=True)
                        break
                    except Exception as e:
                        LOG.exception("avatar %s prepare failed", avatar_id)
                        await fail("avatar_prepare_failed", f"{avatar_id}: {e}", close=True)
                        break
                    sess = {"engine": self.engines[name], "handle": handle, "fps": fps, "avatar": avatar_id}
                    await send_json({"type": "ready", "fps": fps, "crop": handle.crop_json(),
                                     "frame_size": list(handle.frame_size)})
                    LOG.info("%s: hello avatar=%s engine=%s fps=%d crop=%s", peer, avatar_id, name, fps, handle.crop)
                elif ty == "utterance":
                    uid = m.get("id")
                    if not sess:
                        await fail("no_hello", "send hello first", uid)
                        continue
                    if not isinstance(uid, str) or not uid.isascii() or not 0 < len(uid) <= 128 or uid in utts:
                        await fail("bad_id", f"bad or duplicate utterance id {uid!r}", uid if isinstance(uid, str) else None)
                        continue
                    if current is not None and not current.ended:
                        self._end(current)       # a new utterance implies the previous one ended
                    u = Utt(uid, sess["engine"].open_stream(sess["handle"], sess["fps"]), sess["handle"],
                            sess["engine"], sess["fps"], time.monotonic())
                    utts[uid] = current = u
                    self.active[f"{id(ws)}:{uid}"] = u
                    self.stats["utterances"] += 1
                    u.task = asyncio.create_task(self._render(ws, send_lock, u, sess["avatar"]))
                elif ty == "end":
                    u = utts.get(m.get("id"))
                    if u is not None:
                        self._end(u)
                elif ty == "cancel":
                    u = utts.get(m.get("id"))
                    if u is not None and not u.cancelled:
                        self._cancel(u)
                        LOG.info("utt %s cancelled after %d frames", u.id[:8], u.sent)
                        await send_json({"type": "done", "id": u.id, "frames": u.sent, "cancelled": True})
                else:
                    await fail("bad_message", f"unknown type {ty!r}", m.get("id"))
        finally:
            for u in utts.values():
                self._cancel(u)
                self.active.pop(f"{id(ws)}:{u.id}", None)
            self.connections -= 1
            LOG.info("%s: connection closed client=%s (%d utterances)", peer, client, len(utts))
        return ws

    # ------------------------------------------------------------ utterances
    def _end(self, u: Utt) -> None:
        u.ended = True
        u.stream.end()
        u.wake.set()

    def _cancel(self, u: Utt) -> None:
        if not u.cancelled:
            u.cancelled = True
            self.stats["cancelled"] += 1
        u.wake.set()
        if u.task is not None and not u.task.done():
            u.task.cancel()

    def _encode(self, u: Utt, batch) -> list[tuple[int, bytes]]:
        out = []
        q = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        for idx, raw in batch:
            if u.cancelled:
                break
            ok, buf = cv2.imencode(".jpg", u.engine.compose(u.handle, raw), q)
            if ok:
                out.append((idx, buf.tobytes()))
        return out

    async def _send(self, ws, send_lock, u: Utt, enc, prev) -> None:
        frames = await enc
        if prev is not None:
            await prev
        for idx, jpeg in frames:
            if u.cancelled or ws.closed:
                return
            async with send_lock:
                await ws.send_bytes(pack_frame(idx, u.id, jpeg))
            now = time.monotonic()
            if u.t_first_frame is None:
                u.t_first_frame = now
            u.t_last_frame = now
            u.sent += 1
            self.stats["frames"] += 1

    async def _render(self, ws, send_lock, u: Utt, avatar: str) -> None:
        loop = asyncio.get_running_loop()
        prev = None
        key = f"{id(ws)}:{u.id}"
        try:
            while not u.cancelled:
                u.wake.clear()
                batch = await loop.run_in_executor(self.gpu, u.stream.next_batch)
                if u.cancelled:
                    return
                if batch:
                    enc = loop.run_in_executor(self.cpu, self._encode, u, batch)
                    prev = asyncio.create_task(self._send(ws, send_lock, u, enc, prev))
                    continue
                if u.stream.finished:
                    break
                await u.wake.wait()
            if prev is not None:
                await prev
            if u.cancelled or ws.closed:
                return
            total = u.stream.total_frames or u.sent
            async with send_lock:
                await ws.send_json({"type": "done", "id": u.id, "frames": total})
            self._log_done(u, avatar)
        except asyncio.CancelledError:
            if prev is not None:
                prev.cancel()
            raise
        except Exception as e:
            LOG.exception("utt %s render failed", u.id[:8])
            self.stats["errors"] += 1
            if not ws.closed:
                try:
                    async with send_lock:
                        await ws.send_json({"type": "error", "code": "render_failed", "message": str(e), "id": u.id})
                except Exception:
                    pass
        finally:
            self.active.pop(key, None)

    def _log_done(self, u: Utt, avatar: str) -> None:
        audio_s = u.samples / PCM_RATE
        t_pcm = u.t_first_pcm or u.t_start
        ttff = (u.t_first_frame - t_pcm) * 1000 if u.t_first_frame else None
        wall = (u.t_last_frame - t_pcm) if u.t_last_frame else 0.0
        fps = u.sent / wall if wall > 0 else None
        self.last = {"id": u.id, "avatar": avatar, "audio_s": round(audio_s, 2), "frames": u.sent,
                     "ttff_ms": round(ttff) if ttff is not None else None,
                     "render_fps": round(fps, 1) if fps else None, "wall_s": round(wall, 2),
                     "realtime_factor": round(audio_s / wall, 2) if wall > 0 else None}
        LOG.info("utt %s avatar=%s audio=%.2fs frames=%d ttff=%sms fps=%s wall=%.2fs rtf=%s", u.id[:8], avatar,
                 audio_s, u.sent, self.last["ttff_ms"], self.last["render_fps"], wall, self.last["realtime_factor"])
