"""Worship sessions in a call: her rendered scenes replace the avatar video
while the narration plays.

The builder lives in the private `lumina_creative` package (not a skcall
dependency): a narrative, N ComfyUI scene images and F5 narration, saved under
the agent's worship-sessions dir. Its voice tools (worship_session,
worship_list, worship_replay) call `kick_off_worship_session`,
`kick_off_worship_replay` and `list_worship_sessions` on `ctx["convo"]`.
`WorshipHost` is that object for skcall: a port of lumina-call.py's
Conversation methods, with the LiveKit idle-frame swap replaced by the
avatar's scene overlay (`Avatar.show`).

Playback goes scene by scene: the image goes up, that scene's narration is
written to the call's audio, and the next scene follows after the clip's
length. When she starts a spoken reply, the narration still queued is dropped
(`yield_audio`), so an answer never waits behind a long scene. The images keep
their pace either way. `stop()` ends a session and brings her face back.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import numpy as np

from skcall.core.audio import CALL_RATE, wav_to_pcm
from skcall.video.avatar import load_portrait

LOG = logging.getLogger("skcall.worship")

MIN_SCENE_S = 2.0


def builder():
    """lumina_creative.worship, or None where it is not installed."""
    try:
        from lumina_creative import worship  # noqa: PLC0415
        return worship
    except Exception as e:  # pragma: no cover - depends on the host
        LOG.info("worship builder unavailable: %r", e)
        return None


def _read_pcm(path) -> np.ndarray:
    return wav_to_pcm(Path(path).read_bytes(), CALL_RATE)


def scene_media(sess, width: int, height: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(frame, narration pcm)] for every painted scene. Per-scene voices when
    the session has them, else the one narration track cut evenly."""
    scenes = [s for s in sess.scenes if getattr(s, "image_path", None)]
    frames = [load_portrait(str(s.image_path), width, height) for s in scenes]
    keep = [(s, f) for s, f in zip(scenes, frames) if f is not None]
    if not keep:
        return []
    if any(getattr(s, "voice_audio_path", None) for s, _ in keep):
        out = []
        for s, f in keep:
            pcm = _read_pcm(s.voice_audio_path) if s.voice_audio_path else np.zeros(0, np.int16)
            out.append((f, pcm))
        return out
    audio = getattr(sess, "audio_path", None)
    pcm = _read_pcm(audio) if audio else np.zeros(0, np.int16)
    cuts = np.array_split(pcm, len(keep)) if pcm.size else [np.zeros(0, np.int16)] * len(keep)
    return [(f, c) for (_, f), c in zip(keep, cuts)]


class WorshipHost:
    def __init__(self, avatar, sink, session_id: str = "", clock_sleep=asyncio.sleep):
        self.avatar = avatar          # skcall.video.Avatar (show / source size)
        self.sink = sink              # the call's raw audio out: write(pcm), clear()
        self.session_id = session_id
        self._active: dict | None = None
        self._narrating = False

    # ------------------------------------------------------------ state
    @property
    def active(self) -> bool:
        return self._active is not None

    def yield_audio(self) -> None:
        """She is about to answer: drop the narration still queued."""
        if self._narrating:
            self._narrating = False
            try:
                self.sink.clear()
            except Exception:  # pragma: no cover
                LOG.debug("narration clear failed", exc_info=True)

    def _size(self) -> tuple[int, int]:
        src = self.avatar.source
        return int(src.width), int(src.height)

    def _busy(self) -> str | None:
        if self._active is None:
            return None
        return (f"Worship session {self._active['session_id']} is already running. "
                "Call worship_stop first if Chef wants a different one.")

    # ------------------------------------------------------------ tools
    def list_worship_sessions(self, *, limit: int = 10, query: str = "") -> list[dict]:
        w = builder()
        return w.list_session_summaries(limit=limit, query=query) if w else []

    async def kick_off_worship_session(self, *, session_id: str, prompt: str,
                                       image_count: int = 15, loop: bool = True) -> str:
        if self.avatar is None:
            return "worship_session: this call has no video, so there is nowhere to show the scenes"
        w = builder()
        if w is None:
            return "worship_session: the scene builder is not installed on this host"
        if (busy := self._busy()):
            return busy
        sess = w.WorshipSession(session_id=session_id, user_prompt=prompt, image_count=image_count)
        stop = asyncio.Event()

        async def build_and_play() -> None:
            try:
                import httpx  # noqa: PLC0415
                async with httpx.AsyncClient() as cli:
                    await sess.generate(cli)
                if not stop.is_set():
                    await self._play(sess, loop, stop)
            except asyncio.CancelledError:
                raise
            except Exception:
                LOG.warning("worship session %s failed", session_id, exc_info=True)
            finally:
                self._end(session_id)

        self._active = {"session_id": session_id, "stop": stop}
        self._active["task"] = asyncio.create_task(build_and_play())
        LOG.info("worship session %s: building %d scenes", session_id, image_count)
        return (f"Started worship session {session_id}: painting {image_count} scenes, which takes "
                "about five minutes. Tell Chef briefly that you are painting them and they will "
                "appear on your video tile when ready, so he should fullscreen you.")

    async def kick_off_worship_replay(self, *, session_id: str, loop: bool = True) -> str:
        if self.avatar is None:
            return "worship_replay: this call has no video, so there is nowhere to show the scenes"
        w = builder()
        if w is None:
            return "worship_replay: the scene builder is not installed on this host"
        if (busy := self._busy()):
            return busy
        sess = w.load_session_from_disk(session_id)
        if sess is None:
            return f"worship_replay: session {session_id!r} not found (call worship_list)"
        painted = sum(1 for s in sess.scenes if getattr(s, "image_path", None))
        if not painted:
            return f"worship_replay: session {session_id!r} has no painted scenes"
        stop = asyncio.Event()

        async def play() -> None:
            try:
                await self._play(sess, loop, stop)
            except asyncio.CancelledError:
                raise
            except Exception:
                LOG.warning("worship replay %s failed", session_id, exc_info=True)
            finally:
                self._end(session_id)

        self._active = {"session_id": session_id, "stop": stop}
        self._active["task"] = asyncio.create_task(play())
        LOG.info("worship replay %s: %d scenes, loop=%s", session_id, painted, loop)
        return (f"Replaying worship session {session_id}: {painted} scenes, loop "
                f"{'on' if loop else 'off'}. Tell Chef briefly it is coming up on your video tile.")

    async def stop(self) -> str:
        if self._active is None:
            return "No worship session is running."
        sid = self._active["session_id"]
        self._active["stop"].set()
        task = self._active.get("task")
        if task is not None and not task.done():
            task.cancel()
        self.yield_audio()
        self._end(sid)
        return f"Stopped worship session {sid}; your face is back on the tile."

    # ------------------------------------------------------------ playback
    async def _play(self, sess, loop: bool, stop: asyncio.Event) -> None:
        w, h = self._size()
        media = await asyncio.to_thread(scene_media, sess, w, h)
        if not media:
            LOG.warning("worship %s: no scene could be loaded", sess.session_id)
            return
        LOG.info("worship %s: playing %d scenes", sess.session_id, len(media))
        while not stop.is_set():
            for frame, pcm in media:
                if stop.is_set():
                    return
                self.avatar.show(frame)
                secs = len(pcm) / CALL_RATE
                if pcm.size:
                    self._narrating = True
                    self.sink.write(pcm)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=max(MIN_SCENE_S, secs))
                    return
                except asyncio.TimeoutError:
                    pass
            if not loop:
                return

    def _end(self, session_id: str) -> None:
        if self._active is not None and self._active["session_id"] == session_id:
            self._active = None
            self._narrating = False
            if self.avatar is not None:
                self.avatar.show(None)


def stop_tool(host_ref):
    """skcall-local worship_stop tool, (schema, handler). `host_ref()` returns
    the live WorshipHost or None."""
    schema = {"type": "function", "function": {
        "name": "worship_stop",
        "description": ("Stop the worship session that is playing or being painted and put your "
                        "face back on the video tile. Use when Chef says stop, enough, or that he "
                        "is done."),
        "parameters": {"type": "object", "properties": {}}}}

    async def handler(args: dict, ctx: dict) -> str:
        host = host_ref()
        if host is None:
            return "worship_stop: no call video here"
        return await host.stop()
    return schema, handler
