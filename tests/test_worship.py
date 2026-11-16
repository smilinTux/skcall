"""WorshipHost: scenes replace the avatar video while the narration plays."""
import asyncio
import io
import sys
import types
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from skcall import worship as W
from skcall.core.audio import CALL_RATE
from skcall.video import SceneOverlay


def _png(path: Path, color) -> Path:
    import cv2
    cv2.imwrite(str(path), np.full((1216, 832, 3), color, np.uint8))
    return path


def _wav(path: Path, secs: float, sr: int = 24000) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.ones(int(secs * sr)) * 1000).astype("<i2").tobytes())
    return path


@dataclass
class Scene:
    idx: int
    image_path: Path | None = None
    voice_audio_path: Path | None = None


@dataclass
class Sess:
    session_id: str
    scenes: list = field(default_factory=list)
    audio_path: Path | None = None


class FakeSource:
    width, height, fps = 56, 72, 25.0

    def frame(self, t):
        return np.zeros((72, 56, 3), np.uint8)


class FakeAvatar:
    def __init__(self):
        self.source = SceneOverlay(FakeSource())
        self.shown = []

    def show(self, frame):
        self.source.scene = frame
        self.shown.append(None if frame is None else int(frame[0, 0, 0]))


class FakeSink:
    def __init__(self):
        self.writes, self.clears = [], 0

    def write(self, pcm, on_start=None):
        self.writes.append(len(pcm))

    def clear(self):
        self.clears += 1


def _session(tmp_path, per_scene=True):
    scenes = []
    for i, c in enumerate((10, 20, 30)):
        s = Scene(i, _png(tmp_path / f"s{i}.png", c))
        if per_scene:
            s.voice_audio_path = _wav(tmp_path / f"v{i}.wav", 0.05)
        scenes.append(s)
    audio = None if per_scene else _wav(tmp_path / "audio.wav", 0.3)
    return Sess("ws_test", scenes, audio)


def _fake_builder(monkeypatch, sess):
    mod = types.SimpleNamespace(
        load_session_from_disk=lambda sid: sess if sid == sess.session_id else None,
        list_session_summaries=lambda limit=10, query="": [{"session_id": sess.session_id}])
    monkeypatch.setattr(W, "builder", lambda: mod)


def test_overlay_shows_scene_then_face():
    ov = SceneOverlay(FakeSource())
    assert ov.width == 56 and ov.fps == 25.0
    ov.scene = np.full((72, 56, 3), 7, np.uint8)
    assert ov.frame(0)[0, 0, 0] == 7
    ov.scene = None
    assert ov.frame(0)[0, 0, 0] == 0


def test_scene_media_per_scene_and_split(tmp_path):
    m = W.scene_media(_session(tmp_path), 56, 72)
    assert len(m) == 3 and m[0][0].shape == (72, 56, 3)
    assert abs(len(m[0][1]) / CALL_RATE - 0.05) < 0.01
    m = W.scene_media(_session(tmp_path, per_scene=False), 56, 72)
    assert len(m) == 3 and sum(len(p) for _, p in m) == int(0.3 * CALL_RATE)


def test_replay_plays_every_scene_then_restores_face(tmp_path, monkeypatch):
    sess = _session(tmp_path)
    _fake_builder(monkeypatch, sess)
    monkeypatch.setattr(W, "MIN_SCENE_S", 0.01)
    av, sink = FakeAvatar(), FakeSink()
    host = W.WorshipHost(av, sink)

    async def go():
        msg = await host.kick_off_worship_replay(session_id="ws_test", loop=False)
        assert "Replaying" in msg and host.active
        await host._active["task"]
    asyncio.run(go())
    assert av.shown[:3] == [10, 20, 30] and av.shown[-1] is None
    assert len(sink.writes) == 3 and not host.active


def test_stop_ends_a_looping_replay(tmp_path, monkeypatch):
    sess = _session(tmp_path)
    _fake_builder(monkeypatch, sess)
    monkeypatch.setattr(W, "MIN_SCENE_S", 0.01)
    av, sink = FakeAvatar(), FakeSink()
    host = W.WorshipHost(av, sink)

    async def go():
        await host.kick_off_worship_replay(session_id="ws_test", loop=True)
        await asyncio.sleep(0.2)
        assert "Stopped" in await host.stop()
        await asyncio.sleep(0.05)
    asyncio.run(go())
    assert not host.active and av.source.scene is None and sink.clears >= 1


def test_second_session_refused_while_one_runs(tmp_path, monkeypatch):
    sess = _session(tmp_path)
    _fake_builder(monkeypatch, sess)
    host = W.WorshipHost(FakeAvatar(), FakeSink())

    async def go():
        await host.kick_off_worship_replay(session_id="ws_test")
        assert "already running" in await host.kick_off_worship_replay(session_id="ws_test")
        await host.stop()
    asyncio.run(go())


def test_yield_audio_clears_only_while_narrating():
    sink = FakeSink()
    host = W.WorshipHost(FakeAvatar(), sink)
    host.yield_audio()
    assert sink.clears == 0
    host._narrating = True
    host.yield_audio()
    assert sink.clears == 1 and not host._narrating


def test_unknown_session_and_no_video(tmp_path, monkeypatch):
    _fake_builder(monkeypatch, _session(tmp_path))
    assert "not found" in asyncio.run(
        W.WorshipHost(FakeAvatar(), FakeSink()).kick_off_worship_replay(session_id="nope"))
    assert "no video" in asyncio.run(
        W.WorshipHost(None, FakeSink()).kick_off_worship_replay(session_id="ws_test"))


def test_stop_tool_handler():
    schema, handler = W.stop_tool(lambda: None)
    assert schema["function"]["name"] == "worship_stop"
    assert "no call video" in asyncio.run(handler({}, {}))
