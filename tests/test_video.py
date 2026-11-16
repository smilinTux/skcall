"""P5 video tests: portrait, idle loop, compositor timing, lip-sync client and
audio hold against a fake service, barge-in cancel, fallbacks, frame pacing.
No network beyond 127.0.0.1."""
import asyncio
import time

import numpy as np
import pytest

from skcall.core.session import FakeSink
from skcall.video import Avatar, VideoConfig
from skcall.video.avatar import IdleLoop, StaticPortrait, load_portrait
from skcall.video.compositor import Compositor, LipsyncSink, _Play, feather_mask
from skcall.video.fake_lipsync import FakeLipsyncServer, crop_value, flat_jpeg
from skcall.video.lipsync import Crop, LipsyncClient, pack_frame, unpack_frame

cv2 = pytest.importorskip("cv2")

W, H, FPS = 560, 720, 25
CROP = Crop(200, 330, 160, 120)


def portrait(value=100):
    img = np.full((H, W, 3), value, np.uint8)
    img[100:200, 100:200] = 220          # some structure for warp tests
    return img


def probe(frame, box=CROP):
    """Mean grey in the lower middle of the crop box (where the mask is 1)."""
    x, y, w, h = box.x, box.y, box.w, box.h
    return float(frame[y + int(h * 0.65):y + int(h * 0.8), x + w // 3:x + 2 * w // 3].mean())


def frames_for(n, size=(128, 96)):
    return {i: flat_jpeg(*size, crop_value(i)) for i in range(n)}


class PacedSink:
    """Real-time AudioSink stand-in: chunks play back to back on the monotonic
    clock and on_start fires when a chunk actually starts."""
    sample_rate = 48000

    def __init__(self):
        self._end = 0.0
        self._handles = []
        self.starts = []

    def write(self, pcm, on_start=None):
        loop = asyncio.get_running_loop()
        now = time.monotonic()
        start = max(now, self._end)
        self._end = start + len(pcm) / self.sample_rate

        def fire(t=start):
            self.starts.append(t)
            if on_start:
                on_start(t)
        self._handles.append(loop.call_later(max(0.0, start - now), fire))

    def clear(self):
        for h in self._handles:
            h.cancel()
        self._handles.clear()
        self._end = time.monotonic()

    @property
    def buffered_s(self):
        return max(0.0, self._end - time.monotonic())

    async def drained(self):
        while self.buffered_s > 0:
            await asyncio.sleep(0.01)


async def connected_client(server, **kw):
    c = LipsyncClient(server.url, server.token, avatar="lumina", fps=FPS, **kw)
    c.start()
    assert await c.wait_ready(3)
    return c


# ---------------------------------------------------------------- portrait / idle

def test_load_portrait_crops_to_aspect_and_even_size(tmp_path):
    img = np.zeros((1000, 1000, 3), np.uint8)
    img[:, :, 2] = 255                                   # red in BGR
    p = tmp_path / "p.png"
    cv2.imwrite(str(p), img)
    out = load_portrait(str(p), 561, 721)
    assert out.shape == (720, 560, 3)
    assert out[..., 0].mean() > 250                      # RGB order: red first
    assert load_portrait(str(tmp_path / "missing.png")) is None


def test_static_portrait_returns_same_frame():
    s = StaticPortrait(portrait(), fps=5)
    assert s.frame(1.0) is s.frame(2.0)
    assert (s.width, s.height, s.fps) == (W, H, 5)


def test_idle_loop_breathes_and_blinks():
    img = portrait()
    eyes = [(0.3, 0.35, 0.09, 0.03), (0.52, 0.35, 0.09, 0.03)]
    idle = IdleLoop(img, eyes, fps=FPS)
    assert idle.can_blink
    a, b = idle.frame(100.0), idle.frame(102.3)          # half a breath apart
    assert a.shape == (H, W, 3) and a.dtype == np.uint8
    assert np.abs(a.astype(int) - b.astype(int)).mean() > 0.1
    # find a blink within the first 10 s and check the base changes there
    bases = {id(idle.base(100.0 + k / FPS)) for k in range(250)}
    assert len(bases) >= 2
    assert IdleLoop(img, eyes).frame(5.0).tobytes() == IdleLoop(img, eyes).frame(5.0).tobytes()


def test_feather_mask_shape():
    m = feather_mask(160, 120, top_frac=0.35)
    assert m.shape == (120, 160)
    assert m[0].max() == 0 and m[:, 0].max() < 0.1 and m[-1].max() < 0.1
    assert m[int(120 * 0.7), 80] == pytest.approx(1.0)
    assert m[int(120 * 0.2), 80] == 0.0                  # eyes untouched


# ---------------------------------------------------------------- compositor

def test_compositor_shows_frame_for_audio_time():
    comp = Compositor(portrait(), None, fps=FPS)
    comp.set_crop(CROP, (W, H))
    ended = []
    comp.schedule(_Play("u1", frames_for(25), 1.0, t_start=10.0, on_end=lambda: ended.append(1)))
    assert probe(comp.frame(9.9)) == pytest.approx(100, abs=1)        # before playout: portrait
    for k in (0, 1, 7, 24):
        v = probe(comp.frame(10.0 + k / FPS + 0.01))
        assert v == pytest.approx(crop_value(k), abs=3), k
    assert comp.stats.lips == 4 and comp.stats.late == 0
    assert probe(comp.frame(11.2)) == pytest.approx(100, abs=1)       # after: idle mouth
    comp.frame(11.6)
    assert ended == [1] and not comp.speaking


def test_compositor_av_offset_delays_mouth():
    comp = Compositor(portrait(), None, fps=FPS, av_offset_s=0.08)
    comp.set_crop(CROP, (W, H))
    comp.schedule(_Play("u1", frames_for(25), 1.0, t_start=10.0))
    assert probe(comp.frame(10.0 + 5 / FPS + 0.01)) == pytest.approx(crop_value(3), abs=3)


def test_compositor_scales_crop_from_service_frame_size():
    comp = Compositor(portrait(), None, fps=FPS)
    comp.set_crop(Crop(100, 165, 80, 60), (280, 360))                # half-size service frame
    assert comp.box == (200, 330, 160, 120)


def test_compositor_late_frames_hold_then_idle():
    comp = Compositor(portrait(), None, fps=FPS, hold_frames=2)
    comp.set_crop(CROP, (W, H))
    fr = frames_for(3)                                                 # only frames 0..2 arrived
    comp.schedule(_Play("u1", fr, 1.0, t_start=10.0))
    for k in range(3):
        comp.frame(10.0 + k / FPS + 0.01)
    assert probe(comp.frame(10.0 + 4 / FPS + 0.01)) == pytest.approx(crop_value(2), abs=3)  # held
    assert probe(comp.frame(10.0 + 6 / FPS + 0.01)) == pytest.approx(100, abs=1)            # idle
    assert comp.stats.held == 1 and comp.stats.late == 1
    fr[6] = flat_jpeg(128, 96, crop_value(6))                          # a late arrival still lands
    assert probe(comp.frame(10.0 + 6 / FPS + 0.02)) == pytest.approx(crop_value(6), abs=3)


def test_compositor_falls_back_to_portrait_when_idle_fails():
    img = portrait()

    class Broken(IdleLoop):
        def render(self, base, t):
            raise RuntimeError("boom")
    comp = Compositor(img, Broken(img), fps=FPS)
    out = comp.frame(1.0)
    assert out is img and comp.idle is None and comp.stats.errors == 1
    assert comp.frame(1.04) is img


def test_compositor_flush_drops_crops():
    comp = Compositor(portrait(), None, fps=FPS)
    comp.set_crop(CROP, (W, H))
    comp.schedule(_Play("u1", frames_for(25), 1.0, t_start=10.0))
    assert probe(comp.frame(10.1)) != pytest.approx(100, abs=1)
    comp.flush()
    assert probe(comp.frame(10.14)) == pytest.approx(100, abs=1)


# ---------------------------------------------------------------- client + fake service

def test_frame_wire_format_roundtrip():
    data = pack_frame(513, "abc-123", b"\xff\xd8jpeg")
    assert data[:4] == (513).to_bytes(4, "big") and data[4:6] == (7).to_bytes(2, "big")
    assert unpack_frame(data) == (513, "abc-123", b"\xff\xd8jpeg")
    with pytest.raises(ValueError):
        unpack_frame(b"\x00\x00")


async def test_client_handshake_and_full_utterance():
    srv = await FakeLipsyncServer().start()
    c = await connected_client(srv)
    try:
        assert c.crop == Crop(200, 330, 160, 120) and c.frame_size == (560, 720)
        assert srv.hellos == [{"type": "hello", "avatar": "lumina", "fps": 25, "engine": "musetalk"}]
        u = c.submit(np.zeros(16000, np.int16))
        await asyncio.wait_for(u.done.wait(), 3)
        assert u.expected == 25 and sorted(u.frames) == list(range(25))
        img = cv2.imdecode(np.frombuffer(u.frames[7], np.uint8), cv2.IMREAD_GRAYSCALE)
        assert abs(int(img.mean()) - crop_value(7)) <= 2
        assert srv.utterances[0].samples == 16000
    finally:
        await c.aclose()
        await srv.stop()


async def test_client_bad_token_is_unavailable():
    srv = await FakeLipsyncServer().start()
    c = LipsyncClient(srv.url, "wrong", retry_s=(5, 5))
    c.start()
    try:
        assert not await c.wait_ready(0.5)
        assert not c.available and c.submit(np.zeros(1600, np.int16)) is None
    finally:
        await c.aclose()
        await srv.stop()


async def test_client_answers_and_sends_pings():
    srv = await FakeLipsyncServer().start()
    c = await connected_client(srv, ping_s=0.05)
    try:
        await asyncio.sleep(0.2)
        assert srv.pings >= 2 and c.available
    finally:
        await c.aclose()
        await srv.stop()


# ---------------------------------------------------------------- audio hold, sync, cancel

def build(srv_client, hold_s=0.3, inner=None):
    comp = Compositor(portrait(), None, fps=FPS)
    if srv_client is not None and srv_client.crop:
        comp.set_crop(srv_client.crop, srv_client.frame_size)
    else:
        comp.set_crop(CROP, (W, H))
    sink = LipsyncSink(inner or PacedSink(), comp, srv_client, hold_s=hold_s)
    return comp, sink


def speech(seconds):
    return (np.sin(np.arange(int(48000 * seconds)) / 5) * 3000).astype(np.int16)


async def test_first_sentence_held_until_frames_then_in_sync():
    srv = await FakeLipsyncServer(ttff=0.12, rtf=2.0).start()
    c = await connected_client(srv)
    comp, sink = build(c)
    try:
        t_write = time.monotonic()
        starts = []
        sink.write(speech(1.2), on_start=starts.append)
        assert sink.buffered_s == pytest.approx(1.2, abs=0.01)         # she counts as audible while held
        while not starts:
            await asyncio.sleep(0.005)
        held = starts[0] - t_write
        assert 0.12 <= held < 0.3                                       # waited for frames, not the cap
        # sample the compositor at real playout times: every frame is the right one
        seen = []
        for k in range(0, 28, 3):
            due = starts[0] + k / FPS + 0.02
            await asyncio.sleep(max(0, due - time.monotonic()))
            seen.append((k, probe(comp.frame(due))))
        for k, v in seen:
            assert v == pytest.approx(crop_value(k), abs=3), (k, v)
        assert comp.stats.late == 0
        await sink.drained()
    finally:
        await c.aclose()
        await srv.stop()


async def test_hold_is_capped_when_service_is_slow():
    srv = await FakeLipsyncServer(ttff=1.0).start()
    c = await connected_client(srv)
    comp, sink = build(c, hold_s=0.3)
    try:
        t_write = time.monotonic()
        starts = []
        sink.write(speech(0.5), on_start=starts.append)
        while not starts:
            await asyncio.sleep(0.005)
        assert 0.28 <= starts[0] - t_write < 0.4
        assert sink.stats["held_timeout"] == 1
        # audio plays, the mouth stays idle (never frozen) until frames come
        assert probe(comp.frame(starts[0] + 0.1)) == pytest.approx(100, abs=1)
    finally:
        await c.aclose()
        await srv.stop()


async def test_later_sentences_are_not_held():
    srv = await FakeLipsyncServer(ttff=0.1).start()
    c = await connected_client(srv)
    comp, sink = build(c)
    try:
        starts = []
        sink.write(speech(0.6), on_start=starts.append)
        t2 = time.monotonic()
        sink.write(speech(0.6), on_start=starts.append)
        while len(starts) < 2:
            await asyncio.sleep(0.005)
        # the second chunk starts right when the first ends: no extra hold
        assert starts[1] - starts[0] == pytest.approx(0.6, abs=0.03)
        assert sink.stats["held_ok"] == 1 and sink.stats.get("held_timeout", 0) == 0
        assert t2 < starts[0]
    finally:
        await c.aclose()
        await srv.stop()


async def test_barge_in_cancels_utterance_and_drops_crops():
    srv = await FakeLipsyncServer(ttff=0.05, rtf=1.0).start()
    c = await connected_client(srv)
    comp, sink = build(c)
    try:
        starts = []
        sink.write(speech(3.0), on_start=starts.append)
        sink.write(speech(2.0))
        while not starts:
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.3)
        assert comp.speaking and probe(comp.frame(time.monotonic())) != pytest.approx(100, abs=1)
        live_ids = [u.id for u in srv.utterances]
        sink.clear()                                                   # barge-in
        assert not comp.speaking and sink.buffered_s == 0
        assert probe(comp.frame(time.monotonic())) == pytest.approx(100, abs=1)
        assert not c.utts
        await asyncio.sleep(0.15)
        assert set(srv.cancelled) == set(live_ids)
        sent = [u.sent for u in srv.utterances]
        await asyncio.sleep(0.2)
        assert [u.sent for u in srv.utterances] == sent                # rendering stopped
    finally:
        await c.aclose()
        await srv.stop()


async def test_service_down_plays_audio_at_once_over_idle():
    c = LipsyncClient("ws://127.0.0.1:9/v1/lipsync", "x", retry_s=(5, 5))
    c.start()
    comp, sink = build(c)
    try:
        t_write = time.monotonic()
        starts = []
        sink.write(speech(0.4), on_start=starts.append)
        while not starts:
            await asyncio.sleep(0.005)
        assert starts[0] - t_write < 0.05                              # no hold without a service
        assert sink.stats["no_service"] == 1
        assert comp.frame(starts[0] + 0.1).shape == (H, W, 3)
    finally:
        await c.aclose()


async def test_service_dying_mid_call_keeps_video_running():
    srv = await FakeLipsyncServer(ttff=0.05, rtf=1.0).start()
    c = await connected_client(srv, retry_s=(5, 5))
    comp, sink = build(c)
    try:
        starts = []
        sink.write(speech(2.0), on_start=starts.append)
        while not starts:
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.2)
        await srv.kill_connections()
        await asyncio.sleep(0.1)
        assert not c.available
        # frames keep coming: held crop then idle mouth, never an exception
        for k in range(20):
            assert comp.frame(time.monotonic() + k / FPS).shape == (H, W, 3)
        assert comp.stats.late > 0
        starts2 = []
        sink.write(speech(0.3), on_start=starts2.append)
        await asyncio.sleep(0.05)
        await sink.drained()
        assert sink.stats["no_service"] == 1
    finally:
        await c.aclose()
        await srv.stop()


async def test_fillers_prerendered_play_without_hold_or_service():
    srv = await FakeLipsyncServer(ttff=0.2).start()
    c = await connected_client(srv)
    comp, sink = build(c)
    try:
        filler = speech(0.5)
        assert await sink.prerender([filler]) == 1
        n_utts = len(srv.utterances)
        t_write = time.monotonic()
        starts = []
        sink.write(filler.copy(), on_start=starts.append)
        while not starts:
            await asyncio.sleep(0.005)
        assert starts[0] - t_write < 0.05
        assert sink.stats["cached"] == 1 and len(srv.utterances) == n_utts
        assert probe(comp.frame(starts[0] + 3 / FPS + 0.01)) == pytest.approx(crop_value(3), abs=3)
    finally:
        await c.aclose()
        await srv.stop()


async def test_drained_waits_for_held_audio():
    srv = await FakeLipsyncServer(ttff=0.1).start()
    c = await connected_client(srv)
    inner = FakeSink(hold=False)
    comp, sink = build(c, inner=inner)
    try:
        sink.write(speech(0.2))
        assert not inner.written                                       # still held for frames
        await asyncio.wait_for(sink.drained(), 2)
        assert len(inner.written) == 1
    finally:
        await c.aclose()
        await srv.stop()


# ---------------------------------------------------------------- backend track pacing

async def test_avatar_track_paces_25fps():
    from skcall.talk.backend import AvatarTrack
    comp = Compositor(portrait(), IdleLoop(portrait()), fps=FPS)
    tr = AvatarTrack(comp)
    t0 = time.monotonic()
    pts = []
    for _ in range(26):
        f = await tr.recv()
        pts.append(f.pts)
        assert (f.width, f.height) == (W, H)
    took = time.monotonic() - t0
    assert took == pytest.approx(1.0, abs=0.08)
    assert set(np.diff(pts)) == {3600}                                 # 90 kHz / 25
    tr.stop()


async def test_avatar_track_static_converts_once():
    from skcall.talk.backend import AvatarTrack
    tr = AvatarTrack(StaticPortrait(portrait(), fps=50))
    f1 = await tr.recv()
    y1 = tr._last[1]
    await tr.recv()
    assert tr._last[1] is y1 and f1.format.name == "yuv420p"
    tr.stop()


def test_avatar_build_modes(tmp_path, monkeypatch):
    p = tmp_path / "p.png"
    cv2.imwrite(str(p), portrait())
    off = VideoConfig(enabled=False, image=str(p))
    assert Avatar.build(off) is None
    st = Avatar.build(VideoConfig(mode="static", image=str(p)))
    assert isinstance(st.source, StaticPortrait) and st.wrap_sink("inner") == "inner"
    an = Avatar.build(VideoConfig(image=str(p), lipsync_url="ws://127.0.0.1:9/v1/lipsync"))
    assert isinstance(an.source, Compositor) and an.client is not None
    assert Avatar.build(VideoConfig(image=str(tmp_path / "none.png"))) is None
    monkeypatch.setenv("SKCALL_VIDEO", "0")
    monkeypatch.setenv("SKCALL_LIPSYNC_URL", "ws://x:1/v1/lipsync")
    monkeypatch.setenv("SKCALL_AUDIO_HOLD_MS", "250")
    cfg = VideoConfig.from_env("jarvis")
    assert not cfg.enabled and cfg.avatar == "jarvis" and cfg.audio_hold_ms == 250
    assert cfg.lipsync_url == "ws://x:1/v1/lipsync"


# ---------------------------------------------------------------- conversation wiring

async def test_conversation_speaks_through_lipsync_sink_and_prerenders_fillers():
    from skcall.core.loop import Conversation, LoopConfig
    from tests.test_loop import FakeBrain, FakeSession, FakeTTS

    srv = await FakeLipsyncServer(ttff=0.02).start()
    c = await connected_client(srv)
    comp = Compositor(portrait(), None, fps=FPS)
    comp.set_crop(c.crop, c.frame_size)

    class Av:
        sink = None

        def wrap_sink(self, inner):
            self.sink = LipsyncSink(inner, comp, c)
            return self.sink

        async def prerender(self, clips):
            self.clips = clips
            return await self.sink.prerender(clips)

    av = Av()
    sess = FakeSession(time.monotonic)
    conv = Conversation(sess, None, FakeTTS(), FakeBrain(), LoopConfig(), avatar=av)
    try:
        assert conv.sink is av.sink and av.sink.inner is sess.audio_out
        await conv.prerender_fillers()
        assert len(av.clips) == 1 and len(av.sink.cache) == 1
        await conv.speak_line("Hello there.")
        assert len(sess.audio_out.written) == 1 and av.sink.stats["lipsync"] == 1
    finally:
        await c.aclose()
        await srv.stop()
