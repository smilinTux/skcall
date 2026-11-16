"""She must never go deaf silently (2026-10-26, room room0106, twice in one call):

1. Alice's client rejoined with a new session and its publisher took longer
   than the 8 x 2 s requestoffer loop to exist in Janus; the loop logged
   "gave up subscribing" and she never heard him again.
2. Inbound audio stopped with no log line at all while the subscriber peer
   connection still looked connected (an RTCP BYE ends aiortc's track
   without a state change; RTP can stop while ICE consent passes).

Fakes only: no network, no aiortc peer connections."""
import asyncio
import json
import logging

import numpy as np
import pytest

from skcall.core.loop import Conversation, LoopConfig
from skcall.core.session import Participant
from tests.test_loop import ALICE, LOUD, QUIET, Clock, FakeBrain, FakeSession, FakeSTT, FakeTTS

pytest.importorskip("aiortc")

from tests.test_talk_reconnect import FakePC, session  # noqa: E402


# ------------------------------------------------------------ backend
async def test_requestoffer_never_gives_up_while_participant_is_in_call(caplog):
    s = session()
    s._parts["P2"] = Participant(pid="P2", user_id="alice")
    s._requested["P2"] = 0.0
    task = asyncio.ensure_future(s._request_until_offer("P2", tries=3, every_s=0.001, max_every_s=0.004))
    with caplog.at_level(logging.INFO, logger="skcall.talk"):
        await asyncio.sleep(0.15)
    asked = s.sig.requestoffers().count("P2")
    assert asked > 3, asked                       # kept asking past the quick tries
    assert not task.done()
    assert "gave up" not in caplog.text
    assert "still asking" in caplog.text
    s._subs["P2"] = FakePC(state="connected")     # Janus offered at last
    await asyncio.wait_for(task, 1)
    assert s.sig.requestoffers().count("P2") <= asked + 1


async def test_requestoffer_stops_when_participant_leaves():
    s = session()
    s._parts["P2"] = Participant(pid="P2", user_id="alice")
    task = asyncio.ensure_future(s._request_until_offer("P2", tries=2, every_s=0.001, max_every_s=0.002))
    await asyncio.sleep(0.02)
    del s._parts["P2"]
    await asyncio.wait_for(task, 1)
    assert "P2" not in s._requested


async def test_one_requestoffer_loop_per_participant():
    s = session()
    s._parts["P2"] = Participant(pid="P2", user_id="alice")
    a = asyncio.ensure_future(s._request_until_offer("P2", tries=2, every_s=0.01, max_every_s=0.01))
    await asyncio.sleep(0)
    b = asyncio.ensure_future(s._request_until_offer("P2", tries=2, every_s=0.01, max_every_s=0.01))
    await asyncio.wait_for(b, 1)                  # the second returns at once
    assert not a.done()
    del s._parts["P2"]
    await asyncio.wait_for(a, 1)


class EndingTrack:
    """A remote audio track that yields `n` frames, then ends (MediaStreamError)."""

    def __init__(self, n=2):
        self.n = n

    async def recv(self):
        from aiortc.mediastreams import MediaStreamError
        from av import AudioFrame
        if self.n <= 0:
            raise MediaStreamError
        self.n -= 1
        f = AudioFrame.from_ndarray(np.zeros((1, 960), np.int16), format="s16", layout="mono")
        f.sample_rate = 48000
        return f


async def test_track_ending_under_a_live_subscription_resubscribes(monkeypatch, caplog):
    s = session()
    s._parts["P2"] = Participant(pid="P2", user_id="alice")
    pc = FakePC(state="connected")
    s._subs["P2"] = pc
    asked = []

    async def record(sid, *a, **k):
        asked.append(sid)
    monkeypatch.setattr(s, "_request_until_offer", record)
    with caplog.at_level(logging.WARNING, logger="skcall.talk"):
        got = [f async for f in s._frames(EndingTrack(2), "P2", pc)]
        await asyncio.sleep(0.01)
    assert len(got) == 2
    assert asked == ["P2"]
    assert "P2" not in s._subs and pc.closed_by_us
    assert "ended after 2 frames" in caplog.text


async def test_track_of_a_replaced_subscription_ending_does_nothing(monkeypatch):
    s = session()
    s._parts["P2"] = Participant(pid="P2", user_id="alice")
    old, new = FakePC(state="closed"), FakePC(state="connected")
    s._subs["P2"] = new
    asked = []

    async def record(sid, *a, **k):
        asked.append(sid)
    monkeypatch.setattr(s, "_request_until_offer", record)
    _ = [f async for f in s._frames(EndingTrack(0), "P2", old)]
    await asyncio.sleep(0.01)
    assert asked == [] and s._subs["P2"] is new


async def test_resubscribe_skips_agents_and_absent_participants():
    s = session()
    s._parts["J1"] = Participant(pid="J1", user_id="jarvis", is_agent=True)
    assert s.resubscribe("J1") is False
    assert s.resubscribe("nobody") is False


# ------------------------------------------------------------ backend: remote mute
async def test_track_ending_while_muted_does_not_resubscribe_until_unmute(monkeypatch, caplog):
    s = session()
    s._parts["P2"] = Participant(pid="P2", user_id="alice")
    pc = FakePC(state="connected")
    s._subs["P2"] = pc
    asked = []

    async def record(sid, *a, **k):
        asked.append(sid)
    monkeypatch.setattr(s, "_request_until_offer", record)
    s._on_status_message("P2", '{"type":"audioOff"}')
    assert s.audio_muted("P2")
    with caplog.at_level(logging.INFO, logger="skcall.talk"):
        _ = [f async for f in s._frames(EndingTrack(2), "P2", pc)]
        await asyncio.sleep(0.01)
    assert asked == [] and s._subs["P2"] is pc
    assert "while muted" in caplog.text
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    s._on_status_message("P2", '{"type":"audioOn"}')     # unmute: subscribe again
    await asyncio.sleep(0.01)
    assert asked == ["P2"] and not s.audio_muted("P2")


async def test_track_ending_after_participant_left_does_not_resubscribe(monkeypatch):
    s = session()
    pc = FakePC(state="connected")
    s._subs["P2"] = pc                            # still ours, but P2 is not in the call
    asked = []

    async def record(sid, *a, **k):
        asked.append(sid)
    monkeypatch.setattr(s, "_request_until_offer", record)
    _ = [f async for f in s._frames(EndingTrack(1), "P2", pc)]
    await asyncio.sleep(0.01)
    assert asked == []


async def test_mute_signals_log_once_per_transition(caplog):
    s = session()
    s._parts["P2"] = Participant(pid="P2", user_id="alice")
    with caplog.at_level(logging.INFO, logger="skcall.talk"):
        for _ in range(3):
            s._on_status_message("P2", '{"type":"audioOff"}')
        await s._on_msg({"type": "message", "message": {
            "sender": {"type": "session", "sessionid": "P2"},
            "data": {"type": "mute", "payload": {"name": "audio"}}}})
        assert s.audio_muted("P2")
        await s._on_msg({"type": "message", "message": {
            "sender": {"type": "session", "sessionid": "P2"},
            "data": {"type": "unmute", "payload": {"name": "audio"}}}})
        s._on_status_message("P2", '{"type":"audioOn"}')
        s._on_status_message("P2", '{"type":"speaking"}')
        s._on_status_message("P2", "not json")
    assert not s.audio_muted("P2")
    msgs = [r.getMessage() for r in caplog.records]
    assert sum("alice muted" in m for m in msgs) == 1
    assert sum("alice unmuted" in m for m in msgs) == 1


async def test_video_mute_is_not_audio_mute():
    s = session()
    s._parts["P2"] = Participant(pid="P2", user_id="alice")
    s._on_status_message("P2", '{"type":"videoOff"}')
    await s._on_msg({"type": "message", "message": {
        "sender": {"type": "session", "sessionid": "P2"},
        "data": {"type": "mute", "payload": {"name": "video"}}}})
    assert not s.audio_muted("P2")


async def test_in_call_without_audio_flag_counts_as_muted():
    s = session()
    await s._on_participants({"users": [{"sessionId": "P2", "userId": "alice", "inCall": 1}]})
    assert s.audio_muted("P2")
    await s._on_participants({"users": [{"sessionId": "P2", "userId": "alice", "inCall": 3}]})
    assert not s.audio_muted("P2")
    s._on_status_message("P2", '{"type":"audioOff"}')
    await s._on_participants({"users": [{"sessionId": "P2", "userId": "alice", "inCall": 3}]})
    assert s.audio_muted("P2")                    # the flag stays on a plain mute
    await s._drop("P2")
    assert not s.audio_muted("P2")


async def test_subscriber_data_channel_feeds_mute_state():
    s = session()
    s._parts["P2"] = Participant(pid="P2", user_id="alice")
    pcs = []

    class PC(FakePC):
        def __init__(self, cfg=None):
            super().__init__(cfg)
            pcs.append(self)

        def addTransceiver(self, *a, **k):
            pass

        async def setRemoteDescription(self, d):
            raise RuntimeError("stop here")       # the handlers are wired before this
    s._pc_factory = PC
    with pytest.raises(RuntimeError):
        await s._subscribe("P2", {"payload": {"sdp": "v=0"}, "sid": "x"})

    class Ch:
        def __init__(self):
            self.h = {}

        def on(self, ev, cb):
            self.h[ev] = cb
    ch = Ch()
    for cb in pcs[0].handlers["datachannel"]:
        cb(ch)
    ch.h["message"]('{"type":"audioOff"}')
    assert s.audio_muted("P2")


# ------------------------------------------------------------ loop watchdog
class ResubSession(FakeSession):
    def __init__(self, clock, **kw):
        super().__init__(clock, **kw)
        self.resubs: list[str] = []
        self.muted: set[str] = set()

    def audio_muted(self, pid):
        return pid in self.muted

    def resubscribe(self, pid, why=""):
        self.resubs.append(pid)
        return True


def make(**cfg):
    clock = Clock()
    sess = ResubSession(clock)
    c = LoopConfig(filler=False, hangover_ms=500, min_utterance_ms=300, spec_stt=False,
                   stall_s=12.0, resubscribe_every_s=20.0, heartbeat_s=30.0, operators=("alice",))
    for k, v in cfg.items():
        setattr(c, k, v)
    conv = Conversation(sess, FakeSTT([]), FakeTTS(), FakeBrain(), c, clock=clock)
    return conv, sess, clock


async def feed(conv, clock, frames, step=0.02):
    async def gen():
        for f in frames:
            clock.t += step
            yield f
    from skcall.core.session import AudioTrack
    await conv.pump(AudioTrack(ALICE, gen()))


async def test_watchdog_resubscribes_when_inbound_audio_stops(caplog):
    conv, sess, clock = make()
    await feed(conv, clock, [QUIET] * 50)          # audio flowed, then the track went silent
    with caplog.at_level(logging.WARNING, logger="skcall.loop"):
        for _ in range(11):
            clock.t += 1
            conv.check_health(clock())
        assert sess.resubs == []
        clock.t += 1.5
        conv.check_health(clock())
    assert sess.resubs == ["sid-alice"]
    assert "no audio from alice" in caplog.text
    state = json.loads(caplog.text.split("re-subscribing: ", 1)[1].splitlines()[0])
    assert state["listeners"]["alice"]["frames"] == 50
    assert "gated" in state and "sink_buffered_s" in state
    # backoff: not again for resubscribe_every_s, then doubling
    for _ in range(19):
        clock.t += 1
        conv.check_health(clock())
    assert len(sess.resubs) == 1
    clock.t += 1.5
    conv.check_health(clock())
    assert len(sess.resubs) == 2
    for _ in range(39):
        clock.t += 1
        conv.check_health(clock())
    assert len(sess.resubs) == 2


async def test_watchdog_resubscribes_a_human_never_heard_at_all():
    conv, sess, clock = make()
    conv.check_health(clock())                    # first sighting, no listener yet
    clock.t += 13
    conv.check_health(clock())
    assert sess.resubs == ["sid-alice"]


async def test_watchdog_quiet_when_audio_flows():
    conv, sess, clock = make()
    for _ in range(30):
        await feed(conv, clock, [LOUD] * 25 + [QUIET] * 25)   # 1 s per round
        conv.check_health(clock())
    assert sess.resubs == []


async def test_watchdog_clears_a_sink_that_never_drains(caplog):
    conv, sess, clock = make()
    sess.audio_out.hold = True
    sess.audio_out.write(np.ones(48000, np.int16))
    conv.audible(clock())
    with caplog.at_level(logging.WARNING, logger="skcall.loop"):
        for _ in range(13):
            clock.t += 1
            conv.audible(clock())                 # frames keep extending the echo gate
            conv.check_health(clock())
    assert sess.audio_out.cleared == 1
    assert not conv.gated(clock.t + conv.cfg.echo_tail_s + 0.01)
    assert "sink not draining" in caplog.text


async def test_silence_with_frames_flowing_is_not_a_stall(caplog):
    """Alice quiet for minutes: frames (silence, comfort noise) keep arriving
    under the VAD gate. No resubscribe and no gate or segmenter reset."""
    conv, sess, clock = make()
    with caplog.at_level(logging.WARNING, logger="skcall.loop"):
        for _ in range(300):
            await feed(conv, clock, [QUIET] * 50)
            conv.check_health(clock())
    assert sess.resubs == []
    assert "watchdog" not in caplog.text


async def test_muted_participant_never_triggers_resubscribe():
    conv, sess, clock = make()
    sess.muted.add("sid-alice")
    await feed(conv, clock, [QUIET] * 50)
    for _ in range(600):                          # 10 minutes muted, no frames at all
        clock.t += 1
        conv.check_health(clock())
    assert sess.resubs == []
    assert conv.health_state(clock())["muted"] == ["alice"]


async def test_muted_from_the_start_never_triggers_resubscribe():
    conv, sess, clock = make()
    sess.muted.add("sid-alice")
    for _ in range(120):
        clock.t += 1
        conv.check_health(clock())
    assert sess.resubs == []


async def test_unmute_rearms_the_stall_timer_fresh():
    conv, sess, clock = make()
    await feed(conv, clock, [QUIET] * 50)
    sess.muted.add("sid-alice")
    for _ in range(60):
        clock.t += 1
        conv.check_health(clock())
    sess.muted.discard("sid-alice")
    for _ in range(11):                           # unmuted, still no frames: not yet
        clock.t += 1
        conv.check_health(clock())
    assert sess.resubs == []
    clock.t += 1.5
    conv.check_health(clock())                    # unmuted and no frames for stall_s
    assert sess.resubs == ["sid-alice"]


async def test_no_frames_while_unmuted_triggers_resubscribe():
    conv, sess, clock = make()
    await feed(conv, clock, [QUIET] * 50)
    clock.t += 13
    conv.check_health(clock())
    assert sess.resubs == ["sid-alice"]


async def test_heartbeat_logs_counters(caplog):
    conv, sess, clock = make()
    await feed(conv, clock, [LOUD] * 10 + [QUIET] * 10)
    with caplog.at_level(logging.INFO, logger="skcall.loop"):
        clock.t += 30
        conv.check_health(clock())
    line = [r.getMessage() for r in caplog.records if r.getMessage().startswith("HEARTBEAT ")]
    assert len(line) == 1
    hb = json.loads(line[0].split(" ", 1)[1])
    li = hb["listeners"]["alice"]
    assert li["frames"] == 20 and li["peak_rms"] > 1000 and hb["humans"] == ["alice"]


async def test_pump_end_is_logged(caplog):
    conv, sess, clock = make()
    with caplog.at_level(logging.INFO, logger="skcall.loop"):
        await feed(conv, clock, [QUIET] * 3)
    assert "audio from alice ended after 3 frames" in caplog.text
