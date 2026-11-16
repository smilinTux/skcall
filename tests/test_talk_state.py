"""Talk media state messages: the "unmute"/"mute"/"nickChanged" call messages
Talk's web client needs before it shows a remote participant's video, and the
backoff schedule TalkSession sends them on. No network."""
import asyncio

import pytest

pytest.importorskip("aiortc")

from skcall.core.session import Participant  # noqa: E402
from skcall.talk import backend  # noqa: E402
from skcall.talk.backend import TalkSession  # noqa: E402
from skcall.talk.signaling import SignalingClient, media_state_message, nick_message  # noqa: E402


def test_unmute_video_matches_web_client_sendto():
    # spreed src/utils/webrtc/webrtc.js webrtc.sendTo(peerId, 'unmute', {name: 'video'})
    assert media_state_message("S1", "video", True) == {
        "to": "S1", "roomType": "video", "type": "unmute", "payload": {"name": "video"}}


def test_mute_audio():
    assert media_state_message("S1", "audio", False) == {
        "to": "S1", "roomType": "video", "type": "mute", "payload": {"name": "audio"}}


def test_media_name_validated():
    with pytest.raises(ValueError):
        media_state_message("S1", "screen", True)


def test_nick_changed():
    assert nick_message("S1", "Lumina") == {
        "to": "S1", "roomType": "video", "type": "nickChanged", "payload": {"name": "Lumina"}}


class WireSig(SignalingClient):
    """SignalingClient whose websocket is a list."""

    def __init__(self):
        super().__init__("https://hpb.example", "https://nc.example/backend", {})
        self.sessionid = "ME"
        self.wire: list[dict] = []

    async def send(self, msg, *, wait=False, timeout=10):
        self.wire.append(msg)


async def test_signaling_envelope():
    sig = WireSig()
    await sig.send_media_state("S1", "video", True)
    await sig.send_nick("S1", "Lumina")
    assert sig.wire == [
        {"type": "message", "message": {"recipient": {"type": "session", "sessionid": "S1"},
                                        "data": media_state_message("S1", "video", True)}},
        {"type": "message", "message": {"recipient": {"type": "session", "sessionid": "S1"},
                                        "data": nick_message("S1", "Lumina")}},
    ]


def session(video: bool) -> TalkSession:
    s = TalkSession("room1", nick="Lumina", ocs=object(), allowed_rooms=None)
    s.sig = WireSig()
    s._pub = object()                       # publishing
    if not video:
        s.video_out = None
    else:
        s.video_out = object()
    return s


def sent(s: TalkSession) -> list[tuple[str, str]]:
    out = []
    for m in s.sig.wire:
        d = m["message"]["data"]
        out.append((m["message"]["recipient"]["sessionid"], d["type"], d["payload"].get("name")))
    return out


async def test_state_with_video_sends_unmute_audio_video_and_nick():
    s = session(video=True)
    s._parts["P1"] = Participant(pid="P1", user_id="alice")
    await s._send_state_with_backoff("P1", delays=(0,))
    assert sent(s) == [("P1", "unmute", "audio"), ("P1", "unmute", "video"),
                       ("P1", "nickChanged", "Lumina")]


async def test_state_without_video_says_video_off():
    s = session(video=False)
    s._parts["P1"] = Participant(pid="P1", user_id="alice")
    await s._send_state_with_backoff("P1", delays=(0,))
    assert ("P1", "mute", "video") in sent(s)
    assert ("P1", "unmute", "audio") in sent(s)


async def test_state_repeats_on_schedule_and_stops_when_participant_leaves():
    s = session(video=True)
    s._parts["P1"] = Participant(pid="P1", user_id="alice")
    task = asyncio.ensure_future(s._send_state_with_backoff("P1", delays=(0, 0.01, 0.01, 0.5)))
    await asyncio.sleep(0.05)
    assert len(sent(s)) == 9                 # three rounds of three messages
    del s._parts["P1"]
    await asyncio.wait_for(task, 2)
    assert len(sent(s)) == 9


async def test_state_waits_for_publisher():
    s = session(video=True)
    s._pub = None
    s._parts["P1"] = Participant(pid="P1", user_id="alice")
    await s._send_state_with_backoff("P1", delays=(0, 0))
    assert s.sig.wire == []


async def test_announce_on_join_and_cancel_on_drop(monkeypatch):
    monkeypatch.setattr(backend, "STATE_BACKOFF_S", (0, 30))
    s = session(video=True)
    s.sig.sessionid = "ME"

    async def no_request(sid, *a, **k):
        return None
    monkeypatch.setattr(s, "_request_until_offer", no_request)
    await s._on_participants({"users": [{"sessionId": "P1", "inCall": 7, "userId": "alice"}]})
    task = s._state_tasks["P1"]
    await asyncio.sleep(0.02)
    assert ("P1", "unmute", "video") in sent(s)
    await s._drop("P1")
    await asyncio.sleep(0)
    assert task.cancelled() and "P1" not in s._state_tasks


def test_backoff_matches_web_client():
    # ExponentialBackoffCallback: 0, then 1 s doubling while <= 16 s
    assert backend.STATE_BACKOFF_S == (0, 1, 2, 4, 8, 16)


async def test_other_agents_join_but_are_never_subscribed(monkeypatch):
    s = session(video=False)
    s.sig.sessionid = "ME"
    s.agent_users = ("jarvis",)
    asked = []

    async def record(sid, *a, **k):
        asked.append(sid)
    monkeypatch.setattr(s, "_request_until_offer", record)
    await s._on_participants({"users": [{"sessionId": "J1", "inCall": 7, "userId": "jarvis"},
                                        {"sessionId": "C1", "inCall": 7, "userId": "alice"}]})
    await asyncio.sleep(0)
    assert s._parts["J1"].is_agent and asked == ["C1"]
