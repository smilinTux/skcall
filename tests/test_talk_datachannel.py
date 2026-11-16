"""The "status" data channel: the only place talk-ios and talk-android read a
remote participant's audio/video state and nick from. Message formats, the
channel on our publisher offer, and delivery over a real aiortc loopback."""
import asyncio
import json

import pytest

pytest.importorskip("aiortc")

from aiortc import RTCPeerConnection  # noqa: E402

from skcall.core.session import Participant  # noqa: E402
from skcall.talk.backend import TalkSession  # noqa: E402
from skcall.talk.signaling import (SignalingClient, data_channel_message,  # noqa: E402
                                   status_messages)


def test_message_without_payload_matches_web_client():
    # JSON.stringify({type: 'videoOn', payload: undefined}) == '{"type":"videoOn"}'
    assert data_channel_message("videoOn") == '{"type":"videoOn"}'


def test_message_with_payload():
    assert json.loads(data_channel_message("nickChanged", {"name": "Lumina", "userid": "lumina"})) == {
        "type": "nickChanged", "payload": {"name": "Lumina", "userid": "lumina"}}


def test_status_burst_user():
    assert [json.loads(m) for m in status_messages([("audio", True), ("video", True)], "Lumina", "lumina")] == [
        {"type": "audioOn"}, {"type": "videoOn"},
        {"type": "nickChanged", "payload": {"name": "Lumina", "userid": "lumina"}}]


def test_status_burst_guest_nick_is_bare_string_and_video_off():
    assert [json.loads(m) for m in status_messages([("audio", True), ("video", False)], "Lumina")] == [
        {"type": "audioOn"}, {"type": "videoOff"}, {"type": "nickChanged", "payload": "Lumina"}]


def test_status_media_name_validated():
    with pytest.raises(ValueError):
        status_messages([("screen", True)], "Lumina")


class FakeDC:
    def __init__(self, state="open"):
        self.readyState = state
        self.sent: list[str] = []

    def send(self, m):
        self.sent.append(m)


class WireSig(SignalingClient):
    def __init__(self):
        super().__init__("https://hpb.example", "https://nc.example/backend", {})
        self.sessionid = "ME"
        self.wire: list[dict] = []

    async def send(self, msg, *, wait=False, timeout=10):
        self.wire.append(msg)


def session() -> TalkSession:
    s = TalkSession("room1", nick="Lumina", ocs=object(), allowed_rooms=None)
    s.sig = WireSig()
    s.self_user_id = "lumina"
    s._pub = object()
    s.video_out = object()
    return s


async def test_backoff_tick_also_broadcasts_on_data_channel():
    s = session()
    s._status_dc = FakeDC()
    s._parts["P1"] = Participant(pid="P1", user_id="alice")
    await s._send_state_with_backoff("P1", delays=(0, 0))
    types = [json.loads(m)["type"] for m in s._status_dc.sent]
    assert types == ["audioOn", "videoOn", "nickChanged"] * 2
    # the signaling messages the web client relies on are still there
    assert [m["message"]["data"]["type"] for m in s.sig.wire].count("unmute") == 4


async def test_nothing_sent_before_channel_opens():
    s = session()
    s._status_dc = FakeDC(state="connecting")
    s._broadcast_status()
    assert s._status_dc.sent == []


async def test_publisher_offer_carries_status_channel_and_vp8_video():
    import numpy as np

    class Still:
        width, height, fps = 64, 48, 5

        def frame(self, t):
            return np.zeros((48, 64, 3), np.uint8)

    s = TalkSession("room1", nick="Lumina", ocs=object(), allowed_rooms=None, video=Still())
    s.sig = WireSig()
    s._cfg = None
    await s._publish()
    try:
        offer = s.sig.wire[-1]["message"]["data"]["payload"]["sdp"]
        assert "m=application" in offer and "webrtc-datachannel" in offer
        assert "VP8/90000" in offer and "m=audio" in offer
        assert s._status_dc.label == "status"
    finally:
        await s._pub.close()


async def test_state_arrives_over_real_data_channel():
    """Our publisher's "status" channel, answered by a plain aiortc peer: the
    burst sent on open arrives intact, in order."""
    s = session()
    s._pub = None
    pub = RTCPeerConnection()
    s._pub = pub
    dc = pub.createDataChannel("status")
    s._status_dc = dc
    dc.on("open", s._broadcast_status)

    sub = RTCPeerConnection()
    got: list[str] = []
    done = asyncio.Event()

    @sub.on("datachannel")
    def _dc(ch):
        assert ch.label == "status"

        @ch.on("message")
        def _m(m):
            got.append(m)
            if len(got) == 3:
                done.set()

    try:
        await pub.setLocalDescription(await pub.createOffer())
        await sub.setRemoteDescription(pub.localDescription)
        await sub.setLocalDescription(await sub.createAnswer())
        await pub.setRemoteDescription(sub.localDescription)
        await asyncio.wait_for(done.wait(), 15)
        assert [json.loads(m) for m in got] == [
            {"type": "audioOn"}, {"type": "videoOn"},
            {"type": "nickChanged", "payload": {"name": "Lumina", "userid": "lumina"}}]
    finally:
        await pub.close()
        await sub.close()
