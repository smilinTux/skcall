"""Media recovery in TalkSession: re-publish, re-subscribe, signaling
reconnect/resume, the shared attempt budget, and leaving cleanly with a room
note when it runs out. Fakes only: no network, no aiortc peer connections."""
import asyncio
import logging

import pytest

pytest.importorskip("aiortc")

from skcall import cli  # noqa: E402
from skcall.core.session import CallEnded, Participant, ParticipantLeft  # noqa: E402
from skcall.talk import backend  # noqa: E402
from skcall.talk.backend import (CALL_OVER_REASON, LOST_REASON, ReconnectBudget,  # noqa: E402
                                 TalkSession)
from skcall.talk.ocs import OCSError  # noqa: E402
from skcall.talk.signaling import SignalingClient, SignalingError  # noqa: E402


class FakePC:
    """The slice of RTCPeerConnection TalkSession's recovery reads."""

    def __init__(self, cfg=None, state="new"):
        self.connectionState = state
        self.iceConnectionState = "new"
        self.handlers: dict[str, list] = {}
        self.closed_by_us = False

    def on(self, event, cb=None):
        if cb is None:                       # decorator form
            def deco(f):
                self.handlers.setdefault(event, []).append(f)
                return f
            return deco
        self.handlers.setdefault(event, []).append(cb)

    def set_state(self, state):
        self.connectionState = state
        for cb in self.handlers.get("connectionstatechange", []):
            cb()

    async def close(self):
        self.closed_by_us = True
        if self.connectionState != "closed":
            self.set_state("closed")


class WireSig(SignalingClient):
    def __init__(self, sessionid="ME"):
        super().__init__("https://hpb.example", "https://nc.example/backend", {})
        self.sessionid = sessionid
        self.resumeid = "R1"
        self.hello_version = "2.0"
        self.wire: list[dict] = []
        self.calls: list[str] = []
        self.resume_ok = True

    async def connect(self):
        self.calls.append("connect")

    async def send(self, msg, *, wait=False, timeout=10):
        self.wire.append(msg)

    async def hello(self, prefer="2.0"):
        self.calls.append("hello")
        self.sessionid = "NEW"
        return {}

    async def resume(self, resumeid, version="2.0"):
        self.calls.append(f"resume:{resumeid}")
        if not self.resume_ok:
            raise SignalingError('{"code": "no_such_session"}')
        return {}

    async def join_room(self, roomid, nc_sessionid):
        self.calls.append(f"room:{nc_sessionid}")
        return {}

    async def close(self):
        self.calls.append("close")
        self.closed.set()

    def requestoffers(self):
        return [m["message"]["recipient"]["sessionid"] for m in self.wire
                if m["message"]["data"].get("type") == "requestoffer"]


class FakeOCS:
    user = "lumina"
    base_url = "https://nc.example"

    def __init__(self, in_call=("alice",)):
        self.in_call = set(in_call)
        self.joins: list[int] = []
        self.room_joins: list[bool] = []
        self.settings_fetched = 0
        self.participants_error: Exception | None = None

    async def participants(self, room):
        if self.participants_error is not None:
            raise self.participants_error
        return [{"actorType": "users", "actorId": u, "inCall": 7} for u in self.in_call]

    async def signaling_settings(self, room):
        self.settings_fetched += 1
        return {"server": "https://hpb.example",
                "helloAuthParams": {"2.0": {"token": f"jwt-{self.settings_fetched}"}}}

    async def join_room(self, room, force=False):
        self.room_joins.append(force)
        return {"sessionId": f"NC{len(self.room_joins)}"}

    async def join_call(self, room, flags=3, silent=True):
        self.joins.append(flags)


def session(**kw) -> TalkSession:
    s = TalkSession("room1", nick="Lumina", ocs=FakeOCS(**kw), allowed_rooms=None)
    s.sig = WireSig()
    s.self_user_id = "lumina"
    s.video_out = None
    s.settings = {"server": "https://hpb.example", "helloAuthParams": {}}
    s._in_call = True
    s._pc_factory = FakePC
    s._budget = ReconnectBudget(backoff_s=(0.0,))
    s.published: list[FakePC] = []
    return s


def fake_publish(s: TalkSession, outcome="connected"):
    async def _publish():
        pc = FakePC()
        pc.on("connectionstatechange", lambda: s._on_pub_state(pc))
        s._pub = pc
        s.published.append(pc)
        pc.connectionState = outcome() if callable(outcome) else outcome
    s._publish = _publish


async def settle(s: TalkSession, rounds: int = 50):
    for _ in range(rounds):
        await asyncio.sleep(0)
        if not s._bg:
            return
    await asyncio.wait_for(asyncio.gather(*list(s._bg), return_exceptions=True), 5)


def queued(s: TalkSession) -> list:
    out = []
    while not s._q.empty():
        out.append(s._q.get_nowait())
    return out


# ------------------------------------------------------------------ budget
def test_budget_caps_attempts_in_window_and_grows_delay():
    now = [0.0]
    b = ReconnectBudget(max_attempts=5, window_s=120, backoff_s=(1, 2, 4, 8, 15), clock=lambda: now[0])
    assert [b.take() for _ in range(5)] == [1, 2, 4, 8, 15]
    assert b.take() is None
    now[0] = 121.0                      # the first attempts age out of the window
    assert b.take() == 1


def test_lost_note_names_the_agent():
    assert backend.lost_connection_note("lumina") == (
        "I lost the call connection and left; say @lumina join call to bring me back.")


# ------------------------------------------------------------------ publisher
async def test_remote_publisher_close_republishes():
    s = session()
    fake_publish(s)
    old = FakePC(state="connected")
    s._pub = old
    old.on("connectionstatechange", lambda: s._on_pub_state(old))
    old.set_state("closed")                        # aiortc closed it (DTLS closed)
    await settle(s)
    assert len(s.published) == 1 and s._pub is s.published[0]
    assert not s.connection_lost and queued(s) == []
    assert s._budget.used == 1


async def test_failed_publisher_is_closed_and_replaced():
    s = session()
    fake_publish(s)
    old = FakePC(state="connected")
    s._pub = old
    old.on("connectionstatechange", lambda: s._on_pub_state(old))
    old.set_state("failed")
    await settle(s)
    assert old.closed_by_us and s._pub is s.published[0]


async def test_our_own_publisher_close_is_not_recovered():
    s = session()
    fake_publish(s)
    pc = FakePC(state="connected")
    pc.on("connectionstatechange", lambda: s._on_pub_state(pc))
    s._pub = None                                  # every close we make clears _pub first
    pc.set_state("closed")
    s._pub = pc
    s._leaving = True                              # and nothing recovers while leaving
    pc.set_state("failed")
    await settle(s)
    assert s.published == []


async def test_publisher_escalates_to_fresh_session_then_gives_up():
    s = session()
    fake_publish(s, outcome="failed")
    resets = []

    async def reset(resume):
        resets.append(resume)
        await s._publish()
    s._reset_signaling = reset
    old = FakePC(state="connected")
    s._pub = old
    old.on("connectionstatechange", lambda: s._on_pub_state(old))
    old.set_state("closed")
    await settle(s)
    # one plain re-publish, then fresh signaling sessions until the budget is spent
    assert resets == [False] * 4
    assert len(s.published) == 5
    assert s.connection_lost
    ev = queued(s)
    assert len(ev) == 1 and isinstance(ev[0], CallEnded) and ev[0].reason == LOST_REASON


# ------------------------------------------------------------------ subscriber
def add_sub(s: TalkSession, sid="P1", user="alice") -> FakePC:
    s._parts[sid] = Participant(pid=sid, user_id=user)
    pc = FakePC(state="connected")
    s._subs[sid] = pc
    pc.on("connectionstatechange", lambda: s._on_sub_state(sid, pc))
    return pc


async def test_subscriber_close_resubscribes_when_still_in_call(monkeypatch):
    s = session(in_call=("alice",))
    pc = add_sub(s)
    pc.set_state("closed")
    await asyncio.sleep(0.05)
    assert s.sig.requestoffers() == ["P1"]
    assert "P1" not in s._subs and s._budget.used == 1
    for t in list(s._bg):
        t.cancel()


async def test_subscriber_close_after_they_left_drops_them():
    s = session(in_call=())
    pc = add_sub(s)
    pc.set_state("closed")
    await settle(s)
    assert s.sig.requestoffers() == []
    assert "P1" not in s._parts
    assert [type(e) for e in queued(s)] == [ParticipantLeft]


async def test_drop_does_not_trigger_resubscribe():
    s = session()
    add_sub(s)
    await s._drop("P1")                     # pops the pc before closing it
    await settle(s)
    assert s.sig.requestoffers() == [] and s._budget.used == 0


async def test_subscriber_failures_spend_the_shared_budget():
    s = session()
    s._budget = ReconnectBudget(max_attempts=2, backoff_s=(0.0,))
    for i in range(3):
        add_sub(s, sid=f"P{i}", user=f"u{i}").set_state("failed")
    await asyncio.sleep(0.05)
    assert s.connection_lost
    assert any(isinstance(e, CallEnded) for e in queued(s))
    for t in list(s._bg):
        t.cancel()


# ------------------------------------------------------------------ signaling
async def test_websocket_drop_resumes_the_session():
    s = session()
    old = s.sig
    new = WireSig()
    s._new_signaling = lambda: new
    pub = FakePC(state="connected")
    s._pub = pub
    s._spawn(s._watch_signaling(old))
    old.closed.set()                        # websocket gone
    for _ in range(100):
        await asyncio.sleep(0)
        if s.sig is new:
            break
    for t in list(s._bg):                   # the new websocket's own watcher
        t.cancel()
    assert new.calls == ["connect", "resume:R1"]
    assert s.sig is new and s._pub is pub   # same session: streams kept
    assert not s.connection_lost


async def test_refused_resume_starts_fresh_session_and_rejoins():
    s = session()
    fake_publish(s)
    new = WireSig()
    new.resume_ok = False
    s._new_signaling = lambda: new
    sub = add_sub(s)
    oldpub = FakePC(state="connected")
    s._pub = oldpub
    await s._reset_signaling(resume=True)
    assert new.calls == ["connect", "resume:R1", "hello", "room:NC1"]
    assert s.ocs.room_joins == [True] and s.nc_session == "NC1"
    assert s.ocs.joins == [3]               # back in the call (audio, no video)
    assert oldpub.closed_by_us and sub.closed_by_us
    assert s._pub is s.published[0] and s._subs == {}
    assert "P1" in s._parts                 # the participants update re-subscribes
    for t in list(s._bg):
        t.cancel()


async def test_server_bye_forces_fresh_session():
    s = session()
    seen = []
    s._start_signaling_recovery = lambda resume: seen.append(resume)
    await s._on_msg({"type": "bye", "bye": {"reason": "session_expired"}})
    assert seen == [False]


async def test_replaced_own_session_is_not_a_participant():
    s = session()
    s._own_sids = {"OLDME"}
    await s._on_participants({"users": [{"sessionId": "OLDME", "userId": "lumina", "inCall": 7}]})
    assert s._parts == {}


async def test_signaling_gives_up_after_budget():
    s = session()
    s._budget = ReconnectBudget(max_attempts=2, backoff_s=(0.0,))

    async def boom(resume):
        raise OSError("connection refused")
    s._reset_signaling = boom
    await s._recover_signaling(resume=True)
    assert s.connection_lost
    assert [e.reason for e in queued(s)] == [LOST_REASON]


async def test_leave_cancels_recovery():
    s = session()

    async def forever():
        await asyncio.sleep(3600)
    t = s._spawn(forever())
    s.sig = None
    s._in_call = False
    await s.leave()
    await asyncio.sleep(0)
    assert t.cancelled() and s._leaving


# ------------------------------------------------------------------ cli
async def test_lost_note_posted_through_summary_path(monkeypatch):
    posted = []

    async def post_summary(room, msg, ocs=None):
        posted.append((room, msg))
    monkeypatch.setattr(cli.call_summary, "post_summary", post_summary)
    await cli._post_lost_note("room1", backend.lost_connection_note("lumina"))
    assert posted == [("room1", "I lost the call connection and left; say @lumina join call to bring me back.")]


def test_ice_milestones_kept_and_check_noise_dropped():
    cli.keep_ice_milestones()
    lg = logging.getLogger("aioice.ice")
    f = next(f for f in lg.filters if isinstance(f, cli._Milestones))

    def rec(msg, level=logging.INFO):
        return logging.LogRecord("aioice.ice", level, __file__, 1, msg, (), None)
    assert f.filter(rec("Connection(1) Consent to send expired"))
    assert f.filter(rec("Connection(1) Check CandidatePair(relay) nominating pair"))
    assert not f.filter(rec("Connection(1) Check CandidatePair(x) State.WAITING -> State.IN_PROGRESS"))
    assert f.filter(rec("anything", logging.WARNING))


async def test_bye_then_websocket_close_start_one_recovery():
    s = session()
    started = []

    async def recover(resume):
        started.append(resume)
    s._recover_signaling = recover
    await s._on_msg({"type": "bye", "bye": {"reason": "session_expired"}})
    s._start_signaling_recovery(resume=True)         # the websocket close right after
    await settle(s)
    assert started == [False]


async def test_publisher_recovery_adopts_a_publisher_signaling_already_made():
    s = session()
    fake_publish(s)
    old = FakePC(state="connected")
    s._pub = old
    old.on("connectionstatechange", lambda: s._on_pub_state(old))
    old.set_state("closed")
    replacement = FakePC(state="connected")
    s._pub = replacement                              # signaling recovery got there first
    await settle(s)
    assert s.published == [] and s._pub is replacement


# ------------------------------------------------------------------ 2026-10-26
async def test_fresh_session_uses_fresh_signaling_settings():
    """The hello v2.0 JWT lives ~60 s: a fresh session must not reuse the one
    from open() (every attempt was refused with token_expired)."""
    s = session()
    fake_publish(s)
    s.settings = {"server": "https://hpb.example", "helloAuthParams": {"2.0": {"token": "stale"}}}
    built = []

    def new_signaling():
        built.append(s.settings["helloAuthParams"]["2.0"]["token"])
        sig = WireSig()
        sig.resume_ok = False
        return sig
    s._new_signaling = new_signaling
    await s._reset_signaling(resume=False)
    assert built == ["jwt-1"] and s.ocs.settings_fetched == 1
    for t in list(s._bg):
        t.cancel()


async def test_fresh_session_rejoins_room_with_new_nextcloud_session():
    """The old session's bye drops our Nextcloud session from the room, so the
    HPB room join must use a new one ("The user is not invited to this room")."""
    s = session()
    fake_publish(s)
    s.nc_session = "OLDNC"
    new = WireSig()
    s._new_signaling = lambda: new
    await s._reset_signaling(resume=False)
    assert new.calls == ["connect", "hello", "room:NC1"]
    assert s.ocs.room_joins == [True]
    assert s.ocs.joins == [3] and s._pub is s.published[0]
    for t in list(s._bg):
        t.cancel()


async def test_publisher_drop_after_hangup_leaves_quietly():
    s = session(in_call=())                    # Alice hung up: nobody else in the call
    fake_publish(s)
    resets = []

    async def reset(resume):
        resets.append(resume)
    s._reset_signaling = reset
    old = FakePC(state="connected")
    s._pub = old
    old.on("connectionstatechange", lambda: s._on_pub_state(old))
    old.set_state("closed")
    await settle(s)
    assert s.published == [] and resets == []
    assert s.call_over and not s.connection_lost         # no lost-connection note
    assert [e.reason for e in queued(s)] == [CALL_OVER_REASON]
    assert s._budget.used == 1


async def test_only_agents_left_counts_as_call_over():
    s = session(in_call=("lumina", "jarvis"))
    s.agent_users = ("jarvis",)
    assert await s._call_is_over()
    s.ocs.in_call.add("alice")
    assert not await s._call_is_over()


async def test_room_gone_counts_as_call_over_but_unreachable_does_not():
    s = session()
    s.ocs.participants_error = OCSError(404, "GET", "/participants", "")
    assert await s._call_is_over()
    s.ocs.participants_error = OSError("network down")
    assert not await s._call_is_over()


async def test_give_up_after_hangup_is_quiet():
    s = session(in_call=())
    await s._give_up("publisher")
    assert s.call_over and not s.connection_lost
    assert [e.reason for e in queued(s)] == [CALL_OVER_REASON]


async def test_signaling_recovery_stops_when_call_ended():
    s = session(in_call=())
    tried = []

    async def reset(resume):
        tried.append(resume)
    s._reset_signaling = reset
    await s._recover_signaling(resume=True)
    assert tried == [] and s.call_over
    assert [e.reason for e in queued(s)] == [CALL_OVER_REASON]
