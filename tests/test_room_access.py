"""Room allowlist: listed rooms, operator DMs (type 1), everything else rejected."""
import asyncio

import pytest

from skcall.agents import profile_for

SITE = {"SKCALL_NC_URL": "https://example.org"}
from skcall.talk.access import LISTED, OPERATOR_DM, OPERATOR_PAIR, check_room, room_access

ME = "jarvis"
OPS = ("alice", "bob")


class OCS:
    user = ME

    def __init__(self, rtype=1, others=("alice",), parts_fail=False, name=None, extra=()):
        self.rtype, self.others, self.parts_fail, self.extra = rtype, others, parts_fail, list(extra)
        self.name = name if name is not None else (others[0] if others else "")
        self.calls = []

    async def get_room(self, token):
        self.calls.append("room")
        return {"token": token, "type": self.rtype, "name": self.name}

    async def participants(self, token):
        self.calls.append("participants")
        if self.parts_fail:
            raise RuntimeError("boom")
        return [{"actorType": "users", "actorId": ME}] + [
            {"actorType": "users", "actorId": u} for u in self.others] + self.extra


def run(coro):
    return asyncio.run(coro)


def test_listed_room_allowed_without_ocs():
    o = OCS(rtype=2)
    d = run(room_access(o, "room0102", ("room0102",), OPS))
    assert d.allowed and d.reason == LISTED and o.calls == []


def test_dm_with_operator_allowed():
    d = run(room_access(OCS(others=("alice",)), "room0103", ("room0102",), OPS))
    assert d.allowed and d.reason == OPERATOR_DM


def test_dm_with_non_operator_rejected():
    d = run(room_access(OCS(others=("stranger",)), "abcd1234", ("room0102",), OPS))
    assert not d.allowed and "stranger" in d.reason


def test_two_member_group_with_operator_allowed():
    d = run(room_access(OCS(rtype=2, others=("alice",)), "room0103", ("room0102",), OPS))
    assert d.allowed and d.reason == OPERATOR_PAIR


def test_group_with_third_member_rejected():
    for others, extra in ((("alice", "bob"), ()),        # two operators
                          (("alice", "lumina"), ()),       # another agent
                          (("alice",), [{"actorType": "guests", "actorId": "g1"}]),
                          (("alice",), [{"actorType": "groups", "actorId": "admin"}]),
                          (("alice",), [{"actorType": "federated_users",
                                                   "actorId": "x@other.example"}])):
        d = run(room_access(OCS(rtype=2, others=others, extra=extra), "grp12345", ("room0102",), OPS))
        assert not d.allowed, (others, extra)


def test_two_member_group_with_non_operator_rejected():
    d = run(room_access(OCS(rtype=2, others=("stranger",)), "grp12345", ("room0102",), OPS))
    assert not d.allowed and "stranger" in d.reason


def test_group_without_participants_has_no_name_fallback():
    o = OCS(rtype=2, others=("alice",), parts_fail=True, name="alice")
    assert not run(room_access(o, "grp12345", (), OPS)).allowed
    assert not run(room_access(OCS(rtype=2, others=()), "grp12345", (), OPS)).allowed


def test_other_room_types_rejected():
    for t in (3, 4, 6):
        o = OCS(rtype=t)
        d = run(room_access(o, "pub12345", ("room0102",), OPS))
        assert not d.allowed and f"type {t}" in d.reason and "participants" not in o.calls


def test_flag_off_rejects_two_member_group():
    o = OCS(rtype=2)
    assert not run(room_access(o, "room0103", ("room0102",), OPS, operator_dms=False)).allowed
    assert o.calls == []


def test_flag_off_rejects_operator_dm_without_ocs():
    o = OCS()
    d = run(room_access(o, "room0103", ("room0102",), OPS, operator_dms=False))
    assert not d.allowed and o.calls == []


def test_participants_failure_falls_back_to_room_name():
    assert run(room_access(OCS(parts_fail=True), "room0103", (), OPS)).allowed
    assert not run(room_access(OCS(parts_fail=True, name="stranger"), "x1234567", (), OPS)).allowed


def test_dm_other_user_missing_from_participants_uses_room_name():
    assert run(room_access(OCS(others=(), name="bob"), "room0105", (), OPS)).allowed
    assert not run(room_access(OCS(others=(), name="stranger"), "room0105", (), OPS)).allowed
    # a participant who is not the named operator always wins over the name
    assert not run(room_access(OCS(others=("stranger",), name="bob"), "room0105", (), OPS)).allowed


def test_check_room_raises_on_reject():
    with pytest.raises(PermissionError):
        run(check_room(OCS(others=("stranger",)), "abcd1234", (), OPS))


def test_profile_flag_env():
    assert profile_for("jarvis", SITE).operator_dms is True
    assert profile_for("jarvis", {**SITE, "SKCALL_ALLOW_OPERATOR_DMS": "0"}).operator_dms is False
    assert profile_for("jarvis", SITE).tts_voice == "bm_george"      # positional fields unchanged


def test_talk_session_defers_dm_check_to_open(monkeypatch):
    pytest.importorskip("aiortc")
    from skcall.talk.backend import TalkSession
    with pytest.raises(PermissionError):      # no operators: rejected at once
        TalkSession("room0103", ocs=OCS(), allowed_rooms=("room0102",))
    with pytest.raises(PermissionError):      # flag off
        TalkSession("room0103", ocs=OCS(), allowed_rooms=("room0102",), operators=OPS,
                    operator_dms=False)
    TalkSession("room0102", ocs=OCS(), allowed_rooms=("room0102",))

    class Stop(Exception):
        pass

    class Open(OCS):
        async def __aenter__(self):
            return self

        async def signaling_settings(self, token):
            raise Stop

    import skcall.talk.backend as backend
    monkeypatch.setattr(backend, "patch_ice", lambda **kw: None)
    s = TalkSession("room0103", ocs=Open(others=("stranger",)), allowed_rooms=("room0102",),
                    operators=OPS)
    with pytest.raises(PermissionError):
        run(s.open())
    s = TalkSession("room0103", ocs=Open(), allowed_rooms=("room0102",), operators=OPS)
    with pytest.raises(Stop):                 # past the access check, on to signaling
        run(s.open())
