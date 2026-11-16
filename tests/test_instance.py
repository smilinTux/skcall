"""Single-instance guard: the per-(agent, room) flock, the CLI's exit code,
and the 'already in call from another session' check. No network."""
import asyncio
import subprocess
import sys
import textwrap

import pytest

from skcall import cli
from skcall.instance import (ALREADY_IN_CALL, EXIT_ALREADY_IN_CALL, AlreadyInCall, InstanceLock,
                             lock_path, other_session_in_call, valid_room)


@pytest.fixture(autouse=True)
def runtime(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    return tmp_path


def test_room_validation():
    assert valid_room("room0001")
    for bad in ("", "abc", "ROOM0001", "../etc", "a" * 33, "room-1", "room0001\n"):
        assert not valid_room(bad), bad
    with pytest.raises(ValueError):
        lock_path("lumina", "../x")
    with pytest.raises(ValueError):
        lock_path("../lumina", "room0001")


def test_lock_path_per_agent_and_room(runtime):
    assert lock_path("lumina", "room0001") == runtime / "skcall" / "lumina-room0001.lock"
    assert lock_path("jarvis", "room0001") != lock_path("lumina", "room0001")


def test_second_lock_in_same_process_is_refused():
    a, b = InstanceLock("lumina", "room0001"), InstanceLock("lumina", "room0001")
    assert a.acquire()
    assert not b.acquire()
    a.release()
    assert b.acquire()
    b.release()


def test_other_room_and_other_agent_are_independent():
    a = InstanceLock("lumina", "room0001")
    assert a.acquire()
    assert InstanceLock("lumina", "room0106").acquire()
    assert InstanceLock("jarvis", "room0001").acquire()
    a.release()


def test_lock_held_by_another_process(runtime):
    holder = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent("""
            import sys
            from skcall.instance import InstanceLock
            l = InstanceLock("lumina", "room0001")
            assert l.acquire()
            print("held", flush=True)
            sys.stdin.read()
        """)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        env={"XDG_RUNTIME_DIR": str(runtime), "PATH": "/usr/bin:/bin"})
    try:
        assert holder.stdout.readline().strip() == "held"
        assert not InstanceLock("lumina", "room0001").acquire()
    finally:
        holder.stdin.close()
        holder.wait(10)
    # the kernel drops the lock with the process: no stale lock after a crash
    assert InstanceLock("lumina", "room0001").acquire()


def test_context_manager_raises_already_in_call():
    with InstanceLock("lumina", "room0001"):
        with pytest.raises(AlreadyInCall):
            with InstanceLock("lumina", "room0001"):
                pass


def _args(**kw):
    import argparse
    return argparse.Namespace(**{"room": "room0001", "agent": "lumina", **kw})


@pytest.fixture
def site_env(monkeypatch):
    """The site values join needs (nothing is built in)."""
    monkeypatch.setenv("SKCALL_NC_URL", "https://example.org")
    monkeypatch.setenv("SKCALL_ALLOWED_ROOMS", "room0001")
    monkeypatch.setenv("SKCALL_OPERATORS", "alice")


def test_cli_join_exits_already_in_call(monkeypatch, capsys, site_env):
    called = []

    async def fake_join(a, lock=None):
        called.append(a)
        return 0
    monkeypatch.setattr(cli, "_join", fake_join)
    held = InstanceLock("lumina", "room0001")
    assert held.acquire()
    try:
        assert asyncio.run(cli.join(_args())) == EXIT_ALREADY_IN_CALL
    finally:
        held.release()
    assert called == []
    assert ALREADY_IN_CALL in capsys.readouterr().out
    # free again: the join runs, and its lock is released afterwards
    assert asyncio.run(cli.join(_args())) == 0
    assert len(called) == 1
    assert InstanceLock("lumina", "room0001").acquire()


def test_cli_join_maps_remote_session_to_exit_code(monkeypatch, capsys, site_env):
    async def fake_join(a, lock=None):
        raise AlreadyInCall("lumina is in the room0001 call from another session")
    monkeypatch.setattr(cli, "_join", fake_join)
    assert asyncio.run(cli.join(_args())) == EXIT_ALREADY_IN_CALL
    assert ALREADY_IN_CALL in capsys.readouterr().out
    assert InstanceLock("lumina", "room0001").acquire()   # released on the way out


def test_cli_join_rejects_bad_room(monkeypatch):
    monkeypatch.setattr(cli, "_join", lambda a: (_ for _ in ()).throw(AssertionError("joined")))
    assert asyncio.run(cli.join(_args(room="../../x"))) == 2


def test_cli_env_defaults(monkeypatch, site_env):
    seen = {}

    async def fake_join(a, lock=None):
        seen.update(ring=a.ring, greeting=a.greeting, max_call=a.max_call)
        return 0
    monkeypatch.setattr(cli, "_join", fake_join)
    monkeypatch.setattr(cli, "load_env", lambda *a, **k: None)
    monkeypatch.setenv("SKCALL_RING", "1")
    monkeypatch.setenv("SKCALL_GREETING", "Hey Alice")
    monkeypatch.setenv("SKCALL_MAX_CALL_S", "86400")
    assert cli.main(["join", "--room", "room0001"]) == 0
    assert seen == {"ring": True, "greeting": "Hey Alice", "max_call": 86400.0}


# ------------------------------------------------- another session in the call
ME = "lumina"


def test_other_session_in_call_detects_remote_session():
    parts = [{"actorType": "users", "actorId": ME, "inCall": 7, "sessionIds": ["mine", "theirs"]}]
    assert other_session_in_call(parts, ME, {"mine"})


def test_not_in_call_when_only_in_room():
    parts = [{"actorType": "users", "actorId": ME, "inCall": 0, "sessionIds": ["mine", "theirs"]},
             {"actorType": "users", "actorId": "alice", "inCall": 3, "sessionIds": ["alice"]}]
    assert not other_session_in_call(parts, ME, {"mine"})


def test_in_call_flag_without_session_ids_counts():
    assert other_session_in_call([{"actorType": "users", "actorId": ME, "inCall": 1}], ME, {"mine"})


def test_guest_with_same_name_ignored():
    parts = [{"actorType": "guests", "actorId": ME, "inCall": 3, "sessionIds": ["x"]}]
    assert not other_session_in_call(parts, ME, {"mine"})


class FakeOCS:
    user = ME

    def __init__(self, parts):
        self.parts = parts

    async def participants(self, token):
        return self.parts


async def test_talk_session_bails_when_already_in_call():
    pytest.importorskip("aiortc")
    from skcall.talk.backend import TalkSession
    s = TalkSession("room0001", ocs=FakeOCS(
        [{"actorType": "users", "actorId": ME, "inCall": 3, "sessionIds": ["mine", "old"]}]))
    s.nc_session = "mine"
    with pytest.raises(AlreadyInCall):
        await s.check_not_in_call()


async def test_talk_session_joins_when_not_in_call():
    pytest.importorskip("aiortc")
    from skcall.talk.backend import TalkSession
    s = TalkSession("room0001", ocs=FakeOCS(
        [{"actorType": "users", "actorId": ME, "inCall": 0, "sessionIds": ["mine"]}]))
    s.nc_session = "mine"
    await s.check_not_in_call()


def test_lock_aliases_share_one_lock(monkeypatch):
    monkeypatch.setenv("SKCALL_LOCK_ALIASES", "room0002:room0003, abcd1234:bad-room")
    assert lock_path("lumina", "room0002") == lock_path("lumina", "room0003")
    assert lock_path("lumina", "abcd1234").name == "lumina-abcd1234.lock"
    a, b = InstanceLock("lumina", "room0003"), InstanceLock("lumina", "room0002")
    assert a.acquire() and not b.acquire()
    a.release()
