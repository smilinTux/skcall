"""One agent, one room, one call: the single-instance guard.

Two layers, both checked before the agent shows up in a call:

1. A non-blocking flock on a per-(agent, room) lock file under
   $XDG_RUNTIME_DIR/skcall/. A second `skcall join` for the same agent and
   room on this host exits at once with EXIT_ALREADY_IN_CALL. The kernel drops
   the lock when the process dies, so a crash never leaves a stale lock.
   SKCALL_LOCK_ALIASES maps a federated room to its host room, so joining the
   same call natively and over federation is one lock too.
2. TalkSession checks the room's participant list after signaling connects:
   if the agent's Nextcloud user is already in the call from another session
   (another host, a process started before this guard existed), it bails with
   AlreadyInCall instead of adding a second tile.
"""
from __future__ import annotations

import fcntl
import os
import re
import tempfile
from pathlib import Path

EXIT_ALREADY_IN_CALL = 3
ALREADY_IN_CALL = "already in call"
ROOM_RE = re.compile(r"^[a-z0-9]{4,32}$")
_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")


class AlreadyInCall(RuntimeError):
    """The agent is already in this room's call (this host or another session)."""

    def __init__(self, detail: str = ""):
        super().__init__(ALREADY_IN_CALL + (f": {detail}" if detail else ""))


def valid_room(room: str) -> bool:
    return bool(ROOM_RE.fullmatch(room or ""))


def runtime_dir() -> Path:
    base = os.environ.get("XDG_RUNTIME_DIR") or os.path.join(tempfile.gettempdir(), f"skcall-{os.getuid()}")
    return Path(base) / "skcall"


def lock_room(room: str, env: dict | None = None) -> str:
    """The room a lock is keyed on. SKCALL_LOCK_ALIASES ("room0002:room0001,...")
    names rooms that are one call seen from two servers (a federated copy and
    its host room), so the agent's native and federated joins share one lock."""
    env = os.environ if env is None else env
    for pair in (env.get("SKCALL_LOCK_ALIASES") or "").split(","):
        alias, sep, canon = pair.strip().partition(":")
        if sep and alias == room and valid_room(canon.strip()):
            return canon.strip()
    return room


def lock_path(agent: str, room: str) -> Path:
    if not valid_room(room):
        raise ValueError(f"invalid room token {room!r}")
    if not _NAME_RE.fullmatch(agent or ""):
        raise ValueError(f"invalid agent name {agent!r}")
    return runtime_dir() / f"{agent}-{lock_room(room)}.lock"


class InstanceLock:
    """Non-blocking exclusive flock; `acquire()` is False when another process holds it."""

    def __init__(self, agent: str, room: str, path: Path | None = None):
        self.path = Path(path) if path is not None else lock_path(agent, room)
        self._fd: int | None = None

    def acquire(self) -> bool:
        if self._fd is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        self._fd = fd
        return True

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = None

    def __enter__(self) -> "InstanceLock":
        if not self.acquire():
            raise AlreadyInCall(f"lock {self.path} is held")
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def other_session_in_call(participants: list[dict], user: str, my_sessions: set[str]) -> bool:
    """Is `user` in the call from a session that is not one of ours?

    `participants` is Talk's GET /room/{token}/participants (v4) list. Called
    before we join the call ourselves, so any inCall flag on our user's entry
    belongs to another session; sessionIds narrows it when Talk reports them."""
    for p in participants or []:
        if p.get("actorType") != "users" or p.get("actorId") != user:
            continue
        if not int(p.get("inCall") or 0) & 1:
            continue
        sessions = {s for s in (p.get("sessionIds") or []) if s}
        if not sessions or sessions - my_sessions:
            return True
    return False
