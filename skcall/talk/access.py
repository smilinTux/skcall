"""Which rooms skcall may join.

A room is allowed when its token is in the profile's rooms
(SKCALL_ALLOWED_ROOMS), or, with SKCALL_ALLOW_OPERATOR_DMS on (the default),
when it is a private room between the agent and one of its operators
(SKCALL_OPERATORS):

- a one-to-one conversation (Talk room type 1) with an operator, or
- a group conversation (type 2) whose only participants are exactly the agent
  and one operator. Any third participant, human, agent, guest, group or
  federated user, means the room has to be listed.

A new DM from an operator then works without hand-editing the allowlist.

The check reads the room and its participants over OCS as the agent (two
GETs, no join, no ringing), each time a session opens, so a room that gained
a member since the last call is rejected.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

LOG = logging.getLogger("skcall.access")

ROOM_TYPE_ONE_TO_ONE = 1
ROOM_TYPE_GROUP = 2

LISTED = "allowed (listed)"
OPERATOR_DM = "allowed (operator DM)"
OPERATOR_PAIR = "allowed (operator two-member group)"


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str

    def __str__(self) -> str:
        return self.reason


def _other_users(parts: list, me: str) -> list[str]:
    return [p.get("actorId", "") for p in parts or []
            if p.get("actorType") == "users" and p.get("actorId") != me]


def _others(parts: list, me: str) -> list[str]:
    """Every participant except the agent's own user, any actor type."""
    return [f"{p.get('actorType', '?')}/{p.get('actorId', '')}" for p in parts or []
            if not (p.get("actorType") == "users" and p.get("actorId") == me)]


async def _two_member_group(ocs, room: str, operators: tuple[str, ...]) -> Decision:
    """Type 2 room: allowed only if the participants are exactly {agent, one
    operator}. No fallback: without a participant list it is rejected."""
    try:
        others = _others(await ocs.participants(room), ocs.user)
    except Exception as e:
        return Decision(False, f"rejected: {room} is an unlisted group room, participants "
                               f"unavailable ({e})")
    if len(others) == 1 and others[0].startswith("users/") and others[0][6:] in operators:
        return Decision(True, OPERATOR_PAIR)
    return Decision(False, f"rejected: {room} is an unlisted group room with {others or 'nobody'}, "
                           "not exactly the agent and one operator")


async def room_access(ocs, room: str, allowed_rooms: tuple[str, ...] | None,
                      operators: tuple[str, ...] = (), operator_dms: bool = True) -> Decision:
    """Allow or reject `room` for the agent logged in on `ocs` (an open TalkOCS).
    `allowed_rooms=None` means no allowlist (self-tests)."""
    if allowed_rooms is None or room in allowed_rooms:
        return Decision(True, LISTED)
    if not operator_dms:
        return Decision(False, f"rejected: {room} not in the allowlist (operator DMs off)")
    if not operators:
        return Decision(False, f"rejected: {room} not in the allowlist (no operators set)")
    try:
        info = await ocs.get_room(room)
    except Exception as e:
        return Decision(False, f"rejected: {room} not in the allowlist, room info unavailable ({e})")
    rtype = int(info.get("type") or 0)
    if rtype == ROOM_TYPE_GROUP:
        return await _two_member_group(ocs, room, operators)
    if rtype != ROOM_TYPE_ONE_TO_ONE:
        return Decision(False, f"rejected: {room} not in the allowlist and neither a one-to-one "
                               f"nor a group room (type {rtype})")
    # For a one-to-one room `name` is the other user's id, set by the server
    # (such rooms cannot be renamed). Talk can drop that user from the
    # participant list (seen 2026-10-26: an operator's DM with the agent listed
    # only the agent), so the name stands in when the participants
    # show nobody else. A different user there always wins over the name.
    name = [info["name"]] if info.get("name") else []
    try:
        others = _other_users(await ocs.participants(room), ocs.user)
        source = "participants"
        if not others:
            others, source = name, "room name, no other participant"
    except Exception as e:
        LOG.warning("participants of %s unavailable (%s); using the room name", room, e)
        others, source = name, "room name"
    if len(others) == 1 and others[0] in operators:
        return Decision(True, OPERATOR_DM)
    return Decision(False, f"rejected: {room} is a one-to-one room with {others or 'nobody'} "
                           f"({source}), not an operator")


async def check_room(ocs, room: str, allowed_rooms: tuple[str, ...] | None,
                     operators: tuple[str, ...] = (), operator_dms: bool = True) -> Decision:
    """room_access, logged; raises PermissionError when rejected."""
    d = await room_access(ocs, room, allowed_rooms, operators, operator_dms)
    (LOG.info if d.allowed else LOG.warning)("room %s: %s", room, d.reason)
    if not d.allowed:
        raise PermissionError(f"room {room!r} is not in the allowlist ({d.reason})")
    return d
