"""aioice re-binds TURN channels inside the 300 s TURN permission lifetime."""
from aioice import turn

from skcall.talk.media import TURN_CHANNEL_REFRESH_S, patch_turn_refresh


def _client():
    return turn.TurnClientMixin(server=("t", 3478), username="u", password="p",
                                lifetime=600, channel_refresh_time=500)


def test_default_refresh_is_inside_permission_lifetime():
    assert TURN_CHANNEL_REFRESH_S < 300


def test_patch_clamps_channel_refresh_and_is_idempotent():
    patch_turn_refresh()
    patch_turn_refresh()
    c = _client()
    assert c.channel_refresh_time == TURN_CHANNEL_REFRESH_S
    assert c.lifetime == 600


def test_patch_keeps_a_shorter_refresh():
    patch_turn_refresh()
    c = turn.TurnClientMixin(server=("t", 3478), username="u", password="p",
                             lifetime=600, channel_refresh_time=100)
    assert c.channel_refresh_time == 100
