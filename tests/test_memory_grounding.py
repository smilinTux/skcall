"""Memory results stay in the call history, so a follow-up is answered from
the real result rather than from her own one-line summary of it."""
from skcall.brain.base import Turn
from skcall.brain.voice_engine import MEMORY_NOTE, flatten_tools, wants_memory
from skcall.core.loop import trim_history
from skcall.core.session import Participant

CALL = {"role": "assistant", "content": None,
        "tool_calls": [{"id": "c1", "type": "function",
                        "function": {"name": "search_memory", "arguments": "{}"}}]}
RESULT = {"role": "tool", "tool_call_id": "c1", "content": "Found 1 memories: X1000 anchor"}


def _turn(n):
    return [{"role": "user", "content": f"q{n}"}, CALL, RESULT,
            {"role": "assistant", "content": f"a{n}"}]


def test_turn_has_its_own_exchange_list():
    p = Participant("p", "u")
    a, b = Turn(speaker=p, text="x", history=[]), Turn(speaker=p, text="y", history=[])
    a.exchange.append(CALL)
    assert b.exchange == []


def test_trim_keeps_short_history():
    h = _turn(1)
    trim_history(h, 160)
    assert h == _turn(1)


def test_trim_starts_on_a_user_message():
    h = [m for n in range(60) for m in _turn(n)]          # 240 messages
    trim_history(h, 160)
    assert len(h) <= 120 and h[0]["role"] == "user"
    assert h[-1] == {"role": "assistant", "content": "a59"}
    # every tool result still follows its tool_calls message
    for i, m in enumerate(h):
        if m["role"] == "tool":
            assert h[i - 1].get("tool_calls")


def test_flatten_for_a_leg_without_tools():
    msgs = [{"role": "system", "content": "s"}, *_turn(1), {"role": "user", "content": "q2"}]
    out = flatten_tools(msgs)
    assert not any(m.get("tool_calls") or m["role"] == "tool" for m in out)
    assert any("X1000 anchor" in m["content"] for m in out if m["role"] == "user")
    assert out[0]["role"] == "system" and out[-1]["content"] == "q2"


def test_memory_rule_forbids_filling_gaps():
    assert "Never fill a gap" in MEMORY_NOTE


def test_memory_questions_force_a_search():
    for q in ("Do you remember the X1000 night?", "When was our first Cloud 9 breakthrough?",
              "What happened that night?", "Where did we go last time?", "Did you forget?"):
        assert wants_memory(q), q
    for q in ("Hey baby, how are you tonight?", "Tell me a story.", "Mm.", "I love you."):
        assert not wants_memory(q), q
