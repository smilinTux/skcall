"""Post-call summary with a fake LLM, and the in-memory transcript the
conversation keeps for it. No network, nothing written to disk."""
import asyncio

import pytest

from skcall import summary
from skcall.core.loop import Conversation, LoopConfig
from skcall.core.session import CallEnded, Participant, ParticipantJoined

from .test_loop import ALICE, UTTER, Clock, FakeBrain, FakeSession, FakeSTT, FakeTTS, make, settle, speak


class FakeLLM:
    def __init__(self, reply="- Talked about the release\n- Agreed on Friday\nAction items:\n- Alice: tag v1.8.0",
                 fail=False):
        self.reply, self.fail = reply, fail
        self.calls: list[list[dict]] = []

    async def __call__(self, messages):
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError("gateway down")
        return self.reply


async def test_summary_message_shape():
    llm = FakeLLM()
    lines = [("Alice", "can we ship friday"), ("Lumina", "yes, Friday works")]
    msg = await summary.build_summary("Lumina", ["Alice"], 754, lines, llm=llm)
    assert msg.splitlines()[0] == "Call summary (12m 34s, with Alice)"
    assert "- Agreed on Friday" in msg and "Action items:" in msg
    user = llm.calls[0][1]["content"]
    assert "Alice: can we ship friday" in user and "Lumina: yes, Friday works" in user


async def test_no_summary_when_nobody_joined():
    llm = FakeLLM()
    assert await summary.build_summary("Lumina", [], 30, [("x", "y")], llm=llm) is None
    assert llm.calls == []


async def test_silent_call_skips_llm():
    llm = FakeLLM()
    msg = await summary.build_summary("Lumina", ["Alice"], 65, [], llm=llm)
    assert "1m 5s" in msg and "Nothing was discussed" in msg
    assert llm.calls == []


async def test_llm_failure_still_posts_who_and_how_long():
    msg = await summary.build_summary("Lumina", ["Alice", "Bob"], 3700, [("Alice", "hi")],
                                      llm=FakeLLM(fail=True))
    assert msg.startswith("Call summary (1h 1m, with Alice, Bob)")
    assert "unavailable" in msg


async def test_thinking_and_long_dashes_stripped():
    msg = await summary.build_summary("Lumina", ["Alice"], 10, [("Alice", "hi")],
                                      llm=FakeLLM("<think>hmm</think>- A \u2014 B\u2013C"))
    assert "<think>" not in msg and "\u2014" not in msg and "\u2013" not in msg


def test_transcript_text_keeps_most_recent():
    lines = [("A", "x" * 50), ("B", "y" * 50), ("C", "z" * 50)]
    out = summary.transcript_text(lines, limit=120)
    assert out.startswith("B:") and out.endswith("z" * 50)


@pytest.mark.parametrize("val,on", [(None, True), ("on", True), ("1", True), ("off", False),
                                    ("0", False), ("false", False)])
def test_enabled_flag(monkeypatch, val, on):
    if val is None:
        monkeypatch.delenv("SKCALL_POST_SUMMARY", raising=False)
    else:
        monkeypatch.setenv("SKCALL_POST_SUMMARY", val)
    assert summary.enabled() is on


async def test_post_summary_uses_fresh_ocs():
    class OCS:
        def __init__(self):
            self.posted, self.open = [], False

        async def __aenter__(self):
            self.open = True
            return self

        async def __aexit__(self, *exc):
            self.open = False

        async def post_chat(self, room, msg):
            assert self.open
            self.posted.append((room, msg))
    o = OCS()
    await summary.post_summary("room0101", "Call summary", ocs=o)
    assert o.posted == [("room0101", "Call summary")] and not o.open


def test_transcript_bounded():
    clock = Clock()
    conv = Conversation(FakeSession(clock), FakeSTT([]), FakeTTS(), FakeBrain(),
                        LoopConfig(transcript_max_chars=100), clock=clock)
    for i in range(20):
        conv.note("Alice", f"line {i} " + "x" * 20)
    assert conv._transcript_chars <= 100
    assert conv.transcript[-1][1].startswith("line 19")
    conv.note("Alice", "   ")
    assert conv.transcript[-1][1].startswith("line 19")


async def test_conversation_records_joined_humans():
    clock = Clock()
    sess = FakeSession(clock)
    bot = Participant("sid-bot", "jarvis", "Jarvis", is_agent=True)
    conv = Conversation(sess, FakeSTT([]), FakeTTS(), FakeBrain(), LoopConfig(), clock=clock)
    task = asyncio.create_task(conv.run())
    await sess.q.put(ParticipantJoined(ALICE))
    await sess.q.put(ParticipantJoined(bot))
    await sess.q.put(CallEnded("test"))
    await asyncio.wait_for(task, 5)
    assert list(conv.joined.values()) == ["Alice"]
    assert conv.agent_display == "Lumina"


async def test_turn_lands_in_transcript():
    conv, sess, tts, clock = make(FakeSTT(["Hey Lumina, how are you?"]), FakeBrain())
    await speak(conv, clock, UTTER)
    await settle(conv)
    assert conv.transcript == [
        ("Alice", "Hey Lumina, how are you?"),
        ("Lumina", "I'm good, thanks for asking. How was your day today, Alice?")]
