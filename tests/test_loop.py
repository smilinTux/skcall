"""Turn-loop tests with fake audio in/out, fake STT/TTS/brain. No network."""
import asyncio

import numpy as np
import pytest

from skcall.core.loop import Conversation, LoopConfig
from skcall.core.session import (AudioTrack, CallEnded, FakeSink, Participant, ParticipantJoined)

LOUD = (np.sin(np.arange(960) / 3) * 4000).astype(np.int16)
QUIET = np.zeros(960, np.int16)
ALICE = Participant("sid-alice", "alice", "Alice")


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


class FakeSession:
    def __init__(self, clock, hold=False, parts=(ALICE,)):
        self.self_user_id = "lumina"
        self.audio_out = FakeSink(clock=clock, hold=hold)
        self._parts = list(parts)
        self.q: asyncio.Queue = asyncio.Queue()
        self.chat: list[str] = []

    def participants(self):
        return list(self._parts)

    async def events(self):
        while True:
            ev = await self.q.get()
            yield ev
            if isinstance(ev, CallEnded):
                return

    async def post_chat(self, text):
        self.chat.append(text)

    async def leave(self):
        pass


class FakeSTT:
    def __init__(self, texts, delay=0.0):
        self.texts = list(texts)
        self.delay = delay
        self.calls: list[int] = []

    async def transcribe(self, pcm):
        self.calls.append(len(pcm))
        text = self.texts.pop(0) if self.texts else ""
        if self.delay:
            await asyncio.sleep(self.delay)
        return text

    async def warmup(self):
        return 0.0


class FakeTTS:
    def __init__(self):
        self.texts: list[str] = []

    async def synth(self, text):
        self.texts.append(text)
        await asyncio.sleep(0)
        n = 4800 if text == "Mm." else 24000
        return np.full(n, 1000, np.int16)


class FakeBrain:
    def __init__(self, reply="I'm good, thanks for asking. How was your day today, Alice?",
                 first_delay=0.0):
        self.reply = reply
        self.first_delay = first_delay
        self.turns = []
        self.cancelled = 0

    async def stream(self, turn):
        self.turns.append(turn)
        try:
            if self.first_delay:
                await asyncio.sleep(self.first_delay)
            for w in self.reply.split(" "):
                yield w + " "
                await asyncio.sleep(0)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise

    bucket = "lookup"

    def filler(self, text):
        return "Mm.", self.bucket

    def all_fillers(self):
        return ["Mm."]


def make(stt, brain, hold=False, **cfg):
    clock = Clock()
    sess = FakeSession(clock, hold=hold)
    # The classic path (whole-utterance STT after the hangover); streaming STT
    # has its own tests in test_streaming_stt.py.
    c = LoopConfig(filler=False, hangover_ms=500, min_utterance_ms=300, spec_stt=False,
                   operators=("alice",))
    for k, v in cfg.items():
        setattr(c, k, v)
    tts = FakeTTS()
    conv = Conversation(sess, stt, tts, brain, c, clock=clock)
    return conv, sess, tts, clock


async def speak(conv, clock, pattern, who=ALICE):
    async def frames():
        for frame, n in pattern:
            for _ in range(n):
                clock.t += 0.02
                yield frame
                await asyncio.sleep(0)
    await conv.pump(AudioTrack(who, frames()))


async def settle(conv):
    for _ in range(200):
        t = conv.turn
        if t is None or t.task is None or t.task.done():
            break
        await asyncio.sleep(0.005)
    await asyncio.sleep(0)


UTTER = [(QUIET, 10), (LOUD, 50), (QUIET, 30)]


async def test_turn_end_to_end_streams_sentences():
    stt, brain = FakeSTT(["Hey Lumina, how are you?"]), FakeBrain()
    conv, sess, tts, clock = make(stt, brain)
    await speak(conv, clock, UTTER)
    await settle(conv)
    assert tts.texts == ["I'm good, thanks for asking.", "How was your day today, Alice?"]
    assert len(sess.audio_out.written) == 2
    assert brain.turns[0].mode == "private" and brain.turns[0].is_operator
    assert conv.history[0] == {"role": "user", "content": "Hey Lumina, how are you?"}
    assert "How was your day" in conv.history[1]["content"]
    t = conv.timings[0]
    assert t["endpoint"] == pytest.approx(500, abs=30)
    for stage in ("stt", "first_token", "first_sentence", "first_tts", "first_audio"):
        assert stage in t


async def test_non_speech_is_ignored():
    stt, brain = FakeSTT(["[BLANK_AUDIO]"]), FakeBrain()
    conv, sess, tts, clock = make(stt, brain)
    await speak(conv, clock, UTTER)
    await settle(conv)
    assert brain.turns == [] and tts.texts == []


async def test_barge_in_cancels_reply_and_hears_interruption():
    stt = FakeSTT(["Tell me a long story.", "Wait, stop, different question."])
    brain = FakeBrain(reply="Once upon a time there was a fox. It lived in a very big forest.")
    conv, sess, tts, clock = make(stt, brain, hold=True)
    await speak(conv, clock, UTTER)
    for _ in range(100):                       # until she is audible
        if sess.audio_out.buffered_s > 0:
            break
        await asyncio.sleep(0.005)
    assert sess.audio_out.buffered_s > 0
    first = conv.turn
    await speak(conv, clock, [(LOUD, 60), (QUIET, 30)])
    await asyncio.sleep(0.01)
    assert conv.stats["barge_in"] == 1
    assert sess.audio_out.cleared >= 1
    assert first.task.cancelled() or first.task.done()
    assert len(stt.calls) == 2
    assert stt.calls[1] >= 16000 * 1.0         # the interruption from its start, not the tail
    assert conv.turn is not first
    assert "(interrupted)" in conv.history[1]["content"]


async def test_barge_in_can_be_disabled_and_echo_is_gated():
    stt, brain = FakeSTT(["should never be heard"]), FakeBrain()
    conv, sess, tts, clock = make(stt, brain, hold=True, barge_in=False, echo_tail_s=0.6)
    sess.audio_out.write(np.ones(48000 * 3, np.int16))          # she is talking
    await speak(conv, clock, UTTER)                              # her echo / the human
    await asyncio.sleep(0.01)
    assert stt.calls == [] and conv.stats["barge_in"] == 0
    sess.audio_out.release()
    await speak(conv, clock, [(LOUD, 20)])                       # within the 0.6 s tail
    await speak(conv, clock, [(QUIET, 30)])
    assert stt.calls == []
    await speak(conv, clock, UTTER)                              # after the tail
    await settle(conv)
    assert len(stt.calls) == 1


async def test_pause_before_answer_merges_into_one_turn():
    stt = FakeSTT(["So I was thinking", "about the garage cameras."])
    brain = FakeBrain(first_delay=0.3)
    conv, sess, tts, clock = make(stt, brain)
    await speak(conv, clock, UTTER)
    await asyncio.sleep(0.02)                   # STT done, brain still thinking
    await speak(conv, clock, UTTER)
    await settle(conv)
    assert brain.cancelled == 1
    assert brain.turns[-1].text == "So I was thinking about the garage cameras."
    assert len(conv.timings) == 1


async def test_merged_turn_with_noise_tail_still_answers_the_first_words():
    # 2026-10-26: "Give me a full summary of that session." then a noise blip
    # that transcribed empty; the merge dropped the request and nothing answered.
    stt = FakeSTT(["Give me a full summary of that session.", ""])
    brain = FakeBrain(first_delay=0.3)
    conv, sess, tts, clock = make(stt, brain)
    await speak(conv, clock, UTTER)
    await asyncio.sleep(0.02)                   # STT done, brain still thinking
    await speak(conv, clock, UTTER)             # the blip
    await settle(conv)
    assert brain.cancelled == 1
    assert brain.turns[-1].text == "Give me a full summary of that session."
    assert len(conv.timings) == 1


async def test_pause_during_stt_merges_audio():
    stt = FakeSTT(["first half", "So I was thinking about the garage cameras."], delay=0.2)
    brain = FakeBrain()
    conv, sess, tts, clock = make(stt, brain)
    await speak(conv, clock, UTTER)
    await speak(conv, clock, UTTER)             # first STT still running
    await settle(conv)
    assert stt.calls[1] > stt.calls[0] * 1.9    # both pieces transcribed together
    assert brain.turns[-1].text == "So I was thinking about the garage cameras."


async def test_leave_voice_command_ends_conversation():
    stt, brain = FakeSTT(["Lumina, leave the call."]), FakeBrain()
    conv, sess, tts, clock = make(stt, brain)
    await speak(conv, clock, UTTER)
    await settle(conv)
    assert conv.end_reason == "voice command"
    assert tts.texts == [conv.cfg.leave_line]
    assert brain.turns == []


async def test_filler_only_when_slow_and_reply_follows():
    stt, brain = FakeSTT(["What did we talk about yesterday?"]), FakeBrain(first_delay=0.3)
    conv, sess, tts, clock = make(stt, brain, filler=True, filler_delay_s=0.05)
    await conv.prerender_fillers()
    await speak(conv, clock, UTTER)
    await settle(conv)
    lens = [len(x) for x in sess.audio_out.written]
    assert lens[0] == 4800 and len(lens) == 3   # "Mm." then the two sentences
    assert conv.stats["fillers"] == 1

    # Alice: no filler on plain chat, even when the reply is slow.
    stt3, chatty = FakeSTT(["how are you?"]), FakeBrain(first_delay=0.3)
    chatty.bucket = "chat"
    conv3, sess3, _, clock3 = make(stt3, chatty, filler=True, filler_delay_s=0.05)
    await conv3.prerender_fillers()
    await speak(conv3, clock3, UTTER)
    await settle(conv3)
    assert conv3.stats["fillers"] == 0 and len(sess3.audio_out.written) == 2

    stt2, fast = FakeSTT(["hi"]), FakeBrain(reply="Hi there.")
    conv2, sess2, _, clock2 = make(stt2, fast, filler=True, filler_delay_s=0.5)
    await conv2.prerender_fillers()
    await speak(conv2, clock2, UTTER)
    await settle(conv2)
    assert [len(x) for x in sess2.audio_out.written] == [24000]


async def test_group_call_needs_name():
    guest = Participant("sid-guest", "alice", "Alice")
    stt = FakeSTT(["I think the meeting went fine.", "Lumina, what do you think?"])
    brain = FakeBrain(reply="I think it went well.")
    conv, sess, tts, clock = make(stt, brain)
    sess._parts = [ALICE, guest]
    await speak(conv, clock, UTTER, who=guest)
    await settle(conv)
    assert brain.turns == []
    await speak(conv, clock, UTTER, who=guest)
    await settle(conv)
    assert len(brain.turns) == 1 and brain.turns[0].mode == "group"


async def test_run_handles_events_and_call_end():
    stt, brain = FakeSTT(["Hello Lumina."]), FakeBrain(reply="Hi Alice.")
    conv, sess, tts, clock = make(stt, brain)

    async def frames():
        for frame, n in UTTER:
            for _ in range(n):
                clock.t += 0.02
                yield frame
                await asyncio.sleep(0)

    async def script():
        await sess.q.put(ParticipantJoined(ALICE))
        await sess.q.put(AudioTrack(ALICE, frames()))
        for _ in range(200):
            if conv.timings:
                break
            await asyncio.sleep(0.005)
        await sess.q.put(CallEnded("test over"))

    asyncio.ensure_future(script())
    reason = await asyncio.wait_for(conv.run(), 5)
    assert reason == "test over"
    assert conv.summary()["turns"] == 1


JARVIS = Participant("sid-jarvis", "jarvis", "Jarvis", is_agent=True)


async def test_another_agent_is_never_heard():
    stt, brain = FakeSTT(["Lumina, what do you think?"]), FakeBrain()
    conv, sess, tts, clock = make(stt, brain)
    sess._parts = [ALICE, JARVIS]
    await speak(conv, clock, UTTER, who=JARVIS)
    await settle(conv)
    assert stt.calls == [] and brain.turns == [] and JARVIS.pid not in conv.listeners


async def test_one_person_with_two_agents_needs_her_name():
    stt = FakeSTT(["How is everyone doing?", "Lumina, how are you?"])
    brain = FakeBrain(reply="Good, thanks.")
    conv, sess, tts, clock = make(stt, brain)
    sess._parts = [ALICE, JARVIS]
    await speak(conv, clock, UTTER)
    await settle(conv)
    assert brain.turns == []
    await speak(conv, clock, UTTER)
    await settle(conv)
    assert len(brain.turns) == 1


async def test_non_operator_guest_is_answered_when_he_names_her():
    bob = Participant("sid-bob", "bob", "Bob")
    stt, brain = FakeSTT(["Lumina, can you hear me?"]), FakeBrain(reply="Yes, Bob.")
    conv, sess, tts, clock = make(stt, brain, operators=("alice",))
    sess._parts = [Participant("sid-alice", "alice", "Alice"), bob, JARVIS]
    await speak(conv, clock, UTTER, who=bob)
    await settle(conv)
    assert len(brain.turns) == 1 and not brain.turns[0].is_operator
