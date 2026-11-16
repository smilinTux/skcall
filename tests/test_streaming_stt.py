"""Streaming (speculative) STT, transcript-driven endpointing and the hedged
STT client, with fake audio, fake STT and a fake clock. No network."""
import asyncio

import numpy as np
import pytest

from skcall.core.loop import Conversation, LoopConfig
from skcall.core.text import looks_complete
from skcall.core.vad import VADSegmenter
from skcall.speech import HedgedSTT
from tests.test_loop import (ALICE, LOUD, QUIET, Clock, FakeBrain, FakeSession, FakeSTT, FakeTTS,
                             settle, speak)


def make_spec(stt, brain, hold=False, **cfg):
    clock = Clock()
    sess = FakeSession(clock, hold=hold)
    c = LoopConfig(filler=False, hangover_ms=500, min_utterance_ms=300, spec_stt=True,
                   spec_silence_ms=150, endpoint_complete_ms=250, endpoint_incomplete_ms=900,
                   operators=("alice",))
    for k, v in cfg.items():
        setattr(c, k, v)
    conv = Conversation(sess, stt, FakeTTS(), brain, c, clock=clock)
    return conv, sess, clock


# 200 ms pre-roll, 1 s of speech, 1.2 s of silence
SENTENCE = [(QUIET, 10), (LOUD, 50), (QUIET, 60)]


async def test_complete_sentence_endpoints_early_on_the_speculative_transcript():
    stt, brain = FakeSTT(["Hey Lumina, how are you?"]), FakeBrain()
    conv, sess, clock = make_spec(stt, brain)
    await speak(conv, clock, SENTENCE)
    await settle(conv)
    assert len(stt.calls) == 1                     # one transcript, asked at the pause
    assert stt.calls[0] >= 16000 * 1.0
    assert brain.turns[0].text == "Hey Lumina, how are you?"
    t = conv.timings[0]
    assert t["stt_spec_at"] == pytest.approx(160, abs=30)
    assert t["endpoint"] == pytest.approx(260, abs=30)    # not the 500 ms hangover
    assert conv.stats["early_endpoint"] == 1
    assert len(sess.audio_out.written) == 2


async def test_trailing_off_waits_for_the_long_hangover_then_hears_it_whole():
    stt = FakeSTT(["So I was thinking about", "So I was thinking about the garage cameras."])
    brain = FakeBrain()
    conv, sess, clock = make_spec(stt, brain)
    # a 600 ms pause after "about": longer than the 500 ms hangover, but the
    # transcript is unfinished, so the utterance stays open
    await speak(conv, clock, [(QUIET, 10), (LOUD, 50), (QUIET, 30), (LOUD, 50), (QUIET, 60)])
    await settle(conv)
    assert len(brain.turns) == 1 and brain.cancelled == 0
    assert brain.turns[0].text == "So I was thinking about the garage cameras."
    assert stt.calls[1] > stt.calls[0] * 1.5       # the second pass covers both halves
    assert conv.stats["spec_launched"] == 2


async def test_unfinished_phrase_that_really_ended_closes_at_the_backstop():
    stt, brain = FakeSTT(["So I was thinking"]), FakeBrain()
    conv, sess, clock = make_spec(stt, brain)
    await speak(conv, clock, SENTENCE)
    await settle(conv)
    assert brain.turns[0].text == "So I was thinking"
    assert conv.timings[0]["endpoint"] == pytest.approx(900, abs=30)
    assert len(stt.calls) == 1


async def test_slow_transcript_in_flight_is_awaited_not_repeated():
    stt, brain = FakeSTT(["What time is it?"], delay=0.2), FakeBrain()
    conv, sess, clock = make_spec(stt, brain)
    await speak(conv, clock, SENTENCE)             # frames run far faster than 0.2 s
    await settle(conv)
    assert len(stt.calls) == 1
    t = conv.timings[0]
    assert t["endpoint"] == pytest.approx(900, abs=30)    # backstop: a slow transcript is waited for
    assert brain.turns[0].text == "What time is it?"


class GatedSTT:
    """Answers only when released, so a test controls when the transcript lands."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls: list[int] = []
        self.gate = asyncio.Event()

    async def transcribe(self, pcm):
        self.calls.append(len(pcm))
        await self.gate.wait()
        return self.texts.pop(0) if self.texts else ""


async def test_utterance_stays_open_while_the_transcript_is_in_flight():
    stt = GatedSTT(["What kind of", "What kind of music do you like?"])
    brain = FakeBrain()
    conv, sess, clock = make_spec(stt, brain)
    await speak(conv, clock, [(QUIET, 10), (LOUD, 50), (QUIET, 30)])    # 600 ms > 500 ms hangover
    assert conv.turn is None and conv.stats["spec_launched"] == 1
    stt.gate.set()
    await asyncio.sleep(0.01)                       # "What kind of": unfinished, keep waiting
    await speak(conv, clock, [(LOUD, 50), (QUIET, 30)])
    await settle(conv)
    assert len(brain.turns) == 1 and conv.stats.get("merged", 0) == 0
    assert brain.turns[0].text == "What kind of music do you like?"
    assert conv.timings[0]["endpoint"] < 400        # closed early on the finished transcript


async def test_speech_resuming_cancels_the_stale_transcript():
    stt = FakeSTT(["Tell me", "Tell me about the fox in the woods."], delay=0.2)
    brain = FakeBrain()
    conv, sess, clock = make_spec(stt, brain)
    # 300 ms pause: a transcript is launched at 150 ms, then speech resumes
    await speak(conv, clock, [(QUIET, 10), (LOUD, 50), (QUIET, 15), (LOUD, 50)])
    assert conv.stats["spec_launched"] == 1 and conv.stats["spec_stale"] == 1
    await speak(conv, clock, [(QUIET, 60)])
    await settle(conv)
    assert len(stt.calls) == 2 and stt.calls[1] > stt.calls[0] * 1.5
    assert len(brain.turns) == 1
    assert brain.turns[0].text == "Tell me about the fox in the woods."


async def test_speaker_continuing_after_an_early_endpoint_restarts_the_reply():
    """Speculative start: the reply begins on the early transcript; if the
    speaker goes on before she has said anything, it is cancelled and redone
    with both pieces."""
    stt = FakeSTT(["What's the weather like?", "In Chicago tomorrow."])
    brain = FakeBrain(first_delay=0.3)
    conv, sess, clock = make_spec(stt, brain)
    await speak(conv, clock, [(QUIET, 10), (LOUD, 50), (QUIET, 15)])   # early endpoint at 250 ms
    await asyncio.sleep(0.02)
    assert len(brain.turns) == 1                   # already thinking
    await speak(conv, clock, SENTENCE)
    await settle(conv)
    assert brain.cancelled == 1
    assert brain.turns[-1].text == "What's the weather like? In Chicago tomorrow."
    assert len(conv.timings) == 1


async def test_echo_gate_discards_the_open_utterance_and_its_transcript():
    stt, brain = FakeSTT(["should never be used"], delay=0.2), FakeBrain()
    conv, sess, clock = make_spec(stt, brain, hold=True, barge_in=False)
    await speak(conv, clock, [(QUIET, 10), (LOUD, 50), (QUIET, 10)])    # transcript launched
    assert conv.stats["spec_launched"] == 1
    sess.audio_out.write(np.ones(48000 * 3, np.int16))                  # she starts talking
    await speak(conv, clock, [(QUIET, 40)])
    await asyncio.sleep(0.25)
    assert conv.stats["spec_stale"] == 1
    assert brain.turns == [] and conv.turn is None


async def test_non_speech_transcript_falls_back_to_the_hangover_and_is_ignored():
    stt, brain = FakeSTT(["[BLANK_AUDIO]"]), FakeBrain()
    conv, sess, clock = make_spec(stt, brain)
    await speak(conv, clock, SENTENCE)
    await settle(conv)
    assert brain.turns == [] and conv.stats["early_endpoint"] == 0
    assert len(stt.calls) == 1


async def test_barge_in_still_works_with_streaming_stt():
    stt = FakeSTT(["Tell me a long story.", "Wait, stop, different question."])
    brain = FakeBrain(reply="Once upon a time there was a fox. It lived in a very big forest.")
    conv, sess, clock = make_spec(stt, brain, hold=True)
    await speak(conv, clock, SENTENCE)
    for _ in range(100):
        if sess.audio_out.buffered_s > 0:
            break
        await asyncio.sleep(0.005)
    first = conv.turn
    await speak(conv, clock, [(LOUD, 60), (QUIET, 30)])
    await asyncio.sleep(0.01)
    assert conv.stats["barge_in"] == 1
    assert first.task.cancelled() or first.task.done()
    assert stt.calls[-1] >= 16000 * 1.0            # the interruption from its start


# ------------------------------------------------------------------ pieces
def test_looks_complete():
    for t in ("Hey Lumina, how are you?", "What time of day do you like best?",
              "Tell me a story about a fox.", "Okay, stop there.", "What are you thinking about?",
              "Thank you.", "I think so.", "Wow!"):
        assert looks_complete(t), t
    for t in ("", "So I was thinking", "So I was thinking about", "I went to the.", "and then,",
              "Can you check my", "Well...", "Hello", "Because the?", "Okay."):
        assert not looks_complete(t), t


def test_segmenter_snapshot_and_early_endpoint():
    seg = VADSegmenter(rms_threshold=500, silence_hangover_ms=900, min_utterance_ms=300)
    t = 0.0
    for _ in range(10):
        t += 0.02
        seg.push(QUIET, t)
    for _ in range(25):
        t += 0.02
        assert seg.push(LOUD, t) is None
    assert seg.in_speech and seg.onsets == 1
    t += 0.02
    seg.push(QUIET, t)
    assert seg.silence_ms(t) == pytest.approx(20, abs=1)
    assert len(seg.snapshot()) == (10 + 26) * 960       # pre-roll + speech + the quiet frame
    assert seg.speech_ms() >= 300
    u = seg.endpoint(t)
    assert u is not None and u.t_endpoint == t and not seg.in_speech
    assert seg.endpoint(t) is None


class Backend:
    def __init__(self, text="hello there.", delay=0.0, fail=False, url=""):
        self.text, self.delay, self.fail, self.url = text, delay, fail, url
        self.calls = 0
        self.cancelled = 0

    async def transcribe_or_raise(self, pcm):
        self.calls += 1
        try:
            await asyncio.sleep(self.delay)
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        if self.fail:
            raise RuntimeError("down")
        return self.text

    async def warmup(self):
        return 0.0


async def test_hedged_stt_prefers_the_first_backend():
    a, b = Backend("from a", url="http://a:1/x"), Backend("from b", url="http://b:1/x")
    h = HedgedSTT([a, b], hedge_s=0.05)
    assert await h.transcribe(np.zeros(16000, np.int16)) == "from a"
    assert b.calls == 0 and h.last_backend == "a:1"


async def test_hedged_stt_hedges_a_slow_primary_and_benches_it():
    a, b = Backend("from a", delay=0.5), Backend("from b", delay=0.01)
    h = HedgedSTT([a, b], hedge_s=0.05)
    assert await h.transcribe(np.zeros(16000, np.int16)) == "from b"
    await asyncio.sleep(0)
    assert a.cancelled == 1
    assert await h.transcribe(np.zeros(16000, np.int16)) == "from b"    # b is first now
    assert a.calls == 1


async def test_hedged_stt_fails_over_and_survives_all_down():
    a, b = Backend(fail=True), Backend("from b")
    h = HedgedSTT([a, b], hedge_s=1.0)
    assert await h.transcribe(np.zeros(16000, np.int16)) == "from b"
    dead = HedgedSTT([Backend(fail=True), Backend(fail=True)], hedge_s=1.0)
    assert await dead.transcribe(np.zeros(16000, np.int16)) == ""


def test_stt_backend_spec_parsing():
    from skcall.speech import OpenAISTT, WhisperCppSTT, stt_backend
    fw = stt_backend("fw:http://stt-a.example.org:18794/v1/audio/transcriptions")
    cpp = stt_backend("cpp:http://192.0.2.10:18794/v1/audio/transcriptions")
    bare = stt_backend("http://192.0.2.10:18794/v1/audio/transcriptions")
    assert type(fw) is OpenAISTT and fw.url == "http://stt-a.example.org:18794/v1/audio/transcriptions"
    assert type(cpp) is WhisperCppSTT and cpp.url.startswith("http://192.0.2.10")
    assert type(bare) is WhisperCppSTT and bare.url.startswith("http://192.0.2.10")
    assert HedgedSTT([fw, cpp]).name(0) == "stt-a.example.org:18794"
