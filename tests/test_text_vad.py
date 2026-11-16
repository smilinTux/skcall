import numpy as np

from skcall.core.audio import Reframer, normalize_peak, resample, rms, trim_silence, wav_to_pcm, pcm_to_wav
from skcall.core.text import (SentenceChunker, TranscriptDedup, is_leave_command, is_non_speech,
                              is_whisper_repetition, strip_formatting, too_many_words)
from skcall.core.vad import BargeInDetector, VADSegmenter
from skcall.speech import pick_audio_ctx

LOUD = (np.sin(np.arange(320) / 3) * 4000).astype(np.int16)
QUIET = np.zeros(320, np.int16)


def feed(seg, pattern, t0=0.0, gated=False):
    """pattern: list of (frame, count). Returns (utterances, end time)."""
    out, t = [], t0
    for frame, n in pattern:
        for _ in range(n):
            t += 0.02
            u = seg.push(frame, t, gated=gated)
            if u is not None:
                out.append(u)
    return out, t


# ------------------------------------------------------------------ VAD
def test_endpoint_fires_after_hangover():
    seg = VADSegmenter(rms_threshold=500, silence_hangover_ms=500, min_utterance_ms=300)
    utts, _ = feed(seg, [(QUIET, 10), (LOUD, 50), (QUIET, 24)])
    assert utts == []                                  # 480 ms of silence: not yet
    utts, _ = feed(seg, [(QUIET, 2)], t0=1.68)
    assert len(utts) == 1
    u = utts[0]
    assert abs((u.t_endpoint - u.t_last_voice) - 0.5) < 0.03
    assert 1.0 <= u.seconds <= 1.8                     # speech + preroll + hangover


def test_preroll_kept():
    seg = VADSegmenter(rms_threshold=500, silence_hangover_ms=200, min_utterance_ms=100, preroll_ms=200)
    utts, _ = feed(seg, [(QUIET, 20), (LOUD, 20), (QUIET, 12)])
    assert len(utts) == 1
    assert len(utts[0].pcm) >= 320 * (10 + 20 + 10)    # 10 frames of pre-roll


def test_short_blip_dropped():
    seg = VADSegmenter(rms_threshold=500, silence_hangover_ms=300, min_utterance_ms=400)
    utts, _ = feed(seg, [(LOUD, 5), (QUIET, 30)])
    assert utts == []


def test_max_length_forces_flush():
    seg = VADSegmenter(rms_threshold=500, max_utterance_ms=1000)
    utts, _ = feed(seg, [(LOUD, 60)])
    assert len(utts) == 1 and utts[0].forced


def test_gated_frames_are_discarded():
    seg = VADSegmenter(rms_threshold=500, silence_hangover_ms=200, min_utterance_ms=100)
    feed(seg, [(LOUD, 20)])
    utts, _ = feed(seg, [(LOUD, 10), (QUIET, 20)], gated=True)
    assert utts == [] and not seg.in_speech


def test_prime_replays_gated_frames():
    seg = VADSegmenter(rms_threshold=500, silence_hangover_ms=200, min_utterance_ms=100)
    frames = [(LOUD, 0.02 * i) for i in range(1, 30)] + [(QUIET, 0.02 * i) for i in range(30, 45)]
    u = seg.prime(frames)
    assert u is not None and u.t_start <= 0.03


def test_barge_in_dwell_and_decay():
    b = BargeInDetector(rms_threshold=1000, dwell_ms=100)
    assert not any(b.push(LOUD) for _ in range(4))
    assert b.push(LOUD)                                  # 5 x 20 ms
    b.reset()
    for _ in range(4):
        b.push(LOUD)
    for _ in range(4):
        b.push(QUIET)                                    # decays
    assert not b.push(LOUD)
    off = BargeInDetector(enabled=False)
    assert not any(off.push(LOUD) for _ in range(100))


# ------------------------------------------------------------------ chunking
def chunk_all(text, step=3, **kw):
    c = SentenceChunker(**kw)
    out = []
    for i in range(0, len(text), step):
        out += c.push(text[i:i + step])
    return out, c


def test_first_sentence_released_early():
    c = SentenceChunker()
    assert c.push("Hi Alice.") == []                      # might be "Hi Alice.5": wait for the space
    assert c.push(" I") == ["Hi Alice."]


def test_long_first_sentence_speaks_first_clause():
    out, c = chunk_all("Mm, I love the quiet of late night talks, when the world slows down and "
                       "everything feels more honest. You get the real stuff then.")
    out += c.flush()
    assert out[0] == "Mm, I love the quiet of late night talks,"
    assert out[1].startswith("when the world") and out[1].endswith("honest.")
    out, c = chunk_all("Oh I love that question, honestly.")
    assert out + c.flush() == ["Oh I love that question, honestly."]


def test_short_later_sentences_merge():
    out, c = chunk_all("Hello there. Yes. That is right, I think so. Okay.")
    out += c.flush()
    assert out[0] == "Hello there."
    assert out[1] == "Yes. That is right, I think so."
    assert out[-1] == "Okay."


def test_abbreviations_do_not_split():
    out, c = chunk_all("I talked to Dr. Smith today about it. Then we left.", step=2)
    out += c.flush()
    assert out == ["I talked to Dr. Smith today about it.", "Then we left."]


def test_long_run_cut_at_comma_not_mid_word():
    text = "word " * 30 + "and more, " + "tail " * 30
    out, c = chunk_all(text, max_chars=120)
    assert out and all(len(s) <= 121 for s in out)
    assert all(not s.endswith("wor") for s in out)


def test_truncated_tail_dropped():
    c = SentenceChunker()
    out = c.push("This one is done. And this one was cut off by the tok")
    out += c.flush(truncated=True)
    assert out == ["This one is done."]
    c = SentenceChunker()
    c.push("No terminator but complete reply")
    assert c.flush() == ["No terminator but complete reply"]


def test_strip_formatting_removes_dashes_and_markdown():
    s = strip_formatting("**Hey** \u2014 that is _great_ \U0001F600")
    assert "\u2014" not in s and "*" not in s and "\U0001F600" not in s


# ------------------------------------------------------------------ filters
def test_non_speech_and_repetition():
    assert is_non_speech("[BLANK_AUDIO]")
    assert is_non_speech(" you ")
    assert is_non_speech("Thank you.")
    assert not is_non_speech("Thank you for the help with the camera.")
    assert is_whisper_repetition("It's. It's. It's. It's. It's. It's. It's.")
    assert not is_whisper_repetition("I think it is what it is, honestly.")
    assert too_many_words("the " * 40, 1.0)
    assert not too_many_words("what is the weather like today", 2.0)


def test_dedup_window():
    t = [0.0]
    d = TranscriptDedup(window_s=3, clock=lambda: t[0])
    assert not d.is_duplicate("Hello Lumina.")
    t[0] = 1
    assert d.is_duplicate("hello lumina")
    t[0] = 10
    assert not d.is_duplicate("hello lumina")


def test_leave_command():
    assert is_leave_command("Lumina, leave the call.")
    assert is_leave_command("okay you can hang up now")
    assert not is_leave_command("I left the oven on")
    assert not is_leave_command("I have to leave the call soon", one_to_one=False)
    assert is_leave_command("Lumina please leave the call", one_to_one=False)


# ------------------------------------------------------------------ audio / stt ctx
def test_audio_helpers_roundtrip():
    x = (np.sin(np.arange(4800) / 5) * 10000).astype(np.int16)
    assert len(resample(x, 48000, 16000)) == 1600
    assert len(resample(x, 24000, 48000)) == 9600
    assert abs(int(np.abs(normalize_peak(x // 4).astype(np.int32)).max()) / 32767 - 0.97) < 0.01
    back = wav_to_pcm(pcm_to_wav(x, 24000), 48000)
    assert len(back) == 9600
    assert wav_to_pcm(b'{"error": "boom"}', 48000).size == 0
    r = Reframer(320)
    assert len(r.push(np.zeros(1000, np.int16))) == 3 and len(r.push(np.zeros(280, np.int16))) == 1


def test_pick_audio_ctx():
    assert pick_audio_ctx(3.0) == 512
    assert pick_audio_ctx(12.0) == 768
    assert pick_audio_ctx(20.0) == 0
    assert pick_audio_ctx(3.0, "1500") == 1500


def test_trim_silence_lead_tail_and_long_pause():
    sr = 48000
    voice = (np.sin(np.arange(sr // 2) / 4) * 20000).astype(np.int16)       # 0.5 s
    gap = np.zeros(sr * 2, np.int16)                                         # 2 s pause
    x = np.concatenate([np.zeros(sr, np.int16), voice, gap, voice, np.zeros(sr // 2, np.int16)])
    y = trim_silence(x, sr)
    assert len(y) / sr < 1.5                                                 # 4.5 s -> ~1.4 s
    assert np.abs(y[: int(0.07 * sr)].astype(np.int32)).max() > 1000        # voice within pad
    short = np.concatenate([voice, np.zeros(int(0.3 * sr), np.int16), voice])
    assert len(trim_silence(short, sr)) >= len(short) - int(0.01 * sr)       # short pause kept
    assert trim_silence(np.zeros(sr, np.int16), sr).size == 0
