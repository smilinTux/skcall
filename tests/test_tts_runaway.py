"""F5TTS guards against a cloning TTS inventing speech: non-speech chunks are
never sent, and a render far longer than its text is re-rendered, then tried
on the other servers, then dropped (silence beats gibberish)."""
import asyncio
import io
import wave

import httpx
import numpy as np

from skcall.speech import F5TTS, plausible_s

A, B = "http://a/v1/audio/speech", "http://b/v1/audio/speech"
SR = 24000


def _wav(secs: float) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.ones(int(secs * SR)) * 8000).astype("<i2").tobytes())
    return buf.getvalue()


def _tts(handler, url=f"{A},{B}"):
    tts = F5TTS(url=url, model="tts-1", voice="v", speed=1.0, trim=False)
    tts._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return tts


def run(coro):
    return asyncio.run(coro)


def test_plausible_grows_with_text():
    assert plausible_s("a") < 1.5
    assert plausible_s("x" * 70) > 10


def test_non_speech_chunks_never_reach_the_server():
    hits = []

    def h(req):
        hits.append(1)
        return httpx.Response(200, content=_wav(0.3))

    tts = _tts(h)
    for s in ("?", "...", "♡", "*", " ", "", "…"):
        assert len(run(tts.synth(s))) == 0
    assert hits == []


def test_runaway_rerendered_on_same_server_first():
    hits = []

    def h(req):
        hits.append(req.url.host)
        return httpx.Response(200, content=_wav(6.0 if len(hits) == 1 else 0.3))

    out = run(_tts(h).synth("a"))
    assert hits == ["a", "a"] and len(out) > 0


def test_runaway_twice_moves_to_next_server():
    hits = []

    def h(req):
        hits.append(req.url.host)
        return httpx.Response(200, content=_wav(6.0 if req.url.host == "a" else 0.4))

    out = run(_tts(h).synth("Oh!"))
    assert hits == ["a", "a", "b"] and len(out) > 0


def test_all_runaway_gives_silence():
    def h(req):
        return httpx.Response(200, content=_wav(6.0))

    out = run(_tts(h).synth("a"))
    assert out.dtype == np.int16 and len(out) == 0


def test_dead_primary_not_retried_within_one_sentence():
    hits = []

    def h(req):
        hits.append(req.url.host)
        if req.url.host == "a":
            raise httpx.ConnectError("refused")
        return httpx.Response(200, content=_wav(0.3))

    assert len(run(_tts(h).synth("Hello."))) > 0
    assert hits == ["a", "b"]


def test_normal_sentence_one_request():
    hits = []

    def h(req):
        hits.append(req.url.host)
        return httpx.Response(200, content=_wav(4.2))

    text = "I found the bug: the tests were writing to the wrong config folder."
    assert len(run(_tts(h).synth(text))) > 0 and hits == ["a"]
