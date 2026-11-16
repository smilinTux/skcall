"""F5TTS with a comma list of same-voice servers: ordered failover, 30 s sit-out."""
import asyncio
import io
import wave

import httpx
import numpy as np

from skcall.speech import F5TTS

A, B = "http://a/v1/audio/speech", "http://b/v1/audio/speech"


def _wav(n=2400, sr=24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.ones(n) * 8000).astype("<i2").tobytes())
    return buf.getvalue()


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


def _tts(handler, url=f"{A},{B}", clock=None):
    tts = F5TTS(url=url, model="tts-1", voice="bm_george", speed=1.0, trim=False, clock=clock or Clock())
    tts._http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return tts


def run(coro):
    return asyncio.run(coro)


def test_primary_used_when_healthy():
    hits = []

    def h(req):
        hits.append(req.url.host)
        return httpx.Response(200, content=_wav())

    out = run(_tts(h).synth("Hello there."))
    assert len(out) > 0 and hits == ["a"]


def test_fails_over_and_benches_the_dead_server():
    hits, clock = [], Clock()

    def h(req):
        hits.append(req.url.host)
        if req.url.host == "a":
            raise httpx.ConnectError("refused")
        return httpx.Response(200, content=_wav())

    tts = _tts(h, clock=clock)
    assert len(run(tts.synth("One."))) > 0
    assert len(run(tts.synth("Two."))) > 0
    assert hits == ["a", "b", "b"]  # second sentence skips the benched primary
    clock.t += 31
    run(tts.synth("Three."))
    assert hits[-2:] == ["a", "b"]  # primary retried after the sit-out


def test_all_down_returns_silence_not_exception():
    def h(req):
        return httpx.Response(503)

    out = run(_tts(h).synth("Anyone there?"))
    assert out.dtype == np.int16 and len(out) == 0


def test_single_url_unchanged():
    tts = F5TTS(url=A, model="tts-1", speed=1.0)
    assert tts.urls == [A] and tts.url == A
