"""STT and TTS clients for the existing speech services.

  STT  any OpenAI-shaped /v1/audio/transcriptions server: whisper.cpp
       (large-v3-turbo), faster-whisper or HF whisper large-v3, or the
       services/stt dispatcher in front of several of them
  TTS  any OpenAI-shaped /v1/audio/speech server that returns WAV: F5-TTS
       (a cloned voice, 24 kHz) or Kokoro (model tts-1), for example
       services/tts

All stay behind HTTP (no models in the call process).

STT backends measured 2026-10-25 (6 F5-voiced clips of 0.9 to 4.5 s, warm,
medians including ~45 ms network RTT): faster-whisper int8 CUDA 301 ms, HF
whisper fp16 on an RTX 5090 381 ms, whisper on Intel XPU 1288 ms, whisper.cpp
at audio_ctx 512 1283 ms. Same transcripts on all. `HedgedSTT` asks the
fastest first and hedges to the next one when it is slow or down, so a busy
shared transcription pool never stalls a call.

Latency levers measured 2026-10-25 (see README):
- whisper.cpp pads every request to a 30 s window, so a 1.3 s clip costs the
  same ~3.9 s as a 10 s clip. `audio_ctx` shrinks the encoder window: 512
  (10.2 s) brings it to ~1.3 s with the same transcripts, 768 to ~1.85 s.
  Below ~500 large-v3-turbo degenerates into "It's. It's. It's.", so the
  context is picked per clip and a degenerate result is retried at full size.
- F5 `nfe_step` 16 instead of the default 32 halves synthesis (1.75 -> 0.95 s
  for a 51-character sentence) and round-trips through whisper unchanged.
  F5's /stream endpoint is not incremental (first chunk == whole clip), so
  sentence chunking is the streaming lever.
- F5 starts every clip with 0.3 to 1.7 s of silence (worst on short phrases)
  and sometimes pauses 2 s mid-phrase; trimming that is pure first-audio win.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Protocol

import httpx
import numpy as np

from skcall.agents import ConfigError
from skcall.core.audio import CALL_RATE, STT_RATE, normalize_peak, pcm_to_wav, trim_silence, wav_to_pcm
from skcall.core.text import is_whisper_repetition, too_many_words

LOG = logging.getLogger("skcall.speech")

# No built-in endpoints: the site config sets these (join.env).
STT_URL = os.environ.get("SKCALL_STT_URL", "")
# Ordered STT backends, "kind:url" (cpp = whisper.cpp, fw = an OpenAI-shaped
# faster-whisper/HF server). The first is asked; the next one is asked as well
# if the first has not answered within SKCALL_STT_HEDGE_MS.
# Without SKCALL_STT_BACKENDS, SKCALL_STT_URL alone is the one backend.
STT_BACKENDS = os.environ.get("SKCALL_STT_BACKENDS", f"cpp:{STT_URL}" if STT_URL else "")
TTS_URL = os.environ.get("SKCALL_TTS_URL", "")


def bearer(token: str | None) -> dict[str, str]:
    """The Authorization header for `token`, or none when it is empty.

    The per-client tokens (SKCALL_TTS_TOKEN, SKCALL_STT_TOKEN; unset: no
    header) are read when a client is built, so a site env loaded after
    import still counts."""
    token = (token or "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


class STT(Protocol):
    async def transcribe(self, pcm16k: np.ndarray) -> str: ...
    async def warmup(self) -> float: ...


class TTS(Protocol):
    async def synth(self, text: str) -> np.ndarray: ...   # int16 mono CALL_RATE


def pick_audio_ctx(seconds: float, mode: str = "auto") -> int:
    """Encoder context for a clip (whisper frames: 50 per second, 1500 = 30 s)."""
    if mode != "auto":
        return int(mode)
    if seconds <= 9.5:
        return 512
    if seconds <= 14.5:
        return 768
    return 0  # server default (full 30 s window)


class WhisperCppSTT:
    def __init__(self, url: str = STT_URL, language: str = "en",
                 audio_ctx: str = os.environ.get("SKCALL_STT_AUDIO_CTX", "auto"),
                 timeout: float = 60, token: str | None = None,
                 transport: httpx.AsyncBaseTransport | None = None):
        """token: bearer token for the service (default SKCALL_STT_TOKEN).
        transport: an httpx transport (tests)."""
        self.url = url
        self.language = language
        self.audio_ctx = audio_ctx
        token = os.environ.get("SKCALL_STT_TOKEN", "") if token is None else token
        self._http = httpx.AsyncClient(timeout=timeout, headers=bearer(token), transport=transport)
        self.last_ms = 0.0
        self.retries = 0

    async def _post(self, wav: bytes, ctx: int) -> str:
        data = {"response_format": "json", "temperature": "0.0", "temperature_inc": "0.0",
                "no_timestamps": "true"}
        if self.language:
            data["language"] = self.language
        if ctx:
            data["audio_ctx"] = str(ctx)
        r = await self._http.post(self.url, files={"file": ("a.wav", wav, "audio/wav")}, data=data)
        r.raise_for_status()
        return " ".join((r.json().get("text") or "").split())

    async def transcribe_or_raise(self, pcm16k: np.ndarray) -> str:
        secs = len(pcm16k) / STT_RATE
        wav = pcm_to_wav(pcm16k, STT_RATE)
        ctx = pick_audio_ctx(secs, self.audio_ctx)
        t = time.monotonic()
        try:
            text = await self._post(wav, ctx)
            if ctx and text and (is_whisper_repetition(text) or too_many_words(text, secs)):
                LOG.info("stt: degenerate at audio_ctx=%d (%r); retrying full", ctx, text[:60])
                self.retries += 1
                text = await self._post(wav, 0)
        finally:
            self.last_ms = (time.monotonic() - t) * 1000
        return text.strip()

    async def transcribe(self, pcm16k: np.ndarray) -> str:
        try:
            return await self.transcribe_or_raise(pcm16k)
        except Exception as e:
            LOG.error("STT failed: %s: %s (url=%s)", type(e).__name__, e or "(no detail)", self.url)
            return ""

    async def warmup(self) -> float:
        """A tiny request so the first real utterance does not pay a cold start
        (P0 measured 63.7 s for the first call of the day)."""
        noise = (np.random.default_rng(0).standard_normal(STT_RATE // 2) * 30).astype(np.int16)
        t = time.monotonic()
        try:
            await self._post(pcm_to_wav(noise, STT_RATE), 512)
        except Exception as e:
            LOG.warning("STT warmup failed: %s", e)
        return (time.monotonic() - t) * 1000

    async def aclose(self) -> None:
        await self._http.aclose()


class OpenAISTT(WhisperCppSTT):
    """An OpenAI-shaped /v1/audio/transcriptions server (faster-whisper or HF
    whisper): no audio_ctx, the server decides."""

    async def _post(self, wav: bytes, ctx: int) -> str:
        data = {"response_format": "json", "temperature": "0.0"}
        if self.language:
            data["language"] = self.language
        r = await self._http.post(self.url, files={"file": ("a.wav", wav, "audio/wav")}, data=data)
        r.raise_for_status()
        return " ".join((r.json().get("text") or "").split())

    async def transcribe_or_raise(self, pcm16k: np.ndarray) -> str:
        t = time.monotonic()
        try:
            return (await self._post(pcm_to_wav(pcm16k, STT_RATE), 0)).strip()
        finally:
            self.last_ms = (time.monotonic() - t) * 1000

    async def warmup(self) -> float:
        noise = (np.random.default_rng(0).standard_normal(STT_RATE // 2) * 30).astype(np.int16)
        t = time.monotonic()
        try:
            await self._post(pcm_to_wav(noise, STT_RATE), 0)
        except Exception as e:
            LOG.warning("STT warmup failed (%s): %s", self.url, e)
        return (time.monotonic() - t) * 1000


def stt_backend(spec: str, timeout: float = 30, token: str | None = None,
                transport: httpx.AsyncBaseTransport | None = None):
    """'fw:http://host:port/path' or 'cpp:http://...' (a bare URL is whisper.cpp).
    token defaults to SKCALL_STT_TOKEN."""
    kind, sep, url = spec.strip().partition(":")
    if not sep or kind not in ("cpp", "fw"):
        kind, url = "cpp", spec.strip()
    return (OpenAISTT if kind == "fw" else WhisperCppSTT)(url=url.strip(), timeout=timeout,
                                                           token=token, transport=transport)


class HedgedSTT:
    """Several STT backends behind one transcribe().

    The first healthy backend is asked; if it has not answered within
    `hedge_s`, the next one is asked too, and the first answer wins (the
    others are cancelled). A backend that errors, or a primary that lost a
    hedge, sits out for `penalty_s`. Records which backend answered."""

    def __init__(self, backends: list | None = None, hedge_s: float | None = None,
                 penalty_s: float = 30.0, clock=time.monotonic):
        if backends is None:
            backends = [stt_backend(b) for b in STT_BACKENDS.split(",") if b.strip()]
            if not backends:
                raise ConfigError("SKCALL_STT_BACKENDS (or SKCALL_STT_URL) is not set: "
                                  "skcall has no built-in STT server")
        self.backends = backends
        self.hedge_s = (hedge_s if hedge_s is not None
                        else float(os.environ.get("SKCALL_STT_HEDGE_MS", "700")) / 1000)
        self.penalty_s = penalty_s
        self.clock = clock
        self._bad_until: dict[int, float] = {}
        self.last_backend = ""
        self.last_ms = 0.0
        self.log: list[dict] = []

    def _order(self) -> list[int]:
        now = self.clock()
        return sorted(range(len(self.backends)), key=lambda i: (self._bad_until.get(i, 0) > now, i))

    def name(self, i: int) -> str:
        b = self.backends[i]
        url = getattr(b, "url", "") or ""
        return url.split("//")[-1].split("/")[0] if url else f"{type(b).__name__}#{i}"

    async def transcribe(self, pcm16k: np.ndarray) -> str:
        t0 = self.clock()
        order = self._order()
        running: dict[asyncio.Task, int] = {}
        nxt = 0

        def launch() -> None:
            nonlocal nxt
            i = order[nxt]
            nxt += 1
            b = self.backends[i]
            fn = getattr(b, "transcribe_or_raise", None) or b.transcribe
            running[asyncio.ensure_future(fn(pcm16k))] = i

        launch()
        text, winner = "", None
        try:
            while running and winner is None:
                wait = self.hedge_s if nxt < len(order) else None
                done, _ = await asyncio.wait(running, timeout=wait, return_when=asyncio.FIRST_COMPLETED)
                if not done:
                    LOG.info("stt: %s slow (> %.0f ms); hedging to %s", self.name(order[nxt - 1]),
                             self.hedge_s * 1000, self.name(order[nxt]))
                    launch()
                    continue
                for task in done:
                    i = running.pop(task)
                    try:
                        text = task.result()
                        winner = i
                        break
                    except Exception as e:
                        LOG.warning("stt backend %s failed: %s: %s", self.name(i), type(e).__name__,
                                    e or "(no detail)")
                        self._bad_until[i] = self.clock() + self.penalty_s
                if winner is None and not running and nxt < len(order):
                    launch()
        finally:
            for task, i in running.items():
                task.cancel()
                if winner is not None:
                    self._bad_until[i] = self.clock() + self.penalty_s   # lost the hedge: slow now
        self.last_ms = (self.clock() - t0) * 1000
        self.last_backend = self.name(winner) if winner is not None else ""
        self.log.append({"backend": self.last_backend, "ms": round(self.last_ms)})
        if winner is None:
            LOG.error("STT failed on every backend")
        return text

    async def warmup(self) -> float:
        t = self.clock()
        await asyncio.gather(*(b.warmup() for b in self.backends), return_exceptions=True)
        return (self.clock() - t) * 1000

    async def aclose(self) -> None:
        for b in self.backends:
            close = getattr(b, "aclose", None)
            if close is not None:
                await close()


def tts_speed(env=os.environ) -> float:
    """SKCALL_TTS_SPEED, the OpenAI `speed` field (Kokoro honours it). 1.0 when
    unset or unparseable, clamped to 0.5..2.0 so a typo cannot make speech unusable."""
    try:
        v = float(env.get("SKCALL_TTS_SPEED", "") or 1.0)
    except ValueError:
        return 1.0
    return min(2.0, max(0.5, v))


class F5TTS:
    """Any OpenAI-shaped /v1/audio/speech server that returns WAV. `nfe_step`
    is F5-only and is sent only when the model is an F5 one (Kokoro takes
    model "tts-1").

    `url` may be a comma list of servers that render the SAME voice (for
    example Kokoro on two hosts). They are tried in order; one that fails
    sits out `penalty_s` so later sentences go straight to the next. Never
    list a server with a different voice: a call must not change voice.

    `ref` (default SKCALL_TTS_REF) is a reference clip for servers that clone
    from audio per request (ZONOS3-Mini on SGLang-Omni): it is sent as
    `references: [{"audio_path": ref}]`, a path the TTS server can read. Such a
    server ignores `voice` when it gets a reference, and an F5 server ignores
    `references`, so one request body serves both."""

    def __init__(self, url: str = TTS_URL, voice: str = os.environ.get("SKCALL_TTS_VOICE", "lumina"),
                 nfe_step: int | None = int(os.environ.get("SKCALL_TTS_NFE", "16")) or None,
                 timeout: float = 120, trim: bool = True,
                 model: str = os.environ.get("SKCALL_TTS_MODEL", "f5-tts"),
                 speed: float | None = None, penalty_s: float = 30.0, clock=time.monotonic,
                 token: str | None = None, transport: httpx.AsyncBaseTransport | None = None,
                 ref: str | None = None):
        """token: bearer token for the service (default SKCALL_TTS_TOKEN), sent
        to every server in `url`. transport: an httpx transport (tests)."""
        self.urls = [u.strip() for u in (url or TTS_URL).split(",") if u.strip()]
        if not self.urls:
            raise ConfigError("SKCALL_TTS_URL is not set: skcall has no built-in TTS server")
        self.url = self.urls[0]
        self.penalty_s = penalty_s
        self._clock = clock
        self._down_until: dict[str, float] = {}
        self.trim = trim
        self.voice = voice
        self.model = model or "f5-tts"
        self.nfe_step = nfe_step if self.model.startswith("f5") else None
        self.speed = tts_speed() if speed is None else speed
        self.ref = (os.environ.get("SKCALL_TTS_REF", "") if ref is None else ref).strip()
        token = os.environ.get("SKCALL_TTS_TOKEN", "") if token is None else token
        self._http = httpx.AsyncClient(timeout=timeout, headers=bearer(token), transport=transport)

    async def synth(self, text: str) -> np.ndarray:
        if not text.strip():
            return np.zeros(0, np.int16)
        body: dict = {"model": self.model, "input": text, "voice": self.voice, "response_format": "wav"}
        if self.nfe_step:
            body["nfe_step"] = self.nfe_step
        if self.speed != 1.0:
            body["speed"] = self.speed
        if self.ref:
            body["references"] = [{"audio_path": self.ref}]
        content = await self._post(body)
        if content is None:
            return np.zeros(0, np.int16)
        pcm = wav_to_pcm(content, CALL_RATE)
        secs = len(pcm) / CALL_RATE
        if len(text) > 40 and secs < len(text) * 0.02:
            LOG.warning("TTS output looks truncated: %.2fs for %d chars", secs, len(text))
        return trim_silence(normalize_peak(pcm), CALL_RATE) if self.trim else normalize_peak(pcm)

    def _order(self) -> list[str]:
        """Healthy servers first (in configured order), then benched ones as a last resort."""
        now = self._clock()
        up = [u for u in self.urls if self._down_until.get(u, 0) <= now]
        return up + [u for u in self.urls if u not in up]

    async def _post(self, body: dict) -> bytes | None:
        for url in self._order():
            try:
                r = await self._http.post(url, json=body)
                r.raise_for_status()
            except Exception as e:
                LOG.error("TTS failed: %s: %s (url=%s)", type(e).__name__, e or "(no detail)", url)
                if len(self.urls) > 1:
                    self._down_until[url] = self._clock() + self.penalty_s
                continue
            if url != self.url:
                LOG.warning("TTS served by fallback %s", url)
            self._down_until.pop(url, None)
            return r.content
        return None

    async def aclose(self) -> None:
        await self._http.aclose()
