"""OpenAI-shaped text to speech with Kokoro.

  POST /v1/audio/speech   {"input": "...", "voice": "af_bella", "speed": 1.0} -> audio/wav (24 kHz)
  GET  /v1/models         {"object": "list", "data": [{"id": "tts-1", ...}]}
  GET  /health            unauthenticated
  GET  /v1/info           unauthenticated: {"service": "tts", "interface": 1, ...}

Run with `python -m sktts`. Every setting is an environment variable (see
Settings and the README), so a unit only needs an EnvironmentFile.
"""
from __future__ import annotations

import io
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import soundfile as sf
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from . import __version__
from .auth import client_for, load_tokens

LOG = logging.getLogger("sktts")

SAMPLE_RATE = 24000
OPEN_PATHS = frozenset({"/health", "/v1/info"})

VOICE_MAP = {
    "af_bella": ("a", "af_bella"),
    "af_sky": ("a", "af_sky"),
    "af_nova": ("a", "af_nova"),
    "af_sarah": ("a", "af_sarah"),
    "am_adam": ("a", "am_adam"),
    "am_michael": ("a", "am_michael"),
    "bf_emma": ("b", "bf_emma"),
    "bm_george": ("b", "bm_george"),
}


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 18793
    device: str = "auto"  # auto (cuda when torch sees one, else cpu), cuda or cpu
    lang_code: str = "a"
    repo_id: str = ""  # empty: Kokoro's own default (hexgrad/Kokoro-82M)
    default_voice: str = "af_bella"
    tokens_file: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "Settings":
        d = cls()
        return cls(
            host=env.get("SKTTS_HOST", d.host),
            port=int(env.get("SKTTS_PORT", d.port)),
            device=env.get("SKTTS_DEVICE", d.device),
            lang_code=env.get("SKTTS_LANG", d.lang_code),
            repo_id=env.get("SKTTS_REPO_ID", d.repo_id),
            default_voice=env.get("SKTTS_DEFAULT_VOICE", d.default_voice),
            tokens_file=env.get("SKTTS_TOKENS_FILE", d.tokens_file),
        )


def load_pipeline(settings: Settings) -> tuple[Any, str]:
    """Load Kokoro and return (pipeline, device). Tests replace this."""
    import torch
    from kokoro import KPipeline

    device = settings.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    kw = {"repo_id": settings.repo_id} if settings.repo_id else {}
    pipeline = KPipeline(lang_code=settings.lang_code, device=device, **kw)
    LOG.info("Kokoro loaded on device: %s", device)
    return pipeline, device


def resolve_tokens(settings: Settings) -> Optional[dict[str, str]]:
    """{token: client}, or None when the service is open (no tokens file)."""
    if not settings.tokens_file:
        LOG.warning("SKTTS_TOKENS_FILE is not set: the TTS service is open to any caller")
        return None
    tokens = load_tokens(os.path.expanduser(settings.tokens_file))
    if not tokens:
        raise ValueError(f"no tokens in SKTTS_TOKENS_FILE={settings.tokens_file}")
    LOG.info("TTS tokens loaded: %d clients", len(set(tokens.values())))
    return tokens


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    tokens = resolve_tokens(settings)
    state: dict[str, Any] = {"pipeline": None, "device": None}

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        state["pipeline"], state["device"] = load_pipeline(settings)
        yield
        state["pipeline"] = None

    app = FastAPI(title="sktts", version=__version__, lifespan=lifespan)

    @app.middleware("http")
    async def guard(request: Request, call_next):
        t = time.monotonic()
        path = request.url.path
        peer = request.client.host if request.client else "-"
        if tokens is None:
            client = "open"
        else:
            client = client_for(request.headers.get("authorization"), tokens) or "unknown"
            if client == "unknown" and path not in OPEN_PATHS:
                LOG.warning("%s %s %s 401 client=unknown (bad or missing token)", peer, request.method, path)
                return JSONResponse(status_code=401, content={"error": "bad token"})
        response = await call_next(request)
        LOG.info("%s %s %s %d client=%s %.0fms", peer, request.method, path, response.status_code, client,
                 (time.monotonic() - t) * 1000)
        return response

    @app.post("/v1/audio/speech")
    async def tts(request: Request):
        body = await request.json()
        text = body.get("input", "")
        voice = body.get("voice", settings.default_voice)
        speed = float(body.get("speed", 1.0))

        _lang_code, voice_name = VOICE_MAP.get(voice, ("a", settings.default_voice))

        audio_chunks = []
        for _, _, audio in state["pipeline"](text, voice=voice_name, speed=speed):
            if audio is not None:
                audio_chunks.append(np.asarray(audio, dtype=np.float32))

        samples = np.concatenate(audio_chunks) if audio_chunks else np.zeros(100, dtype=np.float32)

        buf = io.BytesIO()
        sf.write(buf, samples, SAMPLE_RATE, format="WAV")
        buf.seek(0)
        return StreamingResponse(buf, media_type="audio/wav")

    @app.get("/health")
    def health():
        # "engine" keeps its old value so existing monitors keep matching.
        return {"status": "healthy", "engine": "kokoro-gpu", "device": state["device"]}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": "tts-1", "object": "model"}]}

    @app.get("/v1/info")
    def info():
        return {"service": "tts", "interface": 1, "version": __version__, "engine": "kokoro",
                "auth": "open" if tokens is None else "tokens"}

    return app
