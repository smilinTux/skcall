#!/usr/bin/env python3
"""OpenAI-shaped speech to text with faster-whisper.

  POST /v1/audio/transcriptions   multipart: file, model, language, prompt, response_format
                                  (json | text | verbose_json), temperature (0..1), task
                                  (transcribe | translate)
  GET  /health                    unauthenticated
  GET  /v1/info                   unauthenticated: {"service": "stt", "interface": 1, "role": "server", ...}

Run with `python -m skstt.server [--host H] [--port P]`. Every setting is an
environment variable (see Settings and the README). One request runs at a time.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from . import __version__
from .auth import client_for, load_tokens

LOG = logging.getLogger("skstt")

OPEN_PATHS = frozenset({"/health", "/v1/info"})


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 18794
    model_id: str = "openai/whisper-large-v3"  # the name clients send in `model`
    model_path: str = "large-v3"  # a faster-whisper size name or a local CTranslate2 model dir
    device: str = "cuda"
    compute_type: str = "int8_float16"
    cpu_threads: int = 4
    max_audio_bytes: int = 512 * 1024 * 1024
    tokens_file: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "Settings":
        d = cls()
        return cls(
            host=env.get("SKSTT_HOST", d.host),
            port=int(env.get("SKSTT_PORT", d.port)),
            model_id=env.get("SKSTT_MODEL_ID", d.model_id),
            model_path=env.get("SKSTT_MODEL_PATH", d.model_path),
            device=env.get("SKSTT_DEVICE", d.device),
            compute_type=env.get("SKSTT_COMPUTE_TYPE", d.compute_type),
            cpu_threads=int(env.get("SKSTT_CPU_THREADS", d.cpu_threads)),
            max_audio_bytes=int(env.get("SKSTT_MAX_AUDIO_BYTES", d.max_audio_bytes)),
            tokens_file=env.get("SKSTT_TOKENS_FILE", d.tokens_file),
        )

    @property
    def model_names(self) -> frozenset[str]:
        """Accepted `model` values: the id, its short form (openai/whisper-large-v3, whisper-large-v3),
        and `whisper-1`, the OpenAI clients' default name, as an alias of the configured model."""
        return frozenset({self.model_id, self.model_id.rsplit("/", 1)[-1], "whisper-1"})


def _patch_av() -> None:
    # faster-whisper 1.2.1 passes metadata_errors, which PyAV 19 removed.
    import av

    if getattr(av.open, "_skstt_compat", False):
        return
    _av_open = av.open

    def _compat_av_open(*args: Any, **kwargs: Any):
        kwargs.pop("metadata_errors", None)
        return _av_open(*args, **kwargs)

    _compat_av_open._skstt_compat = True
    av.open = _compat_av_open


def load_model(settings: Settings) -> Any:
    """Load the faster-whisper model. Tests replace this."""
    _patch_av()
    from faster_whisper import WhisperModel

    LOG.info("loading %s on %s (%s)", settings.model_path, settings.device, settings.compute_type)
    return WhisperModel(settings.model_path, device=settings.device, compute_type=settings.compute_type,
                        cpu_threads=settings.cpu_threads)


def resolve_tokens(settings: Settings) -> Optional[dict[str, str]]:
    """{token: client}, or None when the service is open (no tokens file)."""
    if not settings.tokens_file:
        LOG.warning("SKSTT_TOKENS_FILE is not set: the STT service is open to any caller")
        return None
    tokens = load_tokens(os.path.expanduser(settings.tokens_file))
    if not tokens:
        raise ValueError(f"no tokens in SKSTT_TOKENS_FILE={settings.tokens_file}")
    LOG.info("STT tokens loaded: %d clients", len(set(tokens.values())))
    return tokens


def add_guard(app: FastAPI, tokens: Optional[dict[str, str]]) -> None:
    """Per-client bearer tokens plus the access log. The log names the client, never the token."""

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


class WhisperService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.lock = asyncio.Lock()
        self.started_at = time.time()
        self.last_error: str | None = None
        self.model = load_model(settings)

    def health(self) -> dict[str, Any]:
        s = self.settings
        return {
            "status": "ok" if self.last_error is None else "degraded",
            "model": s.model_id,
            "model_path": s.model_path,
            "device": s.device,
            "compute_type": s.compute_type,
            "serialization": "single-request",
            "uptime_seconds": round(time.time() - self.started_at, 3),
            "last_error": self.last_error,
        }

    def _transcribe_sync(
        self, path: str, language: str | None, task: str, prompt: str | None, temperature: float, verbose: bool
    ) -> dict[str, Any]:
        segments_iter, info = self.model.transcribe(
            path,
            language=language,
            task=task,
            initial_prompt=prompt,
            temperature=temperature,
            beam_size=5,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters={"min_speech_duration_ms": 500},
        )
        raw = list(segments_iter)
        text = "".join(segment.text for segment in raw).strip()
        response: dict[str, Any] = {"text": text}
        if verbose:
            segments = []
            for segment in raw:
                segments.append(
                    {
                        "id": len(segments),
                        "seek": int(getattr(segment, "seek", 0)),
                        "start": float(segment.start),
                        "end": float(segment.end),
                        "text": str(segment.text).strip(),
                        "tokens": list(getattr(segment, "tokens", []) or []),
                        "temperature": temperature,
                        "avg_logprob": float(getattr(segment, "avg_logprob", 0.0)),
                        "compression_ratio": float(getattr(segment, "compression_ratio", 0.0)),
                        "no_speech_prob": float(getattr(segment, "no_speech_prob", 0.0)),
                    }
                )
            response.update(
                {"task": task, "language": language, "duration": float(info.duration), "segments": segments}
            )
        return response

    async def transcribe(self, *args: Any) -> dict[str, Any]:
        async with self.lock:
            try:
                result = await asyncio.to_thread(self._transcribe_sync, *args)
                self.last_error = None
                return result
            except Exception as exc:
                self.last_error = str(exc)
                raise


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    tokens = resolve_tokens(settings)
    holder: dict[str, Optional[WhisperService]] = {"service": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        holder["service"] = WhisperService(settings)
        yield
        holder["service"] = None

    app = FastAPI(title="skstt", version=__version__, lifespan=lifespan)
    add_guard(app, tokens)

    @app.get("/health")
    async def health() -> dict[str, Any]:
        service = holder["service"]
        if service is None:
            return {"status": "loading", "model": settings.model_id}
        return service.health()

    @app.get("/v1/info")
    async def info() -> dict[str, Any]:
        return {"service": "stt", "interface": 1, "role": "server", "version": __version__,
                "engine": "faster-whisper", "model": settings.model_id,
                "auth": "open" if tokens is None else "tokens"}

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        file: UploadFile = File(...),
        model: str = Form(settings.model_id),
        language: str | None = Form(None),
        prompt: str | None = Form(None),
        response_format: str = Form("json"),
        temperature: float = Form(0.0),
        task: str = Form("transcribe"),
    ):
        service = holder["service"]
        if service is None:
            raise HTTPException(status_code=503, detail="model is still loading")
        if model not in settings.model_names:
            raise HTTPException(status_code=400, detail=f"unsupported model: {model}")
        if task not in {"transcribe", "translate"}:
            raise HTTPException(status_code=400, detail="task must be transcribe or translate")
        if response_format not in {"json", "text", "verbose_json"}:
            raise HTTPException(status_code=400, detail="response_format must be json, text, or verbose_json")
        if not 0.0 <= temperature <= 1.0:
            raise HTTPException(status_code=400, detail="temperature must be between 0 and 1")
        payload = await file.read(settings.max_audio_bytes + 1)
        if len(payload) > settings.max_audio_bytes:
            raise HTTPException(status_code=413, detail="audio file exceeds configured limit")
        suffix = Path(file.filename or "audio.bin").suffix or ".bin"
        with tempfile.NamedTemporaryFile(prefix=f"whisper-{uuid.uuid4().hex}-", suffix=suffix) as tmp:
            tmp.write(payload)
            tmp.flush()
            try:
                result = await service.transcribe(
                    tmp.name, language, task, prompt, float(temperature), response_format == "verbose_json"
                )
            except Exception as exc:
                raise HTTPException(status_code=500, detail=f"transcription failed: {exc}") from exc
        if response_format == "text":
            return PlainTextResponse(result["text"])
        return JSONResponse(result)

    return app


def main(argv: Optional[list[str]] = None) -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    env = Settings.from_env()
    parser = argparse.ArgumentParser(prog="skstt.server")
    parser.add_argument("--host", default=env.host, help="env SKSTT_HOST")
    parser.add_argument("--port", type=int, default=env.port, help="env SKSTT_PORT")
    args = parser.parse_args(argv)
    # access_log off: the app logs each request itself, naming the client and never the token.
    uvicorn.run(create_app(env), host=args.host, port=args.port, workers=1, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
