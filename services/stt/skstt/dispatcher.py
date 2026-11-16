#!/usr/bin/env python3
"""A small proxy in front of one or more skstt servers (or any OpenAI-shaped STT backend).

  POST /v1/audio/transcriptions   forwarded as is (every form field, the file and the
                                  client's Authorization header) to the first backend in
                                  SKSTT_BACKENDS that answers; a connection error, a timeout
                                  or a 5xx moves on to the next one. A 4xx is the client's
                                  problem and is returned unchanged.
                                  All down, or none configured: 503 {"error": "no backend"},
                                  always within SKSTT_DISPATCH_TIMEOUT_S.
  GET  /health                    200 when at least one backend is healthy, else 503
  GET  /v1/info                   unauthenticated: {"service": "stt", "interface": 1, "role": "dispatcher", ...}

Run with `python -m skstt.dispatcher`. There is no built-in backend.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.datastructures import UploadFile

from . import __version__
from .server import add_guard, resolve_tokens

LOG = logging.getLogger("skstt.dispatcher")

PATH = "/v1/audio/transcriptions"
NO_BACKEND = {"error": "no backend"}


def _base(url: str) -> str:
    url = url.strip().rstrip("/")
    return url[: -len(PATH)].rstrip("/") if url.endswith(PATH) else url


@dataclass(frozen=True)
class Settings:
    backends: tuple[str, ...] = ()
    timeout_s: float = 30.0
    host: str = "127.0.0.1"
    port: int = 18794
    tokens_file: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> "Settings":
        d = cls()
        raw = env.get("SKSTT_BACKENDS", "")
        return cls(
            backends=tuple(_base(u) for u in raw.split(",") if u.strip()),
            timeout_s=float(env.get("SKSTT_DISPATCH_TIMEOUT_S", d.timeout_s)),
            host=env.get("SKSTT_DISPATCH_HOST", d.host),
            port=int(env.get("SKSTT_DISPATCH_PORT", d.port)),
            tokens_file=env.get("SKSTT_TOKENS_FILE", d.tokens_file),
        )


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or Settings.from_env()
    # Tokens are optional here: with SKSTT_TOKENS_FILE set the dispatcher refuses unknown
    # clients itself (and its log names them); either way it forwards the client's
    # Authorization header, so each backend can check it too.
    tokens = resolve_tokens(settings) if settings.tokens_file else None
    if not settings.backends:
        LOG.warning("SKSTT_BACKENDS is not set: every transcription answers 503 no backend")
    app = FastAPI(title="skstt dispatcher", version=__version__)
    add_guard(app, tokens)

    @app.get("/v1/info")
    async def info() -> dict[str, Any]:
        return {"service": "stt", "interface": 1, "role": "dispatcher", "version": __version__,
                "backends": len(settings.backends), "auth": "forward" if tokens is None else "tokens"}

    @app.get("/health")
    async def health():
        results = []
        async with httpx.AsyncClient(timeout=min(5.0, settings.timeout_s)) as client:
            for i, url in enumerate(settings.backends):
                try:
                    r = await client.get(f"{url}/health")
                    results.append({"backend": i, "status_code": r.status_code, "health": r.json()})
                except Exception as exc:
                    results.append({"backend": i, "error": type(exc).__name__})
        up = any(r.get("status_code") == 200 for r in results)
        # The old single-backend keys stay, about the first backend, so existing monitors keep working.
        legacy: dict[str, Any] = {"backend_url": settings.backends[0] if settings.backends else ""}
        if results and "health" in results[0]:
            legacy["backend"] = results[0]["health"]
        else:
            legacy["backend_error"] = results[0]["error"] if results else NO_BACKEND["error"]
        return JSONResponse(status_code=200 if up else 503,
                            content={"dispatcher": "ok", **legacy, "backends": results,
                                     **({} if up else NO_BACKEND)})

    @app.post(PATH)
    async def transcriptions(request: Request):
        deadline = time.monotonic() + settings.timeout_s
        form = await request.form()
        data: dict[str, list[str]] = {}
        files: list[tuple[str, tuple[str, bytes, str]]] = []
        for key, value in form.multi_items():
            if isinstance(value, UploadFile):
                files.append((key, (value.filename or "audio", await value.read(),
                                    value.content_type or "application/octet-stream")))
            else:
                data.setdefault(key, []).append(value)
        headers = {}
        if request.headers.get("authorization"):
            headers["Authorization"] = request.headers["authorization"]

        for i, url in enumerate(settings.backends):
            left = deadline - time.monotonic()
            if left <= 0:
                LOG.warning("dispatch deadline (%.1fs) reached before backend %d", settings.timeout_s, i)
                break
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(left)) as client:
                    r = await asyncio.wait_for(
                        client.post(f"{url}{PATH}", data=data, files=files, headers=headers), timeout=left)
            except (asyncio.TimeoutError, httpx.HTTPError, OSError) as exc:
                LOG.warning("backend %d failed: %s", i, type(exc).__name__)
                continue
            if r.status_code >= 500:
                LOG.warning("backend %d answered %d", i, r.status_code)
                continue
            content_type = r.headers.get("content-type", "application/json")
            return Response(content=r.content, status_code=r.status_code,
                            media_type=content_type.split(";", 1)[0])
        return JSONResponse(status_code=503, content=NO_BACKEND)

    return app


def main() -> None:
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    settings = Settings.from_env()
    # access_log off: the app logs each request itself, naming the client and never the token.
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, workers=1, access_log=False)


if __name__ == "__main__":
    main()
