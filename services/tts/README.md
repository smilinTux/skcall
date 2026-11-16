# sktts: text to speech (Kokoro)

An OpenAI-shaped speech endpoint backed by [Kokoro](https://github.com/hexgrad/kokoro) (Apache-2.0, 82M parameters). skcall's `F5TTS` client (`skcall/speech.py`) talks to it with model `tts-1`.

- **Provenance:** started as the server that runs in production, imported unchanged and then refactored. The changes are listed at the end of this file.
- **License:** AGPL-3.0-or-later, like the rest of skcall. Kokoro's code and weights are Apache-2.0.

## Interface (interface 1)

```
POST /v1/audio/speech   JSON {"input": "...", "voice": "af_bella", "speed": 1.0}
                        -> 200 audio/wav, 24 kHz mono. "model" and "response_format" are accepted and ignored.
GET  /v1/models         {"object": "list", "data": [{"id": "tts-1", "object": "model"}]}
GET  /health            {"status": "healthy", "engine": "kokoro-gpu", "device": "cuda"}  (no token needed)
GET  /v1/info           {"service": "tts", "interface": 1, "version": ..., "engine": "kokoro",
                         "auth": "open" | "tokens"}                                      (no token needed)
```

Voices: `af_bella`, `af_sky`, `af_nova`, `af_sarah`, `am_adam`, `am_michael`, `bf_emma`, `bm_george`. An unknown voice falls back to `SKTTS_DEFAULT_VOICE`. Empty input returns a short silent WAV.

## Settings

All from the environment, so a unit needs only an `EnvironmentFile`.

| Variable | Default | Meaning |
|---|---|---|
| `SKTTS_HOST` | `127.0.0.1` | Bind address. Set it to the address the clients reach, never `0.0.0.0` on an untrusted network. |
| `SKTTS_PORT` | `18793` | Port. |
| `SKTTS_DEVICE` | `auto` | `auto` (CUDA when torch sees a GPU, else CPU), `cuda` or `cpu`. Pick the card with `CUDA_VISIBLE_DEVICES`. |
| `SKTTS_LANG` | `a` | Kokoro pipeline language (`a` American English, `b` British English). |
| `SKTTS_REPO_ID` | Kokoro's default (`hexgrad/Kokoro-82M`) | Hugging Face repo for the weights. Set `HF_HUB_OFFLINE=1` to use the local cache only. |
| `SKTTS_DEFAULT_VOICE` | `af_bella` | Voice when the request names none, or an unknown one. |
| `SKTTS_TOKENS_FILE` | unset | Per-client tokens, one `client=token` per line (`#` comments). Unset: the service is open, as before, and logs a warning at start. Set but empty or unreadable: the service refuses to start. |

## Tokens and the log

With `SKTTS_TOKENS_FILE` set, `/v1/audio/speech` and `/v1/models` need `Authorization: Bearer <token>`. A missing or unknown token gets 401 `{"error": "bad token"}`. `/health` and `/v1/info` stay open. Each request is logged once, naming the client (`client=alice`, `client=unknown`, or `client=open` without a tokens file), never the token. The token helper is `services/common/sktoken.py`; `sk/setup.sh` copies it to `sktts/_sktoken.py` (git-ignored).

## Install and run

```bash
git archive HEAD | tar -x -C /path/to/export
cd /path/to/export/services/tts
bash sk/setup.sh                      # .venv with the pinned requirements.txt (CUDA 12.8 torch)
SKTTS_PORT=18793 .venv/bin/python -m sktts
curl -s localhost:18793/v1/info
curl -s -o hi.wav localhost:18793/v1/audio/speech -H 'Content-Type: application/json' -d '{"input":"hello"}'
```

`requirements.txt` mirrors the venv the production server runs in (Python 3.12, `torch 2.11.0+cu128`, `kokoro 0.9.4`). The same wheels also run on CPU with `SKTTS_DEVICE=cpu`.

## Tests

The tests use a fake pipeline: no torch, no model.

```bash
cd services/tts
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements-test.txt
.venv/bin/python -m pytest -q
```

## Changes from the imported server

- Settings come from env (above). The bind address defaults to `127.0.0.1` instead of `0.0.0.0`; production sets `SKTTS_HOST`.
- Per-client tokens, `/v1/info`, and an access log that names the client.
- Fixed: empty input used to fail with a `NameError` (`numpy` was imported only on the non-empty path).
- `/health` reports the device actually loaded (the `engine` value stays `"kokoro-gpu"` for existing monitors).
- Run with `python -m sktts` instead of `uvicorn server:app`. Startup uses a lifespan handler instead of the deprecated `on_event`.
