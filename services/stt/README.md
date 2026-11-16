# skstt: speech to text (faster-whisper) and its dispatcher

Two programs in one package and one venv:

- **`skstt.server`**: an OpenAI-shaped `/v1/audio/transcriptions` endpoint backed by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (MIT), Whisper large-v3 on CUDA by default. skcall's `OpenAISTT` client (`fw:` entries in `SKCALL_STT_BACKENDS`) talks to it.
- **`skstt.dispatcher`**: a small proxy that sends each transcription to the first backend in an ordered list that answers.

- **Provenance:** both files started as the servers that run in production, imported unchanged and then refactored. The changes are listed in this file.
- **License:** AGPL-3.0-or-later, like the rest of skcall. faster-whisper and CTranslate2 are MIT; the Whisper weights are MIT.

## Server interface (interface 1)

```
POST /v1/audio/transcriptions   multipart form:
                                  file             (required) any format PyAV decodes
                                  model            openai/whisper-large-v3 (default), whisper-large-v3, or whisper-1
                                                   (the OpenAI default name, an alias of the configured model)
                                  language         e.g. en; empty = detect
                                  prompt           initial prompt
                                  response_format  json (default) | text | verbose_json
                                  temperature      0.0 to 1.0 (default 0.0)
                                  task             transcribe (default) | translate
                                -> json {"text": "..."}; text: the bare text; verbose_json adds
                                   task, language, duration and segments.
                                   400 bad field, 413 too large, 503 still loading, 500 failure.
GET  /health                    status, model, device, compute_type, uptime     (no token needed)
GET  /v1/info                   {"service": "stt", "interface": 1, "role": "server", "version": ...,
                                 "engine": "faster-whisper", "model": ..., "auth": "open" | "tokens"}
```

One request runs at a time (a lock around the model); others wait.

## Dispatcher interface (interface 1)

```
POST /v1/audio/transcriptions   forwarded unchanged (every form field, the file, and the client's
                                Authorization header) to each backend in SKSTT_BACKENDS in order.
                                A connection error, a timeout or a 5xx tries the next one; a 4xx
                                (for example 401 or 400) is returned to the client as is.
                                All down, or none configured: 503 {"error": "no backend"}, always
                                within SKSTT_DISPATCH_TIMEOUT_S (one deadline for all attempts).
GET  /health                    200 if any backend's /health is 200, else 503. Lists `backends` by index,
                                and keeps the old keys about the first one: `backend_url`, and
                                `backend` (its health) or `backend_error`
GET  /v1/info                   {"service": "stt", "interface": 1, "role": "dispatcher",
                                 "backends": <count>, "auth": "forward" | "tokens"}
```

## Settings

All from the environment, so a unit needs only an `EnvironmentFile`.

**Server** (`python -m skstt.server`, which also takes `--host` and `--port`):

| Variable | Default | Meaning |
|---|---|---|
| `SKSTT_HOST` | `127.0.0.1` | Bind address. |
| `SKSTT_PORT` | `18794` | Port. |
| `SKSTT_MODEL_PATH` | `large-v3` | A faster-whisper size name (downloaded on first use) or a local CTranslate2 model directory, such as a Hugging Face snapshot of `Systran/faster-whisper-large-v3`. |
| `SKSTT_MODEL_ID` | `openai/whisper-large-v3` | The name reported and accepted in the `model` field (its short form too, and `whisper-1` as an alias). Any other name gets 400. |
| `SKSTT_DEVICE` | `cuda` | `cuda` or `cpu`. Pick the card with `CUDA_VISIBLE_DEVICES`. |
| `SKSTT_COMPUTE_TYPE` | `int8_float16` | CTranslate2 compute type (`int8_float16`, `float16`, `int8`, ...). |
| `SKSTT_CPU_THREADS` | `4` | CPU threads for decoding. |
| `SKSTT_MAX_AUDIO_BYTES` | `536870912` (512 MiB) | Largest upload. |
| `SKSTT_TOKENS_FILE` | unset | Per-client tokens (below). Unset: open, as before, with a warning at start. Set but empty or unreadable: refuses to start. |

The previous production server read `WHISPER_MODEL_PATH` and `WHISPER_MAX_AUDIO_BYTES`; those are now `SKSTT_MODEL_PATH` and `SKSTT_MAX_AUDIO_BYTES`.

**An int8 variant.** Production also runs a second, int8 instance of Whisper large-v3 as the dispatcher's backend (from an older, simpler script that is not imported). The released server covers it with env alone: `SKSTT_COMPUTE_TYPE=int8 SKSTT_PORT=18795` and the same `SKSTT_MODEL_PATH`. The older script also accepted a `model_name` field and always returned `language`, `duration` and `segments` in `json`; here use `verbose_json` for those (`model_name` is ignored).

**Dispatcher** (`python -m skstt.dispatcher`):

| Variable | Default | Meaning |
|---|---|---|
| `SKSTT_BACKENDS` | unset (none) | Comma list of backend base URLs, tried in order, e.g. `http://192.0.2.10:18795,http://192.0.2.11:18794`. A URL ending in `/v1/audio/transcriptions` is accepted too. There is no built-in backend: unset means every request gets 503 `{"error": "no backend"}`. |
| `SKSTT_DISPATCH_TIMEOUT_S` | `30` | One deadline for the whole request, across all backends. Raise it for long batch files (the old dispatcher allowed 1800 s per backend). |
| `SKSTT_DISPATCH_HOST` | `127.0.0.1` | Bind address. |
| `SKSTT_DISPATCH_PORT` | `18794` | Port. |
| `SKSTT_TOKENS_FILE` | unset | Optional. Set: the dispatcher also refuses unknown clients itself and logs their names. Unset: it does not check. Either way it forwards the client's `Authorization` header, so each backend checks with its own tokens file. |

The previous dispatcher read one backend from `WHISPER_GPU_URL` and its port from `WHISPER_DISPATCHER_PORT`; those are now `SKSTT_BACKENDS` and `SKSTT_DISPATCH_PORT`.

## Tokens and the log

With `SKSTT_TOKENS_FILE` set, `POST /v1/audio/transcriptions` needs `Authorization: Bearer <token>`. A missing or unknown token gets 401 `{"error": "bad token"}`. `/health` and `/v1/info` stay open. The file has one `client=token` per line (`#` comments). Each request is logged once, naming the client (`client=alice`, `client=unknown`, or `client=open` without a tokens file), never the token. The token helper is `services/common/sktoken.py`; `sk/setup.sh` copies it to `skstt/_sktoken.py` (git-ignored).

## Install and run

```bash
git archive HEAD | tar -x -C /path/to/export
cd /path/to/export/services/stt
bash sk/setup.sh                                   # .venv with the pinned requirements.txt
SKSTT_HOST=127.0.0.1 SKSTT_PORT=18794 .venv/bin/python -m skstt.server
curl -s localhost:18794/v1/info
curl -s localhost:18794/v1/audio/transcriptions -F file=@tests/data/hello.wav

SKSTT_BACKENDS=http://192.0.2.10:18795 SKSTT_DISPATCH_PORT=18794 .venv/bin/python -m skstt.dispatcher
```

`requirements.txt` mirrors the venv the production server runs in (Python 3.12, `faster-whisper 1.2.1`, `ctranslate2 4.8.3`, `av 19.0.1`) plus `httpx` for the dispatcher. CTranslate2 needs CUDA 12 cuBLAS and cuDNN 9 from the system for `SKSTT_DEVICE=cuda`; on CPU it needs neither. A quick CPU check: `SKSTT_MODEL_PATH=tiny SKSTT_DEVICE=cpu SKSTT_COMPUTE_TYPE=int8`.

## Tests

The server tests use a fake model (no CTranslate2, no weights). The dispatcher tests use real local sockets: a stub backend, closed ports, and a port that accepts and never answers.

```bash
cd services/stt
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements-test.txt
.venv/bin/python -m pytest -q
```

`tests/data/hello.wav` is 1 s of synthesized tone (`python tests/hello.py` rewrites it; a test checks the bundled file matches), not a recording. Whisper hears no speech in it, so a real model answers 200 with empty or near-empty text.

## Changes from the imported programs

Server:
- Settings come from env (above); the device and compute type were hard-coded. The bind address defaults to `127.0.0.1` instead of `0.0.0.0`.
- Per-client tokens, `/v1/info`, and an access log that names the client.
- faster-whisper and PyAV load on startup, not at import, so the package imports without them (the tests rely on that).

Dispatcher:
- An ordered list of backends with fall-through, instead of one backend with a host name default. No built-in backend.
- One overall deadline (`SKSTT_DISPATCH_TIMEOUT_S`, 30 s) instead of 15 s connect and 1800 s read per request; the all-down answer is 503 `{"error": "no backend"}` (was 502 with the exception text).
- Every form field is forwarded. The old dispatcher forwarded only `model_name`, `language` and `response_format`, so `prompt`, `temperature`, `task` and `model` were dropped.
- The client's `Authorization` header is forwarded; optional tokens check of its own.
- `/health` adds `backends`, every backend by index; the old `backend_url`, `backend` and `backend_error` keys stay, about the first backend.
