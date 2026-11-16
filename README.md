# skcall

skcall puts an AI agent into a Nextcloud Talk call as a real participant. The
agent hears each person separately (speech to text, with who said what),
answers out loud sentence by sentence (text to speech), can be interrupted,
shows a face (a static or animated portrait, optionally lip-synced), and
leaves when asked. It joins through Talk's High Performance Backend, with no
browser.

The repository holds the call client and three optional GPU services:

| Path | What |
|---|---|
| `skcall/` | the call client (Python, aiortc). Runs on CPU. |
| `services/tts/` | sktts: text to speech (Kokoro), OpenAI-shaped `/v1/audio/speech` |
| `services/stt/` | skstt: speech to text (faster-whisper) and a dispatcher in front of several backends |
| `services/lipsync/` | sklipsync: real-time lip-sync (MuseTalk 1.5) over a WebSocket |
| `services/common/` | the per-client token helper the three services share |

## What you need

- **A Nextcloud with Talk and the High Performance Backend** (signaling
  server plus Janus, for example aio-talk). Talk without the HPB (mesh only)
  does not work.
- **A Talk user for the agent**, with an app password, invited to the rooms
  it may join. The agent shows up in the call as that user.
- **A TURN relay of your own** that the agent host can reach, handed out by
  Talk (Talk admin settings). Media to Janus goes through it.
- **Speech services.** Any OpenAI-shaped `/v1/audio/speech` server that
  returns WAV, and any OpenAI-shaped `/v1/audio/transcriptions` server (or
  whisper.cpp). The services in this repo are one way to get them.
- **A GPU or not.** The client needs none. sktts and skstt run on CPU, slowly;
  on CUDA they are fast enough for a call. sklipsync needs a GPU for real
  time (Intel XPU and CUDA were tested). Without lip-sync the agent shows an
  animated portrait.
- **An LLM for the voice layer**: an OpenAI-shaped chat endpoint, set through
  skchat's `SKVOICE_*` variables. Optionally a
  [Hermes](https://github.com/smilinTux/hermes-nextcloud-talk) agent behind it
  for turns that need real tools.
- Linux, Python 3.12, [uv](https://github.com/astral-sh/uv), and a user
  systemd manager for the example units.

## Quick start

```bash
# 1. an export of the repo at ~/skcall (a clone or a release archive)
git clone https://github.com/smilinTux/skcall ~/skcall
# 2. its venv: ~/skcall/.venv (needs uv and PyPI)
bash ~/skcall/sk/setup.sh
# 3. site settings (mode 0600)
mkdir -p ~/.config/skcall
cat > ~/.config/skcall/skcall.env <<'EOF'
SKCALL_NC_URL=https://cloud.example.org
SKCALL_ALLOWED_ROOMS=room0001
SKCALL_OPERATORS=alice
SKCALL_TTS_URL=http://tts.example.org:18793/v1/audio/speech
SKCALL_TTS_MODEL=tts-1
SKCALL_TTS_VOICE=af_bella
SKCALL_STT_BACKENDS=fw:http://stt.example.org:18794/v1/audio/transcriptions
SKCALL_SITE_TURN=turn.example.org
NEXTCLOUD_TALK_USERNAME=lumina
NEXTCLOUD_TALK_APP_PASSWORD=<app password>
EOF
chmod 600 ~/.config/skcall/skcall.env
# 4. check everything before the first call
~/skcall/.venv/bin/skcall smoke --agent lumina
# 5. join a call
cp ~/skcall/systemd/examples/skcall-join@.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user start skcall-join@room0001   # join; returns at once
systemctl --user stop  skcall-join@room0001   # leave cleanly
```

`skcall smoke` prints one `ok` or `FAIL` line per check (tts, stt, lipsync,
talk, turn) and exits 0 only when none fails. `--turn` fails, naming both
hosts, when Talk hands out a TURN server other than `SKCALL_SITE_TURN`.

- **Which checks.** Pick them with `--tts --stt --lipsync --talk --turn`; a
  check you name always runs, and fails naming the variable when its
  settings are missing. No flag runs them all, and then a service whose URL
  is unset (`SKCALL_TTS_URL`, `SKCALL_STT_BACKENDS` or `SKCALL_STT_URL`,
  `SKCALL_LIPSYNC_URL`) prints `ok <check> not configured` instead of
  failing, so a site without lip-sync still passes. talk and turn always run.
- **What it can certify.** The services in this repo answer `GET /v1/info`
  with their interface number, and smoke fails a server whose interface is
  older than this client needs. Other servers (F5-TTS, upstream Kokoro,
  whisper.cpp) answer 404 there: smoke reports them as `unversioned` and only
  the real request (one spoken word, its transcription) decides. Any other
  `/v1/info` error fails.
- **Which settings.** By default smoke reads the files the example unit
  reads, `~/.config/skcall/skcall.env` then `~/.config/skcall/join.env`
  (skipped when absent), and never overrides a variable already set.
  `--env-file PATH` (repeatable, later wins) replaces that list, and a named
  file that does not exist fails. A unit template that reads other files,
  for example a secrets file, passes the same files to smoke as its
  `EnvironmentFile` lines name.

Other commands:

| Command | What it does |
|---|---|
| `skcall join --room R [--agent A] [--ring]` | join a call and talk until it ends (`--no-hermes`, `--no-mcp`, `--no-tools`, `--no-video`, `--operators`) |
| `skcall watch --room R` | join when someone posts "@agent join call" in the room |
| `skcall check-room --agent A --room R` | dry check: would this room be allowed? |
| `skcall prompt --agent A [--search Q]` | dry run: voice prompt sizes and sections, never content |
| `skcall smoke` | service, Talk and TURN checks (above) |

## Configuration

Everything site-specific comes from the environment. skcall has no built-in
Nextcloud, rooms or operators: with no `SKCALL_NC_URL` it refuses to join. The
example unit reads `~/.config/skcall/skcall.env`, then
`~/.config/skcall/join.env` (optional host overrides), then
`$XDG_RUNTIME_DIR/skcall/join-<room>.env` (per call: `SKCALL_RING`,
`SKCALL_GREETING`), then `~/.config/skcall/room-<room>.env` (standing
settings for one room, read last so they win).

A private room, for example: a `room-<room>.env` with its own model and no records.

```sh
SKVOICE_LLM_URL=http://localhost:18780/v1/chat/completions   # a local model
SKVOICE_MODEL=<local model id>
SKCALL_POST_SUMMARY=off      # no summary in the room's chat
SKCALL_LOG_TRANSCRIPT=off    # logs keep timings, never the words
SKCALL_HERMES=off            # no escalation (Hermes keeps its own sessions)
```

Which agent: every command takes `--agent`, and without it uses `SKAGENT`,
else `lumina`. The example unit passes no `--agent`, so put
`SKAGENT=<agent>` in one of its env files (`join.env` is the usual place)
when the agent is not `lumina`.

Per agent (the canonical list is the docstring of `skcall/agents.py`):

| Variable | Meaning |
|---|---|
| `SKCALL_NC_URL` | the Nextcloud the agent's user lives on (required) |
| `SKCALL_ALLOWED_ROOMS` | comma list of room tokens skcall may join or watch |
| `SKCALL_OPERATORS` | comma list of Talk user ids treated as operators (private register, tools) |
| `SKCALL_ALLOW_OPERATOR_DMS` | `1` (default) also allows, unlisted, a one-to-one room with an operator and a group room whose only members are the agent and one operator; `0` turns that off |
| `SKAGENT` | the agent when no `--agent` is given (default `lumina`); the example unit relies on it |
| `SKCALL_NICK`, `SKCALL_AGENT_NAMES`, `SKCALL_PRONOUN` | display name, spoken names ("Jarvis, leave the call"), pronoun for the consent line |
| `SKCALL_AGENT_USERS` | comma list of Talk user ids of other agents (`--agent-users`): they are not counted as people in the call, never get operator tools, and their audio is not answered |
| `SKCALL_TTS_URL` | OpenAI-shaped `/v1/audio/speech` URL, or a comma list of servers with the same voice (required) |
| `SKCALL_TTS_MODEL`, `SKCALL_TTS_VOICE`, `SKCALL_TTS_SPEED` | for example `f5-tts` with a cloned voice, or `tts-1` (Kokoro) with a stock voice; speed 0.5 to 2.0 |
| `SKCALL_TTS_TOKEN` | bearer token sent to the TTS servers (unset: no header) |
| `SKCALL_TTS_REF` | reference clip for a server that clones per request (ZONOS3-Mini on SGLang-Omni), sent as `references`; a path the TTS server can read |
| `SKCALL_STT_BACKENDS` | ordered `kind:url` list, `fw:` (OpenAI-shaped) or `cpp:` (whisper.cpp); or `SKCALL_STT_URL` for one whisper.cpp (one of the two is required) |
| `SKCALL_STT_TOKEN` | bearer token sent to every STT backend (unset: no header) |
| `SKCALL_SITE_TURN` | the site's own TURN host, checked by `skcall smoke --turn` |
| `SKCALL_CONTEXT` | `1` (default) loads the agent's own memory into the voice prompt once at call start |

Host settings:

| Variable | Meaning |
|---|---|
| `NEXTCLOUD_TALK_USERNAME`, `NEXTCLOUD_TALK_APP_PASSWORD` | the agent's Talk login; from the environment, else read from `SKCALL_ENV_FILE` (default `~/.hermes/.env`). Never printed. |
| `SKCALL_ICE_IFACES` | comma list of IP prefixes to gather ICE host candidates on (default: every interface). Use it to keep out a VPN interface that STUN cannot cross, which otherwise costs a 5 s gather timeout. |
| `SKCALL_LIPSYNC_URL`, `SKCALL_LIPSYNC_TOKEN` | sklipsync WebSocket URL and token (unset: no lip-sync) |
| `SKCALL_VIDEO`, `SKCALL_VIDEO_MODE` | `0` for audio only; `static` or `animated` (default) |
| `SKCALL_AVATAR_IMAGE`, `SKCALL_AVATAR_EYES` | portrait (default `~/.skcapstone/agents/<agent>/avatar/portrait.png`), eye boxes for blinking |
| `SKCALL_HERMES_URL`, `SKCALL_HERMES_MODEL` | the Hermes api_server for escalated turns (default `http://127.0.0.1:8642/v1`, `hermes-agent`) |
| `SKCALL_HERMES_KEY` | that api_server's bearer key; when unset, `API_SERVER_KEY` is read from `SKCALL_ENV_FILE` instead. Never printed. |
| `SKCALL_SUMMARY_URL`, `SKCALL_SUMMARY_MODEL` | the OpenAI-shaped LLM that writes the post-call summary to the room chat (default `http://localhost:18780/v1`, model `sk-default`) |
| `SKCALL_SUMMARY_KEY` | bearer token for that LLM (unset: no header) |
| `SKCALL_POST_SUMMARY` | `on` (default) posts the summary; `off` turns it off |
| `SKCALL_LOG_TRANSCRIPT` | `on` (default); `off`: log lines keep timings and lengths, never the words (heard text, tool arguments and results, Hermes requests) |
| `SKCALL_HERMES` | `on` (default); `off`: never escalate to Hermes, like `--no-hermes` |
| `SKCALL_MAX_CALL_S` | hard limit on call length (2 h; the example unit sets 24 h) |

The voice layer's LLM legs come from skchat's `SKVOICE_LLM_URL`,
`SKVOICE_MODEL`, `SKVOICE_FALLBACK_URL`, `SKVOICE_FALLBACK_MODEL` (or the file
named by `SKCALL_VOICE_ENV`).

Setting up an agent's full personality (soul, bond FEB, skmemory, skwhisper,
reflections, face, voice, Hermes) is in
[docs/agent-personality-setup.md](docs/agent-personality-setup.md). The design
behind the client is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## How a call works

- On joining, the agent posts one chat line ("Lumina joined the call; she can
  hear and transcribe what is said here."). If the room does not let it
  publish audio, it joins listen-only and says so.
- Each participant's audio track is transcribed on its own, so every
  transcript is tied to a Talk user id. Transcription starts at the first
  pause, so the turn usually starts as soon as the person stops.
- The voice layer (skchat's voice engine, in-process) streams its reply; each
  sentence goes to TTS as soon as it is complete, so the next one renders
  while the current one plays. A person talking over the agent stops it.
- Requests that need real tools ("check my calendar", "look into ...") go to
  the agent's Hermes. The agent says a short acknowledgement, the call stays
  live, and the answer is spoken in the next gap (or posted to the chat if it
  takes too long).
- Tools run only for operators. A call is private only with exactly one human
  who is an operator; otherwise the agent keeps to a group register, without
  its bond and memory text.
- It leaves when the call ends, when it has been alone 20 s, after 10 min
  without speech, at the call length limit, or on "<name>, leave the call".
  One instance per agent and room: a second join exits 3 with `already in
  call`.

## Services

Each service has its own venv, `sk/setup.sh`, tests with a fake engine, and a
README with its full settings. All three answer `GET /v1/info` without a
token, so a client can check what it talks to; skcall's smoke test uses it.
Tokens are per client: a tokens file holds one `client=token` per line, and
the log names the client, never the token.

| Service | Interface | Default port | Engine |
|---|---|---|---|
| [sktts](services/tts/README.md) | `POST /v1/audio/speech` (JSON in, 24 kHz WAV out), `GET /v1/models`, `/health`, `/v1/info` | 18793 | Kokoro 82M (Apache-2.0), CUDA or CPU |
| [skstt](services/stt/README.md) | `POST /v1/audio/transcriptions` (multipart), `/health`, `/v1/info`; the dispatcher forwards to the first backend that answers, 503 `{"error": "no backend"}` when none does | 18794 (server and dispatcher both) | faster-whisper large-v3 (MIT), CUDA or CPU |
| [sklipsync](services/lipsync/README.md) | WebSocket `/v1/lipsync` (PCM in, JPEG mouth crops out), `/healthz`, `/v1/info` | 18850 | MuseTalk 1.5 (MIT), Intel XPU, CUDA or CPU |

The STT server (`SKSTT_PORT`) and the dispatcher (`SKSTT_DISPATCH_PORT`)
both default to port 18794; on one host, move one of them, for example the
server to 18795 behind a dispatcher on 18794.

Interfaces are additive: `/v1` paths keep working, and a breaking change gets
`/v2` beside them. MuseTalk is fetched by `services/lipsync/sk/setup.sh` at
the commit pinned in `services/lipsync/MUSETALK_REF`, from the fork at
github.com/smilinTux/MuseTalk (upstream PR TMElyralab/MuseTalk#429 adds
the Intel XPU support it needs).

## Deploying

The example units (`systemd/examples/skcall-join@.service`,
`services/lipsync/systemd/sklipsync.service`) are enough for one host. For a
fleet, a cluster manifest lists the Sidekick stack under `stacks:` and gives
each host a role; see skcapstone deploy (smilinTux/skcapstone `docs/DEPLOY.md`,
Stacks). The deploy templates live in
[sidekick-release](https://github.com/smilinTux/sidekick-release).

## Tests

```bash
# client: no network, fake audio, video and lip-sync
uv venv .venv -p 3.12 && uv pip install -p .venv/bin/python -e '.[test,voice]' -c sk/constraints.txt
.venv/bin/python -m pytest -q

# each service, with a fake engine (CPU only, no models)
cd services/tts && uv venv .venv -p 3.12 && uv pip install -p .venv/bin/python -r requirements-test.txt
.venv/bin/python -m pytest -q
# the same for services/stt and services/lipsync; services/common runs in any of those venvs:
cd services/common && ../tts/.venv/bin/python -m pytest -q
```

`scripts/` holds live self-tests against a real Talk room (loopback calls
between two agent sessions, transport and video measurements). They need a
real Nextcloud and are not part of CI.

## Security and contributing

Report vulnerabilities privately, as described in [SECURITY.md](SECURITY.md).
How to send changes: [CONTRIBUTING.md](CONTRIBUTING.md).

## License

AGPL-3.0-or-later, see [LICENSE](LICENSE). The voice engine is skchat's
(GPL-3.0), installed from the `skchat-sovereign` package and loaded
in-process; GPL-3.0 code combines with AGPL-3.0 under section 13 of both
licenses. The services' engines keep their own licenses (Kokoro Apache-2.0,
faster-whisper and Whisper MIT, MuseTalk MIT), noted in each service README.
