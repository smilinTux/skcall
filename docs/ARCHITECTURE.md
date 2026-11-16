# skcall: agents as real participants in Nextcloud Talk (and LiveKit) calls

The original architecture proposal, 2026-10-25, kept for its reasoning. What was built is in the README; where they differ, the README and the code win. Site details of the reference deployment are left out.

## 0. TL;DR

- **The target Nextcloud servers run the High Performance Backend (HPB).** The reference servers report Nextcloud 34.0.5, Talk 24.0.6, `signaling.mode: external`. `/standalone-signaling/api/v1/welcome` returns `nextcloud-spreed-signaling 2.1.1~docker`. The image is `ghcr.io/nextcloud-releases/aio-talk`, which bundles signaling, Janus, NATS and coturn. So the MVP targets the HPB with Janus (an SFU, a server that forwards each participant's stream separately). Mesh mode (every client connected directly to every other client) is a non-goal.
- **The proven pattern for joining a call without a browser is an aiortc client talking to the HPB.** Nextcloud's own `live_transcription` ExApp does exactly this to receive audio, and the open `talk-sip-bridge` does it to send audio into Janus. We lift the protocol, not the browser approach (talk-recording uses Firefox, Xvfb and pulseaudio).
- **Most of the voice code already exists in a transport-free form.** It lives in `skchat`:
  - `skchat.voice_engine` is the brain: persona, memory, LLM, tools, STT and TTS clients.
  - The pure logic classes in `skchat/transports/livekit.py` handle VAD (voice activity detection), barge-in (a human talking over the agent), addressing, dedup and the turn loop.
  - The new project is mostly a **Talk media backend plus a backend-agnostic call core**. LiveKit becomes a second adapter on the same core.
- **Suggested name: `skcall`.** Recommended license: **AGPL-3.0-or-later** (reason in section 7).

## 1. How a Talk call works (technical)

Sources:
- Talk API docs: https://nextcloud-talk.readthedocs.io/ (`spreed/docs/call.md`, `internal-signaling.md`)
- Signaling protocol: https://github.com/strukturag/nextcloud-spreed-signaling/blob/master/docs/standalone-signaling-api-v1.md

Joining has three planes:

1. **Talk REST, run as a Nextcloud user.**
   - `POST /ocs/v2.php/apps/spreed/api/v4/room/{token}/participants/active` joins the conversation and returns the Nextcloud `sessionId`.
   - `POST /ocs/v2.php/apps/spreed/api/v4/call/{token}` with `flags` joins the call. Flags: 1 in call, 2 audio, 4 video. Optional fields: `silent`, `recordingConsent`.
   - `PUT` updates the flags and `DELETE` leaves.
   - `GET /ocs/v2.php/apps/spreed/api/v3/signaling/settings?token=` returns the signaling `server` URL, `stunservers`, `turnservers` (with time-limited credentials) and `helloAuthParams`. Those are `{"1.0":{userid,ticket},"2.0":{token: JWT}}`.
2. **The signaling websocket to the HPB.** Client sends `hello` (v1 ticket or v2 JWT; the v2 public key is `spreed.config.signaling.hello-v2-token-key` in capabilities), then `room {roomid, sessionid}`. Media negotiation travels as `message` envelopes:
   - `offer` sent to your own session id **publishes** (`roomType: video` or `screen`).
   - `requestoffer` sent to a publisher's session id **subscribes**. Janus then sends an offer and the client answers.
   - `candidate` carries ICE candidates (network address options for the connection).
   - `event/participants/update` lists who is in the call and their `inCall` flags. Talk's `BackendNotifier.php` adds `actorId`/`userId` per session, which is how we get speaker attribution.
3. **Media: WebRTC to Janus.** In aio-talk, Janus reaches clients through the bundled coturn TURN relay. The Janus config sets `turn_server=$TURN_DOMAIN:3478` (see `nextcloud/all-in-one Containers/talk/start.sh`). The client must use the TURN credentials from the settings endpoint. A typical aio-talk setup hands out its TURN host on `:3478` (udp, tcp) and `:443` (turns); STUN is the same host plus `stun.nextcloud.com:443`.

**Two ways to authenticate an external client to the HPB:**

| | A. User session (recommended) | B. Internal client |
|---|---|---|
| Credential | Agent's own Nextcloud app password (already used by `hermes-nextcloud-talk/user/adapter.py`) | HPB `internalsecret`: `hello.auth.type=internal`, `token=HMAC-SHA256(secret, random)`, plus `backend` URL |
| Identity in call | Shows as the real user "Lumina", with avatar and name | Internal session; needs `internal/addsession` to show up as a virtual participant |
| Permissions | Talk enforces room permissions (the agent needs "publish audio") | Can join **any** room, uninvited; publishing is allowed by default |
| Blast radius | One user's app password | Root-equivalent for every call on the server |
| Prior art | Talk web and mobile clients | live_transcription, talk-sip-bridge, talk-recording, gocassini |

Pick A. It matches "agent is a real Nextcloud user in the room", needs no new secret on the agent host, and inherits Talk's permission and consent model. Keep B as a fallback if S1 (section 6) shows that the user flow cannot be driven headlessly. Talk's internal signaling (`/api/v3/signaling/{token}` long-poll) returns 400 when the HPB is enabled, so mesh code would be dead code on both servers.

## 2. Prior art

- **nextcloud/live_transcription** (https://github.com/nextcloud/live_transcription, AGPL-3.0-or-later, Python, read at commit `517ddad`). This is the closest blueprint.
  - All signaling is in `ex_app/lib/spreed_client.py` (1486 lines).
  - `send_hello` (lines 443-466): internal hello with HMAC.
  - Lines 468-486: `internal/incall` then `room`.
  - `signalling_monitor` (roughly lines 908-1028): for each user session with `inCall & 1 && inCall & 2` it sends `requestoffer` with `roomType: video`, with a 10 s cooldown.
  - `handle_offer` (lines 1097-1271): an aiortc `RTCPeerConnection` with an `audio recvonly` transceiver. It waits for ICE gathering to finish (aiortc cannot trickle candidates), then sends `a=candidate` lines with `sdpMid 0`.
  - Each remote audio track feeds its own Vosk transcriber, keyed by the publisher's HPB session id.
  - It receives only and publishes no media. Mapping speakers to user ids is a TODO in the code (line 958), even though the data is already in the participants event.
  - It is started through Talk's live-transcription API (`POST /ocs/v2.php/apps/spreed/api/v1/live-transcription/{token}`, which proxies to the ExApp's `/api/v1/call/transcribe`). It requires the HPB.
- **schnebeck/talk-sip-bridge** (GPL-3.0, aiortc). An open replacement for Nextcloud's closed SIP bridge. It **publishes audio into Janus** with `addTrack` (`bridge/call_media.py:117-141`) and uses `internal-incall` and `addsession` (`bridge/talk_messages.py`). This is the reference for sending audio.
- **voxonta/nextcloud-talk-capture** (MIT). A clean re-implementation of the live_transcription protocol in `src/talk_capture/spreed_client.py`. It can be read without AGPL concerns.
- **nextcloud/nextcloud-talk-recording** (AGPL-3.0). Selenium drives Firefox or Chromium on Xvfb, with a pulseaudio null sink and ffmpeg (`Service.py`, `Participant.py:499-545`). It uses the backend secret plus the internalsecret and requires the HPB. Proven, but heavy. Wrong shape for a low-latency duplex agent.
- **anentree/nextcloud-talk-notetaker** (AGPL-3.0). Playwright, logged in as a normal bot user, hooks `RTCPeerConnection`. Shows that the user-session route works without the internalsecret, but through a browser.
- **codemyriad/gocassini** (AGPL, Go, pion) and **silviot/nc_kyutai_live_transcriptions_go**. Go ports of the same internal-client flow.
- **Hermes Agent's own call support** (hermes-agent, MIT):
  - `plugins/google_meet/` uses Playwright and caption scraping, plus a realtime voice through a PulseAudio null-sink mic. It exposes the tools `meet_join / meet_leave / meet_status / meet_transcript / meet_say`. Copy this tool surface.
  - `plugins/platforms/discord/adapter.py` has `join_voice_channel`, `VoiceReceiver`, a per-user PCM buffer and `_process_voice_input`. That path does STT, then calls `_voice_input_callback`. `gateway/run_voice.py:_handle_voice_channel_input` turns the transcript into a synthetic `MessageEvent` on the bound text channel, and the reply goes back out through TTS (`play_in_voice_channel`, `voice_mixer.py`).
  - Hermes therefore already treats "voice is a modality of a chat session". It is batch STT, not a low-latency duplex loop.

**What needs the HPB:** subscribing to individual streams through Janus (`requestoffer`), and internal clients. **What works with internal signaling:** P2P mesh only, where each peer's stream is also separate. Not relevant here.

## 3. What a site needs

| | Needed |
|---|---|
| Nextcloud / Talk | Talk with the HPB (`signaling.mode: external`); the reference servers run Nextcloud 34.0.5 and Talk 24.0.6 |
| HPB | aio-talk or another nextcloud-spreed-signaling setup (2.1.1 was tested) |
| TURN | a TURN relay the agent host can reach on `:3478` udp/tcp, ideally also `:443` turns |
| Agent user | a real Nextcloud user for each agent, with an app password, invited to the rooms it may join |
| live-transcription capability, Talk bots | not needed |

## 4. Reusable code inventory

| Source | What to lift | Verdict |
|---|---|---|
| `skchat/src/skchat/voice_engine/` (GPL-3.0) | `VoiceEngine.respond()`, `STTClient` (whisper hallucination filter), `TTSClient` (batch + stream), `LLMClient` (stream, tools, fallback), `PersonaBuilder` (soul + FEB, private and group modes), `MemoryBridge`, `ToolRegistry`, `caller_profile` | **Core brain**, use as is |
| `skchat/src/skchat/transports/livekit.py` lines 217-940 | `VADSegmenter`, `BargeInDetector`, `AddressingGate` (multi-agent roundtable + turn cap), `TranscriptDedup`, `run_turn`, `mode_ceiling`, `normalize_pcm`, `wav_to_pcm`, `split_for_speech`, `is_non_speech`, `load_avatar_rgba` | **Move into the call core.** They are already pure and unit-tested |
| `skchat/.../livekit.py:build_room_session` (line 993) | LiveKit room loop | Becomes the LiveKit adapter |
| `lumina-creative/scripts/lumina-call.py` (3202 lines) | Original monolith (two-mode prompts, filler phrases, MuseTalk lipsync, `Speaker` class) | Already ported. Mine only `lipsync_frames` and the filler logic |
| `skvoice/` (GPL-3.0) | `emotion.py` (pitch, energy, pace tags) | Move into the core as an optional feature. Keep the skvoice WS service as the no-SFU fallback (per `skchat/docs/superpowers/specs/2026-08-12-voice-transport-consolidation-design.md`) |
| `hermes-nextcloud-talk/user/adapter.py` (MIT) | App-password OCS client, mention parsing (`_is_mention` already accepts `type: call`), room modes, allowlists | Reuse the client. Add `systemMessage` handling (`call_started` / `call_ended`) |
| Hermes `plugins/google_meet` (MIT) | Tool surface and node-routing pattern | Mirror for `call_*` tools |

Speech endpoints of the reference deployment at the time:
- STT: whisper.cpp large-v3-turbo on an Arc iGPU, OpenAI-shaped `/v1/audio/transcriptions`.
- TTS: F5-TTS `/v1/audio/speech`, a cloned voice, 24 kHz WAV, not streaming.
- The GPU host (an RTX 5060 Ti, 16 GB) was mostly taken by a local LLM, so there was no room for lipsync beside it.

## 5. Architecture

```
         Nextcloud Talk                                   LiveKit
  REST (OCS) + HPB signaling ws + Janus via TURN         livekit server + token
                 |                                               |
     +-----------v-----------+                       +-----------v-----------+
     | TalkBackend (aiortc)  |                       | LiveKitBackend (rtc)  |
     | user-session hello,   |                       | wraps rtc.Room        |
     | publisher + N subs    |                       |                       |
     +-----------+-----------+                       +-----------+-----------+
                 \______ CallBackend / CallSession protocol _____/
                                    |
   +--------------------------------v---------------------------------+
   | skcall core (one per agent per call)                             |
   |  per-participant: resample 48k->16k -> VADSegmenter -> STT       |
   |  TranscriptDedup -> AddressingGate (who is talking to me?)       |
   |  Brain.respond(speaker, text) --stream--> split_for_speech       |
   |  TTS per sentence -> resample 24k->48k -> AudioOut (20ms frames) |
   |  BargeInDetector cancels AudioOut + in-flight Brain turn         |
   |  VideoOut (static avatar, later lipsync) / VideoIn (frame tap)   |
   |  Transcript + summary -> Talk chat + skmemory                    |
   +----------+-----------------------------+-------------------------+
              |                             |
     Brain: VoiceEngineBrain        STT / TTS over HTTP (services/)
     (skchat.voice_engine)          vision via an LLM gateway
     or HermesBrain (api_server)
              ^
   +----------+-----------------------------------------------+
   | Call controller                                           |
   |  Hermes plugin tools: call_join/leave/status/say          |
   |  triggers: "@lumina join call" (agent calls tool),        |
   |  auto-join on call_started system msg (allowlisted rooms) |
   +-----------------------------------------------------------+
```

### 5.1 Backend-agnostic interface

```python
class CallBackend(Protocol):
    async def join(self, room: RoomRef, identity: AgentIdentity) -> "CallSession": ...

class CallSession(Protocol):
    def events(self) -> AsyncIterator[CallEvent]  # ParticipantJoined/Left(id, user_id, display, is_agent)
                                                  # AudioTrack(participant, frames: AsyncIterator[PCM48k])
                                                  # VideoTrack(participant, kind="camera"|"screen")
                                                  # CallEnded
    async def audio_out(self) -> AudioSink        # write(pcm48k_mono_20ms); clear() for barge-in
    async def video_out(self) -> VideoSink | None # push RGBA frames
    async def set_status(self, state: str): ...   # speaking/listening (LiveKit metadata, Talk: no-op or chat)
    async def post_chat(self, text: str): ...     # consent notice, transcript links
    async def leave(self): ...
```

The core never imports aiortc or livekit. The current `build_room_session` gets split into a LiveKit adapter plus this core, and the Talk adapter implements the same five methods.

### 5.2 Talk adapter: join sequence (user session)

1. `GET signaling/settings?token=X`, authenticated with the app password. This yields the server, `helloAuthParams` and TURN credentials.
2. `POST room/X/participants/active` returns the Nextcloud `sessionId`.
3. Open the websocket to `<server>/spreed`. Send `hello` v2 with the JWT (fall back to v1 with the ticket), then `room {roomid: X, sessionid}`.
4. `POST call/X {flags: 3}` for audio (7 once video is enabled). Use `silent: true` so joining does not ring everyone.
5. **Publish:** create an aiortc PC with a custom `MediaStreamTrack` (48 kHz Opus, fed from the `AudioSink` ring buffer). Send `offer` to our own session with `roomType: video`, apply Janus's `answer`, and send candidates after gathering completes (live_transcription pattern).
6. **Subscribe:** for every participant with `inCall & 2` (except ourselves), send `requestoffer`. Answer `recvonly` and emit `AudioTrack(participant=userId)`. Attribution comes from `actorId`/`userId` in the participants event.
7. On `update` with `incall == 0` for everyone, or when the agent is the last participant: `DELETE call/X`, then `DELETE participants/active`.

### 5.3 Brain: how turns reach the agent

- **Default: `VoiceEngineBrain`** (`skchat.voice_engine`). It streams, has a voice-length brevity rule, and uses a curated tool set, so the latency is right for speech. Persona mode comes from `mode_ceiling`: group mode whenever anyone other than the operator is in the call, private mode for a 1:1 with the operator. The tool gate uses the HPB `userId`, so attribution is real rather than a display-name guess.
- **Escalation: `HermesBrain`.** When `wants_action()` fires ("book that", "file a ticket"), or on an explicit "ask Hermes", the turn goes to Hermes's OpenAI-compatible `api_server` (`gateway/platforms/api_server_openai_routes.py`, `/v1/chat/completions`, SSE). It is tagged with the Talk room as session key, so the voice turn and the room's text history share one mind.
  - api_server has to be enabled in the agent's Hermes config.
  - While Hermes works, the agent says a filler line ("on it") and posts the result in chat.
- Mirroring Discord's synthetic `MessageEvent` path is the alternative. It is simpler but batch-only, and every utterance would go through the full agent loop (too slow). Use it only for "post this utterance into the chat session" continuity.

### 5.4 Interruption, multi-speaker, multi-agent

- **Barge-in:** a per-participant `BargeInDetector` runs while `AudioOut` is playing. When it fires, the core clears the sink buffer, cancels the TTS task and cancels the streaming brain turn (the same semantics as `lumina-call.py:Conversation.interrupt`). Janus does not loop the agent's own stream back to it, so the agent does not hear itself. Echo from human speakers' devices stays a risk, which is why the barge-in RMS threshold is raised.
- **Multi-speaker:** the HPB gives one track per publisher, so VAD and STT run per participant and every transcript is attributed to a user id. That is better than the mixed audio the SIP route gets.
- **Multi-agent:** each agent is its own process, Nextcloud user and session. `AddressingGate` already does named addressing, follow-up windows and `LUMINA_AGENT_TURN_CAP` loop damping. Agents subscribe to each other's tracks, so a roundtable works.

### 5.5 Video

> **Built (2026-10-25, P5a + P5c):** see README "Avatar video". Core types
> `VideoSource` (pull: `frame(t)` for monotonic time t) and `VideoSink` in
> `skcall/core/session.py`; implementations in `skcall/video/`; Talk's
> `AvatarTrack` pulls at the source's fps on the same clock as the audio
> sink's on_start, which is how the compositor syncs lips to audio playout.
> The lip-sync service contract is the fixed WebSocket protocol in
> `skcall/video/lipsync.py` and `services/lipsync/README.md`.

- **Out, phase 1:** a static portrait at 1 to 5 fps (`load_avatar_rgba` from `~/.skcapstone/agents/$SKAGENT/avatar/portrait.png`), sent as a VP8 track on the same publisher PC. Cheap, and it makes the agent visible in the grid.
- **Out, phase 2:** a speaking or idle loop, then MuseTalk lipsync (`lipsync_frames`). MuseTalk needs about 8.5 GB of VRAM, so it needs a GPU with room to spare, scheduling, or a lighter lipsync model.
- **In:** `requestoffer` already returns video if the publisher has it, so subscribe to the video transceiver. Screen shares use `roomType: screen`. Sample one frame every N seconds, or on demand ("look at my screen"). Down-scale it and send it to a vision model through an LLM gateway as an image part of the next turn. Never stream frames into the LLM continuously.

### 5.6 Latency budget (speech end to first agent audio, estimates to verify in spike S3)

| Stage | Target |
|---|---|
| VAD hangover (end-of-turn detection) | 500-700 ms |
| STT whisper.cpp turbo (Arc), 3-5 s utterance | 300-700 ms |
| LLM first sentence via the gateway (streaming) | 400-1200 ms |
| F5-TTS first sentence (not streaming) | 300-800 ms |
| Resample, Opus, Janus, TURN, jitter buffer | 100-200 ms |
| **Total** | **about 1.6-3.6 s** (goal under 2 s for short replies) |

Levers:
- Sentence-level TTS pipelining (already in `split_for_speech` and `TTSClient.stream`).
- Filler acknowledgements for tool turns.
- A smarter turn detector later (Silero VAD or a turn model).
- Running the media daemon on the same LAN as the speech services, not over a VPN hop through TURN.

### 5.7 Placement

- **Media daemon:** on each agent's host, beside its Hermes. It is CPU-only (aiortc, Opus, resampling).
- **GPU work stays behind HTTP** on the GPU hosts (STT, TTS, lipsync). Do not load models into the call process.
- **TURN:** media to Janus goes through the server's TURN (port 3478). With a public Nextcloud no VPN relay is needed. A private relay only matters for the LiveKit backend.

### 5.8 Security and consent

- Summons only from **allowlisted rooms and users**. Reuse `NEXTCLOUD_TALK_ROOMS_JSON` plus a new `SKCALL_SUMMONERS` list (the operators' Talk user ids). Auto-join on `call_started` is a per-room opt-in.
- **On join, post a chat notice** such as "Lumina joined the call and is listening (transcribing locally, not recording). Say 'Lumina leave' to dismiss." Honor Talk's `recording-consent` capability (0 today). If a room enables consent, do not join until it is given.
- **Voice "leave" command.** Hard limits on call length (for example 2 h) and idle time (for example 10 min alone or with no speech) to cap TURN and GPU use.
- **Group-mode ceiling and tool gating** go through `caller_profile`, keyed on the verified Nextcloud `userId`. Intimate-only topics are deflected whenever anyone other than the operator is in the call (the existing `SYSTEM_PROMPT_GROUP` rule).
- **Storage:** transcripts go to skmemory (the private split) and an optional summary to the room. Raw audio is never stored unless a recording mode is requested explicitly.
- **If option B (internalsecret) is ever used:** keep the secret in a secret store, only on the media host, never in Hermes config, and audit every join.

## 6. Phase plan

| Phase | Scope | Effort |
|---|---|---|
| **P0 spikes** | S1: user-session join plus receive one audio track with the agent's app password (python script, about 300 lines adapted from voxonta/live_transcription). S2: publish a sine wave, then an F5 WAV, into Janus. A human in the browser hears it, and we confirm aiortc gets through the aio-talk TURN on 3478 and 443. S3: measure stage latencies end to end. S4: confirm the agent user's room permissions allow publishing audio. | S (3-5 days) |
| **P1 MVP** | `skcall` repo; core plus TalkBackend, audio only; one agent on one server; joins via the `call_join` tool when @mentioned ("@lumina join call"); VoiceEngineBrain; barge-in; consent notice; leaves on end, empty room or voice command; transcript to skmemory; systemd user unit. | M (2-3 weeks) |
| **P2 Hermes integration** | Hermes plugin `call_join/leave/status/say/transcript` (google_meet pattern, talking to the daemon over local HTTP); `systemMessage` `call_started` handling in `hermes-nextcloud-talk` for opt-in auto-join; HermesBrain escalation through api_server; post-call summary into the room. | M |
| **P3 LiveKit adapter** | Re-home `build_room_session` onto the core; skchat's call answerer spawns `skcall` instead of its silence publisher. | S-M |
| **P4 multi-agent** | Two agents in one call, roundtable tuning, a second server. | M |
| **P5 video** | Static avatar out (S), vision frame tap and screen-share subscribe (M), animated or lipsync avatar with GPU scheduling (L). | M-L |

### Risks and how to de-risk each

| Risk | De-risk |
|---|---|
| The HPB rejects the room join for a session created by an app-password REST client (cookie-bound session, ticket expiry) | Spike S1. Fallback is option B with the internalsecret plus `addsession`, carrying `actorId` |
| aiortc has no ICE trickle and its TURN-over-TLS support is weak; Janus via the aio-talk relay | Spike S2 against both 3478/udp and 443/turns. live_transcription's wait-for-gathering pattern is proven |
| HPB 2.1.1 behavior differs from what live_transcription expects ("HPB released after Sept 2025") | Check the 2.1.1 release date and changelog in S1 |
| Talk web clients may hide or not render audio-only participants, or the agent may lack publish permission | S4: check participant permissions. Set `flags=3`, then add the avatar video early |
| Whisper hallucinates on silence or noise and the agent talks over people | Existing `is_hallucination` filter plus an RMS gate plus `AddressingGate` (speaks only when addressed in group mode) |
| GPU contention with a local LLM on the same card | Keep STT on its own device and TTS resident (1 GB). Video lipsync deferred to P5 with explicit scheduling |
| Echo from human speakerphones triggers barge-in | Raised barge-in RMS plus dwell time (300 ms). Later, WebRTC AEC on the human side is the browser's job |
| AGPL obligations if Nextcloud code is lifted | See section 7 |

## 7. Repo proposal

**Name options:**
1. `skcall`: plain, matches `skchat`, `skvoice` and `skmemory`. Recommended.
2. `skpresence`: "agent presence in rooms".
3. `skhuddle`: friendlier, a little cute.
4. `skrollcall`.

**Layout:**

```
skcall/
  src/skcall/
    core/            session.py (turn loop), vad.py, bargein.py, addressing.py, dedup.py,
                     audio.py (resample/normalize), speech.py (split_for_speech), emotion.py
    brain/           base.py (Brain protocol), voice_engine.py (adapter over skchat.voice_engine),
                     hermes.py (api_server client)
    backends/
      base.py        CallBackend / CallSession / events
      talk/          ocs.py (REST), signaling.py (HPB ws client), media.py (aiortc tracks),
                     backend.py
      livekit/       backend.py (rtc.Room wrapper)
    video/           avatar.py (static), lipsync.py (MuseTalk client, later), frames.py (vision tap)
    controller/      api.py (local HTTP: join/leave/status/say), policy.py (allowlists, consent)
    cli.py           skcall join --backend talk --room <token> --agent lumina
  hermes_plugin/     plugin.yaml + tools (call_join, call_leave, call_status, call_say, call_transcript)
  systemd/           skcall@.service (per agent)
  tests/             core logic tests ported from skchat tests/test_transport_livekit.py
```

**Move vs stay:**
- **Move into `skcall.core`:** the pure classes from `skchat/transports/livekit.py` plus `skvoice/emotion.py`. skchat then imports them back from skcall, or keeps thin re-exports during migration.
- **Stays in skchat:** `voice_engine` (the brain; skcall depends on it, with a later option to split it into an `skvoice-engine` package), the room model (`derive_room`, signed invites) and the call answerer.
- **Stays in skvoice:** the WS service as the no-SFU fallback.
- **Stays in hermes-nextcloud-talk:** chat and the call_started hook.

**License:**
- The deps: skchat and skvoice are GPL-3.0. aiortc is BSD-3. livekit is Apache-2.0. live_transcription, talk-recording, gocassini and nextcloud-spreed-signaling are AGPL-3.0. voxonta's capture client is MIT. talk-sip-bridge is GPL-3.0.
- Copying or adapting `spreed_client.py` makes the Talk backend AGPL.
- **Recommendation: license `skcall` AGPL-3.0-or-later from day one.**
  - GPL-3.0 code can combine with it (GPLv3 section 13).
  - It is a network service anyway.
  - It lets us lift live_transcription and talk-sip-bridge code freely, with attribution in `THIRD_PARTY.md`.
- If GPL-3.0 were preferred, write the Talk signaling client clean-room from the protocol doc and the MIT voxonta code, and do not copy AGPL files.
