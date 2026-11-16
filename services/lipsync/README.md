# sklipsync: real-time lip-sync service

sklipsync turns an agent's reply audio into mouth-region video crops while the call is live. skcall sends 16 kHz PCM for each utterance over a WebSocket. The service streams back JPEG crops of the mouth, one per video frame. skcall pastes each crop onto the avatar's frame (the static portrait or the idle loop) at the matching audio time.

- **Engine today:** MuseTalk 1.5 (MIT code and weights). The reference host runs it on an Intel Arc Pro B70 through PyTorch XPU; CUDA and CPU work too.
- **Endpoints:** `ws://<host>:18850/v1/lipsync`, health at `http://<host>:18850/healthz`, and `GET /v1/info` (`{"service": "lipsync", "interface": 1, "version": ...}`, no token needed).
- **License:** AGPL-3.0-or-later, like the rest of skcall. MuseTalk and its sub-models are MIT, Apache-2.0 or BSD-3.

## Wire protocol (fixed)

The skcall client (`skcall/video/lipsync.py`) is built against this exact text. If you change the protocol, change it here and in the client together, and record the change.

```
WebSocket ws://<host>:<port>/v1/lipsync
header    Authorization: Bearer <SKCALL_LIPSYNC_TOKEN>        (401 without it)

client -> {"type":"hello","avatar":"lumina","fps":25,"engine":"musetalk"}
server -> {"type":"ready","fps":25,"crop":{"x":155,"y":322,"w":250,"h":137},"frame_size":[560,720]}

per utterance:
  client -> {"type":"utterance","id":"<uuid>"}
  client -> binary 16 kHz mono s16le PCM, any chunk size (an odd byte count is carried over)
  client -> {"type":"end","id":"<uuid>"}
  client -> {"type":"cancel","id":"<uuid>"}      barge-in: rendering stops at once
  server -> binary: u32 BE frame index | u16 BE uuid length | uuid ascii | JPEG crop
  server -> {"type":"done","id":"<uuid>","frames":N}

keepalive: {"type":"ping"} / {"type":"pong"}, either side (plus WebSocket heartbeats every 30 s)
errors:    {"type":"error","code":"...","message":"...","id":"<uuid, when it is about one>"}
```

- Frame `i` is the mouth at audio time `i/fps` from the start of the utterance. `N = ceil(samples * fps / 16000)`.
- The crop goes at `crop` on the avatar's full frame (`frame_size`). It is already blended with MuseTalk's face-parsing mask, so it matches the portrait at its edges and a plain paste works. The skcall compositor's feathered paste works too. The JPEG is exactly `w x h`.
- Frames start flowing while audio is still arriving. A frame is rendered once the audio covers it plus a lookahead (0.2 s by default).
- After a `cancel`, at most the batch already on the GPU is still sent (up to 8 frames). The server then sends `{"type":"done","id":..,"frames":<sent>,"cancelled":true}`. Clients can ignore it.
- A new `utterance` implicitly ends the previous one. Several utterances can render at once, on one connection or across connections. The GPU interleaves them batch by batch.
- `fps` is clamped to 5 to 25 (`SKLIPSYNC_MAX_FPS`). The value in `ready` is the one the server uses.
- Avatars are prepared once (face box, mask and latent are cached on disk), then reused. On `hello`, a new or changed portrait is prepared, which takes about 10 s. The skcall client's 5 s ready timeout makes it retry, and the retry hits the cache. Every avatar in the directory is prepared at startup, so this only happens for portraits added while the service is running.

Error codes: `engine_unavailable` (for example `ditto`), `avatar_not_found`, `avatar_prepare_failed`, `no_hello`, `bad_id`, `bad_message`, `render_failed`. The first three close the connection.

## Engines

`sklipsync/engines/__init__.py` defines the interface. A second engine (Ditto) is a new module, not a rewrite.

```python
handle = engine.prepare(avatar)            # once per avatar (cached): frame_size, crop
stream = engine.open_stream(handle, fps)   # once per utterance
stream.push(pcm16k_chunk); stream.end()    # from the event loop
batch  = stream.next_batch()               # [(frame_index, raw), ...]; blocking GPU work
crop   = engine.compose(handle, raw)       # BGR crop for handle.crop; CPU, thread-safe
engine.render(handle, pcm16k)              # offline convenience: yields (index, crop) for a whole clip
```

| Engine | Status | Notes |
|---|---|---|
| `musetalk` | working | MuseTalk 1.5. It repaints only the mouth of a still portrait, so liveliness (blinks, sway) has to come from the skcall idle loop under it. Runs on Intel XPU, CUDA or CPU. |
| `ditto` | stub | Selecting it returns `engine_unavailable` with the reason. See below. |
| `fake` | tests and dev | No model. Flat crops whose grey level encodes the frame index (the same scheme as `skcall/video/fake_lipsync.py`). |

### MuseTalk engine details

- The code comes from the git submodule `third_party/MuseTalk`, pinned to commit `3bd6549` of the fork's `feat/intel-xpu-device` branch (https://github.com/smilinTux/MuseTalk/tree/feat/intel-xpu-device). That branch is upstream PR [TMElyralab/MuseTalk#429](https://github.com/TMElyralab/MuseTalk/pull/429): a device helper with XPU support and an optional `face_alignment` landmark backend, so no mmcv/mmpose is needed. No MuseTalk code is copied into this repo. When #429 merges, the submodule can point at upstream.
- Streaming: whisper-tiny features are computed per window (up to about 28 s, with 1 s of left context). For each frame, the service picks the same 10 whisper rows that MuseTalk's `get_whisper_chunk` would pick, with zero rows before the start and after the end, as upstream does. The UNet and the VAE decode run in batches of 8 (`SKLIPSYNC_BATCH`), and kernels for every batch size from 1 to 8 are compiled at startup, because XPU compiles a kernel the first time it sees each shape.
- `compose` blends only the changed region: the lower face box under the jaw-mode mask. That is about 250x137 px for Lumina, around 5 KB per JPEG at quality 85, or about 1 Mbps at 25 fps.
- Weights (3.8 GB): `musetalkV15/{unet.pth,musetalk.json}`, `sd-vae`, `whisper` (tiny), `face-parse-bisent`, plus face_alignment's s3fd/2DFAN4 in `~/.cache/torch/hub/checkpoints`. `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` is set inside the engine, because torch 2.6+ refuses the legacy face-parse checkpoint (upstream PR #419).

### Ditto (future engine)

[Ditto](https://github.com/antgroup/ditto-talkinghead) (Apache-2.0 code) animates the whole head: pose, blinks and expression. MuseTalk only does the mouth. What adding it needs:

1. **An NVIDIA GPU.** Ditto's real-time path is TensorRT 8.6 and CUDA only. There is an open RTX 5090 / Blackwell issue (#88). Its PyTorch path has not been tried on Intel XPU. It also needs a card with room to spare next to whatever else runs on it.
2. **A license check** of the released weights and every sub-model before shipping.
3. **Code:** `engines/ditto.py` would map Ditto's `stream_pipeline_online.py` onto the interface above. `prepare` registers the source image, `open_stream` starts an online session, and `next_batch` pulls the rendered frames.
4. **A protocol decision.** The head moves, so one fixed `crop` no longer works. Either return full frames (with `crop` covering the whole frame) or add a per-frame crop box. Both change the wire protocol and have to be agreed with the skcall client first.

## Avatars

```
~/skcall-lipsync/avatars/<id>/portrait.png      (or .jpg / .jpeg / .webp)
~/skcall-lipsync/avatars/<id>/avatar.json       optional: {"max_side":720,"bbox_shift":0,"extra_margin":10,"parsing_mode":"jaw","cheek_width":90}
~/skcall-lipsync/cache/musetalk/<id>/<hash>/    frame.png, mask.npy, latent.pt, meta.json (rebuilt when the portrait or settings change)
```

- A frontal, well-lit portrait works best. An 896x1152 portrait is scaled to 560x720; for one such portrait the face box is `[155,217,405,459]` and the crop `x=155 y=322 w=250 h=137`.
- A tilted head is better roll-corrected first, so the eye line is level (for example from insightface landmarks). A re-render that changes the face (InstantID was tried) lost: ArcFace similarity to the original fell from 0.98 to 0.83 and the crop edge showed more seam.
- skcall needs the same portrait (or an idle loop of the same framing) on its side as the base frame. The crop coordinates are in the service's `frame_size` (560x720), and the client scales them to its own frame.

## Install and run

```bash
git archive HEAD | tar -x -C /path/to/export
cd /path/to/export/services/lipsync
bash sk/setup.sh          # .venv with requirements.txt (XPU torch) and MuseTalk at MUSETALK_REF
printf 'alice=%s\n' "$(openssl rand -hex 24)" > ~/.config/sklipsync/tokens   # mode 0600
SKLIPSYNC_TOKENS_FILE=~/.config/sklipsync/tokens .venv/bin/python -m sklipsync serve
curl -s localhost:18850/v1/info
curl -s localhost:18850/healthz   # engine, device, VRAM reserved, avatars, last utterance stats
```

`systemd/sklipsync.service` is an example user unit for the same layout. Startup takes about 20 s: 8 s to load models, 11 s of kernel warm-up, and cached avatars load instantly. On a shared GPU, pin the card (`ZE_AFFINITY_MASK` on Intel, `CUDA_VISIBLE_DEVICES` on NVIDIA) and cap the process with `SKLIPSYNC_VRAM_GB`.

## Tokens

`SKLIPSYNC_TOKENS_FILE` holds one `client=token` per line (`#` comments). Each connection is logged naming its client (`client=alice`, or `client=unknown` on a 401), never the token. The service refuses to start without a token. The old single shared token (`SKLIPSYNC_TOKEN`, or `SKCALL_LIPSYNC_TOKEN` with no tokens file) still works as client `legacy` during a transition, and logs a deprecation warning. skcall sends its token from `SKCALL_LIPSYNC_TOKEN`. To rotate one client's token, change its line and restart the service.

Configuration (environment variables, or the same flags on `python -m sklipsync serve`): `SKLIPSYNC_HOST` (default `127.0.0.1`), `SKLIPSYNC_PORT` (18850), `SKLIPSYNC_TOKENS_FILE`, `SKLIPSYNC_AVATARS`, `SKLIPSYNC_CACHE`, `MUSETALK_MODELS`, `MUSETALK_DIR`, `SKLIPSYNC_ENGINES` (default `musetalk`), `SKLIPSYNC_BATCH` (8), `SKLIPSYNC_LOOKAHEAD_S` (0.2), `SKLIPSYNC_MAX_FPS` (25), `SKLIPSYNC_JPEG_QUALITY` (85), `SKLIPSYNC_VRAM_GB` (12).

## Measured (2026-10-25, client over a VPN at about 44 ms RTT, Intel Arc Pro B70)

12.9 s Lumina F5-TTS clip, 25 fps, batch 8, all 323 frames received each time:

| Case | First frame (client side, from the first PCM byte sent) | Frames/s delivered | Notes |
|---|---|---|---|
| Whole utterance sent at once (how skcall sends sentences) | 169 to 253 ms | 49 to 59 | 1.9 to 2.4x real time, 3 runs |
| Audio paced at real time | 377 ms (275 ms server side) | 25 (follows the audio) | the first frame waits for 0.32 s of audio (0.2 s lookahead plus the frame's own window) |
| Two streams at once | 177 / 295 ms | 30.5 each | both above real time |
| Cancel after 0.5 s | | | 7 frames arrived after the cancel was sent (one batch), then `done` |

- **VRAM:** +9.1 GB with the service loaded and warm (torch reserved 8372 MiB). Flat while rendering.
- **An LLM on the other card of the same host** (4-token completion): 0.63 to 0.64 s before, 0.63 s during rendering, 0.62 to 0.63 s after. No effect.
- **whisper-large-v3 on the same card** (3 s transcription): 0.81 to 0.95 s before, 1.6 to 1.8 s during rendering, 0.82 s after. They share the card's compute, so transcription slows down while an agent is speaking.
- Bandwidth: about 5 KB per crop, 1.05 Mbps per stream at 25 fps.

## Test client and tests

```bash
# the token comes from the env (never pass it on the command line)
export SKCALL_LIPSYNC_TOKEN=...
python scripts/lipsync_client.py --url ws://lipsync.example.org:18850/v1/lipsync --wav clip.wav \
    --base portrait.png --out lumina.mp4      # also: --realtime, --repeat N, --cancel-after S

# protocol tests on the fake engine (no GPU)
cd services/lipsync
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python -r requirements-test.txt
.venv/bin/python -m pytest -q

# local dev server without a GPU
SKCALL_LIPSYNC_TOKEN=dev python -m sklipsync serve --engines fake --avatars-dir /path/to/avatars
```
