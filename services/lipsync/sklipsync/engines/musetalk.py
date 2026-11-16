"""MuseTalk 1.5 engine: repaints the mouth region of a still portrait.

MuseTalk's code comes from the pinned submodule third_party/MuseTalk (the
fork at github.com/smilinTux/MuseTalk with the Intel XPU / face_alignment
changes, upstream PR TMElyralab/MuseTalk#429); nothing is copied here. This module only
re-arranges MuseTalk's own steps (whisper-tiny features, positional
encoding, UNet, VAE decode, mask blend) for streaming:

- Audio arrives in chunks. Whisper features are computed over a window
  (up to about 28 s, with 1 s of left context) as soon as there is enough
  audio for a batch plus a short lookahead, so frames start flowing while
  the client is still sending.
- Frame i uses the same 10 whisper rows MuseTalk's get_whisper_chunk would
  pick for it (zero rows before the start and after the end, as upstream).
- Only the changed region is blended and returned: the lower face box, under
  MuseTalk's own face-parsing mask, so the crop pastes cleanly on the
  avatar's frame.
"""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from ..protocol import PCM_RATE, frames_for
from . import AvatarHandle, Engine, EngineUnavailable, RenderStream

LOG = logging.getLogger("sklipsync.musetalk")

SERVICE_DIR = Path(__file__).resolve().parents[2]          # services/lipsync
DEFAULT_MUSETALK_DIR = SERVICE_DIR / "third_party" / "MuseTalk"

ROWS_PER_S = 50                          # whisper encoder rows per second
SAMPLES_PER_ROW = PCM_RATE // ROWS_PER_S  # 320
WIN_ROWS = 1500                          # whisper-tiny encodes a 30 s window
PAD_LEFT, PAD_RIGHT = 2, 2               # MuseTalk audio_padding_length_left/right
CLIP_ROWS = 2 * (PAD_LEFT + PAD_RIGHT + 1)   # 10 whisper rows per video frame
CTX_ROWS = 50                            # 1 s of left context for later windows
FACE = 256                               # MuseTalk face resolution


class _Stream(RenderStream):
    def __init__(self, eng: "MuseTalkEngine", handle: AvatarHandle, fps: int):
        self.e, self.h, self.fps = eng, handle, fps
        self.mult = ROWS_PER_S / fps
        self.left_rows = math.ceil(self.mult) * PAD_LEFT
        self.lookahead_rows = int(round(eng.lookahead_s * ROWS_PER_S))
        self.lock = threading.Lock()
        self.buf = np.zeros(0, np.float32)
        self.pending: list[np.ndarray] = []
        self.samples = 0
        self.ended = False
        self.next = 0
        self.win = None          # (r0, feats[rows,5,384] on device, frame limit)
        # frames that fit in one window after the left context
        self.max_win_frames = max(1, int((WIN_ROWS - CTX_ROWS - self.left_rows - CLIP_ROWS
                                          - self.lookahead_rows - 4) / self.mult))

    # ---------------------------------------------------------------- input
    def push(self, pcm16k):
        with self.lock:
            self.pending.append(np.asarray(pcm16k, dtype=np.int16))
            self.samples += len(pcm16k)

    def end(self):
        with self.lock:
            self.ended = True

    def _snapshot(self):
        with self.lock:
            if self.pending:
                add = np.concatenate(self.pending).astype(np.float32) / 32768.0
                self.buf = np.concatenate([self.buf, add])
                self.pending.clear()
            return self.buf, self.samples, self.ended

    # ---------------------------------------------------------------- frames
    def _row(self, i: int) -> int:
        """Index of frame i's first whisper row in MuseTalk's padded feature array."""
        return int(math.floor(i * self.mult))

    def _limit(self, samples: int, ended: bool) -> int:
        n_audio = frames_for(samples, self.fps)
        if ended:
            return n_audio
        avail = samples // SAMPLES_PER_ROW
        # frame i needs original rows up to _row(i) - left_rows + CLIP_ROWS, plus lookahead
        hi = avail - self.lookahead_rows - CLIP_ROWS + self.left_rows
        if hi < 0:
            return 0
        n = int(hi / self.mult) + 1
        while n > 0 and self._row(n - 1) > hi:
            n -= 1
        return min(n, n_audio)

    def next_batch(self):
        buf, samples, ended = self._snapshot()
        limit = self._limit(samples, ended)
        if self.next >= limit:
            return []
        if self.win is None or self.next >= self.win[2]:
            r0 = max(0, self._row(self.next) - self.left_rows - CTX_ROWS)
            win_limit = min(limit, self.next + self.max_win_frames)
            s0 = r0 * SAMPLES_PER_ROW
            audio = buf[s0:min(len(buf), s0 + WIN_ROWS * SAMPLES_PER_ROW)]
            self.win = (r0, self.e._encode(audio), win_limit)
        r0, feats, win_limit = self.win
        idx = list(range(self.next, min(self.next + self.e.batch, win_limit)))
        self.next = idx[-1] + 1
        starts = [self._row(i) - r0 for i in idx]   # == MuseTalk's audio_index when r0 == 0
        faces = self.e._render(feats, starts, self.left_rows, self.h.data["latent"])
        return list(zip(idx, faces))

    @property
    def finished(self):
        with self.lock:
            return self.ended and self.next >= frames_for(self.samples, self.fps)

    @property
    def total_frames(self):
        with self.lock:
            return frames_for(self.samples, self.fps) if self.ended else None


class MuseTalkEngine(Engine):
    name = "musetalk"

    def __init__(self, models_dir: str | os.PathLike, cache_dir: str | os.PathLike,
                 musetalk_dir: str | os.PathLike | None = None, batch: int = 8,
                 lookahead_s: float = 0.2, vram_gb: float | None = 12.0, **_cfg):
        self.models = Path(models_dir).expanduser()
        self.cache = Path(cache_dir).expanduser() / "musetalk"
        self.repo = Path(musetalk_dir).expanduser() if musetalk_dir else DEFAULT_MUSETALK_DIR
        self.batch = int(batch)
        self.lookahead_s = float(lookahead_s)
        self.vram_gb = vram_gb
        self.gpu_lock = threading.Lock()     # serialises GPU work across threads
        self._fp = None
        self.loaded = False

    # ---------------------------------------------------------------- load
    def load(self) -> None:
        if self.loaded:
            return
        if not (self.repo / "musetalk").is_dir():
            raise EngineUnavailable(f"MuseTalk code not found at {self.repo} (git submodule update --init)")
        for sub in ("musetalkV15/unet.pth", "musetalkV15/musetalk.json", "sd-vae", "whisper",
                    "face-parse-bisent/79999_iter.pth", "face-parse-bisent/resnet18-5c106cde.pth"):
            if not (self.models / sub).exists():
                raise EngineUnavailable(f"MuseTalk weights missing: {self.models / sub}")
        # face_alignment instead of mmpose/mmcv for avatar preparation (PR #429)
        os.environ.setdefault("MUSETALK_LANDMARK_BACKEND", "face_alignment")
        # the face-parse checkpoints are legacy pickles that torch 2.6+ refuses by default (#419)
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        t0 = time.time()
        import torch
        from transformers import AutoFeatureExtractor, WhisperModel
        from musetalk.models.unet import PositionalEncoding, UNet
        from musetalk.models.vae import VAE
        from musetalk.utils.device import get_device

        self.torch = torch
        dev = self.dev = get_device()
        if self.vram_gb and dev.type in ("xpu", "cuda"):
            mod = getattr(torch, dev.type)
            total = mod.get_device_properties(dev).total_memory
            frac = min(1.0, self.vram_gb * 2**30 / total)
            if hasattr(mod, "set_per_process_memory_fraction"):
                mod.set_per_process_memory_fraction(frac, dev)
        self.vae = VAE(model_path=str(self.models / "sd-vae"))
        self.vae.vae = self.vae.vae.half().to(dev)
        self.unet = UNet(unet_config=str(self.models / "musetalkV15/musetalk.json"),
                         model_path=str(self.models / "musetalkV15/unet.pth"), device=dev)
        self.unet.model = self.unet.model.half().to(dev).eval()
        self.pe = PositionalEncoding(d_model=384).half().to(dev)
        self.whisper = WhisperModel.from_pretrained(str(self.models / "whisper")).to(
            device=dev, dtype=torch.float16).eval()
        self.fe = AutoFeatureExtractor.from_pretrained(str(self.models / "whisper"))
        self.timesteps = torch.tensor([0], device=dev)
        self.load_s = round(time.time() - t0, 1)
        self._warmup()
        self.loaded = True
        LOG.info("musetalk loaded on %s (%s) in %.1f s, warm-up %.1f s, batch %d", dev, self.device_name(),
                 self.load_s, self.warm_s, self.batch)

    def device_name(self) -> str:
        t, d = self.torch, self.dev
        if d.type == "xpu":
            return t.xpu.get_device_name(d)
        if d.type == "cuda":
            return t.cuda.get_device_name(d)
        return d.type

    def _sync(self):
        if self.dev.type == "xpu":
            self.torch.xpu.synchronize()
        elif self.dev.type == "cuda":
            self.torch.cuda.synchronize()

    def _warmup(self):
        """Compile kernels for every batch size up front (XPU JITs each new shape)."""
        t0 = time.time()
        latent = self.torch.zeros(1, 8, 32, 32, dtype=self.torch.float16, device=self.dev)
        feats = self._encode(np.zeros(PCM_RATE, np.float32))
        for n in range(self.batch, 0, -1):
            self._render(feats, [0] * n, 4, latent)
        self._sync()
        self.warm_s = round(time.time() - t0, 1)

    # ---------------------------------------------------------------- GPU steps
    def _encode(self, audio: np.ndarray):
        """whisper-tiny hidden states for <= 30 s of audio: tensor [rows, 5, 384]."""
        torch = self.torch
        rows = len(audio) // SAMPLES_PER_ROW
        mel = self.fe(audio, sampling_rate=PCM_RATE, return_tensors="pt").input_features
        with self.gpu_lock, torch.no_grad():
            mel = mel.to(device=self.dev, dtype=torch.float16)
            hs = self.whisper.encoder(mel, output_hidden_states=True).hidden_states
            return torch.stack(hs, dim=2)[0, :rows]

    def _render(self, feats, starts: list[int], left_rows: int, latent) -> np.ndarray:
        """Faces [n, 256, 256, 3] BGR uint8 for the frames whose clips start at
        `starts` (indices into [left_rows zeros] + feats + [zeros])."""
        torch = self.torch
        with self.gpu_lock, torch.no_grad():
            pad_r = max(starts) + CLIP_ROWS - left_rows - feats.shape[0]
            parts = [feats.new_zeros((left_rows,) + feats.shape[1:]), feats]
            if pad_r > 0:
                parts.append(feats.new_zeros((pad_r,) + feats.shape[1:]))
            padded = torch.cat(parts, 0)
            clips = torch.stack([padded[s:s + CLIP_ROWS] for s in starts])      # n, 10, 5, 384
            audio = self.pe(clips.reshape(len(starts), CLIP_ROWS * clips.shape[2], clips.shape[3]))
            lat = latent.to(device=self.dev, dtype=torch.float16).expand(len(starts), -1, -1, -1).contiguous()
            pred = self.unet.model(lat, self.timesteps, encoder_hidden_states=audio).sample
            return self.vae.decode_latents(pred.to(dtype=self.vae.vae.dtype))

    # ---------------------------------------------------------------- avatars
    def _face_parsing(self):
        if self._fp is None:
            from musetalk.utils.face_parsing import FaceParsing
            fp_dir = self.models / "face-parse-bisent"

            class _FP(FaceParsing):
                def model_init(self, resnet_path=None, model_pth=None):
                    return super().model_init(str(fp_dir / "resnet18-5c106cde.pth"), str(fp_dir / "79999_iter.pth"))

            self._fp = _FP(left_cheek_width=90, right_cheek_width=90, device=self.torch.device("cpu"))
        return self._fp

    def prepare(self, avatar) -> AvatarHandle:
        self.load()
        torch = self.torch
        d = self.cache / avatar.id / avatar.cache_key(self.name)
        if not (d / "meta.json").is_file():
            self._prepare_into(avatar, d)
        meta = json.loads((d / "meta.json").read_text())
        frame = cv2.imread(str(d / "frame.png"))
        mask = np.load(d / "mask.npy")
        latent = torch.load(d / "latent.pt", map_location="cpu").to(device=self.dev, dtype=torch.float16)
        x1, y1, x2, y2 = meta["bbox"]
        rx, ry, rw, rh = meta["crop"]
        xs, ys = meta["crop_box"][:2]
        base = frame[ry:ry + rh, rx:rx + rw].astype(np.float32)
        m = (mask[ry - ys:ry - ys + rh, rx - xs:rx - xs + rw].astype(np.float32) / 255.0)[..., None]
        return AvatarHandle(avatar.id, tuple(meta["frame_size"]), tuple(meta["crop"]),
                            data={"latent": latent, "bbox": (x1, y1, x2, y2), "base": base, "mask": m,
                                  "cache": str(d)})

    def _prepare_into(self, avatar, d: Path) -> None:
        from musetalk.utils.blending import get_image_prepare_material
        from musetalk.utils.preprocessing import coord_placeholder, get_landmark_and_bbox

        t0 = time.time()
        tmp = d.with_name(d.name + f".tmp{os.getpid()}")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        frame = cv2.imread(str(avatar.portrait))
        if frame is None:
            raise ValueError(f"cannot read {avatar.portrait}")
        h, w = frame.shape[:2]
        s = avatar.max_side / max(h, w)
        if s < 1:
            frame = cv2.resize(frame, (int(w * s) // 2 * 2, int(h * s) // 2 * 2), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]
        cv2.imwrite(str(tmp / "frame.png"), frame)
        coords, _ = get_landmark_and_bbox([str(tmp / "frame.png")], avatar.bbox_shift)
        if not coords or coords[0] == coord_placeholder:
            raise ValueError(f"no face found in {avatar.portrait}")
        x1, y1, x2, y2 = (int(v) for v in coords[0])
        y2 = min(y2 + avatar.extra_margin, h)
        fp = self._face_parsing()
        fp.left_cheek_width = fp.right_cheek_width = avatar.cheek_width
        fp.cheek_mask = fp._create_cheek_mask(avatar.cheek_width, avatar.cheek_width)
        mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=fp, mode=avatar.parsing_mode)
        xs, ys = int(crop_box[0]), int(crop_box[1])
        # changed pixels: inside the face box, where the blend mask is non-zero
        sub = mask[y1 - ys:y2 - ys, x1 - xs:x2 - xs]
        rows = np.nonzero(sub.max(axis=1) > 0)[0]
        ry = y1 + (int(rows[0]) if len(rows) else 0)
        crop = (x1, ry, x2 - x1, y2 - ry)
        face = cv2.resize(frame[y1:y2, x1:x2], (FACE, FACE), interpolation=cv2.INTER_LANCZOS4)
        with self.gpu_lock, self.torch.no_grad():
            latent = self.vae.get_latents_for_unet(face).cpu()
        self.torch.save(latent, tmp / "latent.pt")
        np.save(tmp / "mask.npy", mask)
        meta = {"avatar": avatar.id, "portrait": str(avatar.portrait), "frame_size": [w, h],
                "bbox": [x1, y1, x2, y2], "crop_box": [int(v) for v in crop_box], "crop": list(crop),
                "prep_s": round(time.time() - t0, 1)}
        (tmp / "meta.json").write_text(json.dumps(meta, indent=1))
        shutil.rmtree(d, ignore_errors=True)
        tmp.rename(d)
        LOG.info("prepared avatar %s in %.1f s: frame %dx%d face %s crop %s", avatar.id, meta["prep_s"],
                 w, h, meta["bbox"], crop)

    # ---------------------------------------------------------------- per-utterance
    def open_stream(self, handle: AvatarHandle, fps: int) -> RenderStream:
        return _Stream(self, handle, fps)

    def compose(self, handle: AvatarHandle, raw: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = handle.data["bbox"]
        _, ry, _, _ = handle.crop
        face = cv2.resize(raw, (x2 - x1, y2 - y1))[ry - y1:]
        base, m = handle.data["base"], handle.data["mask"]
        return (base + (face.astype(np.float32) - base) * m + 0.5).astype(np.uint8)

    def info(self) -> dict:
        out = {"engine": self.name, "batch": self.batch}
        if self.loaded:
            out["device"] = str(self.dev)
            out["device_name"] = self.device_name()
            mod = getattr(self.torch, self.dev.type, None)
            if mod is not None and hasattr(mod, "memory_reserved"):
                out["vram_reserved_mib"] = round(mod.memory_reserved(self.dev) / 2**20)
        return out
