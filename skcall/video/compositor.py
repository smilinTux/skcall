"""P5c: the lip-sync compositor and the audio hold that keeps lips and voice together.

Two pieces, both backend-agnostic:

Compositor (a core VideoSource, 25 fps). Every frame is the idle loop at
time t; when an utterance is playing, the mouth crop for the current audio
position is pasted at the service's crop box through a feathered mask
(lower part of the box only, so blinks keep working), then the breathing
warp moves the whole picture. "Current audio position" comes from the
AudioSink's on_start timestamp for that utterance: frame index
floor((t - t_start - av_offset) * fps). A missing crop reuses the previous
one for up to `hold_frames`, then shows the idle mouth: video never waits,
never freezes, and never delays the audio. If the idle loop itself fails,
the compositor falls back to the static portrait for the rest of the call.

LipsyncSink (wraps the backend's AudioSink). Each write() goes to the
lip-sync service at once (so sentence N+1 renders while N plays) and is
queued in order. When she starts speaking from silence the audio is held
until the first frames are in, up to `hold_s` (default 0.3 s); later
sentences play straight away (their frames render ahead of them). clear()
(barge-in) drops queued audio, cancels the service-side utterances and
flushes the crops, so the mouth closes the moment she stops. Clips heard
before (fillers) are served from a frame cache with no service round trip
and no hold; prerender() fills it at session start.
"""
from __future__ import annotations

import asyncio
import collections
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from skcall.core.audio import resample
from skcall.video.avatar import IdleLoop, cv2
from skcall.video.lipsync import PCM_RATE, Crop, LipsyncClient, LipsyncUtterance

LOG = logging.getLogger("skcall.video")


def feather_mask(w: int, h: int, top_frac: float = 0.35, feather_frac: float = 0.14) -> np.ndarray:
    """float32 (h, w) in 0..1: 1 in the lower middle of the box, smoothly 0 at
    the left, right and bottom edges and above `top_frac` of the height.
    The service paints only the mouth; the top of its crop is the static
    portrait's eyes, which would overwrite a blink if pasted."""
    f = max(2.0, feather_frac * min(w, h))
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    mx = np.clip(np.minimum(xs + 0.5, w - xs - 0.5) / f, 0, 1)
    my_bottom = np.clip((h - ys - 0.5) / f, 0, 1)
    top = top_frac * h
    my_top = np.clip((ys - top) / f + 0.5, 0, 1)
    m = np.minimum(mx[None, :], (my_bottom * my_top)[:, None])
    return (m * m * (3 - 2 * m)).astype(np.float32)          # smoothstep


@dataclass
class _Play:
    """One utterance on the playout timeline."""
    key: str
    frames: dict[int, bytes]
    duration_s: float
    t_start: float | None = None          # monotonic time its first sample hit the encoder
    on_end: Callable[[], None] | None = None
    last_idx: int = -1


@dataclass
class CompositorStats:
    frames: int = 0
    lips: int = 0          # frames with a fresh crop
    held: int = 0          # previous crop reused (late frame)
    late: int = 0          # speaking but no crop: idle mouth shown
    errors: int = 0
    render_ms: collections.deque = field(default_factory=lambda: collections.deque(maxlen=500))

    def summary(self) -> dict:
        r = sorted(self.render_ms)
        return {"frames": self.frames, "lips": self.lips, "held": self.held, "late": self.late,
                "errors": self.errors,
                "render_ms_p50": round(r[len(r) // 2], 2) if r else None,
                "render_ms_p95": round(r[int(len(r) * 0.95)], 2) if r else None}


class Compositor:
    def __init__(self, portrait: np.ndarray, idle: IdleLoop | None = None, fps: float = 25.0,
                 av_offset_s: float = 0.0, hold_frames: int = 2, mask_top: float = 0.35,
                 clock: Callable[[], float] = time.monotonic):
        self.portrait = portrait
        self.height, self.width = portrait.shape[:2]
        self.idle = idle
        self.fps = fps
        self.av_offset_s = av_offset_s
        self.hold_frames = hold_frames
        self.mask_top = mask_top
        self.clock = clock
        self.box: tuple[int, int, int, int] | None = None   # x, y, w, h in our frame
        self._mask: np.ndarray | None = None
        self._plays: list[_Play] = []
        self._decoded: tuple[str, int, np.ndarray] | None = None
        self.stats = CompositorStats()

    # ------------------------------------------------------------ setup
    def set_crop(self, crop: Crop, frame_size: tuple[int, int] | None) -> None:
        """Map the service's crop box (in its frame_size) onto our frame."""
        fw, fh = frame_size or (self.width, self.height)
        sx, sy = self.width / fw, self.height / fh
        x, y = int(round(crop.x * sx)), int(round(crop.y * sy))
        w, h = int(round(crop.w * sx)), int(round(crop.h * sy))
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.width, x + w), min(self.height, y + h)
        if x1 - x0 < 8 or y1 - y0 < 8:
            LOG.warning("crop box %s outside the %dx%d frame; lips off", crop, self.width, self.height)
            self.box = None
            return
        if (x0, y0, x1, y1) != (x, y, x + w, y + h):
            LOG.warning("crop box %s clipped to the frame", crop)
        self.box = (x0, y0, x1 - x0, y1 - y0)
        self._decoded = None                  # a reconnect may change the box size
        self._mask = feather_mask(x1 - x0, y1 - y0, self.mask_top)[..., None]
        self._full = (x, y, w, h, x0 - x, y0 - y)

    # ------------------------------------------------------------ timeline
    def schedule(self, play: _Play) -> None:
        self._plays.append(play)

    def flush(self) -> None:
        """Barge-in: back to the idle mouth now."""
        plays, self._plays = self._plays, []
        self._decoded = None
        for p in plays:
            if p.on_end:
                p.on_end()

    @property
    def speaking(self) -> bool:
        return bool(self._plays)

    def _crop_for(self, t: float) -> tuple[_Play, int, bytes] | None:
        ta = t - self.av_offset_s
        keep: list[_Play] = []
        hit = None
        for p in self._plays:
            if p.t_start is not None and ta >= p.t_start + p.duration_s + 0.5:
                if p.on_end:
                    p.on_end()
                continue                      # played out
            keep.append(p)
            if hit is not None or p.t_start is None or ta < p.t_start or ta >= p.t_start + p.duration_s:
                continue
            idx = int((ta - p.t_start) * self.fps)
            jpeg = p.frames.get(idx)
            if jpeg is not None:
                p.last_idx = idx
                self.stats.lips += 1
                hit = (p, idx, jpeg)
            elif p.last_idx >= 0 and idx - p.last_idx <= self.hold_frames and p.last_idx in p.frames:
                self.stats.held += 1
                hit = (p, p.last_idx, p.frames[p.last_idx])
            else:
                self.stats.late += 1
        self._plays = keep
        return hit

    # ------------------------------------------------------------ render
    def _decode(self, key: str, idx: int, jpeg: bytes) -> np.ndarray | None:
        if self._decoded is not None and self._decoded[:2] == (key, idx):
            return self._decoded[2]
        bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        x, y, w, h, ox, oy = self._full
        if bgr.shape[1] != w or bgr.shape[0] != h:
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        bw, bh = self.box[2], self.box[3]
        rgb = cv2.cvtColor(bgr[oy:oy + bh, ox:ox + bw], cv2.COLOR_BGR2RGB)
        self._decoded = (key, idx, rgb)
        return rgb

    def _paste(self, base: np.ndarray, crop: np.ndarray) -> np.ndarray:
        x, y, w, h = self.box
        out = base.copy()
        roi = out[y:y + h, x:x + w].astype(np.float32)
        m = self._mask
        out[y:y + h, x:x + w] = (roi + (crop.astype(np.float32) - roi) * m).astype(np.uint8)
        return out

    def frame(self, t: float) -> np.ndarray:
        t0 = time.perf_counter()
        self.stats.frames += 1
        try:
            base = self.idle.base(t) if self.idle is not None else self.portrait
            if self._plays:
                hit = self._crop_for(t)
                if hit is not None and self.box is not None:
                    p, idx, jpeg = hit
                    crop = self._decode(p.key, idx, jpeg)
                    if crop is not None:
                        base = self._paste(base, crop)
            out = self.idle.render(base, t) if self.idle is not None else base
        except Exception:
            self.stats.errors += 1
            if self.idle is not None:
                LOG.warning("idle loop failed; falling back to the static portrait", exc_info=True)
                self.idle = None
            else:
                LOG.debug("compositor frame failed", exc_info=True)
            out = self.portrait
        self.stats.render_ms.append((time.perf_counter() - t0) * 1000)
        return out


def _digest(pcm: np.ndarray) -> str:
    return hashlib.blake2b(pcm.tobytes(), digest_size=10).hexdigest()


@dataclass(eq=False)
class _Item:
    pcm: np.ndarray
    on_start: Callable | None
    play: _Play
    utt: LipsyncUtterance | None = None
    digest: str = ""


class LipsyncSink:
    """AudioSink wrapper: lip-sync each chunk, hold the first sentence briefly,
    schedule the crops on the compositor at the real playout time."""

    def __init__(self, inner, compositor: Compositor, client: LipsyncClient | None,
                 hold_s: float = 0.3, lead_frames: int = 3, cache_max_s: float = 4.0,
                 cache_size: int = 64):
        self.inner = inner
        self.sample_rate = inner.sample_rate
        self.comp = compositor
        self.client = client
        self.hold_s = hold_s
        self.lead_frames = lead_frames
        self.cache_max_s = cache_max_s
        self.cache_size = cache_size
        self.cache: collections.OrderedDict[str, dict[int, bytes]] = collections.OrderedDict()
        self._pending: collections.deque[_Item] = collections.deque()
        self._pending_samples = 0
        self._worker: asyncio.Task | None = None
        self._wake = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._gen = 0
        self._live: set[_Item] = set()           # submitted to the service, not yet played out
        self.holds: list[float] = []             # seconds each held start waited
        self.stats = collections.Counter()

    # ------------------------------------------------------------ AudioSink
    def write(self, pcm: np.ndarray, on_start=None) -> None:
        if pcm.size == 0:
            return
        dur = len(pcm) / self.sample_rate
        item = _Item(pcm, on_start, _Play(key="", frames={}, duration_s=dur))
        if self.comp.box is not None:
            item.digest = _digest(pcm)
            cached = self.cache.get(item.digest)
            if cached is not None:
                self.cache.move_to_end(item.digest)
                item.play.frames = cached
                item.play.key = "c:" + item.digest
                self.stats["cached"] += 1
            elif self.client is not None:
                item.utt = self.client.submit(resample(pcm, self.sample_rate, PCM_RATE))
                if item.utt is not None:
                    item.play.frames = item.utt.frames      # live dict, fills as frames arrive
                    item.play.key = item.utt.id
                    self._live.add(item)
                    self.stats["lipsync"] += 1
                else:
                    self.stats["no_service"] += 1
        item.play.on_end = lambda it=item: self._played(it)
        self._pending.append(item)
        self._pending_samples += len(pcm)
        self._idle.clear()
        self._wake.set()
        if self._worker is None or self._worker.done():
            self._worker = asyncio.ensure_future(self._run())

    def clear(self) -> None:
        self._gen += 1
        self._pending.clear()
        self._pending_samples = 0
        for it in list(self._live):
            if it.utt is not None and self.client is not None and not it.utt.done.is_set():
                self.client.cancel(it.utt)
        self._live.clear()
        self.comp.flush()
        self.inner.clear()
        self._idle.set()
        self._wake.set()

    @property
    def buffered_s(self) -> float:
        return self._pending_samples / self.sample_rate + self.inner.buffered_s

    async def drained(self) -> None:
        while True:
            await self._idle.wait()
            await self.inner.drained()
            if self._idle.is_set() and not self._pending:
                return

    # ------------------------------------------------------------ internals
    async def _run(self) -> None:
        while True:
            if not self._pending:
                self._idle.set()
                self._wake.clear()
                await self._wake.wait()
                continue
            item = self._pending[0]
            gen = self._gen
            if item.utt is not None and self.inner.buffered_s <= 0:
                # starting from silence: give the first frames a moment
                t0 = time.monotonic()
                ok = await item.utt.wait_ready(self.lead_frames, self.hold_s)
                waited = time.monotonic() - t0
                self.holds.append(waited)
                self.stats["held_ok" if ok else "held_timeout"] += 1
                if gen != self._gen:
                    continue                     # barge-in while holding
            if not self._pending or self._pending[0] is not item:
                continue
            self._pending.popleft()
            self._pending_samples -= len(item.pcm)
            self.comp.schedule(item.play)
            user_cb = item.on_start

            def _started(t: float, it=item, cb=user_cb) -> None:
                it.play.t_start = t
                if cb is not None:
                    cb(t)
            self.inner.write(item.pcm, on_start=_started)

    def _played(self, item: _Item) -> None:
        self._live.discard(item)
        u = item.utt
        if u is None or self.client is None:
            return
        self.client.forget(u)
        if (u.done.is_set() and not u.cancelled and item.digest and u.frames
                and len(u.frames) >= u.n_frames and item.play.duration_s <= self.cache_max_s):
            self._remember(item.digest, dict(u.frames))

    def _remember(self, digest: str, frames: dict[int, bytes]) -> None:
        self.cache[digest] = frames
        self.cache.move_to_end(digest)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    async def prerender(self, clips: list[np.ndarray]) -> int:
        """Lip-sync short clips (fillers) once, so they play with lips and no hold."""
        if self.client is None or self.comp.box is None:
            return 0
        n = 0
        for pcm in clips:
            d = _digest(pcm)
            if d in self.cache or not self.client.available:
                continue
            frames = await self.client.render_all(resample(pcm, self.sample_rate, PCM_RATE))
            if frames:
                self._remember(d, frames)
                n += 1
        LOG.info("pre-rendered lips for %d filler clips", n)
        return n
