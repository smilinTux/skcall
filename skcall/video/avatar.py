"""The agent's picture: a static portrait (P5a) and a procedural idle loop.

Both are core VideoSources: frame(t) -> RGB uint8 (h, w, 3). No aiortc here.

The idle loop is a stand-in for a rendered loop asset (P5d, LivePortrait
offline): a slow "breathing" zoom and drift of the whole portrait, plus a
blink when the eye boxes of this portrait are known. It costs one
cv2.warpAffine per frame (about 1 to 3 ms at 560x720), computed on demand, so
nothing is pre-rendered and memory stays at a few copies of the portrait.

The lip-sync compositor (compositor.py) paints mouth crops onto the base
image BEFORE the breathing warp, so the crop moves with her head.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass

import numpy as np

LOG = logging.getLogger("skcall.video")

try:  # OpenCV is optional: without it the portrait still loads (PyAV) and stays static
    import cv2
except Exception:  # pragma: no cover - depends on the install
    cv2 = None

# Eye boxes (normalized x, y, w, h) for known portraits, keyed by the sha256
# of the image file, so a different picture never gets eyelids painted in
# the wrong place. A `portrait.json` next to the image ({"eyes": [[x,y,w,h],
# ...]}) or SKCALL_AVATAR_EYES ("x,y,w,h;x,y,w,h") override this.
KNOWN_EYES = {
    # ~/.skcapstone/agents/lumina/avatar/portrait.png (896x1152, glasses)
    "0b705f92fa3f3e26": [(0.3161, 0.3681, 0.0893, 0.0292), (0.5179, 0.3639, 0.0964, 0.0292)],
    # ~/.skcapstone/agents/jarvis/avatar/portrait.png (896x1152): Jarvis's portrait,
    # roll-corrected 10.9 deg; boxes from insightface 2d106 eye landmarks, padded
    "1597eede59685d3a": [(0.3103, 0.3585, 0.0848, 0.0304), (0.4900, 0.3542, 0.1016, 0.0295)],
}


def portrait_path(agent: str) -> str:
    return os.environ.get("SKCALL_AVATAR_IMAGE") or os.path.expanduser(
        f"~/.skcapstone/agents/{agent}/avatar/portrait.png")


def _even(n: float) -> int:
    return max(2, int(n) // 2 * 2)


def load_portrait(path: str, width: int = 560, height: int = 720) -> np.ndarray | None:
    """RGB uint8 (height, width, 3), center-cropped to the target aspect and
    resized; or None for every way it can fail (a missing face degrades to a
    voice-only call, never to a failed one: skchat's load_avatar_rgba rule)."""
    width, height = _even(width), _even(height)
    try:
        if cv2 is not None:
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("unreadable image")
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            import av
            with av.open(path) as c:
                img = next(c.decode(video=0)).to_ndarray(format="rgb24")
    except Exception as e:
        LOG.warning("avatar image unusable at %s (%r); no video", path, e)
        return None
    h, w = img.shape[:2]
    target = width / height
    if w / h > target:                       # too wide: crop the sides
        nw = int(round(h * target))
        x0 = (w - nw) // 2
        img = img[:, x0:x0 + nw]
    elif w / h < target:                     # too tall: crop the bottom (keep the face)
        nh = int(round(w / target))
        img = img[:nh]
    if img.shape[1] != width or img.shape[0] != height:
        if cv2 is not None:
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        else:
            ys = np.linspace(0, img.shape[0] - 1, height).astype(int)
            xs = np.linspace(0, img.shape[1] - 1, width).astype(int)
            img = img[ys][:, xs]
    return np.ascontiguousarray(img, dtype=np.uint8)


def eye_boxes(path: str) -> list[tuple[float, float, float, float]]:
    env = os.environ.get("SKCALL_AVATAR_EYES", "").strip()
    if env:
        try:
            return [tuple(float(v) for v in b.split(",")) for b in env.split(";") if b.strip()]
        except ValueError:
            LOG.warning("SKCALL_AVATAR_EYES is not 'x,y,w,h;x,y,w,h'; no blink")
            return []
    side = os.path.splitext(path)[0] + ".json"
    try:
        with open(side) as f:
            return [tuple(b) for b in json.load(f).get("eyes", [])]
    except FileNotFoundError:
        pass
    except Exception as e:
        LOG.warning("bad %s (%r); ignoring", side, e)
    try:
        with open(path, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        return []
    return list(KNOWN_EYES.get(digest, []))


class StaticPortrait:
    """P5a: the same frame forever. The track still resends it (WebRTC shows
    a stalled track as frozen), at a low fps."""

    def __init__(self, img: np.ndarray, fps: float = 5.0):
        self.img = img
        self.height, self.width = img.shape[:2]
        self.fps = fps

    def frame(self, t: float) -> np.ndarray:
        return self.img


def _blink_image(img: np.ndarray, box_px: tuple[int, int, int, int], closure: float) -> np.ndarray:
    """Paint an eyelid over one eye: rows from the top of the box down to
    `closure` of its height take the skin colour from just under the eye
    (smoothed per column), a little darker toward the lid edge, with a lash
    line, blended through a feathered ellipse. Crude, but a blink lasts
    4 frames."""
    x, y, w, h = box_px
    H, W = img.shape[:2]
    if w < 4 or h < 4 or x < 0 or y < 0 or x + w > W or y + h + 6 > H:
        return img
    out = img.copy()
    lid_h = int(round(closure * h))
    if lid_h <= 0:
        return out
    roi = out[y:y + h, x:x + w].astype(np.float32)
    below = img[y + h + 1:y + h + 6, x:x + w].astype(np.float32).mean(axis=0)
    k = max(3, w // 3) | 1
    below = np.stack([np.convolve(np.pad(below[:, c], k // 2, mode="edge"), np.ones(k) / k, "valid")
                      for c in range(3)], axis=1)
    new = roi.copy()
    new[:lid_h] = below[None] * np.linspace(0.92, 0.8, lid_h)[:, None, None]
    e = min(lid_h, h - 1)
    new[max(0, e - 2):e + 1] *= 0.45
    yy, xx = np.mgrid[0:h, 0:w]
    m = 1 - (((xx - (w - 1) / 2) / (w / 2)) ** 2 + ((yy - (h - 1) / 2) / (h / 2)) ** 2)
    m = np.clip(m * 3, 0, 1).astype(np.float32)
    if cv2 is not None:
        m = cv2.GaussianBlur(m, (5, 5), 0)
    roi = roi * (1 - m[..., None]) + new * m[..., None]
    out[y:y + h, x:x + w] = np.clip(roi, 0, 255).astype(np.uint8)
    return out


@dataclass
class IdleParams:
    breath_period_s: float = 4.6      # one slow breath
    zoom: float = 0.012               # +-1.2 % scale
    drift_px: float = 2.5             # vertical drift with the breath
    sway_period_s: float = 11.0       # slower sideways sway
    sway_px: float = 2.0
    blink_every_s: tuple[float, float] = (2.5, 6.0)
    blink_s: float = 0.16


class IdleLoop:
    """Procedural idle animation over a portrait. frame(t) is a pure function
    of t (blink times come from a seeded schedule), so it never drifts and is
    easy to test. render(base, t) lets the compositor warp a base it has
    already painted a mouth onto."""

    def __init__(self, img: np.ndarray, eyes_norm=(), fps: float = 25.0,
                 params: IdleParams | None = None, seed: int = 7):
        if cv2 is None:
            raise RuntimeError("the idle loop needs OpenCV (opencv-python-headless)")
        self.img = img
        self.height, self.width = img.shape[:2]
        self.fps = fps
        self.p = params or IdleParams()
        boxes = [(int(round(x * self.width)), int(round(y * self.height)),
                  int(round(w * self.width)), int(round(h * self.height))) for x, y, w, h in eyes_norm]
        # three lid states, painted once: half, nearly closed, closed
        self._lids: list[np.ndarray] = []
        if boxes:
            for c in (0.5, 0.85, 1.0):
                b = img
                for box in boxes:
                    b = _blink_image(b, box, c)
                self._lids.append(b)
        rng = np.random.default_rng(seed)
        lo, hi = self.p.blink_every_s
        self._blinks = np.cumsum(rng.uniform(lo, hi, 4096))   # blink start times, seconds
        self._t0: float | None = None
        self.can_blink = bool(self._lids)

    def base(self, t: float) -> np.ndarray:
        """The un-warped portrait for time t (with the eyelids if blinking)."""
        if not self._lids:
            return self.img
        if self._t0 is None:
            self._t0 = t
        rel = (t - self._t0) % self._blinks[-1]
        i = int(np.searchsorted(self._blinks, rel, side="right")) - 1
        if i < 0:
            return self.img
        ph = (rel - self._blinks[i]) / self.p.blink_s
        if ph >= 1.0:
            return self.img
        # close fast, open a bit slower: half, closed, nearly, half
        k = min(3, int(ph * 4))
        return self._lids[(0, 2, 1, 0)[k]]

    def matrix(self, t: float) -> np.ndarray:
        p = self.p
        br = math.sin(2 * math.pi * t / p.breath_period_s)
        s = 1.0 + p.zoom * (0.5 + 0.5 * br)
        dx = p.sway_px * math.sin(2 * math.pi * t / p.sway_period_s)
        dy = -p.drift_px * (0.5 + 0.5 * br)
        # zoom about a point a third of the way down (her face), so she breathes
        # toward the camera rather than sliding off the top of the frame
        cx, cy = self.width / 2, self.height / 3
        return np.array([[s, 0, (1 - s) * cx + dx], [0, s, (1 - s) * cy + dy]], np.float32)

    def render(self, base: np.ndarray, t: float) -> np.ndarray:
        return cv2.warpAffine(base, self.matrix(t), (self.width, self.height),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def frame(self, t: float) -> np.ndarray:
        return self.render(self.base(t), t)
