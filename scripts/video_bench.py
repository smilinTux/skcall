"""CPU cost of the avatar video on this host, no network.

Renders N seconds of 25 fps through the real pieces: the compositor (idle
loop, plus a mouth crop pasted on every frame when --lips), the YUV
conversion AvatarTrack does, and aiortc's VP8 encoder. Prints per-stage
milliseconds per frame and the share of one core at 25 fps.

  scripts/run.sh scripts/video_bench.py [--seconds 10] [--lips] [--static]
"""
from __future__ import annotations

import argparse
import fractions
import json
import os
import time

import numpy as np

from skcall.video import VideoConfig
from skcall.video.avatar import IdleLoop, StaticPortrait, eye_boxes, load_portrait
from skcall.video.compositor import Compositor, _Play
from skcall.video.fake_lipsync import crop_value, flat_jpeg
from skcall.video.lipsync import Crop


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--lips", action="store_true", help="paste a crop on every frame")
    ap.add_argument("--static", action="store_true", help="static portrait at 5 fps")
    ap.add_argument("--agent", default=os.environ.get("SKAGENT", "lumina"))
    a = ap.parse_args()
    import cv2
    from aiortc.codecs import get_encoder
    from aiortc.rtcrtpparameters import RTCRtpCodecParameters
    from av import VideoFrame

    cfg = VideoConfig.from_env(a.agent)
    img = load_portrait(cfg.image, cfg.width, cfg.height)
    fps = 5.0 if a.static else cfg.fps
    if a.static:
        src = StaticPortrait(img, fps)
    else:
        src = Compositor(img, IdleLoop(img, eye_boxes(cfg.image), fps=fps), fps=fps)
        src.set_crop(Crop(155, 300, 250, 242), (560, 720))
        if a.lips:
            n = int(a.seconds * fps) + 5
            frames = {i: flat_jpeg(256, 256, crop_value(i)) for i in range(n)}
            src.schedule(_Play("bench", frames, n / fps, t_start=0.0))
    enc = get_encoder(RTCRtpCodecParameters(mimeType="video/VP8", clockRate=90000))
    n = int(a.seconds * fps)
    t_comp = t_yuv = t_enc = 0.0
    nbytes = 0
    last = None
    cpu0, wall0 = time.process_time(), time.perf_counter()
    for i in range(n):
        t = i / fps
        s = time.perf_counter()
        rgb = src.frame(t)
        s1 = time.perf_counter()
        if last is not None and last[0] is rgb:
            yuv = last[1]
        else:
            yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420)
            last = (rgb, yuv)
        f = VideoFrame.from_ndarray(yuv, format="yuv420p")
        f.pts = int(t * 90000)
        f.time_base = fractions.Fraction(1, 90000)
        s2 = time.perf_counter()
        payloads, _ = enc.encode(f)
        s3 = time.perf_counter()
        nbytes += sum(len(p) for p in payloads)
        t_comp += s1 - s
        t_yuv += s2 - s1
        t_enc += s3 - s2
    cpu = time.process_time() - cpu0
    wall = time.perf_counter() - wall0
    res = {"mode": "static" if a.static else ("lips" if a.lips else "idle"), "fps": fps, "frames": n,
           "compose_ms": round(t_comp / n * 1000, 2), "yuv_ms": round(t_yuv / n * 1000, 2),
           "vp8_ms": round(t_enc / n * 1000, 2),
           "cpu_s_per_video_s": round(cpu / a.seconds, 3),
           "core_share_at_realtime_pct": round(cpu / a.seconds * 100, 1),
           "kbps": round(nbytes * 8 / a.seconds / 1000), "wall_s": round(wall, 2)}
    if hasattr(src, "stats"):
        res["compositor"] = src.stats.summary()
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
