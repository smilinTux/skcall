"""Measure one-way Talk transport latency for talk spurts after silences of
different lengths: session A writes a 0.3 s tone into its SinkTrack every
`gap` seconds, session B (same process, same clock) reports when it first hears
each one. Both join the spike room's call silently.

Usage: scripts/run.sh scripts/transport_probe.py [--gaps 0.7,3,6] [--count 4]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time

import numpy as np

from skcall.cli import allowed_rooms, load_env
from skcall.core.audio import rms
from skcall.core.session import AudioTrack
from skcall.talk.backend import TalkSession


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", default="room0101")
    ap.add_argument("--gaps", default="0.7,3,6")
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--dither", type=float, default=0.0, help="noise amplitude in silence (int16 units)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    for noisy in ("aioice", "aiortc"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    load_env()
    A = TalkSession(a.room, silent=True, allowed_rooms=allowed_rooms(), name="A")
    B = TalkSession(a.room, silent=True, allowed_rooms=allowed_rooms(), name="B")
    loud: list[float] = []
    tone = (0.3 * 32767 * np.sin(2 * np.pi * 660 * np.arange(14400) / 48000)).astype(np.int16)
    res = {}
    try:
        await A.open()
        await B.open()
        await A.join_call()
        await B.join_call()

        async def hear():
            async for ev in B.events():
                if isinstance(ev, AudioTrack):
                    async for x in ev.frames:
                        if rms(x) > 1000:
                            loud.append(time.monotonic())
        h = asyncio.create_task(hear())
        if a.dither:
            # keep the stream non-silent between spurts
            async def dither():
                while True:
                    if A.audio_out.buffered_s < 0.2:
                        A.audio_out.write((np.random.standard_normal(4800) * a.dither).astype(np.int16))
                    await asyncio.sleep(0.05)
            dt = asyncio.create_task(dither())
        await asyncio.sleep(5)
        for gap in [float(g) for g in a.gaps.split(",")]:
            lat = []
            for _ in range(a.count):
                await asyncio.sleep(gap)
                fut = asyncio.get_running_loop().create_future()
                if a.dither:
                    A.audio_out.clear()
                A.audio_out.write(tone, on_start=lambda t, f=fut: f.done() or f.set_result(t))
                t0 = await fut
                await asyncio.sleep(0.3 + 2.5)
                first = next((x for x in loud if x > t0), None)
                lat.append(round((first - t0) * 1000) if first and first - t0 < 2.8 else None)
            ok = [x for x in lat if x is not None]
            res[f"gap_{gap}s"] = {"ms": lat, "median": round(statistics.median(ok)) if ok else None}
            logging.info("gap %.1fs -> %s", gap, lat)
        h.cancel()
        if a.dither:
            dt.cancel()
    finally:
        await A.leave()
        await B.leave()
    print("RESULT " + json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
