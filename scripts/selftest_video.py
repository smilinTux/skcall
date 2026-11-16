"""Silent loopback check of the avatar video through a real Talk call.

Two sessions of the agent user join the spike room's call SILENTLY (nobody is
rung): `agent` publishes audio + the avatar video; `viewer` subscribes with
video on and measures what arrives through Janus: frame size, frame rate,
and (with --lips) the end-to-end A/V offset.

--lips fake   a local fake lip-sync service (127.0.0.1) returns bright crops;
              the agent plays tone bursts through the LipsyncSink and the
              viewer times the first loud audio against the first frame
              whose mouth region turns bright: video minus audio arrival.
--lips real   the same with SKCALL_LIPSYNC_URL / SKCALL_LIPSYNC_TOKEN and
              real speech (one F5 sentence); reports frames lip-synced.

  scripts/run.sh scripts/selftest_video.py [--mode static|animated] [--lips fake|real] [--seconds 15]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import resource
import statistics
import sys
import time

import numpy as np

from skcall.cli import allowed_rooms, load_env
from skcall.core.audio import CALL_RATE, rms
from skcall.core.session import AudioTrack, VideoTrack
from skcall.talk.backend import TalkSession
from skcall.video import Avatar, VideoConfig
from skcall.video.fake_lipsync import FakeLipsyncServer

LOG = logging.getLogger("selftest_video")


def cpu_s() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


class Viewer:
    def __init__(self):
        self.vt: list[float] = []
        self.sizes: set = set()
        self.bright_t: list[float] = []
        self.loud_t: list[float] = []
        self.box = None
        self.base_level: float | None = None
        self.levels: list[tuple[float, float]] = []

    async def video(self, tr: VideoTrack):
        async for t, img in tr.frames:
            self.vt.append(t)
            self.sizes.add((img.shape[1], img.shape[0]))
            if self.box is not None:
                x, y, w, h = self.box
                sx, sy = img.shape[1] / 560, img.shape[0] / 720
                reg = img[int((y + h * 0.65) * sy):int((y + h * 0.8) * sy),
                          int((x + w / 3) * sx):int((x + 2 * w / 3) * sx)]
                lvl = float(reg.mean())
                self.levels.append((t, lvl))
                if self.base_level is None and len(self.levels) > 10:
                    self.base_level = statistics.median(v for _, v in self.levels)
                if self.base_level is not None and lvl > self.base_level + 60:
                    self.bright_t.append(t)

    async def audio(self, tr: AudioTrack):
        async for x in tr.frames:
            if rms(x) > 1500:
                self.loud_t.append(time.monotonic())

    def fps(self, t0: float, t1: float) -> float:
        n = sum(1 for t in self.vt if t0 <= t < t1)
        return n / (t1 - t0) if t1 > t0 else 0.0

    @staticmethod
    def first_after(ts: list[float], t: float) -> float | None:
        return next((x for x in ts if x > t), None)


def burst(seconds: float, f: float = 660.0) -> np.ndarray:
    n = int(CALL_RATE * seconds)
    return (np.sin(2 * np.pi * f * np.arange(n) / CALL_RATE) * 12000).astype(np.int16)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", default="room0101")
    ap.add_argument("--mode", choices=("static", "animated"), default="animated")
    ap.add_argument("--lips", choices=("none", "fake", "real"), default="none")
    ap.add_argument("--seconds", type=float, default=15)
    ap.add_argument("--bursts", type=int, default=5)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    for noisy in ("aioice", "aiortc", "httpx", "httpcore", "aiohttp.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    load_env()
    rooms = allowed_rooms()
    res: dict = {"mode": a.mode, "lips": a.lips}

    fake = None
    cfg = VideoConfig.from_env(os.environ.get("SKAGENT", "lumina"))
    cfg.mode = a.mode
    if a.lips == "fake":
        fake = await FakeLipsyncServer(value_fn=lambda i: 250, crop_size=(256, 256), ttff=0.1,
                                       rtf=2.0).start()
        cfg.lipsync_url, cfg.lipsync_token = fake.url, fake.token
    elif a.lips == "real":
        if not cfg.lipsync_url:
            print("RESULT " + json.dumps({"skipped": "SKCALL_LIPSYNC_URL not set"}))
            return 0
    else:
        cfg.lipsync_url = ""
    avatar = Avatar.build(cfg)
    if avatar is None:
        print("RESULT " + json.dumps({"error": "no avatar (video off or portrait unusable)"}))
        return 1
    avatar.start()
    agent = TalkSession(a.room, silent=True, allowed_rooms=rooms, name="agent", video=avatar.source)
    viewer = TalkSession(a.room, silent=True, allowed_rooms=rooms, name="viewer", nick="Viewer",
                         receive_video=True)
    sink = avatar.wrap_sink(agent.audio_out)
    v = Viewer()
    tasks: list[asyncio.Task] = []
    try:
        await agent.open()
        await viewer.open()
        await agent.join_call()
        await viewer.join_call()
        res["agent_publishes_video"] = agent.video_out is not None

        async def events():
            async for ev in viewer.events():
                if isinstance(ev, VideoTrack):
                    tasks.append(asyncio.create_task(v.video(ev)))
                elif isinstance(ev, AudioTrack):
                    tasks.append(asyncio.create_task(v.audio(ev)))
        tasks.append(asyncio.create_task(events()))
        t_join = time.monotonic()
        while not v.vt and time.monotonic() - t_join < 20:
            await asyncio.sleep(0.05)
        if not v.vt:
            res["error"] = "no video frames arrived within 20 s"
            return 1
        res["first_video_after_join_ms"] = round((v.vt[0] - t_join) * 1000)
        cpu0, w0 = cpu_s(), time.monotonic()
        if a.lips != "none" and avatar.client is not None:
            if not await avatar.client.wait_ready(10):
                res["lipsync_error"] = "service not ready"
            else:
                comp = avatar.compositor
                v.box = comp.box
                await asyncio.sleep(1.0)                 # baseline mouth level
                offsets = []
                if a.lips == "fake":
                    for k in range(a.bursts):
                        started = asyncio.get_running_loop().create_future()
                        sink.write(burst(0.6), on_start=lambda t, f=started: f.done() or f.set_result(t))
                        t_play = await started
                        await sink.drained()
                        await asyncio.sleep(1.4)
                        ta = v.first_after(v.loud_t, t_play - 0.05)
                        tv = v.first_after(v.bright_t, t_play - 0.05)
                        if ta and tv:
                            offsets.append(round((tv - ta) * 1000))
                        LOG.info("burst %d: audio %s ms, video %s ms after playout", k,
                                 round((ta - t_play) * 1000) if ta else None,
                                 round((tv - t_play) * 1000) if tv else None)
                    res["av_offset_ms"] = offsets
                    if offsets:
                        res["av_offset_median_ms"] = round(statistics.median(offsets))
                else:
                    from skcall.speech import F5TTS
                    tts = F5TTS()
                    pcm = await tts.synth("Hi, this is a quick silent test of my lip sync on the call.")
                    await tts.aclose()
                    sink.write(pcm)
                    await sink.drained()
                    await asyncio.sleep(1.0)
        rest = a.seconds - (time.monotonic() - w0)
        if rest > 0:
            await asyncio.sleep(rest)
        cpu1, w1 = cpu_s(), time.monotonic()
        t_end = v.vt[-1]
        res["video_size"] = sorted(v.sizes)
        res["video_frames"] = len(v.vt)
        res["video_fps_received"] = round(v.fps(t_end - min(10, t_end - v.vt[0]), t_end), 2)
        gaps = np.diff(v.vt[-250:]) * 1000 if len(v.vt) > 2 else np.array([0])
        res["frame_gap_ms_p50_p95_max"] = [round(float(np.percentile(gaps, q)), 1) for q in (50, 95, 100)]
        res["agent_frames_sent"] = agent.video_out.frames_sent if agent.video_out else 0
        res["process_cpu_core_share_pct"] = round((cpu1 - cpu0) / (w1 - w0) * 100, 1)
        res["video"] = avatar.summary()
    finally:
        for t in tasks:
            t.cancel()
        await agent.leave()
        await viewer.leave()
        await avatar.aclose()
        if fake is not None:
            await fake.stop()
    print("RESULT " + json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
