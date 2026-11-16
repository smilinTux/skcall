#!/usr/bin/env python3
"""Stream a WAV to sklipsync and save the returned mouth crops as a video.

  SKCALL_LIPSYNC_TOKEN=... python lipsync_client.py --url ws://lipsync.example.org:18850/v1/lipsync \
      --wav clip.wav --base portrait.png --out lipsync.mp4

The token comes from SKCALL_LIPSYNC_TOKEN or --token-file (a KEY=VALUE env
file). Prints one JSON line per utterance: time to first frame (from the
first PCM byte sent), frames per second, frame count, JPEG sizes. With
--base, crops are pasted on the portrait (resized to the server's
frame_size); without it the video shows the crops alone. Needs aiohttp,
numpy, opencv and ffmpeg.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

import aiohttp
import cv2
import numpy as np

RATE = 16000


def read_token(path: str | None) -> str:
    if path:
        for line in Path(path).expanduser().read_text().splitlines():
            if line.strip().startswith("SKCALL_LIPSYNC_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("SKCALL_LIPSYNC_TOKEN", "")


def load_pcm(path: str) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "s16le", "-ac", "1", "-ar", str(RATE), "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype="<i2")


def unpack(data: bytes):
    idx, n = struct.unpack_from(">IH", data, 0)
    return idx, data[6:6 + n].decode("ascii"), data[6 + n:]


async def utterance(ws, pcm: np.ndarray, chunk: int, realtime: bool, cancel_after: float | None):
    uid = str(uuid.uuid4())
    frames: dict[int, bytes] = {}
    stats = {"id": uid}
    t_first_pcm = time.monotonic()
    t_first = t_last = None
    await ws.send_json({"type": "utterance", "id": uid})

    async def sender():
        t0 = time.monotonic()
        for i in range(0, len(pcm), chunk):
            await ws.send_bytes(pcm[i:i + chunk].tobytes())
            if realtime:
                await asyncio.sleep(max(0.0, t0 + (i + chunk) / RATE - time.monotonic()))
        await ws.send_json({"type": "end", "id": uid})

    send_task = asyncio.create_task(sender())
    cancelled = False
    done = None
    while done is None:
        msg = await asyncio.wait_for(ws.receive(), 30)
        if msg.type == aiohttp.WSMsgType.BINARY:
            idx, u, jpeg = unpack(msg.data)
            if u != uid:
                continue
            now = time.monotonic()
            t_first = t_first or now
            t_last = now
            frames[idx] = jpeg
            if cancel_after is not None and not cancelled and now - t_first >= cancel_after:
                cancelled = True
                send_task.cancel()
                await ws.send_json({"type": "cancel", "id": uid})
                stats["cancel_sent_after_frames"] = len(frames)
        elif msg.type == aiohttp.WSMsgType.TEXT:
            m = json.loads(msg.data)
            if m.get("type") == "done" and m.get("id") == uid:
                done = m
            elif m.get("type") == "error":
                raise SystemExit(f"server error: {m}")
            elif m.get("type") == "ping":
                await ws.send_json({"type": "pong"})
        else:
            raise SystemExit(f"connection closed: {msg.type}")
    if not send_task.done():
        await asyncio.gather(send_task, return_exceptions=True)
    audio_s = len(pcm) / RATE
    sizes = [len(j) for j in frames.values()]
    stats.update({
        "audio_s": round(audio_s, 2), "frames": len(frames), "done_frames": done.get("frames"),
        "cancelled": bool(done.get("cancelled")),
        "ttff_ms": round((t_first - t_first_pcm) * 1000) if t_first else None,
        "fps": round(len(frames) / (t_last - t_first_pcm), 1) if t_last and t_last > t_first_pcm else None,
        "stream_fps": round((len(frames) - 1) / (t_last - t_first), 1) if t_last and t_last > t_first else None,
        "missing": sorted(set(range(done.get("frames") or 0)) - set(frames))[:10] if not done.get("cancelled") else None,
        "jpeg_kb_avg": round(sum(sizes) / len(sizes) / 1024, 1) if sizes else None,
    })
    if t_last:
        stats["mbps"] = round(sum(sizes) * 8 / 1e6 / max(audio_s, 1e-6), 2)
    return frames, stats


def write_video(out: str, frames: dict[int, bytes], n: int, ready: dict, base_path: str | None, fps: int, wav: str):
    fw, fh = ready["frame_size"]
    c = ready["crop"]
    if base_path:
        base = cv2.resize(cv2.imread(base_path), (fw, fh), interpolation=cv2.INTER_AREA)
        size = (fw, fh)
    else:
        base = None
        size = (c["w"], c["h"])
    with tempfile.TemporaryDirectory() as td:
        tmp = os.path.join(td, "v.mp4")
        vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        last = None
        for i in range(n):
            j = frames.get(i)
            crop = cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR) if j else last
            if crop is not None and crop.shape[:2] != (c["h"], c["w"]):
                crop = cv2.resize(crop, (c["w"], c["h"]))
            last = crop
            if base is None:
                img = crop if crop is not None else np.zeros((c["h"], c["w"], 3), np.uint8)
            else:
                img = base.copy()
                if crop is not None:
                    img[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]] = crop
            vw.write(img)
        vw.release()
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", tmp, "-i", wav, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "20", "-c:a", "aac", "-shortest", out], check=True)


async def main_async(a) -> int:
    token = read_token(a.token_file)
    if not token:
        print("no token: set SKCALL_LIPSYNC_TOKEN or pass --token-file", file=sys.stderr)
        return 2
    pcm = load_pcm(a.wav)
    async with aiohttp.ClientSession() as http:
        t0 = time.monotonic()
        ws = await http.ws_connect(a.url, headers={"Authorization": f"Bearer {token}"}, max_msg_size=8 << 20)
        await ws.send_json({"type": "hello", "avatar": a.avatar, "fps": a.fps, "engine": a.engine})
        ready = json.loads((await asyncio.wait_for(ws.receive(), a.ready_timeout)).data)
        if ready.get("type") != "ready":
            print(f"no ready: {ready}", file=sys.stderr)
            return 1
        print(json.dumps({"ready": ready, "connect_ms": round((time.monotonic() - t0) * 1000)}))
        chunk = int(RATE * a.chunk_ms / 1000)
        frames = {}
        for r in range(a.repeat):
            frames, stats = await utterance(ws, pcm, chunk, a.realtime, a.cancel_after)
            stats["run"] = r
            print(json.dumps(stats))
            if a.gap:
                await asyncio.sleep(a.gap)
        await ws.send_json({"type": "ping"})
        pong = json.loads((await asyncio.wait_for(ws.receive(), 5)).data)
        assert pong.get("type") == "pong", pong
        await ws.close()
    if a.out and frames:
        n = max(frames) + 1
        write_video(a.out, frames, n, ready, a.base, ready["fps"], a.wav)
        print(json.dumps({"video": a.out, "frames": n}))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("SKCALL_LIPSYNC_URL", "ws://127.0.0.1:18850/v1/lipsync"))
    ap.add_argument("--token-file", help="env file with SKCALL_LIPSYNC_TOKEN=...")
    ap.add_argument("--wav", required=True, help="any audio file ffmpeg can read")
    ap.add_argument("--avatar", default="lumina")
    ap.add_argument("--engine", default="musetalk")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--chunk-ms", type=int, default=200)
    ap.add_argument("--realtime", action="store_true", help="pace the audio at real time (default: as fast as possible)")
    ap.add_argument("--cancel-after", type=float, help="send cancel this many seconds after the first frame")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--gap", type=float, default=0.0, help="seconds between repeats")
    ap.add_argument("--ready-timeout", type=float, default=60.0)
    ap.add_argument("--base", help="portrait to paste crops on (resized to frame_size)")
    ap.add_argument("--out", help="output video (.mp4), from the last utterance")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    sys.exit(main())
