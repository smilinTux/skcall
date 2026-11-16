"""Fake engine: no model, flat crops whose grey level encodes the frame index
(same scheme as skcall/video/fake_lipsync.py). For tests and local dev."""
from __future__ import annotations

import threading
import time

import numpy as np

from ..protocol import PCM_RATE, frames_for
from . import AvatarHandle, Engine, RenderStream


def crop_value(i: int) -> int:
    return 40 + (i * 23) % 180


class _FakeStream(RenderStream):
    def __init__(self, fps: int, batch: int, delay_s: float):
        self.fps, self.batch, self.delay_s = fps, batch, delay_s
        self.samples = 0
        self.ended = False
        self.next = 0
        self.lock = threading.Lock()

    def push(self, pcm16k):
        with self.lock:
            self.samples += len(pcm16k)

    def end(self):
        with self.lock:
            self.ended = True

    def _limit(self) -> int:
        if self.ended:
            return frames_for(self.samples, self.fps)
        return self.samples * self.fps // PCM_RATE

    def next_batch(self):
        with self.lock:
            n = min(self.batch, self._limit() - self.next)
            start = self.next
            self.next += max(n, 0)
        if n <= 0:
            return []
        if self.delay_s:
            time.sleep(self.delay_s)
        return [(i, crop_value(i)) for i in range(start, start + n)]

    @property
    def finished(self):
        with self.lock:
            return self.ended and self.next >= frames_for(self.samples, self.fps)

    @property
    def total_frames(self):
        with self.lock:
            return frames_for(self.samples, self.fps) if self.ended else None


class FakeEngine(Engine):
    name = "fake"

    def __init__(self, batch: int = 4, delay_s: float = 0.0, crop=(200, 330, 160, 120),
                 frame_size=(560, 720), **_cfg):
        self.batch, self.delay_s = batch, delay_s
        self.crop, self.frame_size = tuple(crop), tuple(frame_size)

    def prepare(self, avatar):
        return AvatarHandle(avatar.id, self.frame_size, self.crop)

    def open_stream(self, handle, fps):
        return _FakeStream(fps, self.batch, self.delay_s)

    def compose(self, handle, raw):
        _, _, w, h = handle.crop
        return np.full((h, w, 3), raw, np.uint8)
