"""Pluggable lip-sync engines.

An engine turns an avatar plus streamed 16 kHz PCM into mouth-region crops:

  handle = engine.prepare(avatar)              # once per avatar, cached
  stream = engine.open_stream(handle, fps)     # once per utterance
  stream.push(pcm16k_chunk); ...; stream.end()
  batch  = stream.next_batch()                 # [(frame_index, raw), ...]; blocking, GPU
  crop   = engine.compose(handle, raw)         # BGR crop for handle.crop; CPU, thread-safe

`Engine.render(handle, pcm16k)` wraps that for offline use and yields
(frame_index, crop) for a whole clip.

Engines: "musetalk" (MuseTalk 1.5, mouth only, runs on Intel XPU / CUDA / CPU),
"ditto" (stub, see ditto.py), "fake" (no model; tests and local dev).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np


class EngineUnavailable(RuntimeError):
    """The engine cannot run here (not implemented, missing GPU, missing weights)."""


@dataclass
class AvatarHandle:
    avatar_id: str
    frame_size: tuple[int, int]               # (w, h) of the avatar's full frame
    crop: tuple[int, int, int, int]           # (x, y, w, h) where crops go on that frame
    data: dict = field(default_factory=dict)  # engine private

    def crop_json(self) -> dict:
        x, y, w, h = self.crop
        return {"x": x, "y": y, "w": w, "h": h}


class RenderStream(ABC):
    """One utterance. push()/end() come from the event loop, next_batch() from
    the GPU thread; implementations guard their buffer with a lock."""

    @abstractmethod
    def push(self, pcm16k: np.ndarray) -> None: ...

    @abstractmethod
    def end(self) -> None: ...

    @abstractmethod
    def next_batch(self) -> list[tuple[int, Any]]:
        """Render the next batch of frames the audio received so far allows.
        Returns [] when nothing is renderable yet (or everything is done)."""

    @property
    @abstractmethod
    def finished(self) -> bool:
        """end() was called and every frame has been returned."""

    @property
    @abstractmethod
    def total_frames(self) -> int | None:
        """Frame count once end() was called, else None."""


class Engine(ABC):
    name = "base"

    def load(self) -> None:
        """Load models. Called once before the server accepts connections."""

    @abstractmethod
    def prepare(self, avatar) -> AvatarHandle: ...

    @abstractmethod
    def open_stream(self, handle: AvatarHandle, fps: int) -> RenderStream: ...

    @abstractmethod
    def compose(self, handle: AvatarHandle, raw: Any) -> np.ndarray:
        """Turn one raw engine output into the BGR crop for handle.crop."""

    def info(self) -> dict:
        return {"engine": self.name}

    def render(self, handle: AvatarHandle, pcm16k: np.ndarray, fps: int = 25) -> Iterator[tuple[int, np.ndarray]]:
        s = self.open_stream(handle, fps)
        s.push(pcm16k)
        s.end()
        while not s.finished:
            for idx, raw in s.next_batch():
                yield idx, self.compose(handle, raw)


def create_engine(name: str, **cfg) -> Engine:
    if name == "musetalk":
        from .musetalk import MuseTalkEngine
        return MuseTalkEngine(**cfg)
    if name == "ditto":
        from .ditto import DittoEngine
        return DittoEngine(**cfg)
    if name == "fake":
        from .fake import FakeEngine
        return FakeEngine(**cfg)
    raise EngineUnavailable(f"unknown engine {name!r} (known: musetalk, ditto, fake)")
