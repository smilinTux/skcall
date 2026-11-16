"""Ditto engine: not implemented yet (selecting it fails cleanly).

Ditto (antgroup/ditto-talkinghead, Apache-2.0, https://github.com/antgroup/ditto-talkinghead)
animates the whole head (pose, blinks, expression) from audio, unlike MuseTalk,
which only repaints the mouth. It would be the second engine once a CUDA GPU
is free. What it needs:

- An NVIDIA GPU. The realtime path is TensorRT 8.6 (CUDA only); there is an
  open issue about RTX 5090 / Blackwell support (#88). The PyTorch path is
  untested on Intel XPU.
- A license check of the released weights and of every sub-model before
  shipping (code is Apache-2.0).
- Its online streaming pipeline (`stream_pipeline_online.py`) mapped onto this
  interface: prepare() = its source-image registration, open_stream() = a new
  online session, next_batch() = pull rendered frames, compose() = crop the
  face region. Because the head moves, either return full frames (and grow
  the crop to the whole frame in `ready`) or extend the protocol with a
  per-frame crop box; both are protocol changes to agree with the skcall client.
"""
from __future__ import annotations

from . import Engine, EngineUnavailable


class DittoEngine(Engine):
    name = "ditto"

    def __init__(self, **_cfg):
        pass

    def load(self) -> None:
        raise EngineUnavailable(
            "engine 'ditto' is not implemented yet: it needs a CUDA GPU (TensorRT), a weights "
            "license check, and a protocol decision for full-head frames; see "
            "services/lipsync/sklipsync/engines/ditto.py")

    def prepare(self, avatar):
        self.load()

    def open_stream(self, handle, fps):
        self.load()

    def compose(self, handle, raw):
        self.load()
