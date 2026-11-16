"""Per-speaker energy VAD with endpointing, and a barge-in detector.

Lifted from skchat (src/skchat/transports/livekit.py, VADSegmenter and
BargeInDetector, GPL-3.0, same authors), ported from raw bytes to numpy
frames, with three additions:

- pre-roll: the ~200 ms before the onset is kept, so a soft first consonant
  is not lost to the gate;
- a clock passed per push (frame timestamps), so tests drive time exactly;
- `prime()`: after a barge-in, frames heard while the agent was speaking (and
  therefore gated as echo) are replayed into the segmenter, so the start of
  the human's interrupting sentence is not chopped off.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass

import numpy as np

from skcall.core.audio import FRAME_MS, STT_RATE, rms


@dataclass
class Utterance:
    pcm: np.ndarray          # int16 mono, STT_RATE
    t_start: float           # monotonic time of the onset frame
    t_last_voice: float      # monotonic time of the last voiced frame
    t_endpoint: float        # monotonic time the endpoint fired
    forced: bool = False     # max-length flush rather than silence
    spec: object = None      # a speculative transcript task that covers this utterance (loop.py)

    @property
    def seconds(self) -> float:
        return len(self.pcm) / STT_RATE


class VADSegmenter:
    def __init__(self, *, rms_threshold: float = 500, silence_hangover_ms: int = 550,
                 min_utterance_ms: int = 400, max_utterance_ms: int = 15000,
                 preroll_ms: int = 200, frame_ms: int = FRAME_MS):
        self.rms_threshold = rms_threshold
        self.hangover_ms = silence_hangover_ms
        self.min_ms = min_utterance_ms
        self.max_ms = max_utterance_ms
        self.frame_ms = frame_ms
        self._pre: collections.deque = collections.deque(maxlen=max(1, preroll_ms // frame_ms))
        self.in_speech = False
        self._voiced: list[np.ndarray] = []
        self._t_start = 0.0
        self._t_last = 0.0
        self.onsets = 0          # increments at every utterance onset (identifies the one in progress)

    # ---- the utterance in progress (for speculative STT while it is still open)
    @property
    def t_last_voice(self) -> float:
        return self._t_last

    def silence_ms(self, now: float) -> float:
        return (now - self._t_last) * 1000 if self.in_speech else 0.0

    def speech_ms(self) -> float:
        return (self._t_last - self._t_start) * 1000 if self.in_speech else 0.0

    def snapshot(self) -> np.ndarray:
        """The audio of the open utterance so far (pre-roll included)."""
        return np.concatenate(self._voiced) if self._voiced else np.zeros(0, np.int16)

    def endpoint(self, now: float) -> Utterance | None:
        """Close the open utterance now, before the hangover runs out (the
        caller decided from a transcript that it is complete)."""
        if not self.in_speech:
            return None
        pcm = np.concatenate(self._voiced)
        self.in_speech = False
        self._voiced = []
        if (self._t_last - self._t_start) * 1000 >= self.min_ms:
            return Utterance(pcm, self._t_start, self._t_last, now)
        return None

    def reset(self) -> None:
        self.in_speech = False
        self._voiced = []
        self._pre.clear()

    def push(self, frame: np.ndarray, now: float, *, gated: bool = False) -> Utterance | None:
        """Feed one frame. gated=True while the agent is speaking (echo guard):
        the frame is dropped and any half-built utterance is discarded."""
        if gated:
            if self.in_speech or self._pre:
                self.reset()
            return None
        voiced = rms(frame) >= self.rms_threshold
        if not self.in_speech:
            if not voiced:
                self._pre.append(frame)
                return None
            self.in_speech = True
            self.onsets += 1
            self._t_start = now - len(self._pre) * self.frame_ms / 1000
            self._voiced = list(self._pre)
            self._pre.clear()
        self._voiced.append(frame)
        if voiced:
            self._t_last = now
        silent_ms = (now - self._t_last) * 1000
        dur_ms = (now - self._t_start) * 1000
        forced = dur_ms >= self.max_ms
        if silent_ms >= self.hangover_ms or forced:
            pcm = np.concatenate(self._voiced)
            self.in_speech = False
            self._voiced = []
            speech_ms = (self._t_last - self._t_start) * 1000
            if speech_ms >= self.min_ms or forced:
                return Utterance(pcm, self._t_start, self._t_last, now, forced=forced)
        return None

    def prime(self, frames: list[tuple[np.ndarray, float]]) -> Utterance | None:
        """Replay (frame, t) pairs that were gated, e.g. right after barge-in."""
        out = None
        for f, t in frames:
            u = self.push(f, t)
            out = out or u
        return out


class BargeInDetector:
    """Sustained loud voice on the HUMAN's own track while the agent speaks.

    Accumulates frame_ms per frame above rms_threshold and decays otherwise;
    fires once dwell_ms is reached. Janus never loops the agent's own stream
    back, so what this hears is the human (or their speaker's echo, which is
    why the threshold sits above the VAD gate and there is a grace period)."""

    def __init__(self, *, rms_threshold: float = 1200, dwell_ms: int = 400,
                 frame_ms: int = FRAME_MS, enabled: bool = True):
        self.rms_threshold = rms_threshold
        self.dwell_ms = dwell_ms
        self.frame_ms = frame_ms
        self.enabled = enabled
        self._ms = 0.0

    def reset(self) -> None:
        self._ms = 0.0

    def push(self, frame: np.ndarray) -> bool:
        if not self.enabled:
            return False
        if rms(frame) >= self.rms_threshold:
            self._ms += self.frame_ms
            if self._ms >= self.dwell_ms:
                self._ms = 0.0
                return True
        else:
            self._ms = max(0.0, self._ms - self.frame_ms)
        return False
