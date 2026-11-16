"""Pure audio helpers: int16 mono numpy arrays in, int16 mono numpy arrays out.

No aiortc, no network. Rates used in skcall:
  48 kHz   what Talk/Janus (Opus) carries, both directions
  16 kHz   what VAD and whisper consume
  24 kHz   what F5-TTS renders
"""
from __future__ import annotations

import io
import wave

import numpy as np

CALL_RATE = 48000
STT_RATE = 16000
FRAME_MS = 20


def rms(pcm: np.ndarray) -> float:
    """RMS of int16 samples, in int16 units (0..32767), like audioop.rms."""
    if pcm.size == 0:
        return 0.0
    x = pcm.astype(np.float64)
    return float(np.sqrt(np.mean(x * x)))


def resample(pcm: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Resample int16 mono. Integer down-ratios average blocks (a crude low-pass,
    fine for speech going to VAD/STT); everything else is linear interpolation."""
    if src == dst or pcm.size == 0:
        return pcm.astype(np.int16, copy=False)
    if src % dst == 0:
        k = src // dst
        n = (len(pcm) // k) * k
        return pcm[:n].reshape(-1, k).astype(np.int32).mean(axis=1).astype(np.int16)
    n = int(round(len(pcm) * dst / src))
    x = np.interp(np.linspace(0, len(pcm) - 1, n), np.arange(len(pcm)), pcm.astype(np.float32))
    return np.clip(x, -32768, 32767).astype(np.int16)


def normalize_peak(pcm: np.ndarray, target_peak: float = 0.97) -> np.ndarray:
    """Peak-normalize one reply. F5 renders about 3x quieter than Piper
    (skchat livekit.py normalize_pcm); peak, not RMS, so it cannot clip."""
    if pcm.size == 0:
        return pcm
    peak = int(np.max(np.abs(pcm.astype(np.int32))))
    if peak <= 0:
        return pcm
    f = target_peak * 32767.0 / peak
    if abs(f - 1.0) < 0.01:
        return pcm
    return np.clip(pcm.astype(np.float32) * f, -32768, 32767).astype(np.int16)


def wav_to_pcm(data: bytes, target_rate: int) -> np.ndarray:
    """Decode a WAV payload to int16 mono at target_rate.

    Refuses a text body (a JSON/HTML error) instead of publishing it as noise,
    the "loud distortion" failure skchat hit (livekit.py wav_to_pcm)."""
    if data[:4] != b"RIFF":
        return np.zeros(0, np.int16)
    with wave.open(io.BytesIO(data)) as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        return np.zeros(0, np.int16)
    a = np.frombuffer(raw, np.int16)
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1).astype(np.int16)
    return resample(a, sr, target_rate)


def pcm_to_wav(pcm: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.astype(np.int16).tobytes())
    return buf.getvalue()


def trim_silence(pcm: np.ndarray, rate: int, *, rel_threshold: float = 0.03, pad_s: float = 0.06,
                 max_gap_s: float = 0.5, keep_gap_s: float = 0.25) -> np.ndarray:
    """Cut leading/trailing silence and shorten long internal pauses.

    Measured 2026-10-25: F5 pads its output with 0.3 to 1.7 s of silence at the
    START (short phrases are mostly silence: "I'm here." is 2.15 s with the
    voice starting at 1.52 s), and occasionally renders a 2 s pause mid-phrase.
    All of that is dead air before and inside her answer."""
    if pcm.size == 0:
        return pcm
    peak = float(np.max(np.abs(pcm.astype(np.int32))))
    if peak <= 0:
        return pcm[:0]
    thr = max(200.0, rel_threshold * peak)
    loud = np.nonzero(np.abs(pcm.astype(np.int32)) > thr)[0]
    if loud.size == 0:
        return pcm[:0]
    pad = int(pad_s * rate)
    x = pcm[max(0, loud[0] - pad): min(len(pcm), loud[-1] + pad)]
    # internal pauses: 20 ms frames below threshold, runs longer than max_gap_s
    n = rate // 50
    frames = len(x) // n
    if frames < 3:
        return x
    env = np.abs(x[: frames * n].astype(np.int32)).reshape(frames, n).max(axis=1)
    quiet = env <= thr
    keep = np.ones(frames, bool)
    max_run, keep_run = int(max_gap_s * 50), int(keep_gap_s * 50)
    i = 0
    while i < frames:
        if quiet[i]:
            j = i
            while j < frames and quiet[j]:
                j += 1
            if j - i > max_run:
                keep[i + keep_run // 2: j - (keep_run - keep_run // 2)] = False
            i = j
        else:
            i += 1
    if keep.all():
        return x
    body = x[: frames * n].reshape(frames, n)[keep].reshape(-1)
    return np.concatenate([body, x[frames * n:]])


class Reframer:
    """Turns arbitrary-length chunks into fixed-size frames (e.g. 320 samples
    = 20 ms at 16 kHz), carrying the remainder between calls."""

    def __init__(self, frame_samples: int):
        self.n = frame_samples
        self._buf = np.zeros(0, np.int16)

    def push(self, pcm: np.ndarray) -> list[np.ndarray]:
        self._buf = np.concatenate([self._buf, pcm.astype(np.int16)])
        out = []
        while len(self._buf) >= self.n:
            out.append(self._buf[: self.n])
            self._buf = self._buf[self.n:]
        return out
