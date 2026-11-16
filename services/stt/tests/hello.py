"""The bundled tests/data/hello.wav: 1 s of synthesized tone, not a recording.

  python tests/hello.py      (rewrites tests/data/hello.wav)
"""
from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np

RATE = 16000
PATH = Path(__file__).resolve().parent / "data" / "hello.wav"


def hello_wav() -> bytes:
    """16 kHz mono s16le, 1 s: a 220 Hz tone with a gentle fade in and out."""
    t = np.arange(RATE) / RATE
    env = np.minimum(1.0, np.minimum(t, 1.0 - t) * 20)
    pcm = (0.3 * 32767 * env * np.sin(2 * np.pi * 220 * t)).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


if __name__ == "__main__":
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_bytes(hello_wav())
    print(PATH)
