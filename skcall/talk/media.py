"""aiortc helpers for talking to Janus behind the Talk HPB.

- ice_config(): build an RTCConfiguration from Talk signaling settings, with an
  optional filter on which TURN URL to use and an optional relay-only policy.
- ToneTrack: an outgoing audio track that plays numbered tone bursts (or a WAV),
  remembering when each burst onset was handed to the encoder (for latency).
- candidates_from_sdp(): the a=candidate lines aiortc puts into a gathered SDP.
"""
from __future__ import annotations

import asyncio
import fractions
import functools
import math
import time
import wave
from typing import Iterable

import numpy as np
from aiortc import (MediaStreamTrack, RTCConfiguration, RTCIceServer)
from av import AudioFrame

SR = 48000
FRAME = 960  # 20 ms at 48 kHz

def patch_ice(relay_only: bool = False, ipv6: bool = False,
              iface_allow: tuple[str, ...] | None = None) -> None:
    """Process-wide aioice tweaks that aiortc 1.15 does not expose.

    relay_only: TransportPolicy.RELAY (aiortc has no iceTransportPolicy).
    ipv6: gather IPv6 host candidates (off by default: Janus is v4 here).
    iface_allow: only gather host candidates on IPs starting with one of these
      prefixes. Measured on a host with tailscale0: STUN through it never
      answers, so any STUN server costs a 5 s gather timeout unless it is
      excluded.
    """
    import aioice
    from aioice import ice as aioice_ice
    from aiortc import rtcicetransport
    kw = {"use_ipv6": ipv6}
    if relay_only:
        kw["transport_policy"] = aioice.TransportPolicy.RELAY
    rtcicetransport.Connection = functools.partial(aioice.Connection, **kw)
    if iface_allow:
        orig = getattr(aioice_ice, "_skcall_orig_gha", aioice_ice.get_host_addresses)
        aioice_ice._skcall_orig_gha = orig

        def filtered(use_ipv4: bool, use_ipv6: bool) -> list[str]:
            return [a for a in orig(use_ipv4, use_ipv6) if a.startswith(iface_allow)]
        aioice_ice.get_host_addresses = filtered


def ice_config(settings: dict, turn_filter: str | None = None,
               stun: str = "turnhost") -> tuple[RTCConfiguration, list[str]]:
    """turn_filter: substring(s) joined by '+' that the chosen TURN URL must
    contain, e.g. 'turn:+transport=udp'; or a full turn:/turns: URL (no '+')
    to use with the advertised credentials. aiortc uses only the FIRST usable
    STUN and TURN server, so we pass exactly the one we want to test.
    stun: 'settings' (what Talk advertises), 'turnhost' (stun: on the TURN
    host, coturn answers binding requests), 'none', or an explicit stun: URL.
    Returns (config, [urls actually configured])."""
    servers: list[RTCIceServer] = []
    used: list[str] = []
    stun_url = None
    if stun == "settings":
        for s in settings.get("stunservers") or []:
            urls = s["urls"] if isinstance(s["urls"], list) else [s["urls"]]
            stun_url = urls[0]
            break
    elif stun == "turnhost":
        for t in settings.get("turnservers") or []:
            u = (t["urls"] if isinstance(t["urls"], list) else [t["urls"]])[0]
            hostport = u.split(":", 1)[1].split("?")[0]
            stun_url = "stun:" + hostport
            break
    elif stun.startswith("stun:"):
        stun_url = stun
    if stun_url:
        servers.append(RTCIceServer(urls=[stun_url]))
        used.append(stun_url)
    for t in settings.get("turnservers") or []:
        urls = t["urls"] if isinstance(t["urls"], list) else [t["urls"]]
        if turn_filter and turn_filter.startswith(("turn:", "turns:")) and "+" not in turn_filter:
            # explicit URL not advertised by Talk (e.g. turns:host:443), same credentials
            urls = [turn_filter]
        elif turn_filter:
            urls = [u for u in urls if all(p in u for p in turn_filter.split("+"))]
        # aiortc only supports turn:udp, turn:tcp and turns:tcp
        urls = [u for u in urls if not (u.startswith("turns:") and "transport=udp" in u)]
        if urls:
            servers.append(RTCIceServer(urls=urls[:1], username=t["username"],
                                        credential=t["credential"]))
            used.append(urls[0])
            break
    return RTCConfiguration(iceServers=servers), used


def candidates_from_sdp(sdp: str) -> list[str]:
    return [line[2:] for line in sdp.splitlines() if line.startswith("a=candidate:")]


def burst_freq(k: int) -> float:
    return 440.0 + 110.0 * (k % 8)


class ToneTrack(MediaStreamTrack):
    """Mono 48 kHz s16 track, paced in real time.

    mode 'bursts': every `period` s a `burst` s sine burst at burst_freq(k),
    silence between. self.onsets[k] = monotonic time the onset frame was
    returned to the encoder.
    mode 'wav': plays `pcm` (int16 48k mono) once, then silence;
    self.onsets[0] = time first wav frame went out.
    """
    kind = "audio"

    def __init__(self, period: float = 1.0, burst: float = 0.2, amp: float = 0.3,
                 pcm: np.ndarray | None = None, lead_silence: float = 0.5,
                 noise: float = 0.0, hold: bool = False):
        super().__init__()
        self.period, self.burst, self.amp = period, burst, amp
        self.pcm = pcm
        self.noise = noise  # dither amplitude added to every frame (0 = digital silence)
        self.lead = int(lead_silence * SR)
        self._n = 0          # samples emitted
        self._t0: float | None = None
        self.onsets: dict[int, float] = {}
        self.frames_sent = 0
        # wav mode: sample index where playback starts; None holds silence
        # until play() is called (so a late subscriber does not miss the start)
        self.play_from: int | None = self.lead if not hold else None

    def play(self) -> None:
        """Start the WAV at the next frame (wav mode with hold=True)."""
        self.play_from = self._n

    async def recv(self) -> AudioFrame:
        if self._t0 is None:
            self._t0 = time.monotonic()
        # pace: frame i is due at t0 + i*20ms
        due = self._t0 + self._n / SR
        delay = due - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        idx = np.arange(self._n, self._n + FRAME)
        samples = np.zeros(FRAME, dtype=np.float32)
        if self.pcm is not None:
            if self.play_from is None:
                rel = np.full(FRAME, -1)
            else:
                rel = idx - self.play_from
            m = (rel >= 0) & (rel < len(self.pcm))
            if m.any():
                samples[m] = self.pcm[rel[m]].astype(np.float32) / 32768.0
                if 0 not in self.onsets:
                    self.onsets[0] = time.monotonic()
        else:
            rel = idx - self.lead
            pos = rel % int(self.period * SR)
            k = rel // int(self.period * SR)
            m = (rel >= 0) & (pos < int(self.burst * SR))
            if m.any():
                kk = int(k[m][0])
                f = burst_freq(kk)
                samples[m] = self.amp * np.sin(2 * math.pi * f * (rel[m] / SR))
                if kk not in self.onsets:
                    self.onsets[kk] = time.monotonic()
        if self.noise:
            samples += np.random.uniform(-self.noise, self.noise, FRAME).astype(np.float32)
        frame = AudioFrame.from_ndarray((samples * 32767).astype(np.int16).reshape(1, -1),
                                        format="s16", layout="mono")
        frame.sample_rate = SR
        frame.pts = self._n
        frame.time_base = fractions.Fraction(1, SR)
        self._n += FRAME
        self.frames_sent += 1
        return frame


def load_wav_48k_mono(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        sr, ch, sw = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw != 2:
        raise ValueError("expected 16-bit wav")
    a = np.frombuffer(raw, dtype=np.int16).reshape(-1, ch).mean(axis=1)
    if sr != SR:
        n = int(len(a) * SR / sr)
        a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a)
    return a.astype(np.int16)


def frame_to_mono(frame: AudioFrame) -> np.ndarray:
    """Decoded Opus frames from aiortc are s16 interleaved (often stereo)."""
    a = frame.to_ndarray()
    ch = len(frame.layout.channels)
    if a.ndim == 2 and a.shape[0] == 1 and ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    else:
        a = a.reshape(-1)
    return a.astype(np.int16)


def write_wav(path: str, pcm: Iterable[np.ndarray], sr: int = SR) -> None:
    data = np.concatenate(list(pcm)) if pcm else np.zeros(0, np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(data.astype(np.int16).tobytes())
