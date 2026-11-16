"""Backend-agnostic call interface (docs/ARCHITECTURE.md section 5.1).

The core (skcall.core.loop) only sees these types. TalkBackend implements
them on aiortc; a LiveKit backend (P3) wraps rtc.Room the same way. Nothing in
skcall.core imports aiortc or livekit.

Audio crossing this boundary is int16 mono numpy at CALL_RATE (48 kHz).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol, Union

import numpy as np


@dataclass(frozen=True)
class Participant:
    pid: str                 # backend id of this participant's session (Talk: HPB session id)
    user_id: str             # verified account id (Talk: Nextcloud userId), "" for guests
    display: str = ""
    is_agent: bool = False


@dataclass
class ParticipantJoined:
    participant: Participant


@dataclass
class ParticipantLeft:
    participant: Participant


@dataclass
class AudioTrack:
    """A remote participant's audio. `frames` yields int16 mono 48 kHz chunks
    (any length; Talk gives 960 samples per 20 ms) until the track ends."""
    participant: Participant
    frames: AsyncIterator[np.ndarray]


@dataclass
class VideoTrack:
    """A remote participant's video, only for sessions that asked for it
    (decoding video costs CPU the voice agent does not need). `frames` yields
    RGB uint8 arrays (h, w, 3) paired with the monotonic arrival time."""
    participant: Participant
    frames: AsyncIterator[tuple[float, np.ndarray]]


@dataclass
class CallEnded:
    reason: str = ""


CallEvent = Union[ParticipantJoined, ParticipantLeft, AudioTrack, VideoTrack, CallEnded]


class AudioSink(Protocol):
    """The agent's outgoing voice. write() queues audio and returns at once;
    the backend paces it out in 20 ms frames and plays silence when empty."""
    sample_rate: int

    def write(self, pcm: np.ndarray, on_start=None) -> None:
        """Queue int16 mono at sample_rate. on_start(t_monotonic) is called when
        the first sample of this chunk is handed to the encoder."""

    def clear(self) -> None:
        """Drop everything queued (barge-in)."""

    @property
    def buffered_s(self) -> float: ...

    async def drained(self) -> None:
        """Return once everything queued so far has been played."""


class VideoSource(Protocol):
    """Where the agent's picture comes from. The backend PULLS: its video sink
    asks for the frame due at monotonic time `t`, at the source's fps, on the
    same clock the AudioSink's on_start timestamps use. That is what keeps
    the lips in step with the audio without any shared buffer.

    frame() must never raise and never block for long: a source that cannot
    render returns its last good frame (or a static portrait)."""
    width: int
    height: int
    fps: float

    def frame(self, t: float) -> np.ndarray:
        """RGB uint8 (height, width, 3) for monotonic time t."""


class VideoSink(Protocol):
    """The agent's outgoing video (a backend track). `source` can be swapped
    at any time (static portrait -> compositor and back); the track keeps its
    negotiated size, so sources must render at the sink's width x height."""
    width: int
    height: int
    source: VideoSource | None
    frames_sent: int


class CallSession(Protocol):
    self_user_id: str
    audio_out: AudioSink
    video_out: VideoSink | None   # None: audio-only session (video off or not permitted)

    def events(self) -> AsyncIterator[CallEvent]: ...
    def participants(self) -> list[Participant]: ...
    async def post_chat(self, text: str) -> None: ...
    async def leave(self) -> None: ...


@dataclass
class FakeSink:
    """In-memory AudioSink for tests. With hold=False it 'plays' instantly;
    with hold=True written audio stays buffered (she is audible) until clear()
    or release(), which is how tests put her mid-sentence."""
    sample_rate: int = 48000
    clock: object = None
    hold: bool = False
    chunks: list = field(default_factory=list)
    written: list = field(default_factory=list)
    cleared: int = 0

    def __post_init__(self):
        import asyncio
        self._drained = asyncio.Event()
        self._drained.set()

    def write(self, pcm, on_start=None):
        import time
        self.chunks.append(pcm)
        self.written.append(pcm)
        if self.hold:
            self._drained.clear()
        if on_start:
            on_start((self.clock or time.monotonic)())

    def release(self):
        self.chunks.clear()
        self._drained.set()

    def clear(self):
        self.cleared += 1
        self.release()

    @property
    def buffered_s(self) -> float:
        return sum(len(c) for c in self.chunks) / self.sample_rate if self.hold else 0.0

    async def drained(self) -> None:
        await self._drained.wait()
