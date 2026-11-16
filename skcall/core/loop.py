"""The conversation loop: per-speaker VAD + endpointing -> STT -> brain (streamed)
-> per-sentence TTS -> AudioSink, with barge-in, echo guard and filler.

Backend-agnostic: it consumes CallSession events and writes to an AudioSink.
The shape follows skchat's LiveKit room loop (transports/livekit.py
build_room_session): the turn runs as a task so the audio pump keeps reading
frames underneath it (that is what makes barge-in possible), the filler only
speaks when the reply is slow, the reply renders while the filler plays, and
sentence N+1 renders while sentence N plays.

Differences from the LiveKit loop, on purpose:
- The reply is streamed from the brain and spoken sentence by sentence, so the
  first sentence is synthesized while the LLM is still writing the second.
- Echo guard: while she is audible, plus `echo_tail_s` after, the segmenter is
  fed nothing (she never transcribes her own voice coming back through a
  human's speaker). Barge-in listens on the human's own track with a higher
  energy gate; when it fires, playback stops and the frames gated during the
  last ~2 s are replayed into the segmenter, so the interrupting sentence is
  transcribed from its first word.
- An utterance that ends while her previous turn is still thinking (nothing
  spoken yet) from the same speaker replaces that turn: the two pieces are
  transcribed together. This is what makes a short endpoint silence safe: a
  mid-sentence pause costs one cancelled LLM call, not an answer to half a
  sentence.
"""
from __future__ import annotations

import asyncio
import collections
import json
import logging
import os
import re
import statistics
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from skcall.privacy import said
from skcall.brain.base import Brain, Turn
from skcall.core.audio import CALL_RATE, FRAME_MS, STT_RATE, Reframer, resample, rms
from skcall.core.session import (AudioTrack, CallEnded, CallSession, Participant,
                                 ParticipantJoined, ParticipantLeft)
from skcall.core.text import (SentenceChunker, TranscriptDedup, is_leave_command, is_non_speech,
                              is_whisper_repetition, looks_complete, strip_formatting)
from skcall.core.vad import BargeInDetector, Utterance, VADSegmenter
from skcall.brain.hermes import (ACK_LINES, is_progress_query, reply_lacks_tool, same_request,
                                 wants_escalation)

LOG = logging.getLogger("skcall.loop")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class LoopConfig:
    vad_rms: float = 500
    hangover_ms: int = 450
    min_utterance_ms: int = 400
    # Streaming STT: the open utterance is transcribed at its first short
    # pause (`spec_silence_ms`), while the endpoint is still undecided, and
    # again at every later pause if the speaker went on. The transcript then
    # decides the endpoint: one that reads as finished closes the turn after
    # `endpoint_complete_ms` of silence, one that trails off ("and...", "so
    # the") waits `endpoint_incomplete_ms`, and so does a transcript still in
    # flight (the turn needs it anyway). `hangover_ms` then only applies to
    # non-speech transcripts and to the classic path (spec_stt off).
    spec_stt: bool = True
    spec_silence_ms: int = 150
    endpoint_complete_ms: int = 250
    endpoint_incomplete_ms: int = 900
    # The first spoken chunk is cut at the first clause mark after this many
    # characters (F5 time grows with length; the first chunk is the first audio).
    first_clause_chars: int = 30
    max_utterance_ms: int = 15000
    barge_in: bool = True
    barge_rms: float = 1200
    barge_dwell_ms: int = 400
    barge_grace_ms: int = 500
    echo_tail_s: float = 0.6
    filler: bool = True
    filler_delay_s: float = 0.3
    # 2026-10-25: filler only when she is working on something. A plain
    # chat turn gets no "mm" (it read as an "um"); lookups, narration and the
    # Hermes hand-off still acknowledge, because those can take seconds.
    filler_on_chat: bool = False
    filler_repeat_s: float = 9.0
    operators: tuple[str, ...] = ()         # Talk user ids; the site config sets them (SKCALL_OPERATORS)
    agent_names: tuple[str, ...] = ("lumina", "luna", "lumi")
    follow_up_s: float = 45.0
    alone_timeout_s: float = 20.0
    wait_join_s: float = 300.0
    idle_timeout_s: float = 600.0
    max_call_s: float = 7200.0
    leave_line: str = "Okay, I'm leaving the call now. Talk soon."
    transcript_max_chars: int = 200_000
    # Voice-layer deadlines (2026-10-25: a "look into the pipeline" turn ran
    # the voice brain's tool loop with no overall deadline and she never
    # spoke). With Hermes available: no first sentence within
    # `voice_deadline_s`, or the voice turn still running after `voice_max_s`
    # (tool rounds included), cancels it, says `voice_deadline_line` and hands
    # the same text to Hermes. 0 disables either.
    voice_deadline_s: float = field(default_factory=lambda: _env_float("SKCALL_VOICE_DEADLINE_S", 8.0))
    voice_max_s: float = field(default_factory=lambda: _env_float("SKCALL_VOICE_MAX_S", 20.0))
    voice_deadline_line: str = "This one needs my full toolbelt, give me a minute."
    # Hand-off to Hermes runs at the OWNER's scope (every tool, her full memory) and the answer is
    # spoken to everyone in the call, so only an operator's request is ever handed off: a guest
    # saying "Lumina, read me the operator's email" stays with the voice layer, whose tools are operator-gated
    # too. In a group call (anyone besides one operator) SKCALL_GROUP_ESCALATION decides:
    # "operator" (default) hands off the operator's requests and tells Hermes who else is listening;
    # "off" never hands off while anyone else is in the call.
    group_escalation: str = field(default_factory=lambda: (
        os.environ.get("SKCALL_GROUP_ESCALATION", "").strip().lower() or "operator"))
    # While a Hermes job runs: a short progress line after `hermes_progress_first_s`,
    # then every `hermes_progress_every_s`, only when nobody is speaking.
    hermes_progress_first_s: float = 45.0
    hermes_progress_every_s: float = 60.0
    hermes_progress_line: str = "Still working on it."
    # A Hermes job with no answer after `hermes_slow_s` (2026-10-26: one hung
    # for 15 min while she said "Still working on it." every minute): say so
    # once, honestly, stop the progress lines, and post the answer to the room
    # chat when it arrives (plus one short spoken pointer if the call is on).
    hermes_slow_s: float = field(default_factory=lambda: _env_float("SKCALL_HERMES_SLOW_S", 90.0))
    hermes_slow_line: str = ("That's taking too long. I'll post the answer in the chat "
                             "when it's ready.")
    hermes_posted_line: str = "I posted that answer in the chat."
    # Asked again (or asked about it) while that job is pending: no new job.
    hermes_pending_line: str = ("I'm still waiting on that one. I'll post the answer in the "
                                "chat when it's ready.")
    # Deafness watchdog (2026-10-26: twice in one call she stopped hearing
    # the operator with no log line at all). A human in the call, not muted, with no
    # inbound audio frames at all for `stall_s` gets a WARNING with the loop
    # state and a fresh subscription (session.resubscribe), retried with a
    # doubling delay from `resubscribe_every_s`. Only media flow counts:
    # frames that arrive and stay under the VAD gate (silence, comfort noise)
    # are a quiet human, not a stall. A muted human (session.audio_muted) is
    # never a stall; unmuting starts the timer over. A sink that stops
    # draining for `stall_s` is cleared. Every `heartbeat_s` one HEARTBEAT
    # line logs the counters. 0 disables.
    stall_s: float = field(default_factory=lambda: _env_float("SKCALL_STALL_S", 12.0))
    resubscribe_every_s: float = 20.0
    heartbeat_s: float = field(default_factory=lambda: _env_float("SKCALL_HEARTBEAT_S", 30.0))


class TurnTimer:
    """Per-turn stage timestamps, reported relative to the end of speech."""

    STAGES = ("endpoint", "stt", "filler_audio", "first_token", "first_sentence", "first_tts", "first_audio")

    def __init__(self, t_speech_end: float, clock: Callable[[], float]):
        self.t0 = t_speech_end
        self.clock = clock
        self.marks: dict[str, float] = {}
        self.extra: dict = {}

    def mark(self, stage: str, t: float | None = None) -> None:
        if stage not in self.marks:
            self.marks[stage] = (t if t is not None else self.clock()) - self.t0

    def report(self) -> dict:
        d = {k: round(v * 1000) for k, v in self.marks.items()}
        d.update(self.extra)
        return d


@dataclass
class _Spec:
    """A transcript of an utterance that was still open when it was asked for."""
    onset: int                   # VADSegmenter.onsets at launch
    t_last: float                # last voiced frame it covers
    t_launch: float
    task: asyncio.Task | None = None
    text: str | None = None      # set when done ("" on failure)
    t_done: float = 0.0
    backend: str = ""


@dataclass
class _Listener:
    participant: Participant
    seg: VADSegmenter
    barge: BargeInDetector
    reframer: Reframer = field(default_factory=lambda: Reframer(STT_RATE * FRAME_MS // 1000))
    ring: collections.deque = field(default_factory=lambda: collections.deque(maxlen=100))
    peak: float = 0.0
    last_report: float = 0.0
    spec: _Spec | None = None
    # inbound counters (watchdog and HEARTBEAT)
    frames: int = 0              # chunks received from the track
    t_frame: float = 0.0         # last chunk
    t_voiced: float = 0.0        # last frame at or above the VAD gate
    utterances: int = 0
    hb_peak: float = 0.0         # loudest frame since the last heartbeat


# Messages kept in a call's history. A turn is 2 messages, or more with tool
# results in it. Was a flat 40 (cut back to 30), which dropped every memory
# result after its own turn; the call models take 131k tokens, so a long call
# with many memory lookups (~1.5k tokens each) still fits.
HISTORY_MAX = int(os.environ.get("SKCALL_HISTORY_MAX", "160") or 160)


def trim_history(history: list[dict], cap: int) -> None:
    """Cut `history` in place to at most ~3/4 of `cap` once it exceeds `cap`,
    always starting at a user message, so a tool result is never left without
    the assistant tool_calls message it answers."""
    if len(history) <= cap:
        return
    keep = history[-max(2, cap * 3 // 4):]
    while keep and keep[0].get("role") != "user":
        keep.pop(0)
    history[:] = keep


@dataclass
class _TurnState:
    speaker: Participant
    pcm: np.ndarray
    spec: _Spec | None = None
    text: str | None = None
    spoke: bool = False          # real reply audio has started (filler does not count)
    spoken: list[str] = field(default_factory=list)
    task: asyncio.Task | None = None
    exchange: list[dict] = field(default_factory=list)


class Conversation:
    def __init__(self, session: CallSession, stt, tts, brain: Brain,
                 cfg: LoopConfig | None = None, clock: Callable[[], float] = time.monotonic,
                 hermes=None, avatar=None):
        self.s = session
        self.hermes = hermes      # HermesBrain-like: .available, .ask(text, who, history)
        self.stt, self.tts, self.brain = stt, tts, brain
        self.cfg = cfg or LoopConfig()
        self.clock = clock
        # avatar (skcall.video.Avatar, duck-typed so the core does not import
        # it): wraps the sink so each sentence is lip-synced and the first one
        # is held briefly for frames; fillers get their lips pre-rendered.
        self.avatar = avatar
        self.worship = None       # WorshipHost: its narration yields when she answers
        self.sink = avatar.wrap_sink(session.audio_out) if avatar is not None else session.audio_out
        self.history: list[dict] = []
        self.timings: list[dict] = []
        self.listeners: dict[str, _Listener] = {}
        self.turn: _TurnState | None = None
        self.dedup = TranscriptDedup(clock=clock)
        self._echo_until = 0.0
        self._speak_start = 0.0
        self._filler_pcm: dict[str, np.ndarray] = {}
        self._engaged: dict[str, float] = {}
        self._tasks: list[asyncio.Task] = []
        # Hermes jobs outlive the call (their answers then go to the room chat),
        # so they are kept apart from the tasks run() cancels.
        self._hermes_jobs: set[asyncio.Task] = set()
        self._hermes_pending: dict[asyncio.Task, str] = {}   # running job -> its request
        # async (message) -> None: posts a Hermes answer that came back after
        # the call ended to the room chat. Set by the CLI.
        self.post_after_call: Callable | None = None
        self._ended = asyncio.Event()
        self.end_reason = ""
        self._last_speech = clock()
        self._had_company = False
        self.stats = collections.Counter()
        # In-memory only (post-call summary); never written to disk.
        self.transcript: list[tuple[str, str]] = []
        self._transcript_chars = 0
        self.joined: dict[str, str] = {}      # user id (or session id) -> display name, humans only
        # watchdog state
        self._seen: dict[str, float] = {}     # pid -> when the watchdog first saw it (no audio yet)
        self._muted: dict[str, float] = {}    # pid -> when the watchdog last saw it muted
        self._resub: dict[str, tuple[float, float]] = {}   # pid -> (last resubscribe, next delay)
        self._sink_seen: tuple[float, float] = (0.0, 0.0)  # (buffered_s, since when unchanged)
        self._hb_at = clock()

    def note(self, who: str, text: str) -> None:
        """Add a line to the in-memory transcript, bounded for long calls."""
        text = (text or "").strip()
        if not text:
            return
        self.transcript.append((who, text))
        self._transcript_chars += len(who) + len(text)
        while self._transcript_chars > self.cfg.transcript_max_chars and len(self.transcript) > 1:
            w, t = self.transcript.pop(0)
            self._transcript_chars -= len(w) + len(t)

    @property
    def agent_display(self) -> str:
        return self.cfg.agent_names[0].capitalize() if self.cfg.agent_names else "Agent"

    # ------------------------------------------------------------ helpers
    def end(self, reason: str) -> None:
        if not self._ended.is_set():
            self.end_reason = reason
            LOG.info("conversation ending: %s", reason)
            self._ended.set()

    def humans(self) -> list[Participant]:
        return [p for p in self.s.participants() if not p.is_agent]

    def other_agents(self) -> list[Participant]:
        return [p for p in self.s.participants() if p.is_agent]

    def mode(self) -> str:
        h = self.humans()
        if len(h) == 1 and h[0].user_id in self.cfg.operators:
            return "private"
        return "private" if not h else "group"

    def audible(self, now: float) -> bool:
        """Is she playing right now? Also tracks when the current stretch of
        her speech began (the barge-in grace runs from there, not from each
        sentence: a per-sentence grace left holes an interruption fell into)."""
        if self.sink.buffered_s > 0:
            if now >= self._echo_until:
                self._speak_start = now
            self._echo_until = now + self.cfg.echo_tail_s
            return True
        return False

    def gated(self, now: float) -> bool:
        return self.audible(now) or now < self._echo_until

    def _on_audio_start(self, t: float) -> None:
        """Playback of a chunk began (hook for status/telemetry)."""

    def may_escalate(self, speaker: Participant) -> bool:
        """May this speaker's turn be handed to Hermes (owner scope, all tools)? Only an operator's,
        and in a group call only with group_escalation "operator" (see LoopConfig)."""
        if self.hermes is None or not getattr(self.hermes, "available", False):
            return False
        if speaker.is_agent or speaker.user_id not in self.cfg.operators:
            return False
        return self.mode() == "private" or self.cfg.group_escalation == "operator"

    def listeners_besides(self, speaker: Participant) -> list[str]:
        """Display names of the other humans in the call (who would hear a hand-off's answer)."""
        return [p.display or p.user_id or "a guest" for p in self.humans()
                if (p.user_id or p.pid) != (speaker.user_id or speaker.pid)]

    def addressed(self, speaker: Participant, text: str) -> bool:
        """1:1: everything is for her. Group: TODO port skchat's AddressingGate
        (transports/livekit.py: named-me / named-other / follow-up window /
        roundtable turn cap, keyed on the verified userId via caller_profile).
        Until then a group turn needs her name, or a follow-up within
        follow_up_s of her last reply to that same person."""
        if speaker.is_agent:
            return False
        # Alone with one person she answers everything, unless another agent
        # is in the call too: then both would answer, so she needs her name.
        if len(self.humans()) <= 1 and not self.other_agents():
            return True
        t = text.lower()
        if any(re.search(rf"\b{re.escape(n)}\b", t) for n in self.cfg.agent_names):
            return True
        last = self._engaged.get(speaker.user_id)
        return last is not None and self.clock() - last < self.cfg.follow_up_s

    # ------------------------------------------------------------ audio in
    async def pump(self, track: AudioTrack) -> None:
        p = track.participant
        if p.is_agent:      # backstop: the backend does not subscribe to agents
            async for _ in track.frames:
                pass
            return
        li = self.listeners.get(p.pid)
        if li is None:
            # With streaming STT the loop closes utterances itself (see
            # _speculate); the segmenter's own hangover is only the backstop.
            hang = (max(self.cfg.hangover_ms, self.cfg.endpoint_incomplete_ms) if self.cfg.spec_stt
                    else self.cfg.hangover_ms)
            li = _Listener(p, VADSegmenter(rms_threshold=self.cfg.vad_rms,
                                           silence_hangover_ms=hang,
                                           min_utterance_ms=self.cfg.min_utterance_ms,
                                           max_utterance_ms=self.cfg.max_utterance_ms),
                           BargeInDetector(rms_threshold=self.cfg.barge_rms,
                                           dwell_ms=self.cfg.barge_dwell_ms,
                                           enabled=self.cfg.barge_in))
            self.listeners[p.pid] = li
        LOG.info("listening to %s (%s)", p.user_id or p.pid[:8], p.display)
        n0 = li.frames
        try:
            async for chunk in track.frames:
                li.frames += 1
                li.t_frame = self.clock()
                for f in li.reframer.push(resample(chunk, CALL_RATE, STT_RATE)):
                    self.process_frame(li, f, self.clock())
        except asyncio.CancelledError:
            raise
        except Exception:
            LOG.warning("audio pump for %s ended on error", p.user_id, exc_info=True)
            return
        LOG.info("audio from %s ended after %d frames", p.user_id or p.pid[:8], li.frames - n0)

    def process_frame(self, li: _Listener, f: np.ndarray, now: float) -> None:
        lvl = rms(f)
        li.peak = max(li.peak, lvl)
        li.hb_peak = max(li.hb_peak, lvl)
        if now - li.last_report >= 5.0:
            LOG.debug("hearing %s peak_rms=%.0f gate=%.0f", li.participant.user_id, li.peak, self.cfg.vad_rms)
            li.peak, li.last_report = 0.0, now
        li.ring.append((f, now))
        if lvl >= self.cfg.vad_rms:
            self._last_speech = now
            li.t_voiced = now
        speaking = self.audible(now)
        if speaking and self.cfg.barge_in and now - self._speak_start >= self.cfg.barge_grace_ms / 1000:
            if li.barge.push(f):
                self.barge_in(li, now)
                return
        elif not speaking:
            li.barge.reset()
        utt = li.seg.push(f, now, gated=self.gated(now))
        if utt is None and self.cfg.spec_stt:
            utt = self._speculate(li, now)
        if utt is not None:
            li.utterances += 1
            self._attach_spec(li, utt)
            self.on_utterance(li.participant, utt)
        elif li.spec is not None and not li.seg.in_speech:
            self._drop_spec(li)          # the utterance was discarded (too short, or echo-gated)

    # ------------------------------------------------------------ streaming STT
    def _spec_valid(self, li: _Listener, t_last: float | None = None) -> bool:
        sp = li.spec
        return (sp is not None and sp.onset == li.seg.onsets
                and sp.t_last == (li.seg.t_last_voice if t_last is None else t_last))

    def _drop_spec(self, li: _Listener) -> None:
        sp, li.spec = li.spec, None
        if sp is not None and sp.task is not None and not sp.task.done():
            sp.task.cancel()
            self.stats["spec_stale"] += 1

    async def _spec_stt(self, sp: _Spec, pcm: np.ndarray) -> str:
        try:
            text = await self.stt.transcribe(pcm)
        except asyncio.CancelledError:
            raise
        except Exception:
            LOG.warning("speculative STT failed", exc_info=True)
            text = ""
        sp.backend = getattr(self.stt, "last_backend", "")
        sp.text, sp.t_done = text, self.clock()
        return text

    def _speculate(self, li: _Listener, now: float) -> Utterance | None:
        """Called per frame while an utterance is open. Launches a transcript
        at a pause, and closes the utterance early when that transcript says
        the speaker has finished. Returns the utterance when it closes."""
        seg = li.seg
        if not seg.in_speech:
            return None
        sil = seg.silence_ms(now)
        if not self._spec_valid(li):
            if li.spec is not None:
                self._drop_spec(li)          # they kept talking: that transcript is short
            if sil >= self.cfg.spec_silence_ms and seg.speech_ms() >= self.cfg.min_utterance_ms:
                sp = _Spec(seg.onsets, seg.t_last_voice, now)
                sp.task = asyncio.ensure_future(self._spec_stt(sp, seg.snapshot()))
                li.spec = sp
                self.stats["spec_launched"] += 1
            return None
        sp = li.spec
        if sp.text is None:
            # Transcript in flight: wait for it rather than close on the plain
            # hangover. That costs nothing (the turn would have to await the
            # same transcript before the LLM can start) and lets an unfinished
            # phrase keep the utterance open. A slow backend is capped by the
            # segmenter's backstop (endpoint_incomplete_ms), and the turn then
            # awaits the transcript in flight.
            return None
        if sp.text and not is_non_speech(sp.text) and looks_complete(sp.text):
            if sil >= self.cfg.endpoint_complete_ms:
                self.stats["early_endpoint"] += 1
                return seg.endpoint(now)
            return None
        if not sp.text or is_non_speech(sp.text):
            if sil >= self.cfg.hangover_ms:
                return seg.endpoint(now)     # the turn's filters drop it
            return None
        # reads unfinished: the segmenter's backstop (endpoint_incomplete_ms) closes it
        return None

    def _attach_spec(self, li: _Listener, utt: Utterance) -> None:
        if self._spec_valid(li, utt.t_last_voice):
            utt.spec, li.spec = li.spec, None
        elif li.spec is not None:
            self._drop_spec(li)

    def barge_in(self, li: _Listener, now: float) -> None:
        t = self.turn
        LOG.info("barge-in from %s: stopping playback", li.participant.user_id)
        self.stats["barge_in"] += 1
        self.sink.clear()
        self._echo_until = now
        if t is not None and t.task is not None and not t.task.done():
            t.task.cancel()
        # replay what the echo guard swallowed, so the interruption is heard whole
        recent = [(f, ts) for f, ts in li.ring if now - ts <= 2.0]
        self._drop_spec(li)
        li.seg.reset()
        li.barge.reset()
        utt = li.seg.prime(recent)
        if utt is not None:
            self.on_utterance(li.participant, utt)

    # ------------------------------------------------------------ turns
    def on_utterance(self, who: Participant, utt: Utterance) -> None:
        self.stats["utterances"] += 1
        prev = self.turn
        prefix_pcm = None
        prefix_text = None
        if prev is not None and prev.task is not None and not prev.task.done():
            if prev.speaker.pid == who.pid and not prev.spoke:
                LOG.info("%s kept talking before she answered; merging into one turn", who.user_id)
                self.stats["merged"] += 1
                prev.task.cancel()
                if prev.text is not None:
                    prefix_text = prev.text
                else:
                    prefix_pcm = prev.pcm
            elif prev.spoke:
                LOG.info("%s spoke over her reply; cancelling it", who.user_id)
                prev.task.cancel()
                self.sink.clear()
            else:
                LOG.info("turn in flight for %s; dropping utterance from %s",
                         prev.speaker.user_id, who.user_id)
                self.stats["dropped"] += 1
                return
        pcm = utt.pcm if prefix_pcm is None else np.concatenate(
            [prefix_pcm, np.zeros(STT_RATE // 5, np.int16), utt.pcm])
        spec = utt.spec
        if spec is not None and prefix_pcm is not None:
            # both pieces get transcribed together; the speculative one is moot
            if spec.task is not None and not spec.task.done():
                spec.task.cancel()
            spec = None
        st = _TurnState(who, pcm, spec=spec)
        st.task = asyncio.create_task(self.handle(st, utt, prefix_text))
        self.turn = st

    async def handle(self, st: _TurnState, utt: Utterance, prefix_text: str | None) -> None:
        tm = TurnTimer(utt.t_last_voice, self.clock)
        tm.mark("endpoint", utt.t_endpoint)
        tm.extra["speech_s"] = round(len(st.pcm) / STT_RATE, 2)
        try:
            sp = st.spec
            if sp is not None and sp.task is not None:
                # streaming STT: the transcript was asked for at the pause, so
                # it is ready (or nearly) by the time the endpoint fires
                tm.extra["stt_spec_at"] = round((sp.t_launch - tm.t0) * 1000)
                try:
                    text = await sp.task
                except asyncio.CancelledError:
                    if asyncio.current_task().cancelling():
                        raise
                    text = await self.stt.transcribe(st.pcm)     # lost it somehow: ask again
                tm.extra["stt_ms"] = round(((sp.t_done or self.clock()) - sp.t_launch) * 1000)
                if sp.backend:
                    tm.extra["stt_backend"] = sp.backend
            else:
                t_ask = self.clock()
                text = await self.stt.transcribe(st.pcm)
                tm.extra["stt_ms"] = round((self.clock() - t_ask) * 1000)
                if getattr(self.stt, "last_backend", ""):
                    tm.extra["stt_backend"] = self.stt.last_backend
            tm.mark("stt")
            if not text or is_non_speech(text) or is_whisper_repetition(text):
                if not prefix_text:
                    LOG.info("ignored non-speech transcript %r", said(text))
                    return
                # 2026-10-26: a merged turn whose new tail was noise used to drop
                # the earlier words too, so the request was never answered.
                LOG.info("ignored non-speech tail %r; answering the merged words", said(text))
                text = prefix_text
            elif prefix_text:
                text = f"{prefix_text} {text}"
            st.text = text
            if text == prefix_text:
                # the same words the cancelled turn already heard and noted;
                # the dedup guard would drop them and nothing would answer
                LOG.info("re-answering %s: %s", st.speaker.user_id, said(text))
            else:
                if self.dedup.is_duplicate(text):
                    return
                LOG.info("heard %s: %s", st.speaker.user_id, said(text))
                self.note(st.speaker.display or st.speaker.user_id or "someone", text)
            if is_leave_command(text, self.cfg.agent_names, one_to_one=len(self.humans()) <= 1):
                await self.speak_line(self.cfg.leave_line)
                self.end("voice command")
                return
            if not self.addressed(st.speaker, text):
                LOG.info("not addressed to her (group call); staying quiet")
                return
            if wants_escalation(text) and self.may_escalate(st.speaker):
                tm.extra["path"] = "hermes"
                await self.escalate(st, text, tm, ACK_LINES[0])
                return
            tm.extra["path"] = "voice"
            timed_out = await self.respond(st, text, tm)
            if timed_out:
                tm.extra["path"] = "voice->hermes"
                tm.extra["voice_deadline"] = timed_out
                self.stats["voice_deadline"] += 1
                await self.escalate(st, text, tm, self.cfg.voice_deadline_line)
                return
            if self.may_escalate(st.speaker) and reply_lacks_tool(" ".join(st.spoken)):
                tm.extra["path"] = "voice->hermes"
                await self.escalate(st, text, tm, "Let me get my full toolbelt on that. One moment.")
        except asyncio.CancelledError:
            tm.extra["cancelled"] = True
            raise
        finally:
            if st.spoken:
                self.note(self.agent_display, " ".join(st.spoken))
            if st.text is not None and "stt" in tm.marks and "first_audio" in tm.marks:
                self._record(st, tm)

    def _record(self, st: _TurnState, tm: TurnTimer) -> None:
        reply = " ".join(st.spoken)
        if tm.extra.get("cancelled"):
            reply += " (interrupted)"
        self.history += [{"role": "user", "content": st.text}, *st.exchange,
                         {"role": "assistant", "content": reply}]
        trim_history(self.history, HISTORY_MAX)
        r = tm.report()
        r["text"] = said(st.text[:80])
        self.timings.append(r)
        LOG.info("TURN %s", json.dumps(r))

    def _voice_deadlines(self) -> tuple[float | None, float | None]:
        """(first-sentence, total) voice-turn deadlines in seconds, or None.
        Only with Hermes behind her: the deadline hands the turn over."""
        if self.hermes is None or not getattr(self.hermes, "available", False):
            return None, None
        return (self.cfg.voice_deadline_s or None), (self.cfg.voice_max_s or None)

    async def respond(self, st: _TurnState, text: str, tm: TurnTimer) -> str | None:
        """Stream the voice brain's reply. Returns None, or which deadline
        cut the turn short ("first_sentence" / "total"); the brain's stream
        (in-flight LLM HTTP, MCP tool calls) is cancelled and closed then."""
        turn = Turn(speaker=st.speaker, text=text, history=list(self.history), mode=self.mode(),
                    is_operator=st.speaker.user_id in self.cfg.operators,
                    participants=self.s.participants())
        st.exchange = turn.exchange
        sentences: asyncio.Queue = asyncio.Queue()
        first_real = asyncio.Event()
        filler_task = asyncio.create_task(self._filler(text, first_real, tm)) if self.cfg.filler else None
        first_s, total_s = self._voice_deadlines() if self.may_escalate(st.speaker) else (None, None)
        loop = asyncio.get_running_loop()
        t_start = loop.time()
        timed_out: list[str] = []

        async def produce() -> None:
            ch = SentenceChunker(first_clause_chars=self.cfg.first_clause_chars)
            got_first = False
            limits = [x for x in (first_s, total_s) if x]
            gen = self.brain.stream(turn)
            try:
                async with asyncio.timeout_at(t_start + min(limits) if limits else None) as to:
                    try:
                        async for d in gen:
                            tm.mark("first_token")
                            for s in ch.push(strip_formatting(d)):
                                tm.mark("first_sentence")
                                if not got_first:
                                    got_first = True
                                    to.reschedule(t_start + total_s if total_s else None)
                                sentences.put_nowait(s)
                    finally:
                        # close the brain's generator here, so its httpx stream
                        # and any MCP call in flight are torn down, not left to GC
                        aclose = getattr(gen, "aclose", None)
                        if aclose is not None:
                            await aclose()
                for s in ch.flush(truncated=turn.truncated):
                    tm.mark("first_sentence")
                    sentences.put_nowait(s)
            except TimeoutError:
                if not to.expired():
                    raise
                timed_out.append("total" if got_first else "first_sentence")
                LOG.warning("voice turn hit its %s deadline after %.1f s; handing it to Hermes",
                            timed_out[0], loop.time() - t_start)
            finally:
                sentences.put_nowait(None)

        tool_log = getattr(self.brain, "tool_log", None)
        n_tools = len(tool_log) if tool_log is not None else 0
        producer = asyncio.create_task(produce())
        # One render at a time, in order: F5 serves requests concurrently but
        # not faster (two parallel renders measured 4.0 s each vs 1.2 + 2.4 s
        # back to back), so the sentence about to play must not share the GPU.
        # sink.write() only queues, so sentence N+1 renders while N plays.
        try:
            while True:
                s = await sentences.get()
                if s is None:
                    break
                pcm = await self._synth(s)
                tm.mark("first_tts")
                if pcm.size == 0:
                    continue
                if not first_real.is_set():
                    first_real.set()
                    if self.worship is not None:
                        self.worship.yield_audio()
                    if filler_task is not None:
                        await asyncio.shield(filler_task)
                st.spoke = True
                self.sink.write(pcm, on_start=lambda t: (tm.mark("first_audio", t), self._on_audio_start(t)))
                st.spoken.append(s)
                self._engaged[st.speaker.user_id] = self.clock()
            await self.sink.drained()
            self._echo_until = self.clock() + self.cfg.echo_tail_s
        finally:
            if tool_log is not None and len(tool_log) > n_tools:
                tm.extra["tools"] = tool_log[n_tools:]
            first_real.set()
            producer.cancel()
            if filler_task is not None and not filler_task.done():
                filler_task.cancel()
            # reap them: no orphan task keeps an HTTP stream or tool call alive
            await asyncio.wait([t for t in (producer, filler_task) if t is not None], timeout=5)
        return timed_out[0] if timed_out else None

    async def _synth(self, text: str) -> np.ndarray:
        t = self.clock()
        pcm = await self.tts.synth(text)
        LOG.info("tts %d chars -> %.1f s audio in %.0f ms", len(text), len(pcm) / CALL_RATE,
                 (self.clock() - t) * 1000)
        return pcm

    async def _filler(self, text: str, first_real: asyncio.Event, tm: TurnTimer | None = None) -> None:
        """Speak a short acknowledgement only if the reply is slow."""
        try:
            await asyncio.wait_for(first_real.wait(), self.cfg.filler_delay_s)
            return
        except asyncio.TimeoutError:
            pass
        phrase, bucket = self.brain.filler(text)
        if bucket == "chat" and not self.cfg.filler_on_chat:
            return
        pcm = self._filler_pcm.get(phrase)
        if pcm is None:
            return  # not pre-rendered yet; rendering now would only delay the reply
        LOG.info("filler (%s): %s", bucket, phrase)
        self.stats["fillers"] += 1
        self.sink.write(pcm, on_start=lambda t: tm.mark("filler_audio", t) if tm else None)
        # A tool turn (mail, calendar, narrate) can take many seconds: keep
        # reassuring so silence does not read as a dropped call.
        waiting = getattr(self.brain, "waiting_filler", None)
        while waiting is not None:
            try:
                await asyncio.wait_for(first_real.wait(), self.cfg.filler_repeat_s)
                return
            except asyncio.TimeoutError:
                pass
            pcm = self._filler_pcm.get(waiting())
            if pcm is not None and self.sink.buffered_s <= 0:
                self.sink.write(pcm)

    # ------------------------------------------------------------ escalation
    async def escalate(self, st: _TurnState, text: str, tm: TurnTimer, ack: str) -> None:
        """Say a short acknowledgement, hand the request to Hermes in the
        background, and speak its answer when it returns. The call stays live
        meanwhile: new turns go to the voice layer as usual."""
        pending = self._pending_match(text)
        if pending is not None:
            # Same question again, or "why is it taking so long?": no second
            # job queued behind the first (Hermes runs one turn per session).
            self.stats["hermes_not_stacked"] += 1
            tm.extra["path"] = "hermes-pending"
            LOG.info("hermes job already pending for %r; not escalating %r", said(pending[:80]), said(text[:80]))
            ack = self.cfg.hermes_pending_line
        else:
            self.stats["hermes"] += 1
        pcm = self._filler_pcm.get(ack)
        if pcm is None:
            pcm = await self.tts.synth(ack)
        if pcm.size:
            self.sink.write(pcm, on_start=lambda t: tm.mark("first_audio", t))
            st.spoken.append(ack)
        if "stt" in tm.marks:
            tm.mark("first_audio")   # ensure the turn is recorded even if playback is cut
        if pending is not None:
            return
        who = st.speaker.display or st.speaker.user_id
        history = list(self.history)
        others = self.listeners_besides(st.speaker)
        if others:
            LOG.info("hermes hand-off in a group call; also listening: %s", ", ".join(others))
        job = asyncio.create_task(self._hermes_job(text, who, history, self.clock(), others))
        self._hermes_jobs.add(job)
        self._hermes_pending[job] = text
        job.add_done_callback(self._hermes_jobs.discard)
        job.add_done_callback(lambda j: self._hermes_pending.pop(j, None))

    def _pending_match(self, text: str) -> str | None:
        """The request of a still-running Hermes job that `text` repeats or
        asks about; None when `text` is a new request."""
        running = [t for j, t in self._hermes_pending.items() if not j.done()]
        if not running:
            return None
        if is_progress_query(text):
            return running[-1]
        return next((t for t in running if same_request(t, text)), None)

    def _quiet(self) -> bool:
        """Nobody is speaking: she is not playing, no turn is being answered,
        no human is mid-utterance."""
        if self.sink.buffered_s > 0:
            return False
        if self.turn is not None and self.turn.task is not None and not self.turn.task.done():
            return False
        return not any(li.seg.in_speech for li in self.listeners.values())

    async def _say_progress(self) -> None:
        line = self.cfg.hermes_progress_line
        pcm = self._filler_pcm.get(line)
        if pcm is None:
            pcm = await self.tts.synth(line)
        if pcm.size and self._quiet() and not self._ended.is_set():
            LOG.info("hermes still working: %s", line)
            self.stats["hermes_progress"] += 1
            self.sink.write(pcm)

    async def _say_line(self, line: str) -> bool:
        pcm = self._filler_pcm.get(line)
        if pcm is None:
            pcm = await self.tts.synth(line)
        if pcm.size and self._quiet() and not self._ended.is_set():
            self.sink.write(pcm)
            return True
        return False

    async def _hermes_job(self, text: str, who: str, history: list[dict], t0: float,
                          others: list[str] | None = None) -> None:
        kw = {"audience": others} if others else {}
        ask = asyncio.create_task(self.hermes.ask(text, who=who, history=history, **kw))
        loop = asyncio.get_running_loop()
        started = loop.time()
        slow_at = started + self.cfg.hermes_slow_s if self.cfg.hermes_slow_s > 0 else float("inf")
        next_at = min(started + self.cfg.hermes_progress_first_s, slow_at)
        deferred = False        # told the caller the answer goes to the chat
        try:
            while not ask.done():
                await asyncio.wait({ask}, timeout=max(0.0, next_at - loop.time()))
                if ask.done():
                    break
                now = loop.time()
                if self._ended.is_set():
                    next_at = now + self.cfg.hermes_progress_every_s
                elif not deferred and now >= slow_at:
                    if self._quiet():
                        try:
                            await self._say_line(self.cfg.hermes_slow_line)
                        except Exception:
                            LOG.debug("slow line failed", exc_info=True)
                        deferred = True
                        self.stats["hermes_deferred"] += 1
                        LOG.warning("Hermes has not answered in %.0f s; the answer will go to the "
                                    "room chat: %s", now - started, text[:80])
                        next_at = float("inf")       # no more progress lines
                    else:
                        next_at = now + 2.0
                elif deferred:
                    next_at = float("inf")
                elif self._quiet():
                    try:
                        await self._say_progress()
                    except Exception:
                        LOG.debug("progress line failed", exc_info=True)
                    next_at = min(now + self.cfg.hermes_progress_every_s, slow_at)
                else:
                    next_at = now + 2.0      # someone is talking: try again shortly
                if next_at == float("inf"):
                    await asyncio.wait({ask})
            answer = ask.result()
        except asyncio.CancelledError:
            ask.cancel()
            LOG.warning("Hermes job abandoned: %s", said(text[:80]))
            if self._ended.is_set():
                await self._post_after_call(
                    f"{self.agent_display}: I was still working on \"{text[:200]}\" from the call "
                    f"and had to stop before Hermes answered.")
            raise
        took = self.clock() - t0
        LOG.info("HERMES %s", json.dumps({"ms": round(took * 1000), "request": said(text[:80]),
                                           "answer": said(answer[:160]), "after_call": self._ended.is_set(),
                                           "deferred": deferred}))
        self.history.append({"role": "assistant", "content": answer})
        self.note(self.agent_display, answer)
        if deferred:
            # Promised to the caller: the answer goes to the chat, with one
            # short spoken pointer if the call is still on.
            await self._post_after_call(
                f"{self.agent_display} (answer to \"{text[:200]}\" from the call): {answer}")
            if not self._ended.is_set():
                await self._wait_for_gap()
                try:
                    await self._say_line(self.cfg.hermes_posted_line)
                except Exception:
                    LOG.debug("posted line failed", exc_info=True)
            return
        await self._wait_for_gap()
        if not self._ended.is_set():
            speak = asyncio.create_task(self.speak_text(answer))
            ended = asyncio.create_task(self._ended.wait())
            await asyncio.wait({speak, ended}, return_when=asyncio.FIRST_COMPLETED)
            ended.cancel()
            if speak.done():
                return
            speak.cancel()                           # the call ended mid-answer
        await self._post_after_call(
            f"{self.agent_display} (answer to \"{text[:200]}\" from the call): {answer}")

    async def _wait_for_gap(self) -> None:
        """Up to 60 s for a gap: not speaking, no turn being answered."""
        for _ in range(300):
            if self._ended.is_set():
                return
            busy = self.sink.buffered_s > 0 or (self.turn is not None and self.turn.task is not None
                                                and not self.turn.task.done())
            if not busy:
                return
            await asyncio.sleep(0.2)

    async def _post_after_call(self, message: str) -> None:
        if self.post_after_call is None:
            LOG.warning("Hermes answer after the call had nowhere to go: %s", message[:160])
            return
        try:
            await self.post_after_call(message)
            LOG.info("posted a Hermes answer to the room chat after the call")
        except Exception as e:
            LOG.warning("Hermes answer not posted after the call: %s", e)

    async def finish_background(self, timeout: float | None = None) -> None:
        """After the call: let running Hermes jobs finish (their answers are
        posted to the room chat), up to `timeout` (default: Hermes' own
        request timeout plus a margin); then cancel what is left, with a trace."""
        jobs = [j for j in self._hermes_jobs if not j.done()]
        if not jobs:
            return
        if timeout is None:
            timeout = float(getattr(self.hermes, "timeout_s", 900.0)) + 30.0
        LOG.info("call over; waiting up to %.0f s for %d Hermes job(s)", timeout, len(jobs))
        _, pending = await asyncio.wait(jobs, timeout=timeout)
        for j in pending:
            j.cancel()
        if pending:
            await asyncio.wait(pending, timeout=10)

    async def speak_text(self, text: str) -> None:
        ch = SentenceChunker()
        for s in ch.push(strip_formatting(text)) + ch.flush():
            pcm = await self._synth(s)
            if pcm.size:
                self.sink.write(pcm)
        await self.sink.drained()
        self._echo_until = self.clock() + self.cfg.echo_tail_s

    async def prerender_fillers(self) -> None:
        phrases = list(self.brain.all_fillers())
        if self.hermes is not None and getattr(self.hermes, "available", False):
            phrases = list(ACK_LINES) + [self.cfg.voice_deadline_line, self.cfg.hermes_progress_line,
                                         self.cfg.hermes_slow_line, self.cfg.hermes_posted_line,
                                         self.cfg.hermes_pending_line] + phrases
        for phrase in phrases:
            # never compete with a live turn for the TTS GPU
            while self.turn is not None and self.turn.task is not None and not self.turn.task.done():
                await asyncio.sleep(0.2)
            try:
                pcm = await self.tts.synth(strip_formatting(phrase))
                if pcm.size:
                    self._filler_pcm[phrase] = pcm
            except Exception:
                LOG.debug("filler prerender failed for %r", phrase, exc_info=True)
        LOG.info("pre-rendered %d filler phrases", len(self._filler_pcm))
        if self.avatar is not None and self._filler_pcm:
            try:
                await self.avatar.prerender(list(self._filler_pcm.values()))
            except Exception:
                LOG.warning("filler lip pre-render failed", exc_info=True)

    async def speak_line(self, text: str) -> None:
        pcm = await self.tts.synth(text)
        if pcm.size:
            self.sink.write(pcm, on_start=self._on_audio_start)
            await self.sink.drained()

    # ------------------------------------------------------------ health
    def _is_muted(self, pid: str) -> bool:
        """The backend's word on whether this participant's audio is off."""
        f = getattr(self.s, "audio_muted", None)
        return bool(f(pid)) if f is not None else False

    def health_state(self, now: float) -> dict:
        """What a hang needs to be diagnosed from the log alone."""
        t = self.turn
        out = {"gated": self.gated(now), "echo_until_in_s": round(self._echo_until - now, 2),
               "sink_buffered_s": round(self.sink.buffered_s, 2),
               "turn_running": bool(t is not None and t.task is not None and not t.task.done()),
               "humans": [p.user_id or p.pid[:8] for p in self.humans()],
               "muted": [p.user_id or p.pid[:8] for p in self.humans() if self._is_muted(p.pid)],
               "hermes_pending": sum(1 for j in self._hermes_pending if not j.done()),
               "listeners": {}}
        for li in self.listeners.values():
            out["listeners"][li.participant.user_id or li.participant.pid[:8]] = {
                "frames": li.frames,
                "since_frame_s": round(now - li.t_frame, 1) if li.frames else None,
                "since_voiced_s": round(now - li.t_voiced, 1) if li.t_voiced else None,
                "peak_rms": round(li.hb_peak), "in_speech": li.seg.in_speech,
                "utterances": li.utterances, "spec_pending": li.spec is not None and li.spec.text is None}
        return out

    def check_health(self, now: float) -> None:
        """Called every second by the watchdog: catch a deaf or stuck loop,
        say so with the state, and heal what can be healed."""
        cfg = self.cfg
        # 1. the sink stopped draining (a sink that never empties gates every frame as echo)
        buf = self.sink.buffered_s
        last_buf, since = self._sink_seen
        if buf <= 0 or buf != last_buf:
            self._sink_seen = (buf, now)
        elif cfg.stall_s and now - since >= cfg.stall_s:
            LOG.warning("watchdog: sink not draining for %.0f s; clearing it: %s",
                        now - since, json.dumps(self.health_state(now)))
            self.stats["watchdog_sink"] += 1
            self.sink.clear()
            self._echo_until = now
            self._sink_seen = (0.0, now)
        # 2. an echo gate far past her last audio (it only ever runs echo_tail_s past it)
        if self.sink.buffered_s <= 0 and self._echo_until - now > cfg.echo_tail_s + 1.0:
            LOG.warning("watchdog: echo gate stuck %.1f s ahead with the sink idle; resetting",
                        self._echo_until - now)
            self.stats["watchdog_gate"] += 1
            self._echo_until = now
        # 3. per human: no inbound frames at all while not muted. Silence is
        # not a stall (frames keep coming, comfort noise included), and a
        # muted human sends nothing on purpose (iOS may even end the track).
        present = {p.pid: p for p in self.humans()}
        for pid in list(self._seen):
            if pid not in present:
                self._seen.pop(pid, None)
                self._resub.pop(pid, None)
                self._muted.pop(pid, None)
        for pid, p in present.items():
            li = self.listeners.get(pid)
            first = self._seen.setdefault(pid, now)
            if self._is_muted(pid):
                self._muted[pid] = now
                self._resub.pop(pid, None)
                continue
            last = li.t_frame if li is not None and li.frames else first
            if pid in self._muted:            # unmuted: the stall timer starts over
                last = max(last, self._muted[pid])
                if last > self._muted[pid] or now - last >= cfg.stall_s:
                    self._muted.pop(pid, None)  # audio is back, or it stalled anyway
            if cfg.stall_s and now - last >= cfg.stall_s:
                last_try, delay = self._resub.get(pid, (-float("inf"), cfg.resubscribe_every_s))
                if now - last_try >= delay:
                    LOG.warning("watchdog: no audio from %s for %.0f s; re-subscribing: %s",
                                p.user_id or pid[:8], now - last, json.dumps(self.health_state(now)))
                    self.stats["watchdog_resubscribe"] += 1
                    resub = getattr(self.s, "resubscribe", None)
                    if resub is not None:
                        try:
                            resub(pid, "no inbound audio")
                        except Exception:
                            LOG.warning("watchdog: resubscribe failed", exc_info=True)
                    nxt = cfg.resubscribe_every_s if last_try == -float("inf") else min(300.0, delay * 2)
                    self._resub[pid] = (now, nxt)
                continue
            self._resub.pop(pid, None)       # audio flows again: the next stall starts over
        # 4. heartbeat
        if cfg.heartbeat_s and now - self._hb_at >= cfg.heartbeat_s:
            self._hb_at = now
            LOG.info("HEARTBEAT %s", json.dumps(self.health_state(now)))
            for li in self.listeners.values():
                li.hb_peak = 0.0

    # ------------------------------------------------------------ main
    async def _watchdog(self) -> None:
        t_start = self.clock()
        alone_since: float | None = None
        while not self._ended.is_set():
            await asyncio.sleep(1.0)
            now = self.clock()
            try:
                self.check_health(now)
            except Exception:
                LOG.warning("watchdog: health check failed", exc_info=True)
            if self.humans():
                self._had_company = True
                alone_since = None
            elif self._had_company:
                alone_since = alone_since or now
                if now - alone_since >= self.cfg.alone_timeout_s:
                    self.end("everyone else left")
            elif now - t_start >= self.cfg.wait_join_s:
                self.end("nobody joined")
            if now - self._last_speech >= self.cfg.idle_timeout_s:
                self.end("idle")
            if now - t_start >= self.cfg.max_call_s:
                self.end("max call length")

    async def run(self, greeting: str | None = None) -> str:
        self._tasks.append(asyncio.create_task(self._watchdog()))
        self._tasks.append(asyncio.create_task(self.prerender_fillers()))
        greeted = False

        async def events() -> None:
            nonlocal greeted
            async for ev in self.s.events():
                if isinstance(ev, AudioTrack):
                    self._tasks.append(asyncio.create_task(self.pump(ev)))
                elif isinstance(ev, ParticipantJoined):
                    LOG.info("joined: %s", ev.participant.user_id)
                    p = ev.participant
                    if not p.is_agent:
                        self.joined.setdefault(p.user_id or p.pid, p.display or p.user_id or "a guest")
                    if greeting and not greeted and not ev.participant.is_agent:
                        greeted = True
                        self._tasks.append(asyncio.create_task(self._greet(greeting)))
                elif isinstance(ev, ParticipantLeft):
                    LOG.info("left: %s", ev.participant.user_id)
                    li = self.listeners.pop(ev.participant.pid, None)
                    t = self.turn
                    if li and t and t.speaker.pid == ev.participant.pid and t.task and not t.task.done():
                        t.task.cancel()
                elif isinstance(ev, CallEnded):
                    self.end(ev.reason or "call ended")
                    return
            self.end("event stream closed")

        self._tasks.append(asyncio.create_task(events()))
        try:
            await self._ended.wait()
        finally:
            t = self.turn
            if t and t.task and not t.task.done():
                t.task.cancel()
            for task in self._tasks:
                task.cancel()
            await asyncio.gather(*self._tasks, return_exceptions=True)
        return self.end_reason

    async def _greet(self, text: str) -> None:
        # wait for the joiner's subscription to us to start (P0: ~2 s hold)
        await asyncio.sleep(2.5)
        if self.turn is None:
            await self.speak_line(text)

    def summary(self) -> dict:
        out: dict = {"turns": len(self.timings), **self.stats}
        for st in TurnTimer.STAGES:
            vals = [t[st] for t in self.timings if st in t]
            if vals:
                out[f"median_{st}_ms"] = round(statistics.median(vals))
        return out
