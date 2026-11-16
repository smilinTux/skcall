"""Automated two-way conversation test through a real Talk call (no human).

Two sessions of the agent user join the spike room's call SILENTLY (nobody is
notified):
  agent  runs the real skcall Conversation (whisper, voice-engine brain, F5)
  human  a scripted caller: plays pre-rendered F5 questions, listens to the
         agent's answers through Janus, interrupts one answer (barge-in), and
         finally says "Lumina, please leave the call".

Reports the agent's per-turn stage timings and the caller-side latency: from
the end of each question's audio to the first agent audio heard back.

Usage: scripts/run.sh scripts/selftest_loopback.py [--room room0101]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import statistics
import sys
import time

import numpy as np

from skcall.brain.voice_engine import VoiceEngineBrain
from skcall.cli import allowed_rooms, load_env
from skcall.core.audio import CALL_RATE, STT_RATE, pcm_to_wav, resample, rms
from skcall.core.loop import Conversation, LoopConfig
from skcall.core.session import AudioTrack
from skcall.speech import F5TTS, HedgedSTT, WhisperCppSTT
from skcall.talk.backend import TalkSession

LOG = logging.getLogger("selftest")

SCRIPT = [
    ("ask", "Hi Lumina, can you hear me okay?"),
    ("ask", "What is one thing you love about late night conversations?"),
    ("interrupt", "Tell me a story about a fox who finds a lantern in the woods."),
    ("ask", "Okay, sorry, stop there. What time of day do you like best?"),
    ("hermes", "Can you tell me your skcapstone status?"),
    ("ask", "Lumina, please leave the call."),
]
# --more: extra plain chat turns before the leave line, for per-stage medians
# over 10+ turns in one run (no tools, no Hermes).
MORE = [
    "What's your favorite color, and why?",
    "Do you like rainy days?",
    "If you could visit any city tonight, which one would it be?",
    "Tell me one word that describes your mood right now.",
    "What kind of music do you like late at night?",
    "Do you prefer mornings or evenings?",
    "What's a small thing that always makes you smile?",
    "Would you rather read a book or watch a movie?",
]


class Ear:
    """The scripted caller's ear on the agent's stream."""

    def __init__(self):
        self.chunks: list[np.ndarray] = []
        self.loud_t: list[float] = []
        self.last_loud = 0.0

    async def listen(self, track: AudioTrack):
        async for x in track.frames:
            now = time.monotonic()
            self.chunks.append(x)
            if rms(x) > 300:
                self.loud_t.append(now)
                self.last_loud = now

    def first_loud_after(self, t: float) -> float | None:
        for x in self.loud_t:
            if x > t:
                return x
        return None


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", default="room0101")
    ap.add_argument("--hangover-ms", type=int, default=450)
    ap.add_argument("--stt", choices=("hedged", "cpp"), default="hedged",
                    help="hedged: SKCALL_STT_BACKENDS in order; cpp: whisper.cpp at SKCALL_STT_URL only")
    ap.add_argument("--no-streaming-stt", action="store_true", help="the pre-streaming turn path")
    ap.add_argument("--spec-silence-ms", type=int, default=150)
    ap.add_argument("--first-clause-chars", type=int, default=30)
    ap.add_argument("--more", type=int, default=0, help="extra chat turns (up to %d)" % len(MORE))
    ap.add_argument("--no-hermes", action="store_true", help="skip the Hermes escalation turn")
    ap.add_argument("--out", default=os.environ.get("SKCALL_SCRATCH", "/tmp/skcall-selftest"))
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    for noisy in ("aioice", "aiortc", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    load_env()
    os.makedirs(a.out, exist_ok=True)
    rooms = allowed_rooms()

    script = [x for x in SCRIPT if not (a.no_hermes and x[0] == "hermes")]
    script[-1:-1] = [("ask", q) for q in MORE[:a.more]]
    tts = F5TTS()
    phrases = {}
    for _, text in script:
        pcm = await tts.synth(text)
        loud = np.nonzero(np.abs(pcm.astype(np.int32)) > 1000)[0]
        # trim F5's leading/trailing silence so 'end of question' is the last word
        phrases[text] = pcm[max(0, loud[0] - 2400): loud[-1] + 2400] if loud.size else pcm
    agent = TalkSession(a.room, silent=True, allowed_rooms=rooms, name="agent")
    human = TalkSession(a.room, silent=True, allowed_rooms=rooms, name="human", nick="Caller")
    stt = HedgedSTT() if a.stt == "hedged" else WhisperCppSTT()
    brain = VoiceEngineBrain()
    cfg = LoopConfig(operators=("alice", "lumina"), hangover_ms=a.hangover_ms,
                     spec_stt=not a.no_streaming_stt, spec_silence_ms=a.spec_silence_ms,
                     first_clause_chars=a.first_clause_chars)
    from skcall.brain.hermes import HermesBrain
    hermes = None if a.no_hermes else HermesBrain(session_key=f"skcall:talk:{a.room}")
    conv = Conversation(agent, stt, F5TTS(), brain, cfg, hermes=hermes)
    ear = Ear()
    # transport check: when each agent chunk reached the encoder vs when the
    # caller first heard sound after it (same process, same clock)
    agent_starts: list[float] = []
    _write = agent.audio_out.write

    def write(pcm, on_start=None):
        def cb(t):
            agent_starts.append(t)
            if on_start:
                on_start(t)
        _write(pcm, on_start=cb)
    agent.audio_out.write = write
    caller: list[dict] = []
    res: dict = {}
    try:
        warm = asyncio.create_task(stt.warmup())
        # all sessions join the REST room before any joins the call (409 rule)
        await agent.open()
        await human.open()
        await agent.join_call()
        await human.join_call()
        res["stt_warmup_ms"] = round(await warm)
        await brain.start()
        res["tools_available"] = brain.tool_names()
        run = asyncio.create_task(conv.run())

        async def human_events():
            async for ev in human.events():
                if isinstance(ev, AudioTrack):
                    asyncio.ensure_future(ear.listen(ev))
        hev = asyncio.create_task(human_events())
        await asyncio.sleep(6)          # both subscriptions past the ~2 s hold
        for kind, text in script:
            if run.done():
                break
            started = asyncio.get_running_loop().create_future()
            pcm = phrases[text]
            human.audio_out.write(pcm, on_start=lambda t, f=started: f.done() or f.set_result(t))
            t_start = await started
            t_end = t_start + len(pcm) / CALL_RATE
            await human.audio_out.drained()
            rec = {"q": text, "kind": kind}
            # wait for her answer to start
            t0 = time.monotonic()
            while ear.first_loud_after(t_end) is None and time.monotonic() - t0 < 25 and not run.done():
                await asyncio.sleep(0.05)
            heard = ear.first_loud_after(t_end)
            rec["caller_first_audio_ms"] = round((heard - t_end) * 1000) if heard else None
            if kind == "interrupt" and heard:
                await asyncio.sleep(3.0)            # let her get into the story
                rec["interrupted_after_s"] = 3.0
            elif kind == "hermes" and heard:
                # acknowledgement first; then Hermes' answer after a gap
                while not run.done() and time.monotonic() - ear.last_loud < 3.0:
                    await asyncio.sleep(0.05)
                gap_start = time.monotonic()
                while not run.done() and ear.first_loud_after(gap_start) is None and time.monotonic() - gap_start < 120:
                    await asyncio.sleep(0.1)
                ans = ear.first_loud_after(gap_start)
                rec["hermes_answer_heard_ms"] = round((ans - t_end) * 1000) if ans else None
                while not run.done() and time.monotonic() - ear.last_loud < 3.0:
                    await asyncio.sleep(0.05)
            else:
                # wait until she has been quiet for 3 s (the gap between filler and answer is not the end)
                while not run.done() and time.monotonic() - max(ear.last_loud, t_end) < 3.0:
                    await asyncio.sleep(0.05)
            caller.append(rec)
            LOG.info("CALLER %s", json.dumps(rec))
        try:
            await asyncio.wait_for(run, 20)
        except asyncio.TimeoutError:
            conv.end("selftest timeout")
            await run
        hev.cancel()
    finally:
        await agent.leave()
        await human.leave()
        await stt.aclose()
        await brain.aclose()
        if hermes is not None:
            await hermes.aclose()
        await tts.aclose()
    res["end_reason"] = conv.end_reason
    res["agent"] = conv.summary()
    res["turns"] = conv.timings
    res["caller"] = caller
    res["config"] = {"stt": a.stt, "streaming_stt": not a.no_streaming_stt,
                     "spec_silence_ms": a.spec_silence_ms, "hangover_ms": a.hangover_ms,
                     "first_clause_chars": a.first_clause_chars}
    res["stt_log"] = getattr(stt, "log", None)
    for key in ("stt_ms", "stt_spec_at"):   # not median_stt_ms: that is the stage
        vals = [t[key] for t in conv.timings if key in t]
        if vals:
            res["agent"][f"median_{key}_raw"] = round(statistics.median(vals))
    res["hermes"] = hermes.log if hermes is not None else None
    res["tool_calls"] = brain.tool_log
    vals = [c["caller_first_audio_ms"] for c in caller if c.get("caller_first_audio_ms")]
    if vals:
        res["caller_first_audio_median_ms"] = round(statistics.median(vals))
    lat = []
    for t in agent_starts:
        h = ear.first_loud_after(t)
        if h is not None and h - t < 3:
            lat.append(round((h - t) * 1000))
    res["agent_to_caller_transport_ms"] = lat
    if lat:
        res["agent_to_caller_transport_median_ms"] = round(statistics.median(lat))
    res["agent_subscription_hold"] = [
        round((t["first_frame"] - t["connected"]) * 1000) for t in agent.sub_timing.values()
        if "first_frame" in t and "connected" in t]
    res["caller_subscription_hold"] = [
        round((t["first_frame"] - t["connected"]) * 1000) for t in human.sub_timing.values()
        if "first_frame" in t and "connected" in t]
    if ear.chunks:
        heard = np.concatenate(ear.chunks)
        path = os.path.join(a.out, "agent-heard.wav")
        with open(path, "wb") as f:
            f.write(pcm_to_wav(heard, CALL_RATE))
        s2 = WhisperCppSTT(audio_ctx="0")
        res["agent_said_transcript"] = await s2.transcribe(resample(heard[-CALL_RATE * 28:], CALL_RATE, STT_RATE))
        await s2.aclose()
    print("RESULT " + json.dumps(res), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
