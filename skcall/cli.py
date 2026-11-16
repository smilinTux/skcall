"""skcall command line.

  skcall join --room <token>     join the room's call and converse until it ends
                                 (exit 3, "already in call", if this agent is already in it)
  skcall watch --room <token>    join when someone posts "@<agent> join call" there
  skcall prompt [--search Q]     dry run: build the voice system prompt and print its
                                 size and section headers (never its content); with
                                 --search, run one search_memory and print hit titles
  skcall smoke [--tts] [--stt] [--lipsync] [--talk] [--turn] [--env-file PATH]...
                                 after a start: prove the services, Talk and the
                                 site's TURN work, one ok/FAIL line per check
                                 (no flag: all, an unset service URL is "not
                                 configured"); exit 0 only if none FAIL (skcall.smoke)

The agent is --agent (default $SKAGENT, else lumina); its name, nick, voice,
and site (Nextcloud URL, rooms, operators) come from skcall.agents and SKCALL_*.

Credentials: NEXTCLOUD_TALK_USERNAME / NEXTCLOUD_TALK_APP_PASSWORD from the
environment, or read (only those keys, plus API_SERVER_KEY for Hermes) from ~/.hermes/.env. Never printed.
Rooms: only tokens in SKCALL_ALLOWED_ROOMS (default: the agent profile's rooms) are used,
plus rooms that are just the agent and one operator (SKCALL_ALLOW_OPERATOR_DMS, default on).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time

from skcall import summary as call_summary
from skcall.privacy import hermes_enabled
from skcall.agents import ConfigError, profile_for
from skcall.core.loop import Conversation, LoopConfig
from skcall.instance import (ALREADY_IN_CALL, EXIT_ALREADY_IN_CALL, AlreadyInCall, InstanceLock,
                             valid_room)

LOG = logging.getLogger("skcall")



def _env_from_file(path: str, keys: tuple[str, ...]) -> None:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            k, sep, v = line.strip().partition("=")
            if sep and k in keys and not os.environ.get(k):
                os.environ[k] = v.strip().strip("\"'")


def _agent(agent: str | None) -> str:
    return agent or os.environ.get("SKAGENT") or "lumina"


def load_env(agent: str | None = None) -> None:
    agent = _agent(agent)
    _env_from_file(os.environ.get("SKCALL_ENV_FILE", "~/.hermes/.env"),
                   ("NEXTCLOUD_TALK_USERNAME", "NEXTCLOUD_TALK_APP_PASSWORD"))
    # The same voice-LLM legs skchat's LiveKit call uses (only SKVOICE_ LLM keys).
    _env_from_file(os.environ.get("SKCALL_VOICE_ENV", f"~/.config/skchat/webui-{agent}.env"),
                   ("SKVOICE_LLM_URL", "SKVOICE_MODEL", "SKVOICE_FALLBACK_URL",
                    "SKVOICE_FALLBACK_MODEL", "SKVOICE_MAX_TOKENS"))
    # The agent's own Hermes api_server key (loopback :8642), for escalated turns.
    if not os.environ.get("SKCALL_HERMES_KEY"):
        _env_from_file(os.environ.get("SKCALL_ENV_FILE", "~/.hermes/.env"), ("API_SERVER_KEY",))
        if os.environ.get("API_SERVER_KEY"):
            os.environ["SKCALL_HERMES_KEY"] = os.environ.pop("API_SERVER_KEY")


def _site_problem(a) -> str:
    """Why this agent must not join: no rooms and no operators means nothing
    it may do, and skcall has no built-in estate to fall back on."""
    prof = profile_for(a.agent)
    if not prof.rooms and not operators(a):
        return (f"no rooms or operators configured for {a.agent}: set SKCALL_ALLOWED_ROOMS "
                f"and/or SKCALL_OPERATORS in the site config (join.env)")
    return ""


def allowed_rooms(agent: str | None = None) -> tuple[str, ...]:
    return profile_for(_agent(agent)).rooms


def operators(a) -> tuple[str, ...]:
    return tuple(u for u in (getattr(a, "operators", "") or "").split(",") if u) or profile_for(a.agent).operators


async def check_room(a) -> int:
    """Dry check: would the agent be allowed into this room? Two OCS GETs as
    the agent (room + participants); no join, no call, no ringing."""
    from skcall.talk.access import room_access
    from skcall.talk.ocs import TalkOCS
    prof = profile_for(a.agent)
    async with TalkOCS() as o:
        d = await room_access(o, a.room, prof.rooms, operators(a), prof.operator_dms)
    print(f"{a.agent} {a.room}: {d.reason}")
    return 0 if d.allowed else 1


def loop_config(a) -> LoopConfig:
    c = LoopConfig()
    c.hangover_ms = a.hangover_ms
    c.spec_stt = not a.no_streaming_stt
    c.spec_silence_ms = a.spec_silence_ms
    c.vad_rms = a.vad_rms
    c.barge_in = not a.no_barge_in
    c.filler = not a.no_filler
    c.filler_delay_s = a.filler_delay
    prof = profile_for(a.agent)
    c.operators = operators(a)
    c.agent_names = prof.names
    c.wait_join_s = a.wait
    c.max_call_s = a.max_call
    return c


# aioice INFO lines worth keeping (the rest is per-check noise): which pair got
# nominated (host, srflx or TURN relay), ICE completed/failed, consent expiry,
# TURN allocation lifecycle. The 2026-10-25 media drop left no trace because
# all of these were filtered out.
ICE_MILESTONES = ("nominating pair", "ICE completed", "ICE failed", "Consent to send expired",
                  "TURN allocation", "TURN channel bound")


class _Milestones(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        msg = record.getMessage()
        return any(m in msg for m in ICE_MILESTONES)


def keep_ice_milestones() -> None:
    for name in ("aioice.ice", "aioice.turn"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        if not any(isinstance(f, _Milestones) for f in lg.filters):
            lg.addFilter(_Milestones())


def _truthy(v: str | None) -> bool:
    return (v or "").strip().lower() in ("1", "on", "true", "yes")


async def join(a) -> int:
    """Single-instance wrapper: one call per (agent, room) on this host."""
    if not valid_room(a.room):
        print(f"invalid room token {a.room!r}", file=sys.stderr)
        return 2
    if problem := _site_problem(a):
        print(problem, file=sys.stderr)
        return 2
    lock = InstanceLock(a.agent, a.room)
    if not lock.acquire():
        LOG.warning("%s: %s holds %s", ALREADY_IN_CALL, a.agent, lock.path)
        print(ALREADY_IN_CALL, flush=True)
        return EXIT_ALREADY_IN_CALL
    try:
        return await _join(a, lock)
    except AlreadyInCall as e:
        LOG.warning("%s", e)
        print(ALREADY_IN_CALL, flush=True)
        return EXIT_ALREADY_IN_CALL
    finally:
        lock.release()


async def _join(a, lock: InstanceLock | None = None) -> int:
    from skcall.brain.voice_engine import VoiceEngineBrain
    from skcall.speech import F5TTS, HedgedSTT
    from skcall.talk.backend import TalkSession

    from skcall.video import Avatar, VideoConfig
    vcfg = VideoConfig.from_env(a.agent)
    if a.no_video:
        vcfg.enabled = False
    if a.video_mode:
        vcfg.mode = a.video_mode
    avatar = Avatar.build(vcfg)
    prof = profile_for(a.agent)
    sess = TalkSession(a.room, nick=prof.nick, silent=not a.ring, allowed_rooms=prof.rooms,
                       operators=operators(a), operator_dms=prof.operator_dms,
                       agent_users=tuple(u for u in a.agent_users.split(",") if u), name="agent",
                       video=avatar.source if avatar is not None else None)
    stt, tts = HedgedSTT(), F5TTS(url=prof.tts_url, voice=prof.tts_voice, model=prof.tts_model)
    brain = VoiceEngineBrain(agent=a.agent, tools=not a.no_tools, mcp=not a.no_mcp)
    from skcall.brain.hermes import HermesBrain
    # X-Hermes-Session-Id selects the Hermes transcript; one per call, so every
    # escalation in this call continues the same Hermes conversation.
    hermes = None if a.no_hermes else HermesBrain(
        session_key=f"skcall:talk:{a.room}",
        session_id=f"skcall-talk-{a.room}-{time.strftime('%Y%m%dT%H%M%S')}")
    if hermes is not None and not hermes.available:
        LOG.warning("no Hermes API key; escalation disabled")
    if avatar is not None:
        avatar.start()
    conv = Conversation(sess, stt, tts, brain, loop_config(a), hermes=hermes, avatar=avatar)
    if avatar is not None:
        from skcall.worship import WorshipHost
        conv.worship = brain.convo = WorshipHost(avatar, sess.audio_out, session_id=a.room)
    conv.post_after_call = lambda msg: call_summary.post_summary(a.room, msg)
    t0 = time.monotonic()
    warm = asyncio.create_task(stt.warmup())
    reason = "error"
    tools_ready = None
    t_in_call: float | None = None
    try:
        await sess.open()
        if not sess.can_speak:
            LOG.warning("no publish-audio permission in %s: listening only", a.room)
        if not a.no_consent:
            await sess.post_chat(prof.consent_line if sess.can_speak else
                                 prof.consent_line + prof.listen_only_note)
        await sess.join_call()
        t_in_call = time.monotonic()
        LOG.info("in call after %.0f ms; STT warmup %.0f ms", (time.monotonic() - t0) * 1000, await warm)
        tools_ready = asyncio.create_task(brain.start())    # MCP servers spawn after the join
        reason = await conv.run(greeting=a.greeting or None)
    finally:
        if not warm.done():
            warm.cancel()
        if tools_ready is not None and not tools_ready.done():
            tools_ready.cancel()
        await sess.leave()
        if lock is not None:
            lock.release()      # out of the call: a new join may start while Hermes finishes
        await stt.aclose()
        await tts.aclose()
        await brain.aclose()
        if avatar is not None:
            await avatar.aclose()
        summary = {"video": avatar.summary() if avatar is not None else None,
                   "video_frames_sent": sess.video_out.frames_sent if sess.video_out is not None else 0,"end_reason": reason, **conv.summary(), "subscriptions": _sub_hold(sess),
                   "tools_available": len(brain.tool_names()), "tool_calls": brain.tool_log,
                   "hermes": hermes.log if hermes is not None else None}
        print("SUMMARY " + json.dumps(summary), flush=True)
        if getattr(sess, "connection_lost", False):
            await _post_lost_note(a.room, sess.lost_connection_note())
        if t_in_call is not None and call_summary.enabled():
            await _post_call_summary(a, conv, time.monotonic() - t_in_call)
        if hermes is not None:
            # A Hermes job still running when the call ended finishes in the
            # background (up to its timeout) and posts its answer to the room.
            try:
                await conv.finish_background()
            finally:
                await hermes.aclose()
    return 0


async def _post_lost_note(room: str, note: str) -> None:
    """Tell the room why she left (the media could not be recovered)."""
    try:
        await call_summary.post_summary(room, note)
        LOG.info("posted lost-connection note to %s", room)
    except Exception as e:
        LOG.warning("lost-connection note not posted: %s", e)


async def _post_call_summary(a, conv, duration_s: float) -> None:
    """Who joined, how long, what was discussed: posted to the room's chat.
    The transcript stays in memory; it is dropped with the process."""
    try:
        msg = await call_summary.build_summary(conv.agent_display, list(conv.joined.values()),
                                               duration_s, conv.transcript)
        if msg:
            await call_summary.post_summary(a.room, msg)
            LOG.info("posted call summary to %s", a.room)
    except Exception as e:
        LOG.warning("call summary not posted: %s", e)


def _sub_hold(sess) -> list[dict]:
    out = []
    for sid, t in sess.sub_timing.items():
        if "first_frame" in t and "connected" in t:
            out.append({"connect_to_first_audio_ms": round((t["first_frame"] - t["connected"]) * 1000),
                        "requestoffer_to_first_audio_ms": round(
                            (t["first_frame"] - t.get("requestoffer", t["offer"])) * 1000)})
    return out


async def watch(a) -> int:
    """Read-only poll of one room's chat as the agent user. Joins the call when
    a message mentions the agent and says 'join call'; ignores everything else."""
    import re

    from skcall.talk.access import room_access
    from skcall.talk.ocs import SPREED, TalkOCS
    if problem := _site_problem(a):
        print(problem, file=sys.stderr)
        return 2
    pat = re.compile(rf"@?{re.escape(a.agent)}\b.*\bjoin\b.*\bcall\b", re.I)
    prof = profile_for(a.agent)
    async with TalkOCS() as o:
        d = await room_access(o, a.room, prof.rooms, operators(a), prof.operator_dms)
        if not d.allowed:
            raise SystemExit(f"room {a.room}: {d.reason}")
        last = (await o.get_room(a.room)).get("lastMessage", {}).get("id", 0)
        LOG.info("watching %s from message %s", a.room, last)
        while True:
            msgs = await o.request("GET", f"{SPREED}/v1/chat/{a.room}", ok=(200, 304),
                                   params={"lookIntoFuture": 1, "lastKnownMessageId": last,
                                           "timeout": 15, "limit": 50})
            for m in msgs or []:
                last = max(last, int(m.get("id", 0)))
                if m.get("actorId") == o.user or m.get("systemMessage"):
                    continue
                if pat.search(m.get("message", "")) or any(
                        p.get("type") == "user" and p.get("id") == a.agent
                        for p in (m.get("messageParameters") or {}).values()
                        if isinstance(p, dict)) and re.search(r"\bjoin\b.*\bcall\b", m.get("message", ""), re.I):
                    LOG.info("summoned by %s", m.get("actorId"))
                    await join(a)
                    if a.once:
                        return 0


async def prompt_dry_run(a) -> int:
    """Build the brain as a call would and report the system prompt's shape."""
    from skcall.brain.voice_engine import VoiceEngineBrain
    brain = VoiceEngineBrain(agent=a.agent, mcp=False)
    try:
        print(f"agent {a.agent}  model {brain.cfg.model}  tools {', '.join(brain.tool_names()) or 'none'}")
        print(f"memory context: {json.dumps(brain.context_report)}")
        for mode in ("private", "group"):
            text = brain.system_prompt(mode, ["Operator"])
            print(f"{mode}: {len(text)} chars, ~{round(len(text) / 2.7)} tokens (est.)")
            for para in text.split("\n\n"):
                head = para.strip().splitlines()[0] if para.strip() else ""
                print(f"  [{len(para):5d}] {' '.join(head.split()[:6])[:48]}")
        if a.search:
            t0 = time.monotonic()
            await brain.memory.warm()
            res = await brain.memory.search_tool({"query": a.search, "limit": 5}, {"agent": a.agent})
            print(f"search_memory backend {brain.memory.backend}, {(time.monotonic() - t0) * 1000:.0f} ms "
                  "(store open + search)")
            for line in res.splitlines():
                print("  " + line[:60])
    finally:
        await brain.aclose()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="skcall")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prompt", help="dry run: voice system prompt size and sections")
    pp.add_argument("--agent", default=os.environ.get("SKAGENT", "lumina"))
    pp.add_argument("--search", default="", help="also run one search_memory with this query")
    pp.add_argument("-v", "--verbose", action="store_true")
    cp = sub.add_parser("check-room", help="dry check: is this room allowed (listed, operator DM or two-member group)?")
    cp.add_argument("--room", required=True)
    cp.add_argument("--agent", default=os.environ.get("SKAGENT", "lumina"))
    cp.add_argument("--operators", default="")
    cp.add_argument("-v", "--verbose", action="store_true")
    sp = sub.add_parser("smoke", help="after a start: services, Talk and the site's TURN work")
    sp.add_argument("--agent", default=os.environ.get("SKAGENT", "lumina"))
    for check in ("tts", "stt", "lipsync", "talk", "turn"):
        sp.add_argument(f"--{check}", action="store_true", help=f"run the {check} check")
    sp.add_argument("--env-file", action="append", default=[], metavar="PATH",
                    help="settings file to read (repeatable; replaces the default "
                         "~/.config/skcall/skcall.env and join.env; must exist)")
    sp.add_argument("-v", "--verbose", action="store_true")
    for name in ("join", "watch"):
        p = sub.add_parser(name)
        p.add_argument("--room", required=True)
        p.add_argument("--agent", default=os.environ.get("SKAGENT", "lumina"))
        p.add_argument("--ring", action="store_true", default=_truthy(os.environ.get("SKCALL_RING")),
                       help="join non-silently (notifies the room); default from SKCALL_RING")
        p.add_argument("--no-consent", action="store_true", help="skip the chat consent line (self-tests)")
        p.add_argument("--no-barge-in", action="store_true")
        p.add_argument("--no-filler", action="store_true")
        p.add_argument("--no-tools", action="store_true", help="talk without her voice tools")
        p.add_argument("--no-hermes", action="store_true", default=not hermes_enabled(),
                       help="never escalate turns to Hermes (default from SKCALL_HERMES)")
        p.add_argument("--no-mcp", action="store_true", help="built-in voice tools only, no MCP servers")
        p.add_argument("--no-video", action="store_true", help="audio only (also SKCALL_VIDEO=0)")
        p.add_argument("--video-mode", choices=("static", "animated"), default="",
                       help="static portrait, or idle loop + lip-sync (default from SKCALL_VIDEO_MODE)")
        p.add_argument("--filler-delay", type=float, default=float(os.environ.get("SKCALL_FILLER_DELAY_S", "0.3")),
                       help="speak an acknowledgement if no reply audio within this many s of the transcript")
        p.add_argument("--hangover-ms", type=int, default=int(os.environ.get("SKCALL_HANGOVER_MS", "450")))
        p.add_argument("--no-streaming-stt", action="store_true",
                       help="transcribe only after the endpoint (no speculative STT at pauses)")
        p.add_argument("--spec-silence-ms", type=int, default=int(os.environ.get("SKCALL_SPEC_SILENCE_MS", "150")),
                       help="streaming STT: transcribe the open utterance after this much silence")
        p.add_argument("--vad-rms", type=float, default=float(os.environ.get("SKCALL_VAD_RMS", "500")))
        p.add_argument("--operators", default="",
                       help="comma list of operator user ids (default: SKCALL_OPERATORS, else the agent profile's)")
        p.add_argument("--agent-users", default=os.environ.get("SKCALL_AGENT_USERS", ""))
        p.add_argument("--wait", type=float, default=300, help="leave if nobody joins within this")
        p.add_argument("--greeting", default=os.environ.get("SKCALL_GREETING", ""),
                       help="say this when the first person joins; default from SKCALL_GREETING")
        p.add_argument("--max-call", type=float, default=float(os.environ.get("SKCALL_MAX_CALL_S", "7200")),
                       help="leave after this many seconds in the call")
        p.add_argument("--once", action="store_true", help="watch: exit after one call")
        p.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("aioice", "aiortc", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    keep_ice_milestones()
    if a.cmd == "smoke":
        from skcall import smoke
        if not a.verbose:
            logging.getLogger().setLevel(logging.WARNING)
        try:
            smoke.load_site_env(a.env_file)
        except smoke.EnvFileMissing as e:
            print(f"FAIL env-file {e}: no such file (named with --env-file)", flush=True)
            return 1
        load_env(a.agent)
        return asyncio.run(smoke.run([c for c in smoke.CHECKS if getattr(a, c)], a.agent))
    load_env(a.agent)
    try:
        if a.cmd == "prompt":
            return asyncio.run(prompt_dry_run(a))
        if a.cmd == "check-room":
            return asyncio.run(check_room(a))
        return asyncio.run(join(a) if a.cmd == "join" else watch(a))
    except ConfigError as e:
        print(f"skcall: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
