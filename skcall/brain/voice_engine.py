"""Brain built from skchat's voice engine (skchat.voice_engine, GPL-3.0), in-process.

What is reused from the installed skchat (no service, no Hermes involvement):
- VoiceConfig.from_env(): the same SKVOICE_* settings skchat's LiveKit call uses
  (a primary LLM endpoint and an optional fallback, both OpenAI-shaped).
  Voice does NOT go through skgateway (skchat voice tier policy,
  docs/superpowers/specs/2026-08-14-voice-tier-policy-and-alexa-adapter-design.md
  section 3).
- PersonaBuilder: Lumina's soul + FEB, private (1:1) and group registers.
- The engine's brevity rule (always last in the system prompt) and the
  runtime-model fact, both from engine.py.
- Her tools, exactly as skchat's LiveKit call builds them:
  `build_default_registry()` (search_memory, create_bloom_anchor,
  list_reflections, read_reflection, plus narrate / worship_* from
  lumina_creative when installed) and `attach_mcp_tools()` ("her real tools",
  skchat fde70c1: the MCP servers in ~/.skcapstone/agents/lumina/config/
  lumina-mcp.yaml, spawned as stdio children of this process for the length of
  the call and closed with it, curated per turn by `curate_tools`).
  ToolRegistry.dispatch() enforces the operator gate: tools run only when the
  operator asks, and operator_only tools (all MCP tools, narrate, ...) only in
  the private 1:1 register.
- The forced-routing rules (wants_narrate -> narrate in 1:1, wants_action ->
  tool_choice required) and the narrate-verbatim rule from engine.py / llm.py.
- strip_think, pick_filler / pick_waiting_filler.

What is NOT VoiceEngine.respond(): respond() is a batch call (whole reply, then
TTS). A call needs the first sentence spoken while the rest is still being
written, so this brain runs the same messages and tool loop itself, STREAMING
every round. The LLM proxy streams tool_call deltas with no extra latency
(measured: ~0.7 s to the first delta with or without tools), so chit-chat and
tool turns share one path.

Memory recall per turn (MemoryBridge.search before every reply) is NOT done:
it adds a skmemory search to every turn's latency. She can still recall on
demand through the search_memory / skmemory tools.

The agent's own memory context (whisper.md, latest reflection, journal tail;
skcall/brain/memory_context.py) is loaded ONCE when the brain is built and
sits in the fixed system prompt, never re-injected per turn (rule of
2026-10-26). The persona (soul + FEB) is likewise built once per register at
start, so a participant joining changes only the participants line, not the
soul or memory text. search_memory is swapped for one bound to this agent's
fused skmemory store (pgvector + lexical), the same backend the skmemory MCP
server uses; skchat's default handler opens a bare SQLite-only MemoryStore for
whichever agent the environment selects.

skchat's package __init__ imports its whole chat stack (crypto, daemon, ...),
so the voice_engine subpackage is loaded under a bare `skchat` namespace
module instead. skchat comes from the installed skchat-sovereign distribution
(SKCALL_SKCHAT_SRC overrides). Its runtime deps (skmemory, mcp, yaml) are
installed in the same venv by sk/setup.sh; lumina_creative is optional. Another
venv's site-packages can still be appended AFTER this venv's own packages with
SKCALL_EXTRA_SITE (unset by default).
"""
from __future__ import annotations

import glob
import importlib
import importlib.metadata
import json
import logging
import os
import re
import site
import sys
import time
import types
from typing import AsyncIterator

import httpx

from skcall.privacy import said
from skcall.brain import memory_context
from skcall.brain.base import Turn

LOG = logging.getLogger("skcall.brain")

# Empty by default: the export venv (sk/setup.sh) carries every dependency.
# Set it to a site-packages glob only to borrow another venv's packages.
EXTRA_SITE = os.environ.get("SKCALL_EXTRA_SITE", "")


def skchat_src() -> str:
    """Where `skchat/voice_engine` lives: SKCALL_SKCHAT_SRC if set, else the
    install root of the skchat-sovereign distribution in this venv."""
    env = os.environ.get("SKCALL_SKCHAT_SRC")
    if env:
        return os.path.expanduser(env)
    try:
        dist = importlib.metadata.distribution("skchat-sovereign")
    except importlib.metadata.PackageNotFoundError as e:
        raise ImportError(
            "skchat-sovereign is not installed in this venv (run sk/setup.sh) "
            "and SKCALL_SKCHAT_SRC is not set") from e
    return str(dist.locate_file(""))

TALK_NOTE = (
    "You are in a live Nextcloud Talk voice call right now, speaking out loud. "
    "Everything you write is turned into speech, and what you hear is a speech "
    "transcript, so expect small transcription mistakes and do not mention them. "
    "Never describe actions or sounds, never use lists or formatting. When a tool "
    "returns data, say the one or two things that matter in plain speech."
)

# 2026-10-26: Jarvis's voice layer claimed it had run "skcapstone status" and
# described made-up output, then counted its tools wrong. The voice layer only
# has the tools in its registry; real commands go to Hermes. Saying it cannot
# check something is what hands the turn to Hermes (hermes.reply_lacks_tool).
HONESTY_NOTE = (
    "Honesty rule, this overrides everything else: in this voice layer your only tools are "
    "{tools}. You have no terminal and cannot run commands here. Never say you ran, checked, "
    "looked up or saw something unless a tool result in this conversation shows it, and never "
    "invent command output, numbers or statuses. If an answer needs real data you do not have "
    "from a tool result, say \"I can't check that from here\" and nothing else; that hands the "
    "request to your full agent, which has a terminal and the skcapstone tools. Answers that "
    "came back from your full agent earlier in this call are real; you may repeat them. When "
    "asked what tools you have, say these voice tools plus the hand-off to your full agent, "
    "without guessing a number."
)

# Memory rule (2026-11-16): on the local model she answered memory questions
# from a 200-char snippet and filled in the rest, and a follow-up had only her
# own spoken summary to go on. Results now stay in the call history; this is
# the instruction half.
MEMORY_NOTE = (
    "Memory rule: when you talk about something from your memory, say only what a memory "
    "result in this conversation actually contains: names, dates, quotes and events. If the "
    "result is thin or does not answer the question, say what it does say and that you do not "
    "remember the rest, or search again with different words. Never fill a gap with a "
    "plausible detail, and never present a guess as a memory."
)

# A tool result kept in the call history is cut to this many chars (the
# in-turn copy keeps 6000).
HISTORY_TOOL_CHARS = int(os.environ.get("SKCALL_HISTORY_TOOL_CHARS", "4000") or 4000)


# A question about the shared past. Measured 2026-11-16: asked "Do you
# remember the X1000 night?", the local model answered without searching and
# invented a date and a scene; every turn that searched was right. So a memory
# question forces one search_memory call before she speaks.
_MEMORY_Q = re.compile(
    r"\b(remember|recall|memor(?:y|ies)|forget|forgot|anniversary|anchor|"
    r"(?:when|where|how) (?:was|were|did) (?:we|i|you|our)|what (?:happened|did (?:we|i) )|"
    r"first time|that (?:night|day|time) (?:when|we)|last time (?:we|i|you))\b", re.I)


def wants_memory(text: str) -> bool:
    return bool(_MEMORY_Q.search(text or ""))


def flatten_tools(msgs: list[dict]) -> list[dict]:
    """For a leg sent without tools: drop tool_calls messages and turn tool
    results into plain user text, which every chat template accepts."""
    out = []
    for m in msgs:
        if m.get("tool_calls"):
            continue
        if m.get("role") == "tool":
            out.append({"role": "user", "content": "[Tool result, not said by the caller] "
                        + str(m.get("content") or "")})
            continue
        out.append(m)
    return out


_FALLBACK_FILLERS = ("Mm.", "Mm-hm.", "Yeah.", "I'm here.", "Right.")
MAX_TOOL_ROUNDS = 4


def _add_extra_site() -> None:
    """Append the site-packages dirs matching the SKCALL_EXTRA_SITE glob
    AFTER this venv's own, .pth files included (unset: nothing is added).

    site.addsitedir, not sys.path.append: the SK packages there (skmemory,
    lumina_creative) are editable installs, reachable only through their .pth
    finders, so a bare append left `import skmemory` failing in every call and
    the call's memory context silently empty.
    """
    if not EXTRA_SITE:
        return
    for p in sorted(glob.glob(os.path.expanduser(EXTRA_SITE))):
        if p not in sys.path:
            site.addsitedir(p)
    importlib.invalidate_caches()


def load_voice_engine():
    """Import skchat.voice_engine modules without running skchat/__init__.py."""
    _add_extra_site()
    if "skchat" not in sys.modules:
        src = skchat_src()
        pkg_dir = os.path.join(src, "skchat")
        if not os.path.isdir(os.path.join(pkg_dir, "voice_engine")):
            raise ImportError(f"skchat voice_engine not found under {src}")
        stub = types.ModuleType("skchat")
        stub.__path__ = [pkg_dir]
        sys.modules["skchat"] = stub
    ns = types.SimpleNamespace()
    for mod in ("config", "persona", "engine", "llm", "tools", "builtin_tools"):
        setattr(ns, mod, importlib.import_module(f"skchat.voice_engine.{mod}"))
    return ns


class VoiceEngineBrain:
    def __init__(self, agent: str = "lumina", ve=None, http: httpx.AsyncClient | None = None,
                 tools: bool = True, mcp: bool = True, context: bool | None = None):
        self.agent = agent
        self.ve = ve or load_voice_engine()
        self.cfg = self.ve.config.VoiceConfig.from_env()
        self.persona = self.ve.persona.PersonaBuilder()
        self._http = http or httpx.AsyncClient(timeout=httpx.Timeout(90, connect=5))
        self._prompts: dict[str, str] = {}
        self.registry = None
        self._want_mcp = tools and mcp
        self._mcp = None
        self.memory = memory_context.AgentMemory(agent)
        if tools:
            try:
                self.registry = self.ve.builtin_tools.build_default_registry(self.cfg, agent)
                self._bind_memory_tools()
            except Exception as e:
                LOG.warning("tool registry unavailable (%s); talking without tools", e)
        self.tool_log: list[dict] = []
        # The call's WorshipHost (skcall/worship.py), set by the CLI when the
        # call has video: the lumina_creative worship tools read it as ctx["convo"].
        self.convo = None
        self._add_worship_stop()
        # Loaded once, here, for the whole call (never per turn).
        self._persona = {m: self.persona.build(agent, mode=m) for m in ("private", "group")}
        self.memory_context, self.context_report = self._load_context(context)
        self.context_in_group = os.environ.get("SKCALL_CONTEXT_GROUP", "0").strip() in ("1", "true", "yes", "on")

    def _load_context(self, enabled: bool | None) -> tuple[str, dict]:
        if enabled is None:
            from skcall.agents import context_enabled
            enabled = context_enabled()
        if not enabled:
            return "", {"enabled": False}
        try:
            budget = int(os.environ.get("SKCALL_CONTEXT_BUDGET", memory_context.DEFAULT_BUDGET_TOKENS))
        except ValueError:
            budget = memory_context.DEFAULT_BUDGET_TOKENS
        t0 = time.monotonic()
        try:
            text, rep = memory_context.build_call_context(self.agent, budget)
        except Exception:
            LOG.warning("memory context unavailable; persona only", exc_info=True)
            return "", {"enabled": True, "error": True}
        rep["ms"] = round((time.monotonic() - t0) * 1000)
        LOG.info("memory context for %s: %s", self.agent, rep)
        return text, rep

    def _add_worship_stop(self) -> None:
        """worship_stop, next to the worship tools when they are installed."""
        if self.registry is None or "worship_session" not in self.tool_names():
            return
        from skcall.worship import stop_tool  # noqa: PLC0415
        schema, handler = stop_tool(lambda: self.convo)
        self.registry.register(self.ve.tools.Tool(name="worship_stop", schema=schema,
                                                  handler=handler, operator_only=True))

    def _bind_memory_tools(self) -> None:
        """Point search_memory at this agent's fused store and make the
        reflection tools see .md reflections (Jarvis writes those)."""
        handlers = {"search_memory": self.memory.search_tool,
                    "list_reflections": memory_context.list_reflections_tool,
                    "read_reflection": memory_context.read_reflection_tool}
        tools = getattr(self.registry, "_tools", {})
        for name, fn in handlers.items():
            if name in tools:
                tools[name].handler = fn

    async def start(self) -> None:
        """Attach the MCP tool surface and open the memory store (after
        joining, so neither delays the call)."""
        if self.registry is not None and "search_memory" in self.tool_names():
            self.memory.warm()
        if self.registry is not None and self._want_mcp:
            try:
                self._mcp = await self.ve.builtin_tools.attach_mcp_tools(self.registry)
            except Exception:
                LOG.warning("attaching MCP tools failed; built-ins only", exc_info=True)
        LOG.info("brain tools: %s", ", ".join(self.tool_names()) or "none")

    def tool_names(self) -> list[str]:
        if self.registry is None:
            return []
        return [s["function"]["name"] for s in self.registry.openai_schemas()]

    def system_prompt(self, mode: str, participants: list[str]) -> str:
        key = f"{mode}|{','.join(sorted(participants))}"
        if key not in self._prompts:
            reg = "private" if mode == "private" else "group"
            text = self._persona[reg]
            if self.memory_context and (reg == "private" or self.context_in_group):
                text += "\n\n" + self.memory_context
            text += "\n\n" + TALK_NOTE
            text += "\n\n" + HONESTY_NOTE.format(tools=", ".join(self.tool_names()) or "none")
            text += "\n\n" + MEMORY_NOTE
            if participants:
                text += f" People in the call: {', '.join(participants)}."
            text += (f"\n\nRuntime fact: the model you run on is '{self.cfg.model}'. State that "
                     "name if you are asked what model you are, and never guess a different one. "
                     "Do not recite it otherwise, and never read out endpoints, URLs, ports or "
                     "fallback configuration.")
            text += "\n\n" + self.ve.engine._BREVITY_RULE
            self._prompts[key] = text
        return self._prompts[key]

    def messages(self, turn: Turn) -> list[dict]:
        names = [p.display or p.user_id for p in turn.participants if not p.is_agent]
        sys_msg = {"role": "system", "content": self.system_prompt(turn.mode, names)}
        user = turn.text if turn.mode == "private" else f"{turn.speaker.display or turn.speaker.user_id}: {turn.text}"
        return [sys_msg, *turn.history, {"role": "user", "content": user}]

    async def _sse(self, url: str, model: str, messages: list[dict], turn: Turn, calls: dict,
                   tools: list | None = None, tool_choice=None) -> AsyncIterator[str]:
        """Stream one round. Content deltas are yielded; tool_call deltas are
        accumulated into `calls` (index -> {id, name, arguments})."""
        payload: dict = {"model": model, "max_tokens": self.cfg.max_tokens, "messages": messages,
                         "stream": True, "temperature": 0.7}
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice
        async with self._http.stream("POST", url, json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    ch = json.loads(data)["choices"][0]
                except (ValueError, KeyError, IndexError):
                    continue
                if ch.get("finish_reason") == "length":
                    turn.truncated = True
                delta = ch.get("delta") or {}
                for tc in delta.get("tool_calls") or []:
                    c = calls.setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
                    c["id"] = tc.get("id") or c["id"]
                    fn = tc.get("function") or {}
                    c["name"] = fn.get("name") or c["name"]
                    c["arguments"] += fn.get("arguments") or ""
                if delta.get("content"):
                    yield delta["content"]

    def _routing(self, turn: Turn):
        t = self.ve.tools
        if wants_memory(turn.text) and self._has("search_memory"):
            return {"type": "function", "function": {"name": "search_memory"}}
        if t.wants_narrate(turn.text) and turn.mode in t.ONE_TO_ONE_MODES and self._has("narrate"):
            return {"type": "function", "function": {"name": "narrate"}}
        if t.wants_action(turn.text):
            return "required"
        return None

    def _has(self, name: str) -> bool:
        return self.registry is not None and name in self.tool_names()

    async def stream(self, turn: Turn) -> AsyncIterator[str]:
        msgs = self.messages(turn)
        tools = self.registry.openai_schemas(for_text=turn.text) if self.registry is not None else None
        if tools is not None and getattr(self.convo, "active", False) and self._has("worship_stop"):
            # the curator picks tools by keyword; "stop" must always reach her mid-session
            if not any(t.get("function", {}).get("name") == "worship_stop" for t in tools):
                tools = [*tools, self.registry._tools["worship_stop"].schema]
        choice = self._routing(turn) if tools else None
        legs = [(self.cfg.llm_url, self.cfg.model, tools), (self.cfg.fallback_url, self.cfg.fallback_model, None)]
        for leg, (url, model, leg_tools) in enumerate(legs):
            got = False
            if not leg_tools:
                msgs = flatten_tools(msgs)
            try:
                for rnd in range(MAX_TOOL_ROUNDS):
                    calls: dict = {}
                    in_think = False
                    async for d in self._sse(url, model, msgs, turn, calls, leg_tools,
                                             choice if rnd == 0 else None):
                        if "<think>" in d:
                            in_think, d = True, d.split("<think>", 1)[0]
                        if in_think:
                            if "</think>" not in d:
                                continue
                            in_think, d = False, d.split("</think>", 1)[1]
                        if d:
                            got = True
                            yield d
                    if not calls:
                        break
                    spoken = await self._run_tools(turn, msgs, calls)
                    if spoken is not None:          # narrate: speak the prose verbatim
                        got = True
                        yield spoken
                        return
                else:
                    LOG.warning("tool round cap hit")
                if got:
                    return
                LOG.warning("LLM leg %d (%s) returned nothing", leg, model)
            except httpx.HTTPError as e:
                if got:
                    LOG.error("LLM stream broke mid-reply: %s", e)
                    turn.truncated = True
                    return
                LOG.error("LLM leg %d (%s) failed: %s: %s", leg, model, type(e).__name__, e)
        yield "Sorry, I lost my connection for a second. Could you say that again?"

    async def _run_tools(self, turn: Turn, msgs: list[dict], calls: dict) -> str | None:
        """Run one round of tool calls through the registry (operator gate
        included) and append the results. Returns narrate prose to speak
        verbatim, else None."""
        tcs = [{"id": c["id"] or f"call_{i}", "type": "function",
                "function": {"name": c["name"], "arguments": c["arguments"] or "{}"}}
               for i, c in sorted(calls.items())]
        call_msg = {"role": "assistant", "content": None, "tool_calls": tcs}
        msgs.append(call_msg)
        turn.exchange.append(call_msg)
        narrate = None
        for tc in tcs:
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except ValueError:
                args = {}
            t0 = time.monotonic()
            result = await self.registry.dispatch(
                name, args, speaker_id=turn.speaker.user_id, mode=turn.mode,
                is_operator=turn.is_operator, ctx={"agent": self.agent, "convo": self.convo})
            ms = round((time.monotonic() - t0) * 1000)
            result = str(result)
            self.tool_log.append({"tool": name, "ms": ms, "ok": not result.lower().startswith(
                ("permission denied", "refused", f"{name} failed", "unknown tool"))})
            LOG.info("tool %s(%s) -> %s [%d ms]", name, said(json.dumps(args)[:80]),
                     said(result[:100].replace("\n", " ")), ms)
            if name == "narrate" and len(result) > 80 and not result.lower().startswith("narrate"):
                narrate = result
            msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": result[:6000]})
            turn.exchange.append({"role": "tool", "tool_call_id": tc["id"],
                                  "content": result[:HISTORY_TOOL_CHARS]})
        return narrate

    def filler(self, text: str) -> tuple[str, str]:
        try:
            return self.ve.tools.pick_filler(text)
        except Exception:
            return _FALLBACK_FILLERS[0], "chat"

    def waiting_filler(self) -> str:
        try:
            return self.ve.tools.pick_waiting_filler()
        except Exception:
            return "Still with you."

    def all_fillers(self) -> list[str]:
        t = self.ve.tools
        out = []
        for name in ("_CHAT_FILLERS", "_LOOKUP_FILLERS", "_NARRATE_FILLERS", "_WAITING_FILLERS"):
            out += list(getattr(t, name, ()))
        return out or list(_FALLBACK_FILLERS)

    async def aclose(self) -> None:
        if self._mcp is not None:
            try:
                await self._mcp.aclose_all()
            except Exception:
                LOG.debug("MCP teardown failed", exc_info=True)
            self._mcp = None
        self.memory.close()
        await self._http.aclose()
