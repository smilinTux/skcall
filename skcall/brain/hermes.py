"""HermesBrain: Lumina's full assistant behind the fast voice layer.

The voice engine (VoiceEngineBrain) answers normal turns in about a second.
When a request needs real actions or her full toolbelt, the conversation loop
says a short acknowledgement in her voice, hands the request to Hermes, keeps
the call responsive meanwhile (she can keep chatting), and speaks a concise
summary when Hermes returns. Like ChatGPT voice: a voice layer plus the full
assistant behind it.

Hermes side: the agent's Hermes gateway api_server on the same host, OpenAI-compatible,
loopback only (http://127.0.0.1:8642/v1, model `hermes-agent`, Bearer
API_SERVER_KEY from ~/.hermes/.env, read into the environment as
SKCALL_HERMES_KEY by skcall.cli.load_env and never printed). Requests go to
/v1/chat/completions (non-streaming: the call only needs the final answer,
and the agent loop can run for minutes).

Session continuity (read from hermes-agent gateway/platforms/
api_server_openai_routes.py::_handle_chat_completions): Chat Completions keys
the TRANSCRIPT on `X-Hermes-Session-Id`. Without it, the session id is derived
from sha256(system prompt + first user message in the body) (api_server.py::
_derive_chat_session_id, "api-<16 hex>"), so a body whose history window had
moved on got a new session on every escalation. `X-Hermes-Session-Key` only
scopes long-term memory (Honcho etc.) and never selects the transcript on this
route. So each call sends one stable `X-Hermes-Session-Id` (per call) plus the
per-room `X-Hermes-Session-Key`. With an explicit id Hermes loads the history
from its own state.db and ignores history in the body, so the recent voice
exchange (which Hermes never saw) rides in the system message instead.

Escalation rule (simple on purpose, logged per turn):
- explicit action intents: send/email/schedule/book/create/file/remind/check
  my .../look up/search the web/status/run/deploy ..., or "ask Hermes";
- investigative or multi-step asks: look into, dig into, investigate, figure
  out, debug, troubleshoot, diagnose, why is/isn't/did/didn't/hasn't ..., what
  happened, get it working/running, set up, build, "nothing has run", "stopped
  working" (2026-10-25: "look into the ingestion pipeline" went to the voice
  layer, which ran tools with no deadline and never answered);
- or the voice engine's reply says it lacks the capability ("I don't have
  access to that", "I can't check that") and Hermes is available.
"""
from __future__ import annotations

import logging
import os
import re
import time

import httpx

LOG = logging.getLogger("skcall.hermes")

HERMES_URL = os.environ.get("SKCALL_HERMES_URL", "http://127.0.0.1:8642/v1")
HERMES_MODEL = os.environ.get("SKCALL_HERMES_MODEL", "hermes-agent")

_VERBS = (r"send|email|e-mail|text|message|schedule|book|reschedule|cancel|create|make|file|add|"
          r"check|look up|find|search|run|open|post|remind|deploy|restart|update|order|buy|"
          r"capture|draft|reply to|forward|summarize|fix|build|set up|setup|install|configure|"
          r"get (?:it|that|this) (?:working|running|going|fixed)")
_ESCALATE = re.compile(
    rf"\b(?:ask hermes|have hermes|hermes,|use your (?:full )?tools|go ahead and (?:{_VERBS})"
    rf"|(?:can|could|would|will) you (?:please )?(?:go )?(?:{_VERBS})\b"
    rf"|please (?:{_VERBS})\b"
    rf"|remind me|set a reminder|add (?:it|that|this) to"
    rf"|what'?s on my|what do i have (?:on|today|tomorrow)|my (?:calendar|schedule|inbox|emails?)"
    rf"|look (?:it|that|this) up|search (?:the web|online|for)"
    rf"|status\b)",
    re.I,
)
# Investigative / multi-step asks: these need many tool calls and real time, so
# they go to Hermes, never to the voice layer's short tool loop.
_INVESTIGATE = re.compile(
    r"\b(?:look(?:ing)? into|dig(?:ging)? into|investigate|figure out|debug|troubleshoot|diagnose"
    r"|track down|find out why|root cause"
    r"|why (?:is|isn'?t|is not|are|aren'?t|did|didn'?t|did not|does|doesn'?t|has|hasn'?t|has not"
    r"|have|haven'?t|was|wasn'?t|won'?t|can'?t)\b(?!\s+(?:that|it|this)\s*[?.!]*\s*$)"
    r"|what happened (?:to|with)|what went wrong"
    r"|get (?:it|that|this|them|the [\w-]+(?: [\w-]+)?) (?:working|running|going|fixed|back up)"
    r"|set (?:it |that |this |them |me |us )?up\b|build (?:me |us |a |an |the |out )"
    r"|nothing (?:has|have|is|was) (?:run|ran|running|worked|working|happened)"
    r"|(?:hasn'?t|has not|haven'?t|have not|didn'?t|did not) (?:run|ran|fired|worked|updated|synced)"
    r"|(?:stopped|isn'?t|is not|it'?s not|aren'?t) (?:working|running|syncing|updating))",
    re.I,
)
# Things the voice layer says when it lacks a tool (checked on her reply).
_LACKS = re.compile(
    r"\b(?:(?:i )?(?:don'?t|do not) have (?:access|a way|the ability|that (?:tool|capability))"
    r"|(?:i )?(?:can'?t|cannot|am not able to|'m not able to) (?:check|access|see|look|do that|send|"
    r"reach|pull|get to|run)|not something i (?:can|have access)|no access to)",
    re.I,
)

ACK_LINES = ("On it, give me a moment.", "Okay, I'm on it. Give me a moment.")

INSTRUCTIONS = (
    "This request comes from a live Nextcloud Talk VOICE call. {who} said it out loud; it is a "
    "speech transcript, so allow for small transcription mistakes. Do the task with your tools. "
    "Then reply with what should be SPOKEN back: one or two short plain sentences with the "
    "result, no markdown, no lists, no URLs, no emoji. Report only what your tool output actually "
    "showed; never claim a command ran or invent its output. If the task cannot be done, say why "
    "in one sentence."
)

# Added when others are in the call: the answer is spoken to all of them.
GROUP_NOTE = (
    "\n\nThis is a GROUP call: {others} can hear your answer too. Do only what {who} asked and "
    "speak only what that needs. Never volunteer private material beyond the request (memories, "
    "messages, email, contacts, calendar details, health, money, credentials, other people's "
    "business). If the answer itself is private, say so in one sentence and offer to tell {who} "
    "after the call instead of saying it now."
)


# While a Hermes job is pending: a question ABOUT that job ("why is it taking
# so long?", "what are you working on?") is answered from the job state, never
# sent to Hermes as a new job (2026-10-26: that question matched "why is" and
# queued a second job behind the hung one).
_PROGRESS_QUERY = re.compile(
    r"\b(?:(?:why is|why's) (?:it|this|that) taking|taking (?:so|this|that|too) long"
    r"|what are you (?:working on|doing)|what'?s taking (?:so|this|that) long"
    r"|are you still (?:working|on it|there|doing)|still working on"
    r"|how'?s (?:it|that) (?:going|coming)|how is (?:it|that) (?:going|coming)"
    r"|any (?:update|news|progress)|(?:is|was) it done|are you done|did (?:it|you|hermes) finish"
    r"|is (?:it|hermes) (?:stuck|hung|frozen))",
    re.I,
)
_STOP = frozenset("a an the to of and or for on in my me you your it is are be can could would "
                  "will please just right still do does did that this with what".split())


def is_progress_query(text: str) -> bool:
    return bool(_PROGRESS_QUERY.search(text or ""))


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", (text or "").lower()) if w not in _STOP}


def same_request(a: str, b: str, threshold: float = 0.6) -> bool:
    """Two transcripts ask the same thing (a repeat while the first is pending)."""
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return False
    return len(wa & wb) / len(wa | wb) >= threshold


def wants_escalation(text: str) -> bool:
    text = text or ""
    return bool(_ESCALATE.search(text) or _INVESTIGATE.search(text))


def reply_lacks_tool(reply: str) -> bool:
    return bool(_LACKS.search(reply or ""))


DEFAULT_TIMEOUT_S = 900.0


def timeout_from_env() -> float:
    try:
        return float(os.environ.get("SKCALL_HERMES_TIMEOUT_S", DEFAULT_TIMEOUT_S))
    except ValueError:
        return DEFAULT_TIMEOUT_S


def _voice_context(history: list[dict]) -> str:
    lines = []
    for m in history:
        c = (m.get("content") or "").strip()
        if c and m.get("role") in ("user", "assistant"):
            lines.append(f"{'Caller' if m['role'] == 'user' else 'You'}: {c[:400]}")
    return "\n".join(lines)


class HermesBrain:
    def __init__(self, url: str = HERMES_URL, key: str | None = None, model: str = HERMES_MODEL,
                 session_key: str = "skcall", session_id: str | None = None,
                 timeout_s: float | None = None):
        self.url = url.rstrip("/")
        self.key = key if key is not None else os.environ.get("SKCALL_HERMES_KEY", "")
        self.model = model
        self.session_key = session_key
        # One Hermes transcript per call (see the module docstring).
        self.session_id = session_id
        self.timeout_s = timeout_s if timeout_s is not None else timeout_from_env()
        self._http = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_s, connect=5))
        self.log: list[dict] = []

    @property
    def available(self) -> bool:
        return bool(self.key)

    def headers(self) -> dict[str, str]:
        h = {"Authorization": f"Bearer {self.key}", "X-Hermes-Session-Key": self.session_key}
        if self.session_id:
            h["X-Hermes-Session-Id"] = self.session_id
        return h

    def build_messages(self, text: str, who: str, history: list[dict] | None,
                       audience: list[str] | None = None) -> list[dict]:
        system = INSTRUCTIONS.format(who=who)
        if audience:
            system += GROUP_NOTE.format(others=", ".join(audience), who=who)
        recent = (history or [])[-6:]
        if self.session_id:
            # Hermes replays its own transcript for this id and ignores body
            # history, so what was said to the voice layer goes in here.
            ctx = _voice_context(recent)
            if ctx:
                system += "\n\nRecent spoken exchange in this call, for context:\n" + ctx
            return [{"role": "system", "content": system}, {"role": "user", "content": text}]
        return [{"role": "system", "content": system}, *recent, {"role": "user", "content": text}]

    async def ask(self, text: str, who: str = "the operator", history: list[dict] | None = None,
                  audience: list[str] | None = None) -> str:
        """Run one request through Hermes; returns text to speak. ``audience``: the other people
        in the call, who will hear the answer (group calls)."""
        msgs = self.build_messages(text, who, history, audience)
        t0 = time.monotonic()
        rec = {"text": text[:80]}
        try:
            r = await self._http.post(
                f"{self.url}/chat/completions", headers=self.headers(),
                json={"model": self.model, "messages": msgs, "stream": False})
            r.raise_for_status()
            out = (r.json()["choices"][0]["message"].get("content") or "").strip()
            rec.update(ok=True, completed=r.headers.get("X-Hermes-Completed", "true"),
                       session=r.headers.get("X-Hermes-Session-Id", ""))
        except Exception as e:
            LOG.error("Hermes request failed: %s: %s", type(e).__name__, e or "(no detail)")
            rec.update(ok=False, error=type(e).__name__)
            out = "I couldn't get that done just now; my full assistant didn't answer."
        rec["ms"] = round((time.monotonic() - t0) * 1000)
        self.log.append(rec)
        LOG.info("hermes %s in %d ms (session %s): %s", "ok" if rec.get("ok") else "FAILED",
                 rec["ms"], rec.get("session") or "-", out[:160])
        return out

    async def aclose(self) -> None:
        await self._http.aclose()
