"""Post-call summary: who joined, how long, what was discussed, action items.

The transcript lives only in memory (Conversation.transcript) and is handed
to an OpenAI-compatible endpoint (skgateway, model sk-default by default) for
the bullet summary. Nothing here writes it to disk. If the LLM is down, the
chat still gets who and how long.

Env:
  SKCALL_POST_SUMMARY   on (default) / off
  SKCALL_SUMMARY_URL    default http://localhost:18780/v1
  SKCALL_SUMMARY_MODEL  default sk-default
  SKCALL_SUMMARY_KEY    optional bearer token
"""
from __future__ import annotations

import logging
import os
import re
from typing import Awaitable, Callable

import httpx

LOG = logging.getLogger("skcall.summary")

DEFAULT_URL = "http://localhost:18780/v1"
DEFAULT_MODEL = "sk-default"
MAX_PROMPT_CHARS = 24000

SYSTEM_PROMPT = (
    "You summarize a voice call transcript for the room's chat. Reply with plain "
    "markdown only, no preamble, in exactly this shape:\n"
    "- 3 to 6 short bullets of what was discussed\n"
    "then, only if there were any, a line 'Action items:' followed by bullets "
    "naming who does what. Never invent anything that is not in the transcript. "
    "Do not use em dashes or en dashes."
)

LLM = Callable[[list[dict]], Awaitable[str]]


def enabled() -> bool:
    return os.environ.get("SKCALL_POST_SUMMARY", "on").strip().lower() not in ("0", "off", "false", "no")


def fmt_duration(seconds: float) -> str:
    s = max(0, int(round(seconds)))
    h, rem = divmod(s, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def transcript_text(lines: list[tuple[str, str]], limit: int = MAX_PROMPT_CHARS) -> str:
    """'Who: text' lines, keeping the most recent `limit` characters."""
    out: list[str] = []
    total = 0
    for who, text in reversed(lines):
        line = f"{who}: {text}"
        if total + len(line) > limit:
            break
        out.append(line)
        total += len(line) + 1
    return "\n".join(reversed(out))


def clean(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    text = text.replace("\u2014", ", ").replace("\u2013", "-")
    return text.strip()


def make_llm(url: str | None = None, model: str | None = None, key: str | None = None,
             timeout: float = 90.0) -> LLM:
    url = (url or os.environ.get("SKCALL_SUMMARY_URL") or DEFAULT_URL).rstrip("/")
    model = model or os.environ.get("SKCALL_SUMMARY_MODEL") or DEFAULT_MODEL
    key = key if key is not None else os.environ.get("SKCALL_SUMMARY_KEY", "")

    async def call(messages: list[dict]) -> str:
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.post(f"{url}/chat/completions", headers=headers,
                             json={"model": model, "messages": messages, "max_tokens": 600,
                                   "temperature": 0.2})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"] or ""
    return call


async def build_summary(agent: str, joined: list[str], duration_s: float,
                        lines: list[tuple[str, str]], llm: LLM | None = None) -> str | None:
    """The chat message, or None when there is nothing worth posting (nobody came)."""
    if not joined:
        return None
    head = f"Call summary ({fmt_duration(duration_s)}, with {', '.join(joined)})"
    if not lines:
        return head + "\n- Nothing was discussed."
    llm = llm or make_llm()
    try:
        body = clean(await llm([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Call with {', '.join(joined)} and {agent}.\n\n"
                                        f"Transcript:\n{transcript_text(lines)}"},
        ]))
    except Exception as e:
        LOG.warning("summary LLM failed: %s", e)
        body = ""
    if not body:
        body = "- (Summary unavailable: the summarizer did not answer.)"
    return f"{head}\n{body}"


async def post_summary(room: str, message: str, ocs=None) -> None:
    """Post with a fresh OCS session (the call's own is closed by then)."""
    from skcall.talk.ocs import TalkOCS
    o = ocs or TalkOCS()
    async with o:
        await o.post_chat(room, message)
