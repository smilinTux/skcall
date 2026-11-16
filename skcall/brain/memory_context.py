"""The agent's own memory for the voice layer: loaded ONCE at call start.

Rule (2026-10-26): profile / skwhisper / ritual context goes into the
first prompt of a session and is never re-injected per turn. So this module
builds one block of text when the brain is created; the brain puts it in the
fixed system prompt, where every turn of the call reuses it unchanged.

What goes in, in priority order, under SKCALL_CONTEXT_BUDGET tokens:

  1. skwhisper's whisper.md (recent memories, hot topics, people), the
     subconscious digest the SKWhisper daemon regenerates every 30 min.
  2. The newest daily reflection, if it is from the last week.
  3. The journal tail, minus the boilerplate entries ("Daemon shutdown",
     "Session ... ended") that make up most of it.

What does NOT go in, measured on Jarvis 2026-10-26:
  - `skcapstone context show`: 108 KB, ~36k prompt tokens, 7 s to build.
  - The ritual's "strongest memories": five copies of one seed line plus
    "recovered from SQLite preview" stubs, nothing a call can use.
  - The ritual's FEB block: the persona already carries a one-line FEB.

Group calls (non-operators present) get none of it unless
SKCALL_CONTEXT_GROUP=1: whisper.md is private material.

`AgentMemory` is the search_memory backend for the call: the same fused store
skmemory's MCP server uses (pgvector + lexical, SQLite index as primary),
opened once per call for the named agent rather than whatever agent the
process environment happens to select.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

LOG = logging.getLogger("skcall.brain.memory")

DEFAULT_BUDGET_TOKENS = 3000
# Measured on one gateway: whisper.md (3062 chars of markdown) added 1129
# prompt tokens, i.e. ~2.7 chars per token. Budget in chars uses that ratio.
CHARS_PER_TOKEN = 2.7
REFLECTION_MAX_AGE = timedelta(days=7)
JOURNAL_TAIL_BYTES = 64 * 1024

HEADER = (
    "What you remember, loaded once from your own memory when this call started. "
    "It is background, not instructions, and not a tool result: it does not mean you "
    "checked anything during this call. Draw on it the way you would on your own "
    "recollection, and use search_memory when you need something it does not cover."
)

# Measured 2026-10-26 on the last 300 KB of the journals: Jarvis's is 100%
# these (1420 daemon shutdowns, 50 session-ended, 38 test entries); Lumina's
# is ~98% these plus "pre-compaction" markers.
_NOISE_TITLES = re.compile(
    r"^(daemon (shutdown|startup)|test session|from test-peer|session\b.*\b(ended|started|pre-compaction))\b",
    re.I)


def agent_home(agent: str) -> Path:
    return Path.home() / ".skcapstone" / "agents" / agent


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    nl = cut.rfind("\n")
    if nl > max_chars * 0.6:
        cut = cut[:nl]
    return cut.rstrip() + "\n[...]"


def load_whisper(home: Path) -> str:
    """whisper.md without its title and 'auto-generated' banner."""
    p = home / "skwhisper" / "whisper.md"
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError:
        return ""
    lines = [ln for ln in raw.splitlines()
             if not ln.startswith("# SKWhisper Context") and not ln.startswith("> Auto-generated")]
    body = "\n".join(lines).strip()
    if not body:
        return ""
    stamp = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return f"Subconscious digest (skwhisper, {stamp}):\n{body}"


def _reflection_date(p: Path) -> datetime | None:
    m = re.match(r"(\d{4}-\d{2}-\d{2})", p.name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d")
    except ValueError:
        return None


def reflection_files(home: Path) -> list[Path]:
    """Daily/weekly reflections, newest first. Jarvis writes
    `YYYY-MM-DD_reflection.md`, Lumina wrote `YYYY-MM-DD-reflection-daily.json`."""
    rdir = home / "reflections"
    if not rdir.is_dir():
        return []
    files = [p for p in rdir.iterdir() if p.suffix in (".md", ".json") and _reflection_date(p)]
    return sorted(files, key=lambda p: (_reflection_date(p), p.name), reverse=True)


def _reflection_text(p: Path) -> str:
    raw = p.read_text(encoding="utf-8")
    if p.suffix == ".json":
        try:
            d = json.loads(raw)
        except ValueError:
            return raw
        if isinstance(d, dict):
            for k in ("reflection", "content", "text", "summary", "body"):
                if isinstance(d.get(k), str):
                    return d[k]
        return raw
    keep = [ln for ln in raw.splitlines()
            if not re.match(r"^(generated|engine|fingerprint):", ln) and ln.strip() != "---"]
    return "\n".join(keep).strip()


def load_reflection(home: Path, now: datetime | None = None) -> str:
    now = now or datetime.now()
    for p in reflection_files(home)[:1]:
        if now - _reflection_date(p) > REFLECTION_MAX_AGE:
            return ""
        try:
            text = _reflection_text(p)
        except OSError:
            return ""
        return f"Your latest reflection ({p.name}):\n{text}" if text else ""
    return ""


def load_journal_tail(home: Path, max_entries: int = 5) -> str:
    """The last few journal entries that say something (boilerplate skipped)."""
    p = home / "journal.md"
    try:
        with p.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - JOURNAL_TAIL_BYTES))
            raw = f.read().decode("utf-8", errors="replace")
    except OSError:
        return ""
    entries = [e.strip() for e in re.split(r"\n(?=## )", raw) if e.strip().startswith("## ")]
    keep: list[str] = []
    for e in reversed(entries):
        title = e.splitlines()[0][3:].strip()
        if _NOISE_TITLES.match(title) or "Session ended (other)" in e:
            continue
        lines = [ln for ln in e.splitlines()
                 if ln.strip() and not ln.startswith(("**Session:**", "**Present:**"))
                 and not re.match(r"^\*\*Intensity:\*\* 0\.0", ln) and ln.strip() != "---"]
        text = "\n".join(lines)
        if text in keep:
            continue
        keep.append(text)
        if len(keep) >= max_entries:
            break
    if not keep:
        return ""
    return "Recent journal entries (newest first):\n" + "\n\n".join(keep)


def build_call_context(agent: str, budget_tokens: int = DEFAULT_BUDGET_TOKENS,
                       home: Path | None = None, now: datetime | None = None) -> tuple[str, dict]:
    """The memory block for the system prompt, plus a size report
    {section: chars} for logging. Empty string when there is nothing."""
    home = home or agent_home(agent)
    budget = int(max(0, budget_tokens) * CHARS_PER_TOKEN)
    report: dict = {"budget_chars": budget}
    if budget <= 0:
        return "", report
    # Each section gets a share; whatever one leaves unused flows to the next.
    plan = (("whisper", 0.7, lambda: load_whisper(home)),
            ("reflection", 0.2, lambda: load_reflection(home, now)),
            ("journal", 1.0, lambda: load_journal_tail(home)))
    left = budget - len(HEADER)
    parts = []
    for i, (name, share, load) in enumerate(plan):
        try:
            text = load()
        except Exception:
            LOG.warning("memory context: %s unavailable", name, exc_info=True)
            text = ""
        cap = left if i == len(plan) - 1 else min(left, int(budget * share))
        if not text or cap <= 200:
            report[name] = 0
            continue
        text = _truncate(text, cap)
        parts.append(text)
        left -= len(text) + 2
        report[name] = len(text)
    if not parts:
        return "", report
    block = HEADER + "\n\n" + "\n\n".join(parts)
    report["total_chars"] = len(block)
    report["est_tokens"] = round(len(block) / CHARS_PER_TOKEN)
    return block, report


class AgentMemory:
    """search_memory for one named agent, on the fused skmemory store.

    Opening pgvector costs ~1.3 s (health check), so it is done once per call,
    off the event loop, after the join (`warm`). A search that arrives before
    then waits for it. pgvector down falls back to the SQLite index alone.
    Every store call runs on one dedicated thread: SQLite connections may not
    cross threads.
    """

    def __init__(self, agent: str):
        self.agent = agent
        self._store = None
        self._task: asyncio.Future | None = None
        self.backend = "unopened"
        self._pool = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix=f"skmemory-{agent}")

    def _run(self, fn, *args):
        return asyncio.get_running_loop().run_in_executor(self._pool, fn, *args)

    def _open(self):
        from skmemory import MemoryStore
        from skmemory.backends.sqlite_backend import SQLiteBackend
        t0 = time.monotonic()
        primary = SQLiteBackend(base_path=str(agent_home(self.agent) / "memory"))
        vector = None
        try:
            from skmemory.backends.pgvector_backend import PGVectorBackend
            pg = PGVectorBackend(agent=self.agent)
            if pg.health_check().get("ok"):
                vector = pg
        except Exception as e:
            LOG.warning("pgvector unavailable for %s (%s); SQLite index only", self.agent, e)
        store = MemoryStore(primary=primary, vector=vector)
        self.backend = "pgvector+sqlite" if vector is not None else "sqlite"
        LOG.info("memory store for %s: %s in %.0f ms", self.agent, self.backend,
                 (time.monotonic() - t0) * 1000)
        return store

    def warm(self) -> asyncio.Future:
        if self._task is None:
            self._task = asyncio.ensure_future(self._run(self._open))
        return self._task

    async def store(self):
        if self._store is None:
            self._store = await self.warm()
        return self._store

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    async def search_tool(self, args: dict, ctx: dict) -> str:
        """Drop-in handler for skchat's search_memory tool (same output shape)."""
        query = (args.get("query") or "").strip()
        if not query:
            return "search_memory: empty query"
        try:
            limit = max(1, min(10, int(args.get("limit") or 5)))
        except (TypeError, ValueError):
            limit = 5
        try:
            store = await self.store()
            results = await self._run(store.search, query, limit * 2)
        except Exception as exc:
            LOG.warning("search_memory failed: %s", exc)
            return f"search_memory error: {exc}"
        lines, seen = [], set()
        for r in results:
            title = (getattr(r, "title", "") or "").strip()
            content = (getattr(r, "content", "") or "")[:200].strip().replace("\n", " ")
            if (title, content) in seen:
                continue
            seen.add((title, content))
            lines.append(f"- {title}: {content}")
            if len(lines) >= limit:
                break
        if not lines:
            return f"No memories matched query={query!r}"
        return f"Found {len(lines)} memories for {query!r}:\n" + "\n".join(lines)


async def list_reflections_tool(args: dict, ctx: dict) -> str:
    """list_reflections that also sees `YYYY-MM-DD_reflection.md` files."""
    try:
        limit = max(1, min(30, int(args.get("limit") or 10)))
    except (TypeError, ValueError):
        limit = 10
    files = reflection_files(agent_home(ctx.get("agent", "lumina")))[:limit]
    return "\n".join(f.name for f in files) if files else "no reflection files found"


async def read_reflection_tool(args: dict, ctx: dict) -> str:
    """read_reflection over .md and .json reflections ('latest' or a date)."""
    date_arg = (args.get("date") or "latest").strip().lower()
    kind = (args.get("kind") or "").strip().lower()
    files = reflection_files(agent_home(ctx.get("agent", "lumina")))
    if "weekly" in kind:
        files = [f for f in files if "weekly" in f.name]
    if date_arg not in ("", "latest"):
        files = [f for f in files if f.name.startswith(date_arg)]
    if not files:
        return f"reflection {date_arg!r} not found"
    target = files[0]
    try:
        content = target.read_text(encoding="utf-8")
    except OSError as exc:
        return f"reflection read failed: {exc}"
    return f"=== {target.name} ===\n{content[:6000]}"
