"""The agent's memory in the voice prompt: loaded once, budgeted, private only."""
import types
from datetime import datetime

import pytest

from skcall.agents import profile_for

SITE = {"SKCALL_NC_URL": "https://example.org"}
from skcall.brain import memory_context as mc
from skcall.brain import voice_engine
from skcall.brain.voice_engine import HONESTY_NOTE, TALK_NOTE, VoiceEngineBrain

WHISPER = """# SKWhisper Context - 2026-10-26T17:09:15

> Auto-generated subconscious context. Read-only.

## Recent Memories (latest activity)

- [This week] **Fixed the gateway pool ceiling**
## Hot Topics (recurring)
- skcall, voice latency
"""

JOURNAL = """# Journal
## Daemon shutdown

**Date:** 2026-10-26T04:25:40+00:00
**Intensity:** 0.0/10 []

### Key Moments
- Graceful shutdown - drained, flushed, and signed off.

---
## Shipped the Jarvis voice profile

**Date:** 2026-10-25T10:00:00+00:00
**Intensity:** 7.0/10 []

### Key Moments
- First call on the second server worked.

---
## From test-peer

hello

---
## Session unknown - ended

**Intensity:** 0.0/10 []

---
## Session 0a1b2c3d - pre-compaction

x

---
## Session 0a1b2c3d - ended

**Session:** `0a1b2c3d`
**Intensity:** 0.0/10 []

### Key Moments
- Session ended (other), 7 turns
"""


@pytest.fixture
def home(tmp_path):
    (tmp_path / "skwhisper").mkdir()
    (tmp_path / "skwhisper" / "whisper.md").write_text(WHISPER)
    (tmp_path / "reflections").mkdir()
    (tmp_path / "reflections" / "2026-10-24_reflection.md").write_text("# old\nolder one")
    (tmp_path / "reflections" / "2026-10-25_reflection.md").write_text(
        "# Jarvis daily reflection\n\ngenerated: x\nengine: y\nfingerprint: z\n\n---\n\nCalls feel real now.")
    (tmp_path / "journal.md").write_text(JOURNAL)
    return tmp_path


NOW = datetime(2026, 10, 26, 12, 0)


def test_context_sections_and_noise_filtering(home):
    text, rep = mc.build_call_context("jarvis", 2500, home=home, now=NOW)
    assert text.startswith(mc.HEADER)
    assert "Fixed the gateway pool ceiling" in text
    assert "SKWhisper Context" not in text and "Auto-generated" not in text
    assert "2026-10-25_reflection.md" in text and "Calls feel real now." in text
    assert "fingerprint:" not in text
    assert "Shipped the Jarvis voice profile" in text
    assert "Daemon shutdown" not in text and "Session ended (other)" not in text
    assert "test-peer" not in text and "Session unknown" not in text and "pre-compaction" not in text
    assert rep["whisper"] and rep["reflection"] and rep["journal"]
    assert rep["total_chars"] == len(text)


def test_context_respects_budget(home):
    (home / "skwhisper" / "whisper.md").write_text(WHISPER + ("- filler line\n" * 2000))
    text, rep = mc.build_call_context("jarvis", 1000, home=home, now=NOW)
    assert len(text) <= 1000 * mc.CHARS_PER_TOKEN + 10
    assert "[...]" in text


def test_context_zero_budget_and_missing_files(tmp_path):
    assert mc.build_call_context("x", 0, home=tmp_path)[0] == ""
    assert mc.build_call_context("x", 2500, home=tmp_path)[0] == ""


def test_stale_reflection_is_skipped(home):
    text, _ = mc.build_call_context("jarvis", 2500, home=home, now=datetime(2026, 12, 1))
    assert "reflection.md" not in text


def test_profile_context_toggle():
    assert profile_for("jarvis", SITE).context is True
    assert profile_for("lumina", SITE).context is True
    assert profile_for("jarvis", {**SITE, "SKCALL_CONTEXT": "0"}).context is False
    assert profile_for("jarvis", {**SITE, "SKCALL_CONTEXT": "1"}).context is True


async def test_reflection_tools_read_md(home, monkeypatch):
    monkeypatch.setattr(mc, "agent_home", lambda agent: home)
    listing = await mc.list_reflections_tool({}, {"agent": "jarvis"})
    assert listing.splitlines()[0] == "2026-10-25_reflection.md"
    got = await mc.read_reflection_tool({"date": "latest"}, {"agent": "jarvis"})
    assert "Calls feel real now." in got
    assert "older one" in await mc.read_reflection_tool({"date": "2026-10-24"}, {"agent": "jarvis"})


# --- the brain -------------------------------------------------------------

class _Persona:
    calls = 0

    def build(self, agent, mode="private"):
        _Persona.calls += 1
        return f"You are {agent.capitalize()} ({mode})."


def _fake_ve():
    from dataclasses import dataclass

    @dataclass
    class Tool:
        name: str
        schema: dict
        handler: object = None

    class Registry:
        def __init__(self):
            self._tools = {n: Tool(n, {"function": {"name": n}})
                           for n in ("search_memory", "list_reflections", "read_reflection")}

        def openai_schemas(self, for_text=None):
            return [t.schema for t in self._tools.values()]

    cfg = types.SimpleNamespace(model="m", llm_url="u", fallback_url="f", fallback_model="fm", max_tokens=150)
    return types.SimpleNamespace(
        config=types.SimpleNamespace(VoiceConfig=types.SimpleNamespace(from_env=lambda: cfg)),
        persona=types.SimpleNamespace(PersonaBuilder=_Persona),
        builtin_tools=types.SimpleNamespace(build_default_registry=lambda cfg, agent: Registry()),
        engine=types.SimpleNamespace(_BREVITY_RULE="BREVITY LAST."))


@pytest.fixture
def brain(home, monkeypatch):
    monkeypatch.setattr(mc, "agent_home", lambda agent: home)
    _Persona.calls = 0
    b = VoiceEngineBrain(agent="jarvis", ve=_fake_ve(), mcp=False, context=True)
    yield b
    b.memory.close()


def test_prompt_order_memory_then_rules(brain):
    p = brain.system_prompt("private", ["Alice"])
    i_persona, i_mem = p.index("You are Jarvis"), p.index(mc.HEADER)
    i_talk, i_hon = p.index(TALK_NOTE), p.index(HONESTY_NOTE.split("{tools}")[0])
    assert i_persona < i_mem < i_talk < i_hon
    assert p.rstrip().endswith("BREVITY LAST.")
    assert "Fixed the gateway pool ceiling" in p


def test_group_register_gets_no_memory(brain, monkeypatch):
    assert mc.HEADER not in brain.system_prompt("group", ["Alice", "Bob"])
    brain.context_in_group = True
    assert mc.HEADER in brain.system_prompt("group", ["Alice", "Bob", "X"])


def test_loaded_once_not_per_turn(brain, home):
    first = brain.system_prompt("private", ["Alice"])
    calls = _Persona.calls
    (home / "skwhisper" / "whisper.md").write_text("# changed\n- something new")
    again = brain.system_prompt("private", ["Alice", "Bob"])     # new participants: new key
    assert _Persona.calls == calls                                 # soul/FEB not re-read
    assert "something new" not in again                            # memory not re-read
    assert first.split(TALK_NOTE)[0] == again.split(TALK_NOTE)[0]  # identical cacheable prefix


def test_memory_tools_bound_to_agent(brain):
    tools = brain.registry._tools
    assert tools["search_memory"].handler == brain.memory.search_tool
    assert tools["list_reflections"].handler is mc.list_reflections_tool
    assert brain.memory.agent == "jarvis"


async def test_search_tool_dedupes_and_limits(brain):
    class Store:
        def search(self, q, limit):
            m = types.SimpleNamespace
            return [m(title="I am sovereign", content="I am sovereign")] * 4 + \
                   [m(title=f"t{i}", content=f"c{i}") for i in range(6)]
    brain.memory._store = Store()
    out = await brain.memory.search_tool({"query": "q", "limit": 3}, {"agent": "jarvis"})
    lines = out.splitlines()
    assert lines[0] == "Found 3 memories for 'q':"
    assert lines[1:] == ["- I am sovereign: I am sovereign", "- t0: c0", "- t1: c1"]


def test_context_off(home, monkeypatch):
    monkeypatch.setattr(mc, "agent_home", lambda agent: home)
    b = VoiceEngineBrain(agent="jarvis", ve=_fake_ve(), mcp=False, context=False)
    assert mc.HEADER not in b.system_prompt("private", ["Alice"])
    assert b.context_report == {"enabled": False}
    b.memory.close()
