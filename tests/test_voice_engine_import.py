"""The voice engine comes from the installed skchat-sovereign, never a source tree under ~."""
import sys
import types

from skcall.brain import memory_context as mc
from skcall.brain import voice_engine
from skcall.brain.voice_engine import VoiceEngineBrain


class _FakeDist:
    def __init__(self, root):
        self.root = root

    def locate_file(self, path):
        return self.root


def test_skchat_src_from_distribution(tmp_path, monkeypatch):
    monkeypatch.delenv("SKCALL_SKCHAT_SRC", raising=False)
    monkeypatch.setattr(voice_engine.importlib.metadata, "distribution",
                        lambda name: _FakeDist(tmp_path) if name == "skchat-sovereign" else None)
    assert voice_engine.skchat_src() == str(tmp_path)


def test_skchat_src_env_wins(tmp_path, monkeypatch):
    monkeypatch.setenv("SKCALL_SKCHAT_SRC", str(tmp_path / "custom"))
    monkeypatch.setattr(voice_engine.importlib.metadata, "distribution",
                        lambda name: _FakeDist(tmp_path))
    assert voice_engine.skchat_src() == str(tmp_path / "custom")


def test_no_home_tree_default(tmp_path, monkeypatch):
    monkeypatch.delenv("SKCALL_SKCHAT_SRC", raising=False)
    monkeypatch.setattr(voice_engine.importlib.metadata, "distribution",
                        lambda name: _FakeDist(tmp_path / "venv" / "site-packages"))
    assert "clawd" not in voice_engine.skchat_src()


def test_extra_site_defaults_empty():
    assert voice_engine.EXTRA_SITE == ""


def _fake_ve():
    class Persona:
        def build(self, agent, mode="private"):
            return f"You are {agent} ({mode})."

    class Registry:
        _tools: dict = {}

        def openai_schemas(self, for_text=None):
            return []

    cfg = types.SimpleNamespace(model="m", llm_url="u", fallback_url="f", fallback_model="fm", max_tokens=150)
    return types.SimpleNamespace(
        config=types.SimpleNamespace(VoiceConfig=types.SimpleNamespace(from_env=lambda: cfg)),
        persona=types.SimpleNamespace(PersonaBuilder=Persona),
        builtin_tools=types.SimpleNamespace(build_default_registry=lambda cfg, agent: Registry()),
        engine=types.SimpleNamespace(_BREVITY_RULE="BREVITY LAST."))


def test_engine_without_optional_deps(tmp_path, monkeypatch):
    """No lumina_creative, no skmemory: the call still works, with no memory context."""
    for name in ("lumina_creative", "skmemory", "skmemory.backends.sqlite_backend",
                 "skmemory.backends.pgvector_backend"):
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setattr(mc, "agent_home", lambda agent: tmp_path)
    b = VoiceEngineBrain(agent="alice", ve=_fake_ve(), mcp=False, context=True)
    try:
        assert b.memory_context == ""
        assert mc.build_call_context("alice", 3000, home=tmp_path)[0] == ""
        assert "You are alice" in b.system_prompt("private", [])
    finally:
        b.memory.close()


def test_real_registry_builds_without_optional_deps(monkeypatch):
    """The real skchat build_default_registry tolerates a missing lumina_creative and skmemory."""
    import importlib.util
    import pytest
    if importlib.util.find_spec("skchat") is None:
        pytest.skip("skchat-sovereign not installed")
    for name in ("lumina_creative", "skmemory"):
        monkeypatch.setitem(sys.modules, name, None)
    ve = voice_engine.load_voice_engine()
    cfg = ve.config.VoiceConfig.from_env()
    reg = ve.builtin_tools.build_default_registry(cfg, "alice")
    names = [s["function"]["name"] for s in reg.openai_schemas()]
    assert names
    assert "narrate" not in names
