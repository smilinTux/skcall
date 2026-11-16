"""Per-agent settings (skcall.agents) and where cli/backend/text use them."""

import pytest

from skcall import agents, cli
from skcall.agents import PROFILES, ConfigError, ice_ifaces, profile_for
from skcall.core.text import is_leave_command


SITE = {"SKCALL_NC_URL": "https://example.org", "SKCALL_ALLOWED_ROOMS": "room0001",
        "SKCALL_OPERATORS": "alice"}


def test_builtin_profiles_hold_no_site_values():
    for p in PROFILES.values():
        assert p.nc_url == ""
        assert p.rooms == ()
        assert p.operators == ()
        assert p.tts_url == ""


def test_profile_requires_nc_url():
    with pytest.raises(ConfigError, match="SKCALL_NC_URL"):
        profile_for("jarvis", {})
    with pytest.raises(ConfigError, match="SKCALL_NC_URL"):
        profile_for("lumina", {"SKCALL_NC_URL": "  "})


def test_env_supplies_site_values():
    p = profile_for("jarvis", SITE)
    assert (p.nc_url, p.rooms, p.operators) == ("https://example.org", ("room0001",), ("alice",))


def test_lumina_profile_defaults():
    p = profile_for("lumina", SITE)
    assert p.nick == "Lumina"
    assert p.names == ("lumina", "luna", "lumi")
    assert (p.tts_voice, p.tts_model, p.tts_url) == ("lumina", "f5-tts", "")
    assert p.consent_line == "Lumina joined the call; she can hear and transcribe what is said here."


def test_jarvis_profile_defaults():
    p = profile_for("jarvis", SITE)
    assert p.nick == "Jarvis"
    assert p.names == ("jarvis",)
    assert (p.tts_model, p.tts_voice) == ("tts-1", "bm_george")
    assert p.tts_url == ""
    assert "he can hear" in p.consent_line and p.consent_line.startswith("Jarvis joined")
    assert p.listen_only_note == " (He can listen but not talk in this room.)"


def test_unknown_agent_gets_generic_profile():
    p = profile_for("Steward", SITE)
    assert (p.agent, p.nick, p.names, p.pronoun) == ("steward", "Steward", ("steward",), "they")


def test_env_overrides():
    env = {"SKCALL_NICK": "J.A.R.V.I.S.", "SKCALL_AGENT_NAMES": "Jarvis, Jarv",
           "SKCALL_PRONOUN": "They", "SKCALL_NC_URL": "https://example.test/",
           "SKCALL_ALLOWED_ROOMS": "room0001,room0002", "SKCALL_OPERATORS": "alice,bob",
           "SKCALL_TTS_VOICE": "jarvis", "SKCALL_TTS_MODEL": "f5-tts", "SKCALL_TTS_URL": "http://tts.test/v1/audio/speech"}
    p = profile_for("jarvis", env)
    assert p.nick == "J.A.R.V.I.S."
    assert p.names == ("jarvis", "jarv")
    assert p.pronoun == "they"
    assert p.nc_url == "https://example.test"
    assert p.rooms == ("room0001", "room0002")
    assert p.operators == ("alice", "bob")
    assert p.tts_voice == "jarvis"
    assert (p.tts_model, p.tts_url) == ("f5-tts", "http://tts.test/v1/audio/speech")


def test_empty_room_allowlist_means_no_rooms():
    assert profile_for("jarvis", {**SITE, "SKCALL_ALLOWED_ROOMS": ""}).rooms == ()


def test_join_refuses_without_rooms_or_operators(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SKCALL_ENV_FILE", str(tmp_path / "none.env"))
    monkeypatch.setenv("SKCALL_VOICE_ENV", str(tmp_path / "none.env"))
    monkeypatch.setenv("SKCALL_NC_URL", "https://example.org")
    for k in ("SKCALL_ALLOWED_ROOMS", "SKCALL_OPERATORS"):
        monkeypatch.delenv(k, raising=False)
    for cmd in ("join", "watch"):
        assert cli.main([cmd, "--room", "room0001", "--agent", "jarvis"]) != 0
        assert "no rooms or operators configured" in capsys.readouterr().err


def test_join_refuses_without_nc_url(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SKCALL_ENV_FILE", str(tmp_path / "none.env"))
    monkeypatch.setenv("SKCALL_VOICE_ENV", str(tmp_path / "none.env"))
    monkeypatch.delenv("SKCALL_NC_URL", raising=False)
    monkeypatch.setenv("SKCALL_ALLOWED_ROOMS", "room0001")
    assert cli.main(["join", "--room", "room0001", "--agent", "jarvis"]) != 0
    assert "SKCALL_NC_URL" in capsys.readouterr().err


def test_speech_has_no_builtin_endpoints(monkeypatch):
    import importlib

    from skcall import speech
    for k in ("SKCALL_STT_URL", "SKCALL_STT_BACKENDS", "SKCALL_TTS_URL"):
        monkeypatch.delenv(k, raising=False)
    try:
        s = importlib.reload(speech)
        assert (s.STT_URL, s.STT_BACKENDS, s.TTS_URL) == ("", "", "")
        with pytest.raises(ConfigError, match="SKCALL_TTS_URL"):
            s.F5TTS(url="")
        with pytest.raises(ConfigError, match="SKCALL_STT_BACKENDS"):
            s.HedgedSTT()
    finally:
        monkeypatch.undo()
        importlib.reload(speech)


def test_ice_ifaces():
    assert ice_ifaces({}) is None                       # default: every interface
    assert ice_ifaces({"SKCALL_ICE_IFACES": "192.0.2., 198.51.100."}) == ("192.0.2.", "198.51.100.")
    assert ice_ifaces({"SKCALL_ICE_IFACES": "*"}) is None
    assert ice_ifaces({"SKCALL_ICE_IFACES": ""}) is None


def test_leave_command_uses_agent_names():
    assert is_leave_command("Jarvis, leave the call", ("jarvis",), one_to_one=False)
    assert not is_leave_command("Lumina, leave the call", ("jarvis",), one_to_one=False)
    assert is_leave_command("goodbye jarvis", ("jarvis",), one_to_one=False)
    assert not is_leave_command("goodbye lumina", ("jarvis",), one_to_one=False)
    assert is_leave_command("goodbye, luna", ("lumina", "luna"), one_to_one=False)


def test_cli_allowed_rooms_and_loop_config(monkeypatch):
    for k in ("SKCALL_ALLOWED_ROOMS", "SKCALL_OPERATORS", "SKCALL_AGENT_NAMES", "SKAGENT"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SKCALL_NC_URL", "https://example.org")
    monkeypatch.setenv("SKCALL_ALLOWED_ROOMS", "room0001")
    monkeypatch.setenv("SKCALL_OPERATORS", "alice")
    assert cli.allowed_rooms("jarvis") == ("room0001",)
    assert cli.allowed_rooms() == ("room0001",)
    monkeypatch.setenv("SKCALL_ALLOWED_ROOMS", "room0001,room0002")
    assert cli.allowed_rooms("jarvis") == ("room0001", "room0002")

    a = cli.main.__globals__["argparse"].Namespace(
        agent="jarvis", hangover_ms=450, no_streaming_stt=False, spec_silence_ms=150, vad_rms=500,
        no_barge_in=False, no_filler=False, filler_delay=0.3, operators="", wait=300, max_call=7200)
    c = cli.loop_config(a)
    assert c.agent_names == ("jarvis",)
    assert c.operators == ("alice",)
    monkeypatch.setenv("SKCALL_OPERATORS", "bob")
    assert cli.loop_config(a).operators == ("bob",)
    a.operators = "alice,bob"
    assert cli.loop_config(a).operators == ("alice", "bob")


def test_load_env_reads_agent_voice_env(monkeypatch, tmp_path):
    home = tmp_path / "home"
    (home / ".config" / "skchat").mkdir(parents=True)
    (home / ".config" / "skchat" / "webui-jarvis.env").write_text("SKVOICE_MODEL=jarvis-model\nOTHER=x\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("SKCALL_VOICE_ENV", raising=False)
    monkeypatch.delenv("SKVOICE_MODEL", raising=False)
    monkeypatch.delenv("OTHER", raising=False)
    monkeypatch.setenv("SKCALL_ENV_FILE", str(tmp_path / "none.env"))
    monkeypatch.setenv("SKCALL_NC_URL", "https://example.org")
    cli.load_env("jarvis")
    assert agents.os.environ["SKVOICE_MODEL"] == "jarvis-model"
    assert "OTHER" not in agents.os.environ


def test_backend_patches_ice_with_configured_ifaces(monkeypatch):
    from skcall.talk import backend
    seen = {}
    monkeypatch.setattr(backend, "patch_ice", lambda **kw: seen.update(kw))
    monkeypatch.setenv("SKCALL_ICE_IFACES", "192.0.2.")

    class Stop(Exception):
        pass

    class OCS:
        async def __aenter__(self):
            raise Stop

    s = backend.TalkSession("room1", nick="Jarvis", ocs=OCS(), allowed_rooms=None)
    import asyncio
    try:
        asyncio.run(s.open())
    except Stop:
        pass
    assert seen == {"iface_allow": ("192.0.2.",)}
    assert s.nick == "Jarvis"


def test_tts_body_per_model(monkeypatch):
    import asyncio

    from skcall import speech
    sent = []

    class R:
        content = b"not a wav"

        def raise_for_status(self):
            pass

    class H:
        async def post(self, url, json):
            sent.append((url, json))
            return R()

        async def aclose(self):
            pass

    f5 = speech.F5TTS(url="http://f5/v1/audio/speech", voice="lumina", nfe_step=16, model="f5-tts")
    ko = speech.F5TTS(url="http://ko/v1/audio/speech", voice="bm_george", nfe_step=16, model="tts-1")
    for t in (f5, ko):
        t._http = H()
        asyncio.run(t.synth("Hello there."))
    assert sent[0] == ("http://f5/v1/audio/speech", {"model": "f5-tts", "input": "Hello there.", "voice": "lumina",
                                                     "response_format": "wav", "nfe_step": 16})
    assert sent[1] == ("http://ko/v1/audio/speech", {"model": "tts-1", "input": "Hello there.", "voice": "bm_george",
                                                     "response_format": "wav"})
    assert speech.F5TTS(url="http://a/x, http://b/x").urls == ["http://a/x", "http://b/x"]


def test_tts_reference_clip_is_sent_as_references(monkeypatch):
    """SKCALL_TTS_REF adds `references` and keeps `voice`, so an F5 fallback in the list still works."""
    import asyncio
    from skcall import speech
    sent = []

    class R:
        content = b""

        def raise_for_status(self):
            pass

    class H:
        async def post(self, url, json):
            sent.append(json)
            return R()

        async def aclose(self):
            pass

    monkeypatch.setenv("SKCALL_TTS_REF", " /voices/lumina.wav ")
    t = speech.F5TTS(url="http://z/v1/audio/speech", voice="lumina", model="f5-tts", nfe_step=16)
    t._http = H()
    asyncio.run(t.synth("Hi."))
    assert sent[0] == {"model": "f5-tts", "input": "Hi.", "voice": "lumina", "response_format": "wav",
                       "nfe_step": 16, "references": [{"audio_path": "/voices/lumina.wav"}]}
    monkeypatch.delenv("SKCALL_TTS_REF")
    assert speech.F5TTS(url="http://z/x").ref == ""


def test_known_eye_boxes_are_sane():
    from skcall.video.avatar import KNOWN_EYES
    assert "1597eede59685d3a" in KNOWN_EYES          # Jarvis's portrait.png
    for boxes in KNOWN_EYES.values():
        assert len(boxes) == 2
        for x, y, w, h in boxes:
            assert 0 < x < 1 and 0 < y < 1 and 0 < w < 0.2 and 0 < h < 0.1 and x + w < 1


def test_jarvis_video_defaults(monkeypatch):
    from skcall.video import VideoConfig
    for k in ("SKCALL_AVATAR_IMAGE", "SKCALL_LIPSYNC_AVATAR"):
        monkeypatch.delenv(k, raising=False)
    v = VideoConfig.from_env("jarvis")
    assert v.avatar == "jarvis"
    assert v.image.endswith("/.skcapstone/agents/jarvis/avatar/portrait.png")
