"""skcall smoke: one line per check, exit 0 only when every selected check passes.

httpx is faked with MockTransport, Talk with a fake TalkOCS, the lip-sync
service with a fake LipsyncClient, and the TURN allocation with a fake
aioice create_turn_endpoint. Nothing here opens a socket.
"""
import io
import os
import wave

import httpx
import numpy as np
import pytest

from skcall import cli, smoke

TTS = "http://192.0.2.10:18793/v1/audio/speech"
STT = "http://192.0.2.11:18794/v1/audio/transcriptions"
LIPS = "ws://192.0.2.12:18799/v1/lipsync"


def _wav(n=24000, sr=24000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        t = np.arange(n) / sr
        w.writeframes((np.sin(2 * np.pi * 220 * t) * 8000).astype("<i2").tobytes())
    return buf.getvalue()


def _settings(*hosts):
    return {"server": "wss://example.org/standalone-signaling",
            "stunservers": [{"urls": ["stun:example.org:443"]}],
            "turnservers": [{"urls": [f"turn:{h}:3478?transport=udp"], "username": "u",
                             "credential": "c"} for h in hosts]}


@pytest.fixture
def env(monkeypatch, tmp_path):
    for k in list(os.environ):
        if k.startswith(("SKCALL_", "NEXTCLOUD_TALK_")):
            monkeypatch.delenv(k)
    monkeypatch.setattr(smoke, "SITE_ENV_FILES", (str(tmp_path / "none.env"), str(tmp_path / "none-join.env")))
    monkeypatch.setenv("SKCALL_ENV_FILE", str(tmp_path / "none-login.env"))
    monkeypatch.setenv("SKCALL_VOICE_ENV", str(tmp_path / "none-voice.env"))
    full = {"SKCALL_NC_URL": "https://example.org", "NEXTCLOUD_TALK_USERNAME": "alice",
            "NEXTCLOUD_TALK_APP_PASSWORD": "secret", "SKCALL_ALLOWED_ROOMS": "room0001",
            "SKCALL_TTS_URL": TTS, "SKCALL_STT_BACKENDS": f"fw:{STT}",
            "SKCALL_LIPSYNC_URL": LIPS, "SKCALL_LIPSYNC_TOKEN": "tok",
            "SKCALL_SITE_TURN": "turn.example.org"}
    for k, v in full.items():
        monkeypatch.setenv(k, v)
    saved = dict(os.environ)
    yield monkeypatch
    # smoke.load_site_env writes os.environ directly: drop what a test loaded.
    os.environ.clear()
    os.environ.update(saved)


@pytest.fixture
def fakes(monkeypatch):
    state = {"interface": {"tts": 1, "stt": 1, "lipsync": 1}, "settings": _settings("turn.example.org"),
             "rooms": [{"token": "room0001"}], "allocs": [], "requests": [], "down": set(),
             "info": {}, "heard": "This is a quick speech test."}

    def handler(req: httpx.Request) -> httpx.Response:
        state["requests"].append(req)
        port = req.url.port
        svc = {18793: "tts", 18794: "stt", 18799: "lipsync"}[port]
        if req.url.path == "/v1/info":
            if req.url.host in state["info"]:
                return state["info"][req.url.host]
            return httpx.Response(200, json={"service": svc, "interface": state["interface"][svc],
                                             "version": "1.0.0"})
        if req.url.host in state["down"]:
            return httpx.Response(500, text="engine down")
        if svc == "tts":
            return httpx.Response(200, content=_wav())
        return httpx.Response(200, json={"text": state["heard"]})

    monkeypatch.setattr(smoke, "_transport", lambda: httpx.MockTransport(handler))

    class FakeOCS:
        user = "alice"

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return None

        async def list_rooms(self):
            return state["rooms"]

        async def signaling_settings(self, token):
            assert token == "room0001"
            return state["settings"]

    monkeypatch.setattr(smoke, "TalkOCS", FakeOCS)

    class FakeLips:
        def __init__(self, url, token, avatar="lumina", **kw):
            self.url, self.token, self.closed = url, token, False

        def start(self):
            pass

        async def wait_ready(self, timeout):
            return True

        async def aclose(self):
            self.closed = True

    monkeypatch.setattr(smoke, "LipsyncClient", FakeLips)

    class FakeTurnTransport:
        def get_extra_info(self, name, default=None):
            return ("192.0.2.99", 49152) if name == "sockname" else default

        def close(self):
            state["allocs"].append("closed")

    async def fake_turn(factory, server_addr, username, password, **kw):
        state["allocs"].append((server_addr, username, password, kw.get("transport", "udp")))
        state.setdefault("ssl", []).append(kw.get("ssl"))
        return FakeTurnTransport(), factory()

    async def fake_resolve(host, port):
        return "192.0.2.50"

    monkeypatch.setattr(smoke, "create_turn_endpoint", fake_turn)
    monkeypatch.setattr(smoke, "_resolve", fake_resolve)
    return state


def _lines(out: str) -> list[str]:
    return [line for line in out.splitlines() if line.strip()]


def test_all_ok_exit_0(env, fakes, capsys):
    code = cli.main(["smoke", "--agent", "lumina"])
    lines = _lines(capsys.readouterr().out)
    assert code == 0, lines
    assert [line.split()[:2] for line in lines] == [
        ["ok", "tts"], ["ok", "stt"], ["ok", "lipsync"], ["ok", "talk"], ["ok", "turn"]]
    turn = lines[-1]
    assert "turn.example.org" in turn and "192.0.2.99:49152" in turn
    assert fakes["allocs"][0] == (("192.0.2.50", 3478), "u", "c", "udp")
    assert "c" not in turn.split() and "secret" not in "\n".join(lines)


def test_only_selected_checks_run(env, fakes, capsys):
    assert cli.main(["smoke", "--talk"]) == 0
    lines = _lines(capsys.readouterr().out)
    assert len(lines) == 1 and lines[0].startswith("ok talk ")


def test_interface_too_old_fails(env, fakes, capsys):
    fakes["interface"]["tts"] = 0
    code = cli.main(["smoke", "--tts"])
    lines = _lines(capsys.readouterr().out)
    assert code == 1
    assert len(lines) == 1 and lines[0].startswith("FAIL tts ")
    assert "interface 0" in lines[0] and "1" in lines[0]


def test_turn_host_mismatch_fails(env, fakes, capsys):
    fakes["settings"] = _settings("turn.other.example")
    code = cli.main(["smoke", "--turn"])
    lines = _lines(capsys.readouterr().out)
    assert code == 1
    assert len(lines) == 1 and lines[0].startswith("FAIL turn ")
    assert "turn.other.example" in lines[0] and "turn.example.org" in lines[0]
    assert fakes["allocs"] == []


def test_no_turn_offered_fails(env, fakes, capsys):
    fakes["settings"] = _settings()
    code = cli.main(["smoke", "--turn"])
    lines = _lines(capsys.readouterr().out)
    assert code == 1
    assert len(lines) == 1 and lines[0].startswith("FAIL turn ")
    assert "no TURN" in lines[0]


def test_talk_missing_room_fails(env, fakes, capsys):
    env.setenv("SKCALL_ALLOWED_ROOMS", "room0001,room0002")
    assert cli.main(["smoke", "--talk"]) == 1
    line = _lines(capsys.readouterr().out)[0]
    assert line.startswith("FAIL talk ") and "room0002" in line


@pytest.mark.parametrize("check,var", [
    ("tts", "SKCALL_TTS_URL"), ("stt", "SKCALL_STT_BACKENDS"), ("lipsync", "SKCALL_LIPSYNC_TOKEN"),
    ("lipsync", "SKCALL_LIPSYNC_URL"),
    ("talk", "SKCALL_NC_URL"), ("turn", "NEXTCLOUD_TALK_APP_PASSWORD")])
def test_missing_config_fails_naming_the_variable(env, fakes, capsys, check, var):
    env.delenv(var)
    assert cli.main(["smoke", f"--{check}"]) == 1
    lines = _lines(capsys.readouterr().out)
    assert len(lines) == 1 and lines[0].startswith(f"FAIL {check} ") and var in lines[0]


def test_site_env_files_are_loaded_like_the_unit(env, fakes, capsys, tmp_path):
    env.delenv("SKCALL_NC_URL")
    env.delenv("SKCALL_ALLOWED_ROOMS")
    (tmp_path / "skcall.env").write_text("SKCALL_NC_URL=https://example.org\nSKCALL_ALLOWED_ROOMS=room0009\n")
    (tmp_path / "join.env").write_text("SKCALL_ALLOWED_ROOMS='room0001'\n")
    env.setattr(smoke, "SITE_ENV_FILES", (str(tmp_path / "skcall.env"), str(tmp_path / "join.env")))
    assert cli.main(["smoke", "--talk"]) == 0, capsys.readouterr().out


def test_env_file_stays_the_talk_login_file(env, fakes, capsys, tmp_path):
    """SKCALL_ENV_FILE keeps its one meaning (the Talk login, as for a call);
    the site files are the unit's fixed ones."""
    env.delenv("NEXTCLOUD_TALK_USERNAME")
    env.delenv("NEXTCLOUD_TALK_APP_PASSWORD")
    (tmp_path / "login.env").write_text("NEXTCLOUD_TALK_USERNAME=alice\nNEXTCLOUD_TALK_APP_PASSWORD=secret\n")
    env.setenv("SKCALL_ENV_FILE", str(tmp_path / "login.env"))
    assert cli.main(["smoke", "--talk"]) == 0, capsys.readouterr().out


def test_tts_each_server_must_speak(env, fakes, capsys):
    other = "http://192.0.2.13:18793/v1/audio/speech"
    env.setenv("SKCALL_TTS_URL", f"{TTS},{other}")
    fakes["down"].add("192.0.2.13")
    assert cli.main(["smoke", "--tts"]) == 1
    lines = _lines(capsys.readouterr().out)
    assert len(lines) == 1 and lines[0].startswith("FAIL tts ")
    assert "192.0.2.13" in lines[0] and "192.0.2.10" not in lines[0]


def test_turns_connects_by_host_name_with_tls(env, fakes, capsys):
    fakes["settings"] = {"turnservers": [{"urls": ["turns:turn.example.org:443?transport=tcp"],
                                          "username": "u", "credential": "c"}]}
    assert cli.main(["smoke", "--turn"]) == 0, capsys.readouterr().out
    addr, _, _, transport = fakes["allocs"][0]
    assert addr == ("turn.example.org", 443) and transport == "tcp"
    assert fakes["ssl"] == [True]


def _auth(fakes, path_part):
    return [r.headers.get("authorization") for r in fakes["requests"] if path_part in r.url.path]


def test_tokens_sent_when_set_and_never_printed(env, fakes, capsys):
    env.setenv("SKCALL_TTS_TOKEN", "tts-tok-7f3a")
    env.setenv("SKCALL_STT_TOKEN", "stt-tok-91c2")
    assert cli.main(["smoke", "--tts", "--stt"]) == 0
    out = capsys.readouterr()
    assert set(_auth(fakes, "/audio/speech")) == {"Bearer tts-tok-7f3a"}
    assert set(_auth(fakes, "/audio/transcriptions")) == {"Bearer stt-tok-91c2"}
    assert set(_auth(fakes, "/v1/info")) == {None}
    assert "tok-" not in out.out + out.err


def test_no_token_no_header(env, fakes, capsys):
    assert cli.main(["smoke", "--tts", "--stt"]) == 0
    assert all("authorization" not in r.headers for r in fakes["requests"])


def test_call_clients_send_the_tokens(monkeypatch):
    """Calls build F5TTS and the STT backends the same way: the header comes from the env."""
    from skcall.speech import F5TTS, stt_backend
    monkeypatch.setenv("SKCALL_TTS_TOKEN", "t1")
    monkeypatch.setenv("SKCALL_STT_TOKEN", "s1")
    assert F5TTS(url=TTS, model="tts-1")._http.headers.get("authorization") == "Bearer t1"
    assert stt_backend(f"fw:{STT}")._http.headers.get("authorization") == "Bearer s1"
    assert stt_backend(STT)._http.headers.get("authorization") == "Bearer s1"
    monkeypatch.delenv("SKCALL_TTS_TOKEN")
    monkeypatch.delenv("SKCALL_STT_TOKEN")
    assert "authorization" not in F5TTS(url=TTS, model="tts-1")._http.headers
    assert "authorization" not in stt_backend(f"fw:{STT}")._http.headers


# ------------------------------------------------------------ R14: no flags
def test_no_flags_unconfigured_services_are_ok_not_configured(env, fakes, capsys):
    """No flag: a service whose URL is unset is not configured, not a failure."""
    for var in ("SKCALL_TTS_URL", "SKCALL_STT_BACKENDS", "SKCALL_LIPSYNC_URL", "SKCALL_LIPSYNC_TOKEN"):
        env.delenv(var)
    code = cli.main(["smoke"])
    lines = _lines(capsys.readouterr().out)
    assert code == 0, lines
    assert lines[:3] == ["ok tts not configured", "ok stt not configured", "ok lipsync not configured"]
    assert lines[3].startswith("ok talk ") and lines[4].startswith("ok turn ")


def test_no_flags_without_lipsync_still_checks_the_rest(env, fakes, capsys):
    env.delenv("SKCALL_LIPSYNC_URL")
    env.delenv("SKCALL_LIPSYNC_TOKEN")
    assert cli.main(["smoke"]) == 0
    lines = _lines(capsys.readouterr().out)
    assert lines[2] == "ok lipsync not configured"
    assert [line.split()[:2] for line in lines] == [
        ["ok", "tts"], ["ok", "stt"], ["ok", "lipsync"], ["ok", "talk"], ["ok", "turn"]]


def test_no_flags_partly_configured_service_fails(env, fakes, capsys):
    """The URL is set but the token is not: configured, so it is checked and FAILs."""
    env.delenv("SKCALL_LIPSYNC_TOKEN")
    assert cli.main(["smoke"]) == 1
    line = [x for x in _lines(capsys.readouterr().out) if " lipsync " in f" {x} "][0]
    assert line.startswith("FAIL lipsync ") and "SKCALL_LIPSYNC_TOKEN" in line


def test_no_flags_talk_and_turn_always_run(env, fakes, capsys):
    for var in ("SKCALL_NC_URL", "SKCALL_TTS_URL", "SKCALL_STT_BACKENDS", "SKCALL_LIPSYNC_URL"):
        env.delenv(var)
    assert cli.main(["smoke"]) == 1
    lines = _lines(capsys.readouterr().out)
    assert lines[3].startswith("FAIL talk ") and "SKCALL_NC_URL" in lines[3]
    assert lines[4].startswith("FAIL turn ") and "SKCALL_NC_URL" in lines[4]


# ------------------------------------------------------- R15: /v1/info 404
def test_info_404_is_unversioned_and_the_request_decides(env, fakes, capsys):
    fakes["info"]["192.0.2.10"] = httpx.Response(404, text="Not Found")
    fakes["info"]["192.0.2.11"] = httpx.Response(404, text="Not Found")
    assert cli.main(["smoke", "--tts", "--stt"]) == 0
    lines = _lines(capsys.readouterr().out)
    assert lines[0].startswith("ok tts ") and "unversioned" in lines[0] and "192.0.2.10" in lines[0]
    assert lines[1].startswith("ok stt ") and "unversioned" in lines[1] and "192.0.2.11" in lines[1]
    assert any(r.url.path == "/v1/audio/speech" for r in fakes["requests"])
    assert any(r.url.path == "/v1/audio/transcriptions" for r in fakes["requests"])


def test_info_404_and_the_request_fails_is_a_fail(env, fakes, capsys):
    fakes["info"]["192.0.2.10"] = httpx.Response(404, text="Not Found")
    fakes["down"].add("192.0.2.10")
    assert cli.main(["smoke", "--tts"]) == 1
    line = _lines(capsys.readouterr().out)[0]
    assert line.startswith("FAIL tts ") and "192.0.2.10" in line


def test_info_versioned_server_says_interface_ok(env, fakes, capsys):
    assert cli.main(["smoke", "--tts"]) == 0
    line = _lines(capsys.readouterr().out)[0]
    assert "interface ok" in line and "unversioned" not in line


@pytest.mark.parametrize("resp,why", [
    (httpx.Response(500, text="boom"), "HTTP 500"),
    (httpx.Response(401, text="no"), "HTTP 401"),
    (httpx.Response(200, json={"service": "stt", "interface": 1}), "not 'tts'"),
])
def test_info_other_errors_still_fail(env, fakes, capsys, resp, why):
    fakes["info"]["192.0.2.10"] = resp
    assert cli.main(["smoke", "--tts"]) == 1
    line = _lines(capsys.readouterr().out)[0]
    assert line.startswith("FAIL tts ") and why in line
    assert not any(r.url.path == "/v1/audio/speech" for r in fakes["requests"])


# ---------------------------------------------------------- R16: --env-file
def test_env_file_flags_replace_the_default_files(env, fakes, capsys, tmp_path):
    env.delenv("SKCALL_NC_URL")
    env.delenv("SKCALL_ALLOWED_ROOMS")
    default = tmp_path / "default.env"
    default.write_text("SKCALL_ALLOWED_ROOMS=room0009\n")
    env.setattr(smoke, "SITE_ENV_FILES", (str(default),))
    (tmp_path / "a.env").write_text("SKCALL_NC_URL=https://example.org\nSKCALL_ALLOWED_ROOMS=room0008\n")
    (tmp_path / "secrets.env").write_text("export SKCALL_ALLOWED_ROOMS=room0001\n")
    code = cli.main(["smoke", "--talk", "--env-file", str(tmp_path / "a.env"),
                     "--env-file", str(tmp_path / "secrets.env")])
    assert code == 0, capsys.readouterr().out
    assert os.environ["SKCALL_ALLOWED_ROOMS"] == "room0001"


def test_env_file_never_overrides_the_environment(env, fakes, capsys, tmp_path):
    (tmp_path / "a.env").write_text("SKCALL_ALLOWED_ROOMS=room0009\n")
    assert cli.main(["smoke", "--talk", "--env-file", str(tmp_path / "a.env")]) == 0
    assert os.environ["SKCALL_ALLOWED_ROOMS"] == "room0001"


def test_env_file_missing_fails_clearly(env, fakes, capsys, tmp_path):
    gone = tmp_path / "secrets.env"
    assert cli.main(["smoke", "--talk", "--env-file", str(gone)]) == 1
    lines = _lines(capsys.readouterr().out)
    assert len(lines) == 1 and lines[0].startswith("FAIL env-file ") and str(gone) in lines[0]
    assert fakes["requests"] == []


# ------------------------------------------------------------- m1: TURN timeout
def test_turn_allocation_timeout_names_the_turn_timeout(env, fakes, capsys):
    import asyncio

    async def hang(*a, **kw):
        await asyncio.sleep(60)

    env.setattr(smoke, "create_turn_endpoint", hang)
    env.setattr(smoke, "TURN_TIMEOUT_S", 0.05)
    assert cli.main(["smoke", "--turn"]) == 1
    line = _lines(capsys.readouterr().out)[0]
    assert line.startswith("FAIL turn ") and "TURN allocation timed out after 0.05 s" in line
    assert "120" not in line


def test_stt_matches_the_end_of_the_phrase_not_its_first_word(fakes, env):
    """The recognizer often drops the first words of a cloned voice: hearing the
    end of the phrase passes, a mishearing of it fails and says what was heard."""
    fakes["heard"] = "this is a quick Speech Test."
    assert cli.main(["smoke", "--tts", "--stt"]) == 0
    fakes["heard"] = "Tess?"
    assert cli.main(["smoke", "--tts", "--stt"]) == 1
