"""skstt.server against a fake faster-whisper model (no ctranslate2, no weights).

  cd services/stt && .venv/bin/python -m pytest -q
"""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from hello import PATH as HELLO, hello_wav
from skstt import server


class FakeModel:
    """faster-whisper's WhisperModel shape: transcribe(path, **kw) -> (segments, info)."""

    def __init__(self):
        self.calls = []

    def transcribe(self, path, **kw):
        with open(path, "rb") as fh:
            self.calls.append((fh.read(), kw))
        seg = SimpleNamespace(text=" hello from the fake", start=0.0, end=1.0, seek=0, tokens=[1, 2],
                              avg_logprob=-0.1, compression_ratio=1.0, no_speech_prob=0.01)
        return iter([seg]), SimpleNamespace(duration=1.0, language="en")


@pytest.fixture
def fake(monkeypatch):
    model = FakeModel()
    loaded = {}

    def load_model(settings):
        loaded["settings"] = settings
        return model

    monkeypatch.setattr(server, "load_model", load_model)
    model.loaded = loaded
    return model


def _client(**env):
    return TestClient(server.create_app(server.Settings.from_env(env)))


def _post(c, headers=None, **fields):
    files = {"file": ("hello.wav", HELLO.read_bytes(), "audio/wav")}
    return c.post("/v1/audio/transcriptions", files=files, data={"response_format": "json", **fields},
                  headers=headers or {})


def test_bundled_hello_wav_is_the_synthesized_one():
    assert HELLO.read_bytes() == hello_wav()


def test_stt_transcribe_fake_model(fake):
    with _client() as c:
        r = _post(c, language="en", temperature="0.0")
    assert r.status_code == 200
    assert r.json() == {"text": "hello from the fake"}
    audio, kw = fake.calls[0]
    assert audio == HELLO.read_bytes()
    assert kw["language"] == "en" and kw["task"] == "transcribe" and kw["temperature"] == 0.0


def test_stt_text_and_verbose_formats(fake):
    with _client() as c:
        assert _post(c, response_format="text").text == "hello from the fake"
        v = _post(c, response_format="verbose_json").json()
    assert v["duration"] == 1.0 and v["segments"][0]["text"] == "hello from the fake"


def test_stt_rejects_bad_fields(fake):
    with _client() as c:
        assert _post(c, model="gpt-9").status_code == 400
        assert _post(c, task="dance").status_code == 400
        assert _post(c, response_format="srt").status_code == 400
        assert _post(c, temperature="2").status_code == 400
        assert _post(c, model="whisper-large-v3").status_code == 200
        assert _post(c, model="whisper-1").status_code == 200
        assert _post(c, model="whisper-2").status_code == 400


def test_stt_info_and_health(fake):
    with _client() as c:
        info = c.get("/v1/info").json()
        health = c.get("/health").json()
    assert info["service"] == "stt" and info["interface"] == 1 and info["role"] == "server"
    assert info["auth"] == "open"
    assert health["status"] == "ok" and health["device"] == "cuda" and health["compute_type"] == "int8_float16"


def test_stt_settings_reach_the_model(fake):
    with _client(SKSTT_MODEL_PATH="tiny", SKSTT_DEVICE="cpu", SKSTT_COMPUTE_TYPE="int8") as c:
        assert _post(c).status_code == 200
    s = fake.loaded["settings"]
    assert (s.model_path, s.device, s.compute_type) == ("tiny", "cpu", "int8")


def test_stt_tokens_enforced_when_file_set(fake, tmp_path, caplog):
    tokens = tmp_path / "tokens"
    tokens.write_text("alice=alice-secret-0001\n")
    caplog.set_level(logging.INFO, logger="skstt")
    with _client(SKSTT_TOKENS_FILE=str(tokens)) as c:
        assert _post(c).status_code == 401
        assert _post(c, headers={"Authorization": "Bearer old-shared-token"}).status_code == 401
        assert _post(c, headers={"Authorization": "Bearer alice-secret-0001"}).status_code == 200
        assert c.get("/v1/info").json()["auth"] == "tokens"
    assert "client=unknown" in caplog.text and "client=alice" in caplog.text
    assert "old-shared-token" not in caplog.text and "alice-secret-0001" not in caplog.text


def test_settings_defaults():
    s = server.Settings.from_env({})
    assert (s.host, s.port, s.model_path, s.device, s.compute_type) == (
        "127.0.0.1", 18794, "large-v3", "cuda", "int8_float16")
    assert s.model_id == "openai/whisper-large-v3" and s.tokens_file == ""
