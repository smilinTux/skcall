"""sktts against a fake Kokoro pipeline (no torch, no model).

  cd services/tts && .venv/bin/python -m pytest -q
"""
from __future__ import annotations

import io
import logging

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from sktts import server

SPEECH = {"model": "tts-1", "input": "hello", "voice": "af_bella"}


class FakePipeline:
    """Kokoro's KPipeline shape: called with text, yields (graphemes, phonemes, audio)."""

    def __init__(self):
        self.calls = []

    def __call__(self, text, voice, speed):
        self.calls.append((text, voice, speed))
        if text:
            yield text, "h@loU", np.full(2400, 0.1, np.float32)


@pytest.fixture
def fake(monkeypatch):
    pipe = FakePipeline()
    monkeypatch.setattr(server, "load_pipeline", lambda settings: (pipe, "cpu"))
    for name in ("SKTTS_TOKENS_FILE", "SKTTS_DEFAULT_VOICE"):
        monkeypatch.delenv(name, raising=False)
    return pipe


def _client(**env):
    return TestClient(server.create_app(server.Settings.from_env(env)))


def test_tts_speech_returns_audio(fake):
    with _client() as c:
        r = c.post("/v1/audio/speech", json=SPEECH)
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/")
    assert len(r.content) > 0
    samples, rate = sf.read(io.BytesIO(r.content))
    assert rate == 24000 and len(samples) == 2400
    assert fake.calls == [("hello", "af_bella", 1.0)]


def test_tts_unknown_voice_and_speed(fake):
    with _client() as c:
        r = c.post("/v1/audio/speech", json={"input": "hi", "voice": "nobody", "speed": 1.25})
    assert r.status_code == 200
    assert fake.calls == [("hi", "af_bella", 1.25)]


def test_tts_empty_input_still_returns_wav(fake):
    with _client() as c:
        r = c.post("/v1/audio/speech", json={"input": ""})
    assert r.status_code == 200
    samples, _ = sf.read(io.BytesIO(r.content))
    assert len(samples) == 100


def test_tts_info(fake):
    with _client() as c:
        r = c.get("/v1/info")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "tts" and body["interface"] == 1
    assert body["auth"] == "open"


def test_tts_models_and_health(fake):
    with _client() as c:
        assert c.get("/v1/models").json() == {"object": "list", "data": [{"id": "tts-1", "object": "model"}]}
        h = c.get("/health").json()
    assert h["status"] == "healthy" and h["device"] == "cpu"


def test_tts_open_without_tokens_file_warns(fake, caplog):
    caplog.set_level(logging.INFO, logger="sktts")
    with _client() as c:
        assert c.post("/v1/audio/speech", json=SPEECH).status_code == 200
    assert "SKTTS_TOKENS_FILE" in caplog.text and "open" in caplog.text


def test_tts_tokens_enforced_when_file_set(fake, tmp_path, caplog):
    tokens = tmp_path / "tokens"
    tokens.write_text("alice=alice-secret-0001\nbob=bob-secret-0002\n")
    caplog.set_level(logging.INFO, logger="sktts")
    with _client(SKTTS_TOKENS_FILE=str(tokens)) as c:
        assert c.post("/v1/audio/speech", json=SPEECH).status_code == 401
        bad = c.post("/v1/audio/speech", json=SPEECH, headers={"Authorization": "Bearer old-shared-token"})
        assert bad.status_code == 401
        assert c.get("/v1/models").status_code == 401
        ok = c.post("/v1/audio/speech", json=SPEECH, headers={"Authorization": "Bearer bob-secret-0002"})
        assert ok.status_code == 200 and ok.headers["content-type"].startswith("audio/")
        assert c.get("/v1/info").json()["auth"] == "tokens"
        assert c.get("/health").status_code == 200
    assert "client=unknown" in caplog.text
    assert "client=bob" in caplog.text
    assert "old-shared-token" not in caplog.text and "bob-secret-0002" not in caplog.text


def test_tts_empty_tokens_file_refuses_to_start(fake, tmp_path):
    tokens = tmp_path / "tokens"
    tokens.write_text("# nobody yet\n")
    with pytest.raises(ValueError, match="no tokens"):
        server.create_app(server.Settings.from_env({"SKTTS_TOKENS_FILE": str(tokens)}))


def test_settings_defaults():
    s = server.Settings.from_env({})
    assert (s.host, s.port, s.device, s.lang_code, s.default_voice) == ("127.0.0.1", 18793, "auto", "a", "af_bella")
    assert s.tokens_file == ""
