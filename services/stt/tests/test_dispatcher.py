"""skstt.dispatcher against real local sockets: a stub backend, closed ports and a silent one."""
from __future__ import annotations

import json
import logging
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from hello import PATH as HELLO
from skstt import dispatcher, server
from test_server import FakeModel


def _closed_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class Backend:
    """A stand-in skstt.server: records each request, answers with fixed JSON."""

    def __init__(self, status=200, text="from backend"):
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _reply(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                self._reply(200, {"status": "ok"})

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                outer.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})
                self._reply(status, {"text": text} if status == 200 else {"detail": "boom"})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def backend():
    b = Backend()
    yield b
    b.close()


@pytest.fixture
def silent():
    """Accepts TCP connections and never answers (a hung backend)."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(16)
    yield f"http://127.0.0.1:{s.getsockname()[1]}"
    s.close()


def _client(**env):
    return TestClient(dispatcher.create_app(dispatcher.Settings.from_env(env)))


def _post(c, headers=None, **fields):
    files = {"file": ("hello.wav", HELLO.read_bytes(), "audio/wav")}
    return c.post("/v1/audio/transcriptions", files=files,
                  data={"response_format": "json", **fields}, headers=headers or {})


def test_dispatch_falls_through_to_second_backend(backend):
    dead = f"http://127.0.0.1:{_closed_port()}"
    with _client(SKSTT_BACKENDS=f"{dead},{backend.url}", SKSTT_DISPATCH_TIMEOUT_S="5") as c:
        r = _post(c, language="en", temperature="0.0", prompt="hi there",
                  headers={"Authorization": "Bearer alice-secret-0001"})
    assert r.status_code == 200 and r.json() == {"text": "from backend"}
    assert len(backend.requests) == 1
    req = backend.requests[0]
    assert req["path"] == "/v1/audio/transcriptions"
    assert req["headers"]["Authorization"] == "Bearer alice-secret-0001"
    for field in (b'name="language"', b'name="temperature"', b'name="prompt"', b'name="response_format"'):
        assert field in req["body"]
    assert HELLO.read_bytes() in req["body"]


def test_dispatch_skips_a_5xx_backend(backend):
    broken = Backend(status=500)
    try:
        with _client(SKSTT_BACKENDS=f"{broken.url},{backend.url}") as c:
            r = _post(c)
        assert r.status_code == 200 and len(broken.requests) == 1 and len(backend.requests) == 1
    finally:
        broken.close()


def test_dispatch_returns_a_4xx_as_is():
    picky = Backend(status=401)
    try:
        with _client(SKSTT_BACKENDS=picky.url) as c:
            r = _post(c)
        assert r.status_code == 401
    finally:
        picky.close()


def test_dispatch_all_down_503():
    timeout = 3.0
    urls = f"http://127.0.0.1:{_closed_port()},http://127.0.0.1:{_closed_port()}"
    with _client(SKSTT_BACKENDS=urls, SKSTT_DISPATCH_TIMEOUT_S=str(timeout)) as c:
        t = time.monotonic()
        r = _post(c)
        took = time.monotonic() - t
    assert r.status_code == 503 and r.json() == {"error": "no backend"}
    assert took < timeout + 2


def test_dispatch_hung_backend_503_within_timeout(silent):
    timeout = 1.0
    with _client(SKSTT_BACKENDS=f"{silent},{silent}", SKSTT_DISPATCH_TIMEOUT_S=str(timeout)) as c:
        t = time.monotonic()
        r = _post(c)
        took = time.monotonic() - t
    assert r.status_code == 503 and r.json() == {"error": "no backend"}
    assert took < timeout + 2


def test_dispatch_no_backends_configured_503():
    with _client() as c:
        r = _post(c)
        assert r.status_code == 503 and r.json() == {"error": "no backend"}
        assert c.get("/health").status_code == 503


def test_dispatch_info_and_health(backend):
    with _client(SKSTT_BACKENDS=backend.url) as c:
        info = c.get("/v1/info").json()
        health = c.get("/health")
    assert info["service"] == "stt" and info["interface"] == 1 and info["role"] == "dispatcher"
    assert info["backends"] == 1 and info["auth"] == "forward"
    h = health.json()
    assert health.status_code == 200 and h["dispatcher"] == "ok"
    assert h["backend_url"] == backend.url and h["backend"] == {"status": "ok"}
    assert h["backends"] == [{"backend": 0, "status_code": 200, "health": {"status": "ok"}}]


def test_dispatch_health_keeps_old_keys_when_first_is_down(backend):
    dead = f"http://127.0.0.1:{_closed_port()}"
    with _client(SKSTT_BACKENDS=f"{dead},{backend.url}") as c:
        h = c.get("/health")
    assert h.status_code == 200
    body = h.json()
    assert body["backend_url"] == dead and "backend_error" in body and "backend" not in body


@pytest.fixture
def real_server(monkeypatch):
    """A real skstt.server on a local port, with the fake model."""
    import uvicorn

    monkeypatch.setattr(server, "load_model", lambda settings: FakeModel())
    port = _closed_port()
    srv = uvicorn.Server(uvicorn.Config(server.create_app(server.Settings.from_env({})), host="127.0.0.1",
                                        port=port, log_level="warning"))
    t = threading.Thread(target=srv.run, daemon=True)
    t.start()
    deadline = time.monotonic() + 10
    while not srv.started and time.monotonic() < deadline:
        time.sleep(0.02)
    assert srv.started
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    t.join(5)


def test_dispatch_to_server_openai_default_model_name(real_server):
    with _client(SKSTT_BACKENDS=real_server) as c:
        ok = _post(c, model="whisper-1")
        unknown = _post(c, model="whisper-2")
    assert ok.status_code == 200 and ok.json() == {"text": "hello from the fake"}
    assert unknown.status_code == 400


def test_dispatch_checks_tokens_when_file_set(backend, tmp_path, caplog):
    tokens = tmp_path / "tokens"
    tokens.write_text("alice=alice-secret-0001\n")
    caplog.set_level(logging.INFO, logger="skstt")
    with _client(SKSTT_BACKENDS=backend.url, SKSTT_TOKENS_FILE=str(tokens)) as c:
        assert _post(c, headers={"Authorization": "Bearer old-shared-token"}).status_code == 401
        assert backend.requests == []
        assert _post(c, headers={"Authorization": "Bearer alice-secret-0001"}).status_code == 200
        assert c.get("/v1/info").json()["auth"] == "tokens"
    assert backend.requests[0]["headers"]["Authorization"] == "Bearer alice-secret-0001"
    assert "client=unknown" in caplog.text and "client=alice" in caplog.text
    assert "old-shared-token" not in caplog.text and "alice-secret-0001" not in caplog.text


def test_settings():
    s = dispatcher.Settings.from_env({})
    assert (s.backends, s.timeout_s, s.port, s.host) == ((), 30.0, 18794, "127.0.0.1")
    s = dispatcher.Settings.from_env({"SKSTT_BACKENDS": " http://192.0.2.1:18795/ , ,"
                                                         "http://192.0.2.2:18795/v1/audio/transcriptions"})
    assert s.backends == ("http://192.0.2.1:18795", "http://192.0.2.2:18795")
