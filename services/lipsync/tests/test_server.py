"""Protocol tests against the fake engine (no GPU, no model).

  cd services/lipsync && python -m pytest tests -q
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from aiohttp import ClientSession, WSMsgType, WSServerHandshakeError
from aiohttp.test_utils import TestServer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklipsync.engines import create_engine  # noqa: E402
from sklipsync.engines.fake import crop_value  # noqa: E402
from sklipsync.protocol import frames_for, unpack_frame  # noqa: E402
from sklipsync.server import LipsyncServer  # noqa: E402

TOKEN = "test-token"


@pytest.fixture
def avatars(tmp_path):
    d = tmp_path / "avatars" / "lumina"
    d.mkdir(parents=True)
    cv2.imwrite(str(d / "portrait.png"), np.zeros((72, 56, 3), np.uint8))
    return tmp_path / "avatars"


ALICE, BOB = "alice-secret-0001", "bob-secret-0002"


@pytest.fixture
async def server(avatars):
    eng = create_engine("fake", batch=4, delay_s=0.002)
    s = LipsyncServer({"fake": eng}, avatars, {ALICE: "alice", BOB: "bob", TOKEN: "legacy"})
    ts = TestServer(s.app())
    await ts.start_server()
    yield s, ts
    await ts.close()


async def _connect(http, ts, token=TOKEN):
    return await http.ws_connect(ts.make_url("/v1/lipsync"), headers={"Authorization": f"Bearer {token}"})


async def _hello(ws, **kw):
    await ws.send_json({"type": "hello", "avatar": "lumina", "fps": 25, "engine": "fake", **kw})
    return json.loads((await ws.receive()).data)


async def _collect(ws, uid, timeout=5.0):
    frames, done = {}, None
    while done is None:
        msg = await asyncio.wait_for(ws.receive(), timeout)
        if msg.type == WSMsgType.BINARY:
            i, u, jpeg = unpack_frame(msg.data)
            assert u == uid
            frames[i] = jpeg
        elif msg.type == WSMsgType.TEXT:
            m = json.loads(msg.data)
            if m["type"] == "done" and m["id"] == uid:
                done = m
        else:
            raise AssertionError(f"unexpected {msg.type}")
    return frames, done


async def test_rejects_bad_token(server):
    _, ts = server
    async with ClientSession() as http:
        with pytest.raises(WSServerHandshakeError) as e:
            await _connect(http, ts, token="nope")
        assert e.value.status == 401


async def test_hello_ready_and_utterance(server):
    s, ts = server
    async with ClientSession() as http:
        ws = await _connect(http, ts)
        ready = await _hello(ws)
        assert ready["type"] == "ready" and ready["fps"] == 25
        assert ready["crop"] == {"x": 200, "y": 330, "w": 160, "h": 120}
        assert ready["frame_size"] == [560, 720]
        pcm = np.zeros(int(16000 * 1.3) + 1, np.int16)     # 1.3 s + 1 sample -> 33 frames
        await ws.send_json({"type": "utterance", "id": "u1"})
        raw = pcm.tobytes()
        for i in range(0, len(raw), 3333):                  # odd chunk sizes split samples
            await ws.send_bytes(raw[i:i + 3333])
        await ws.send_json({"type": "end", "id": "u1"})
        frames, done = await _collect(ws, "u1")
        assert done["frames"] == frames_for(len(pcm), 25) == 33
        assert sorted(frames) == list(range(33))
        img = cv2.imdecode(np.frombuffer(frames[7], np.uint8), cv2.IMREAD_COLOR)
        assert img.shape == (120, 160, 3)
        assert abs(int(img.mean()) - crop_value(7)) <= 2
        assert s.last["frames"] == 33
        await ws.close()


async def test_frames_flow_before_end(server):
    _, ts = server
    async with ClientSession() as http:
        ws = await _connect(http, ts)
        await _hello(ws)
        await ws.send_json({"type": "utterance", "id": "u2"})
        await ws.send_bytes(np.zeros(16000, np.int16).tobytes())   # 1 s, no end yet
        msg = await asyncio.wait_for(ws.receive(), 2)
        assert msg.type == WSMsgType.BINARY
        assert unpack_frame(msg.data)[0] == 0
        await ws.close()


async def test_cancel_stops_rendering(avatars):
    eng = create_engine("fake", batch=2, delay_s=0.05)     # 25 frames take about 0.6 s
    s = LipsyncServer({"fake": eng}, avatars, TOKEN)
    ts = TestServer(s.app())
    await ts.start_server()
    try:
        async with ClientSession() as http:
            ws = await _connect(http, ts)
            await _hello(ws)
            await ws.send_json({"type": "utterance", "id": "c1"})
            await ws.send_bytes(np.zeros(16000 * 4, np.int16).tobytes())
            await ws.send_json({"type": "end", "id": "c1"})
            first = await asyncio.wait_for(ws.receive(), 2)
            assert first.type == WSMsgType.BINARY
            await ws.send_json({"type": "cancel", "id": "c1"})
            got_done = None
            frames_after = 0
            while True:
                try:
                    msg = await asyncio.wait_for(ws.receive(), 0.5)
                except asyncio.TimeoutError:
                    break
                if msg.type == WSMsgType.BINARY:
                    frames_after += 1
                else:
                    got_done = json.loads(msg.data)
            assert got_done and got_done.get("cancelled") is True
            assert frames_after <= 4          # at most the batch that was already in flight
            assert s.stats["cancelled"] == 1
            assert not s.active
            await ws.close()
    finally:
        await ts.close()


async def test_ping_pong_and_errors(server):
    _, ts = server
    async with ClientSession() as http:
        ws = await _connect(http, ts)
        await ws.send_json({"type": "ping"})
        assert json.loads((await ws.receive()).data) == {"type": "pong"}
        await ws.send_json({"type": "utterance", "id": "x"})
        err = json.loads((await ws.receive()).data)
        assert err["type"] == "error" and err["code"] == "no_hello"
        await ws.close()


async def test_ditto_and_unknown_avatar_error_cleanly(server):
    _, ts = server
    async with ClientSession() as http:
        ws = await _connect(http, ts)
        err = await _hello(ws, engine="ditto")
        assert err["type"] == "error" and err["code"] == "engine_unavailable"
        assert "not implemented" in err["message"]
        assert (await ws.receive()).type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.CLOSING)
        ws = await _connect(http, ts)
        err = await _hello(ws, avatar="jarvis")
        assert err["code"] == "avatar_not_found"


async def test_new_avatar_dir_is_picked_up(server, avatars):
    _, ts = server
    d = avatars / "jarvis"
    d.mkdir()
    cv2.imwrite(str(d / "portrait.jpg"), np.zeros((72, 56, 3), np.uint8))
    async with ClientSession() as http:
        ws = await _connect(http, ts)
        ready = await _hello(ws, avatar="jarvis")
        assert ready["type"] == "ready"
        async with http.get(ts.make_url("/healthz")) as r:
            h = await r.json()
        assert h["avatars"]["jarvis"] == "ready" and h["ok"]
        await ws.close()


async def test_two_clients_each_accepted(server, caplog):
    _, ts = server
    caplog.set_level(logging.INFO, logger="sklipsync")
    async with ClientSession() as http:
        for tok in (ALICE, BOB):
            ws = await _connect(http, ts, token=tok)
            assert (await _hello(ws))["type"] == "ready"
            await ws.close()
    text = caplog.text
    assert "client=alice" in text and "client=bob" in text
    assert ALICE not in text and BOB not in text


async def test_unknown_token_401(server, caplog):
    _, ts = server
    caplog.set_level(logging.INFO, logger="sklipsync")
    old_shared = "old-shared-lipsync-token"
    async with ClientSession() as http:
        with pytest.raises(WSServerHandshakeError) as e:
            await _connect(http, ts, token=old_shared)
        assert e.value.status == 401
    assert "client=unknown" in caplog.text
    assert old_shared not in caplog.text


async def test_info_unauthenticated(server):
    from sklipsync import __version__
    _, ts = server
    async with ClientSession() as http:
        async with http.get(ts.make_url("/v1/info")) as r:
            assert r.status == 200
            assert await r.json() == {"service": "lipsync", "interface": 1, "version": __version__}


def test_main_token_sources(tmp_path, monkeypatch):
    from sklipsync.__main__ import resolve_tokens
    f = tmp_path / "tokens"
    f.write_text("# clients\nalice=aaa\n\nbob=bbb\n")
    for k in ("SKLIPSYNC_TOKENS_FILE", "SKLIPSYNC_TOKEN", "SKCALL_LIPSYNC_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    assert resolve_tokens() == {}
    monkeypatch.setenv("SKLIPSYNC_TOKEN", "old")
    assert resolve_tokens() == {"old": "legacy"}
    monkeypatch.setenv("SKLIPSYNC_TOKENS_FILE", str(f))
    assert resolve_tokens() == {"aaa": "alice", "bbb": "bob", "old": "legacy"}
    monkeypatch.delenv("SKLIPSYNC_TOKEN")
    monkeypatch.setenv("SKCALL_LIPSYNC_TOKEN", "unit")
    assert resolve_tokens() == {"aaa": "alice", "bbb": "bob"}       # ignored beside a tokens file
    monkeypatch.delenv("SKLIPSYNC_TOKENS_FILE")
    assert resolve_tokens() == {"unit": "legacy"}                 # what the shipped unit sets


OLD = "old-shared-lipsync-token"


@pytest.fixture
async def env_server(avatars, tmp_path, monkeypatch):
    """Server built from resolve_tokens(); the test sets the environment first via the returned starter."""
    f = tmp_path / "tokens"
    f.write_text(f"alice={ALICE}\nbob={BOB}\n")
    for k in ("SKLIPSYNC_TOKENS_FILE", "SKLIPSYNC_TOKEN", "SKCALL_LIPSYNC_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SKLIPSYNC_TOKENS_FILE", str(f))
    started = []

    async def start():
        from sklipsync.__main__ import resolve_tokens
        s = LipsyncServer({"fake": create_engine("fake", batch=4, delay_s=0.002)}, avatars, resolve_tokens())
        ts = TestServer(s.app())
        await ts.start_server()
        started.append(ts)
        return ts

    yield start, monkeypatch
    for ts in started:
        await ts.close()


async def _expect_401(ts, caplog):
    async with ClientSession() as http:
        with pytest.raises(WSServerHandshakeError) as e:
            await _connect(http, ts, token=OLD)
        assert e.value.status == 401
    assert "client=unknown" in caplog.text and OLD not in caplog.text


async def test_old_token_refused_with_tokens_file(env_server, caplog):
    start, _ = env_server
    caplog.set_level(logging.INFO, logger="sklipsync")
    await _expect_401(await start(), caplog)


async def test_unit_token_ignored_beside_tokens_file(env_server, caplog):
    start, mp = env_server
    caplog.set_level(logging.INFO, logger="sklipsync")
    mp.setenv("SKCALL_LIPSYNC_TOKEN", OLD)
    await _expect_401(await start(), caplog)


async def test_explicit_legacy_switch_accepted(env_server, caplog):
    start, mp = env_server
    caplog.set_level(logging.INFO, logger="sklipsync")
    mp.setenv("SKLIPSYNC_TOKEN", OLD)
    ts = await start()
    async with ClientSession() as http:
        ws = await _connect(http, ts, token=OLD)
        assert (await _hello(ws))["type"] == "ready"
        await ws.close()
    assert "client=legacy" in caplog.text and OLD not in caplog.text
