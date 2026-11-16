"""Client for the Nextcloud Talk High Performance Backend (nextcloud-spreed-signaling).

Clean implementation from the public protocol doc
(strukturag/nextcloud-spreed-signaling docs/standalone-signaling-api-v1.md),
cross-checked against nextcloud/live_transcription (AGPL) and
voxonta/nextcloud-talk-capture (MIT).

User-session flow (option A in docs/ARCHITECTURE.md):
  1. GET signaling/settings -> server URL + helloAuthParams
  2. ws connect <server>/spreed
  3. hello v2.0 with the JWT (fallback v1.0 with userid + ticket)
  4. room {roomid, sessionid=<Nextcloud sessionId from REST participants/active>}
"""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
import time
from typing import Any, Awaitable, Callable

import aiohttp

LOG = logging.getLogger("skcall.signaling")

Handler = Callable[[dict], Awaitable[None] | None]


class SignalingError(RuntimeError):
    pass


class SignalingClient:
    def __init__(self, server: str, backend_url: str, auth_params: dict,
                 name: str = "sig", features: list[str] | None = None):
        self.server = server.rstrip("/")
        self.ws_url = self.server.replace("https://", "wss://").replace("http://", "ws://") + "/spreed"
        self.backend_url = backend_url
        self.auth_params = auth_params
        self.name = name
        self.features = features if features is not None else []
        self.sessionid: str | None = None     # our HPB session id
        self.userid: str | None = None
        self.hello_version: str | None = None
        self.resumeid: str | None = None      # lets a new websocket take this session over
        self.server_info: dict = {}
        self._http: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._ids = itertools.count(1)
        self._pending: dict[str, asyncio.Future] = {}
        self._handlers: list[Handler] = []
        self._reader: asyncio.Task | None = None
        self.closed = asyncio.Event()

    # --- plumbing -------------------------------------------------------
    def on(self, handler: Handler) -> None:
        """Register a handler called for every unsolicited message."""
        self._handlers.append(handler)

    async def connect(self) -> None:
        self._http = aiohttp.ClientSession()
        self._ws = await self._http.ws_connect(self.ws_url, heartbeat=30)
        self._reader = asyncio.create_task(self._read_loop(), name=f"{self.name}-reader")

    async def _read_loop(self) -> None:
        assert self._ws is not None
        try:
            async for msg in self._ws:
                if msg.type != aiohttp.WSMsgType.TEXT:
                    continue
                data = json.loads(msg.data)
                LOG.debug("%s <- %s", self.name, _short(data))
                mid = data.get("id")
                if mid and mid in self._pending:
                    fut = self._pending.pop(mid)
                    if not fut.done():
                        fut.set_result(data)
                    continue
                for h in list(self._handlers):
                    try:
                        r = h(data)
                        if asyncio.iscoroutine(r):
                            await r
                    except Exception:
                        LOG.exception("%s handler failed", self.name)
        finally:
            self.closed.set()
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(SignalingError("websocket closed"))

    async def send(self, msg: dict, *, wait: bool = False, timeout: float = 10) -> dict | None:
        assert self._ws is not None
        fut = None
        if wait:
            msg = {"id": f"{self.name}-{next(self._ids)}", **msg}
            fut = asyncio.get_running_loop().create_future()
            self._pending[msg["id"]] = fut
        LOG.debug("%s -> %s", self.name, _short(msg))
        await self._ws.send_str(json.dumps(msg))
        if fut is None:
            return None
        resp = await asyncio.wait_for(fut, timeout)
        if resp.get("type") == "error":
            raise SignalingError(json.dumps(resp.get("error")))
        return resp

    # --- protocol -------------------------------------------------------
    async def hello(self, prefer: str = "2.0") -> dict:
        order = [prefer] + [v for v in ("2.0", "1.0") if v != prefer]
        last: Exception | None = None
        for version in order:
            params = self.auth_params.get(version)
            if not params:
                continue
            try:
                resp = await self.send({"type": "hello", "hello": {
                    "version": version,
                    "features": self.features,
                    "auth": {"url": self.backend_url, "params": params},
                }}, wait=True)
            except SignalingError as e:
                LOG.warning("%s hello %s refused: %s", self.name, version, e)
                last = e
                continue
            h = resp["hello"]
            self.sessionid = h["sessionid"]
            self.userid = h.get("userid")
            self.server_info = h.get("server", {})
            self.hello_version = version
            self.resumeid = h.get("resumeid")
            return resp
        raise SignalingError(f"all hello versions failed: {last}")

    async def resume(self, resumeid: str, version: str = "2.0") -> dict:
        """Take over an existing HPB session on this (new) websocket: hello with
        only ``resumeid`` (standalone-signaling-api-v1 "Resuming sessions").
        The server keeps the session, its room and its streams for a short
        while after a websocket drops; past that it answers ``no_such_session``
        (raised as SignalingError) and the caller starts a fresh session."""
        resp = await self.send({"type": "hello", "hello": {"version": version, "resumeid": resumeid}},
                               wait=True)
        h = resp["hello"]
        self.sessionid = h["sessionid"]
        self.userid = h.get("userid")
        self.server_info = h.get("server", {})
        self.hello_version = version
        self.resumeid = h.get("resumeid") or resumeid
        return resp

    async def join_room(self, roomid: str, nc_sessionid: str) -> dict:
        return await self.send({"type": "room", "room": {
            "roomid": roomid, "sessionid": nc_sessionid}}, wait=True)

    async def leave_room(self) -> None:
        try:
            await self.send({"type": "room", "room": {"roomid": ""}}, wait=True, timeout=5)
        except Exception as e:  # best effort on teardown
            LOG.warning("%s leave_room: %s", self.name, e)

    async def message_session(self, sessionid: str, data: dict) -> None:
        await self.send({"type": "message", "message": {
            "recipient": {"type": "session", "sessionid": sessionid}, "data": data}})

    async def request_offer(self, publisher_sid: str, room_type: str = "video") -> None:
        await self.message_session(publisher_sid, {"type": "requestoffer", "roomType": room_type})

    async def send_sdp(self, to_sid: str, kind: str, sid: str, sdp: str,
                       room_type: str = "video", nick: str = "skcall") -> None:
        """kind is 'offer' (publish: to_sid == our own session) or 'answer' (subscribe)."""
        await self.message_session(to_sid, {
            "to": to_sid, "type": kind, "roomType": room_type, "sid": sid,
            "payload": {"type": kind, "sdp": sdp, "nick": nick}})

    async def send_candidate(self, to_sid: str, sid: str, candidate: str,
                             mid: str = "0", mline: int = 0, room_type: str = "video") -> None:
        await self.message_session(to_sid, {
            "to": to_sid, "type": "candidate", "sid": sid, "roomType": room_type,
            "payload": {"candidate": {"candidate": candidate, "sdpMid": mid,
                                      "sdpMLineIndex": mline}}})

    async def send_media_state(self, to_sid: str, name: str, on: bool) -> None:
        """Tell one session our audio or video is on ("unmute") or off ("mute")."""
        await self.message_session(to_sid, media_state_message(to_sid, name, on))

    async def send_nick(self, to_sid: str, nick: str) -> None:
        await self.message_session(to_sid, nick_message(to_sid, nick))

    async def close(self) -> None:
        if self._ws is not None and not self._ws.closed:
            try:
                await self.send({"type": "bye", "bye": {}}, wait=True, timeout=3)
            except Exception:
                pass
            await self._ws.close()
        if self._reader:
            self._reader.cancel()
            try:
                await self._reader
            except (asyncio.CancelledError, Exception):
                pass
        if self._http:
            await self._http.close()


def media_state_message(to_sid: str, name: str, on: bool) -> dict:
    """The "unmute"/"mute" call message Talk's web client sends with sendTo()
    (spreed src/utils/webrtc/webrtc.js webrtc.sendTo, LocalStateBroadcasterMcu
    _sendCurrentStateTo). The receiving client turns it into videoAvailable /
    audioAvailable on our CallParticipantModel; without it, video stays hidden."""
    if name not in ("audio", "video"):
        raise ValueError(f"media name must be audio or video, not {name!r}")
    return {"to": to_sid, "roomType": "video", "type": "unmute" if on else "mute",
            "payload": {"name": name}}


def nick_message(to_sid: str, nick: str) -> dict:
    return {"to": to_sid, "roomType": "video", "type": "nickChanged", "payload": {"name": nick}}


def data_channel_message(kind: str, payload: Any = None) -> str:
    """One message for the "status" data channel, in the JSON shape Talk's web
    client sends (spreed src/utils/webrtc/simplewebrtc/peer.js sendDirectly:
    JSON.stringify({type, payload}), payload left out when undefined).

    The mobile apps read media state ONLY from these: talk-ios starts every
    remote peer with isRemoteVideoDisabled = true and clears it only on a
    "videoOn" data channel message (NCPeerConnection.swift
    setStatusForDataChannelMessageType); its handler for the signaling
    "unmute" message sends a data channel message back instead of applying it
    (NCCallController.swift processMuteUnmute). talk-android has no handler for
    signaling "unmute" at all."""
    msg: dict[str, Any] = {"type": kind}
    if payload is not None:
        msg["payload"] = payload
    return json.dumps(msg, separators=(",", ":"))


def status_messages(media: list[tuple[str, bool]], nick: str, userid: str | None = None) -> list[str]:
    """The data channel state burst Talk's web client sends to everyone
    (LocalStateBroadcasterMcu._sendCurrentStateToAll): audioOn/audioOff,
    videoOn/videoOff, then nickChanged. The nickChanged payload is the bare
    name for a guest and {name, userid} for a user
    (LocalStateBroadcaster._getNickChangedDataChannelMessagePayload)."""
    out = []
    for name, on in media:
        if name not in ("audio", "video"):
            raise ValueError(f"media name must be audio or video, not {name!r}")
        out.append(data_channel_message(name + ("On" if on else "Off")))
    out.append(data_channel_message("nickChanged", {"name": nick, "userid": userid} if userid else nick))
    return out


def _short(d: Any, n: int = 600) -> str:
    s = json.dumps(d)
    # never log credentials that ride in hello
    if '"hello"' in s and '"auth"' in s:
        return '{"type":"hello", ...auth redacted...}'
    return s if len(s) <= n else s[:n] + "..."


def now_ms() -> float:
    return time.monotonic() * 1000
