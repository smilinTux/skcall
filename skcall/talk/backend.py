"""TalkBackend: a Nextcloud Talk call as a skcall CallSession (aiortc + HPB/Janus).

Built from the P0 spike code, keeping every rule P0 learned:
- ICE host candidates only on SKCALL_ICE_IFACES prefixes when it is set
  (default every interface; STUN through a VPN interface such as tailscale0
  can stall gathering 5 s, so a host with one lists its LAN prefix);
- join the REST room with every session before any joins the call (409);
- aiortc cannot trickle: wait for gathering, then send a=candidate lines;
- buffer subscriber candidates until the remote offer is applied;
- answers/candidates for our own publisher arrive from our own HPB session id;
- each new subscription has a ~2 s server-side hold before the first RTP, so
  we subscribe to every audio publisher the moment it shows up in the call
  (and to everyone already in it right after joining), paying the hold before
  anyone speaks rather than on their first sentence.

Speaker attribution is the Nextcloud userId from the HPB participants event.
"""
from __future__ import annotations

import asyncio
import collections
import contextlib
import json
import fractions
import logging
import secrets
import time
from typing import AsyncIterator

import numpy as np
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from aiortc.sdp import candidate_from_sdp
from av import AudioFrame, VideoFrame

from skcall.agents import ice_ifaces
from skcall.core.audio import CALL_RATE
from skcall.core.session import (AudioTrack, CallEnded, CallEvent, Participant,
                                 ParticipantJoined, ParticipantLeft, VideoSource, VideoTrack)
from skcall.instance import AlreadyInCall, other_session_in_call
from skcall.talk.access import check_room
from skcall.talk.media import (FRAME, candidates_from_sdp, frame_to_mono, ice_config,
                               patch_ice)
from skcall.talk.ocs import OCSError, TalkOCS
from skcall.talk.signaling import SignalingClient, SignalingError, status_messages

LOG = logging.getLogger("skcall.talk")

PERM_JOIN_CALL = 4
PERM_PUBLISH_AUDIO = 16
PERM_PUBLISH_VIDEO = 32
VIDEO_CLOCK = 90000
# when Talk's web client re-sends its media state to a newly joined participant
# (ExponentialBackoffCallback in spreed src/utils/webrtc/LocalStateBroadcasterMcu.ts)
STATE_BACKOFF_S = (0, 1, 2, 4, 8, 16)

# Media recovery (2026-10-25: both peer connections closed 5.5 min into a call,
# the signaling session stayed up, and skcall sat silent for 8 minutes). Every
# recovery attempt (re-publish, re-subscribe, signaling reconnect) takes one
# slot of a shared budget; when it runs out the call is left cleanly.
RECONNECT_MAX = 5
RECONNECT_WINDOW_S = 120.0
RECONNECT_BACKOFF_S = (1.0, 2.0, 4.0, 8.0, 15.0)
PC_CONNECT_TIMEOUT_S = 15.0
LOST_REASON = "lost the call connection"
# The media dropped because the call itself is over on the Talk side (they hung
# up): leave at once, quietly, with no lost-connection note.
CALL_OVER_REASON = "the call ended"
# What _audio_off records for a participant in the call without the audio flag.
NO_AUDIO_FLAG = "in the call without audio"


def lost_connection_note(agent_user: str) -> str:
    """The room-chat line after giving up on a dropped call."""
    who = agent_user or "lumina"
    return f"I lost the call connection and left; say @{who} join call to bring me back."


class ReconnectBudget:
    """At most ``max_attempts`` recovery attempts in any ``window_s`` window,
    with a growing delay before each one."""

    def __init__(self, max_attempts: int = RECONNECT_MAX, window_s: float = RECONNECT_WINDOW_S,
                 backoff_s: tuple[float, ...] = RECONNECT_BACKOFF_S, clock=time.monotonic):
        self.max_attempts = max_attempts
        self.window_s = window_s
        self.backoff_s = backoff_s
        self.clock = clock
        self._attempts: collections.deque[float] = collections.deque()

    def _prune(self) -> None:
        now = self.clock()
        while self._attempts and now - self._attempts[0] > self.window_s:
            self._attempts.popleft()

    def take(self) -> float | None:
        """Claim an attempt: the delay to wait before it, or None when spent."""
        self._prune()
        if len(self._attempts) >= self.max_attempts:
            return None
        self._attempts.append(self.clock())
        return self.backoff_s[min(len(self._attempts), len(self.backoff_s)) - 1]

    @property
    def used(self) -> int:
        self._prune()
        return len(self._attempts)

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


class SinkTrack(MediaStreamTrack):
    """Outgoing audio: a queue of int16 mono 48 kHz segments, paced in real
    time as 20 ms frames, silence when empty. Implements core AudioSink."""
    kind = "audio"
    sample_rate = CALL_RATE

    def __init__(self):
        super().__init__()
        self._q: collections.deque = collections.deque()   # [pcm, offset, on_start]
        self._queued = 0
        self._n = 0
        self._t0: float | None = None
        self._drained = asyncio.Event()
        self._drained.set()
        self.frames_sent = 0

    # AudioSink -------------------------------------------------------
    def write(self, pcm: np.ndarray, on_start=None) -> None:
        if pcm.size == 0:
            return
        self._q.append([pcm.astype(np.int16), 0, on_start])
        self._queued += len(pcm)
        self._drained.clear()

    def clear(self) -> None:
        self._q.clear()
        self._queued = 0
        self._drained.set()

    @property
    def buffered_s(self) -> float:
        return self._queued / CALL_RATE

    async def drained(self) -> None:
        await self._drained.wait()

    # MediaStreamTrack -----------------------------------------------
    def _take(self) -> np.ndarray:
        out = np.zeros(FRAME, np.int16)
        filled = 0
        while filled < FRAME and self._q:
            seg = self._q[0]
            pcm, off, cb = seg
            if off == 0 and cb is not None:
                seg[2] = None
                try:
                    cb(time.monotonic())
                except Exception:
                    LOG.debug("on_start callback failed", exc_info=True)
            k = min(FRAME - filled, len(pcm) - off)
            out[filled:filled + k] = pcm[off:off + k]
            filled += k
            seg[1] = off + k
            self._queued -= k
            if seg[1] >= len(pcm):
                self._q.popleft()
        if not self._q:
            self._queued = 0
            self._drained.set()
        return out

    async def recv(self) -> AudioFrame:
        if self._t0 is None:
            self._t0 = time.monotonic()
        delay = self._t0 + self._n / CALL_RATE - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        elif delay < -0.5:          # fell far behind (event loop stall): resync, do not burst
            self._t0 = time.monotonic() - self._n / CALL_RATE
        samples = self._take()
        frame = AudioFrame.from_ndarray(samples.reshape(1, -1), format="s16", layout="mono")
        frame.sample_rate = CALL_RATE
        frame.pts = self._n
        frame.time_base = fractions.Fraction(1, CALL_RATE)
        self._n += FRAME
        self.frames_sent += 1
        return frame


class AvatarTrack(MediaStreamTrack):
    """Outgoing video: pulls the frame due at each tick from a core
    VideoSource, paced on time.monotonic() (the clock SinkTrack's on_start
    uses, which is what lets the compositor line the lips up with the audio).
    Implements core VideoSink. The source can be swapped while live."""
    kind = "video"

    def __init__(self, source: VideoSource):
        super().__init__()
        self.source = source
        self.width, self.height = source.width, source.height
        self._t0: float | None = None
        self._n = 0
        self._last: tuple[np.ndarray, np.ndarray] | None = None   # (source frame, yuv)
        self.frames_sent = 0
        self.late_ticks = 0

    def _yuv(self, rgb: np.ndarray) -> np.ndarray | None:
        # Same array object as last tick (static portrait): convert once. The
        # reference is held, so a fresh frame can never reuse its identity.
        if self._last is not None and self._last[0] is rgb:
            return self._last[1]
        src = rgb
        if cv2 is None:
            return None
        if rgb.shape[0] != self.height or rgb.shape[1] != self.width:
            rgb = cv2.resize(rgb, (self.width, self.height))
        yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV_I420)
        self._last = (src, yuv)
        return yuv

    async def recv(self) -> VideoFrame:
        fps = max(0.5, float(self.source.fps))
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
        due = self._t0 + self._n / fps
        delay = due - now
        if delay > 0:
            await asyncio.sleep(delay)
        elif delay < -0.5:                     # event loop stalled: resync, do not burst
            self.late_ticks += 1
            self._t0 = now - self._n / fps
            due = now
        rgb = self.source.frame(due)
        yuv = self._yuv(rgb)
        if yuv is not None:
            frame = VideoFrame.from_ndarray(yuv, format="yuv420p")
        else:
            frame = VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
        frame.pts = int(round((due - self._t0) * VIDEO_CLOCK))
        frame.time_base = fractions.Fraction(1, VIDEO_CLOCK)
        self._n += 1
        self.frames_sent += 1
        return frame


async def _add_candidate(pc: RTCPeerConnection, cand: dict) -> None:
    c = cand.get("candidate", "")
    if not c:
        return
    if c.startswith("candidate:"):
        c = c[len("candidate:"):]
    ic = candidate_from_sdp(c)
    ic.sdpMid = cand.get("sdpMid")
    ic.sdpMLineIndex = cand.get("sdpMLineIndex")
    await pc.addIceCandidate(ic)


async def _close_quietly(pc) -> None:
    try:
        await pc.close()
    except Exception:
        LOG.debug("peer connection close failed", exc_info=True)


class TalkSession:
    """One Talk session in one room's call: publishes `audio_out`, subscribes
    to every other participant with audio, and yields CallEvents."""

    def __init__(self, room: str, *, nick: str = "Lumina", silent: bool = True,
                 agent_users: tuple[str, ...] = (), allowed_rooms: tuple[str, ...] | None = None,
                 ocs: TalkOCS | None = None, name: str = "talk",
                 video: VideoSource | None = None, receive_video: bool = False,
                 operators: tuple[str, ...] = (), operator_dms: bool = True):
        # An unlisted room may still be a one-to-one DM with an operator; that
        # needs OCS, so it is checked in open(). Without the DM rule, fail now.
        unlisted = allowed_rooms is not None and room not in allowed_rooms
        if unlisted and not (operator_dms and operators):
            raise PermissionError(f"room {room!r} is not in the allowlist")
        self._access = (allowed_rooms, tuple(operators), operator_dms) if unlisted else None
        self.room = room
        self.nick = nick
        self.silent = silent
        self.agent_users = agent_users
        self.name = name
        self.ocs = ocs or TalkOCS()
        self.audio_out = SinkTrack()
        self.video_out: AvatarTrack | None = AvatarTrack(video) if video is not None else None
        self.receive_video = receive_video
        self.sig: SignalingClient | None = None
        self.settings: dict = {}
        self.self_user_id = ""
        self.nc_session = ""
        self.permissions = 0
        self._cfg = None
        self._pub: RTCPeerConnection | None = None
        self._subs: dict[str, RTCPeerConnection] = {}
        self._remote_set: dict[str, asyncio.Event] = {}
        self._requested: dict[str, float] = {}
        self._offer_tasks: dict[str, asyncio.Task] = {}   # hpb sid -> its requestoffer loop
        self._parts: dict[str, Participant] = {}       # hpb sid -> participant (in call)
        self._audio_off: dict[str, str] = {}   # hpb sid -> what said its audio is off (muted)
        self._ended_muted: set[str] = set()    # tracks that ended while muted: resubscribe on unmute
        self._q: asyncio.Queue = asyncio.Queue()
        self._in_room = self._in_call = False
        self._ocs_open = False
        self.sub_timing: dict[str, dict] = {}
        self._names: dict[str, str] = {}
        self._state_tasks: dict[str, asyncio.Task] = {}   # hpb sid -> media state sender
        self._status_dc = None    # "status" data channel on our publisher
        self._pc_factory = RTCPeerConnection
        self._leaving = False
        self.connection_lost = False
        self._budget = ReconnectBudget()
        self._bg: set[asyncio.Task] = set()       # recovery tasks, cancelled on leave
        self._pub_recovering = False
        self._sig_recovering = False
        self._own_sids: set[str] = set()          # every HPB session id we have had
        self.call_over = False                    # Talk says the call ended under us

    # ---------------------------------------------------------- setup
    async def open(self) -> None:
        """REST room join + signaling hello + room join (not in the call yet)."""
        patch_ice(iface_allow=ice_ifaces())
        await self.ocs.__aenter__()
        self._ocs_open = True
        if self._access is not None:
            await check_room(self.ocs, self.room, *self._access)
        room = await self.ocs.get_room(self.room)
        self.permissions = int(room.get("permissions") or 0)
        try:
            for p in await self.ocs.participants(self.room):
                if p.get("actorType") == "users":
                    self._names[p["actorId"]] = p.get("displayName") or p["actorId"]
        except Exception as e:
            LOG.debug("participant names unavailable: %s", e)
        self.settings = await self.ocs.signaling_settings(self.room)
        joined = await self.ocs.join_room(self.room)
        self._in_room = True
        backend = self.ocs.base_url + "/ocs/v2.php/apps/spreed/api/v3/signaling/backend"
        self.sig = SignalingClient(self.settings["server"], backend, self.settings["helloAuthParams"],
                                   name=self.name)
        await self.sig.connect()
        await self.sig.hello()
        self.self_user_id = self.sig.userid or self.ocs.user
        self.sig.on(self._on_msg)
        await self.sig.join_room(self.room, joined["sessionId"])
        self.nc_session = joined.get("sessionId", "")
        self._own_sids.add(self.sig.sessionid or "")
        self._spawn(self._watch_signaling(self.sig))
        await self.check_not_in_call()
        self._cfg, used = ice_config(self.settings)
        LOG.info("%s: in room %s as %s (perms %d), ice %s", self.name, self.room,
                 self.self_user_id, self.permissions, used)

    async def check_not_in_call(self) -> None:
        """Bail (AlreadyInCall) if our Nextcloud user is already in this room's
        call from another session: a second join would add a second tile."""
        try:
            parts = await self.ocs.participants(self.room)
        except Exception as e:
            LOG.warning("%s: participant check failed (%s); joining anyway", self.name, e)
            return
        if other_session_in_call(parts, self.ocs.user, {self.nc_session} - {""}):
            raise AlreadyInCall(f"{self.ocs.user} is in the {self.room} call from another session")

    @property
    def can_speak(self) -> bool:
        return bool(self.permissions & PERM_PUBLISH_AUDIO) or self.permissions == 0

    @property
    def can_show(self) -> bool:
        return bool(self.permissions & PERM_PUBLISH_VIDEO) or self.permissions == 0

    async def join_call(self) -> None:
        if self.permissions and not self.permissions & PERM_JOIN_CALL:
            raise PermissionError("no join-call permission in this room")
        if self.video_out is not None and not (self.can_speak and self.can_show):
            LOG.warning("%s: no publish-video permission in %s; audio only", self.name, self.room)
            self.video_out = None
        flags = (7 if self.video_out is not None else 3) if self.can_speak else 1
        await self.ocs.join_call(self.room, flags=flags, silent=self.silent)
        self._in_call = True
        if self.can_speak:
            await self._publish()
        # Publishers already in the call: our own join makes the HPB send a
        # participants update listing everyone in it, and _on_participants
        # sends requestoffer for each one right away (pre-subscribe).

    async def _publish(self) -> None:
        assert self.sig is not None
        pub = self._pc_factory(self._cfg)
        self._pub = pub
        pub.on("connectionstatechange", lambda: self._on_pub_state(pub))
        pub.on("iceconnectionstatechange",
               lambda: LOG.info("%s publisher ice %s", self.name, pub.iceConnectionState))
        pub.addTransceiver(self.audio_out, direction="sendonly")
        # The "status" data channel every Talk client opens on its publisher.
        # Janus relays what we send on it to every subscriber that negotiated
        # data, and the iOS and Android apps take audio/video on/off and the
        # nick ONLY from it: without "videoOn" here talk-ios never shows our
        # video, whatever the signaling "unmute" says.
        dc = pub.createDataChannel("status")
        self._status_dc = dc
        dc.on("open", self._broadcast_status)
        if self.video_out is not None:
            from aiortc.rtcrtpsender import RTCRtpSender
            vt = pub.addTransceiver(self.video_out, direction="sendonly")
            # VP8: what Talk's web client and Janus both take without question
            vp8 = [c for c in RTCRtpSender.getCapabilities("video").codecs
                   if c.mimeType.lower() in ("video/vp8", "video/rtx")]
            if vp8:
                vt.setCodecPreferences(vp8)
        await pub.setLocalDescription(await pub.createOffer())
        me = self.sig.sessionid
        await self.sig.send_sdp(me, "offer", secrets.token_hex(8), pub.localDescription.sdp, nick=self.nick)
        for sid in list(self._parts):
            self._announce(sid)

    # ---------------------------------------------------------- media state
    def media_state(self) -> list[tuple[str, bool]]:
        """What we tell other clients: [(name, on)] for audio and video."""
        return [("audio", self._pub is not None),
                ("video", self._pub is not None and self.video_out is not None)]

    def _announce(self, sid: str) -> None:
        """Send our media state + nick to one participant, repeated with backoff.

        Talk's web client keeps a remote participant's video hidden (avatar
        placeholder) until it gets an "unmute" for video: videoAvailable starts
        undefined and only the "unmute"/"videoOn" messages, or a getStats probe
        that gives up for good on a first sample under 2000 bytes, set it. Its
        own publishers send the state to each participant that joins at 0, 1,
        2, 4, 8 and 16 s (LocalStateBroadcasterMcu), because a message that
        arrives before that client has subscribed to us is dropped. Same here."""
        old = self._state_tasks.pop(sid, None)
        if old is not None:
            old.cancel()
        self._state_tasks[sid] = asyncio.ensure_future(self._send_state_with_backoff(sid))

    def status_messages(self) -> list[str]:
        return status_messages(self.media_state(), self.nick, self.self_user_id or None)

    def _broadcast_status(self) -> None:
        """Our state on the "status" data channel, to everyone at once (Janus
        fans it out). Sent when the channel opens and on every backoff tick of
        _announce, since a subscriber whose own data channel is not open yet
        misses it, just like the web client does it."""
        dc = self._status_dc
        if dc is None or dc.readyState != "open":
            return
        for m in self.status_messages():
            dc.send(m)

    async def _send_state_with_backoff(self, sid: str, delays: tuple[float, ...] | None = None) -> None:
        try:
            for d in (STATE_BACKOFF_S if delays is None else delays):
                await asyncio.sleep(d)
                if sid not in self._parts or self.sig is None:
                    return
                if self._pub is None:
                    continue
                self._broadcast_status()
                for name, on in self.media_state():
                    await self.sig.send_media_state(sid, name, on)
                await self.sig.send_nick(sid, self.nick)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            LOG.warning("%s: media state to %s failed: %s", self.name, self._who(sid), e)
        finally:
            if self._state_tasks.get(sid) is asyncio.current_task():
                self._state_tasks.pop(sid, None)

    # ---------------------------------------------------------- signaling
    async def _on_msg(self, msg: dict) -> None:
        assert self.sig is not None
        ty = msg.get("type")
        if ty == "event":
            ev = msg["event"]
            if ev.get("target") == "participants" and ev.get("type") == "update":
                await self._on_participants(ev.get("update") or {})
            elif ev.get("target") == "room" and ev.get("type") == "leave":
                for sid in ev.get("leave") or []:
                    await self._drop(sid if isinstance(sid, str) else sid.get("sessionid", ""))
            else:
                LOG.debug("%s: event %s/%s", self.name, ev.get("target"), ev.get("type"))
            return
        if ty == "bye":
            # The HPB ended our session (bye.reason, e.g. "session_expired"). A
            # resume cannot bring it back: start a fresh one.
            LOG.warning("%s: signaling bye from server: %s", self.name, (msg.get("bye") or {}).get("reason"))
            self._start_signaling_recovery(resume=False)
            return
        if ty != "message":
            # Anything else unsolicited (error, welcome, control, ...): keep a
            # trace, the 2026-10-25 drop left none.
            LOG.info("%s: unhandled signaling %s: %s", self.name, ty, str(msg)[:300])
            return
        data = msg["message"]["data"]
        sender = msg["message"]["sender"].get("sessionid", "")
        kind = data.get("type")
        if sender == self.sig.sessionid:
            if self._pub is None:
                return
            if kind == "answer":
                await self._pub.setRemoteDescription(RTCSessionDescription(sdp=data["payload"]["sdp"], type="answer"))
            elif kind == "candidate":
                await _add_candidate(self._pub, data["payload"]["candidate"])
            return
        if kind in ("mute", "unmute"):
            if (data.get("payload") or {}).get("name") == "audio":
                self._set_audio(sender, kind == "unmute", "signaling " + kind)
            return
        if kind == "offer":
            asyncio.ensure_future(self._subscribe(sender, data))
        elif kind == "candidate":
            evt = self._remote_set.setdefault(sender, asyncio.Event())
            try:
                await asyncio.wait_for(evt.wait(), 10)
                if sender in self._subs:
                    await _add_candidate(self._subs[sender], data["payload"]["candidate"])
            except asyncio.TimeoutError:
                LOG.warning("candidate for %s without an offer", sender[:8])

    async def _on_participants(self, upd: dict) -> None:
        assert self.sig is not None
        if upd.get("all") and upd.get("incall") == 0:
            await self._q.put(CallEnded("call ended for everyone"))
            return
        for u in upd.get("users") or []:
            sid = u.get("sessionId")
            if not sid or sid == self.sig.sessionid or sid in self._own_sids or u.get("internal"):
                continue     # ourselves, including a session we replaced
            in_call = int(u.get("inCall") or 0)
            if in_call & 1:
                if sid not in self._parts:
                    uid = u.get("userId") or u.get("actorId") or ""
                    p = Participant(pid=sid, user_id=uid, display=u.get("displayName") or self._names.get(uid, uid),
                                    is_agent=uid in self.agent_users)
                    self._parts[sid] = p
                    await self._q.put(ParticipantJoined(p))
                    self._announce(sid)
                # Another agent's audio is never transcribed: subscribing would
                # only feed its voice to STT and let two agents answer each other.
                if (in_call & 2 and not self._parts[sid].is_agent
                        and sid not in self._subs and sid not in self._requested):
                    self._requested[sid] = time.monotonic()
                    self._spawn(self._request_until_offer(sid))
                # inCall without the audio flag (2): no microphone, nothing to hear.
                # The flag stays set on a plain mute; that comes from the data
                # channel or the signaling "mute" message.
                if not in_call & 2:
                    self._set_audio(sid, False, NO_AUDIO_FLAG)
                elif self._audio_off.get(sid) == NO_AUDIO_FLAG:
                    self._set_audio(sid, True, "inCall audio flag set")
            elif sid in self._parts:
                await self._drop(sid)

    async def _request_until_offer(self, sid: str, tries: int = 8, every_s: float = 2.0,
                                   max_every_s: float = 15.0) -> None:
        """requestoffer, repeated until Janus offers. A participant shows up in
        the call (inCall) a moment before its publisher exists, and a
        requestoffer sent in that gap is simply lost: the first self-test run
        never heard the agent because of exactly this.

        Never gives up while the participant is still in the call: after
        `tries` quick attempts it goes on with a doubling delay capped at
        `max_every_s`. 2026-10-26: an operator's client rejoined and its publisher
        took longer than 16 s to exist in Janus (requestoffer answered
        "processing_failed"); the old loop gave up after 8 tries and she was
        deaf for the rest of the call."""
        me = asyncio.current_task()
        other = self._offer_tasks.get(sid)
        if other is not None and other is not me and not other.done():
            return                       # one loop per participant
        self._offer_tasks[sid] = me
        delay = every_s
        try:
            i = 0
            while not self._leaving:
                if sid not in self._parts or sid in self._subs:
                    return
                if self.sig is None or self._sig_recovering:
                    await asyncio.sleep(every_s)     # the signaling reset asks again itself
                    continue
                if i == tries:
                    LOG.warning("%s: no offer from %s after %d requests; still asking, up to every "
                                "%.0f s, while they are in the call", self.name, self._who(sid), tries,
                                max_every_s)
                elif i:
                    LOG.info("%s: no offer from %s yet, requesting again", self.name, self._who(sid))
                self.sub_timing[sid] = {"requestoffer": time.monotonic()}
                try:
                    await self.sig.request_offer(sid)
                except Exception as e:
                    LOG.warning("requestoffer to %s failed: %s", self._who(sid), e)
                i += 1
                if i >= tries:
                    delay = min(max_every_s, delay * 2)
                await asyncio.sleep(delay)
        finally:
            if self._offer_tasks.get(sid) is me:
                self._offer_tasks.pop(sid, None)
            if sid not in self._subs and sid not in self._parts:
                self._requested.pop(sid, None)

    # ---------------------------------------------------------- remote mute
    def audio_muted(self, sid: str) -> bool:
        """Has this participant said its audio is off? The watchdog never
        treats a muted participant's silence as a dead subscription."""
        return sid in self._audio_off

    def _set_audio(self, sid: str, on: bool, via: str) -> None:
        """Record a mute or unmute, logging only real transitions. On unmute
        after the track ended while muted, subscribe again."""
        if on == (sid not in self._audio_off):
            return
        if on:
            self._audio_off.pop(sid, None)
            LOG.info("%s: %s unmuted (%s)", self.name, self._who(sid), via)
            if sid in self._ended_muted:
                self._ended_muted.discard(sid)
                self.resubscribe(sid, "unmuted after the track ended")
        else:
            self._audio_off[sid] = via
            LOG.info("%s: %s muted (%s)", self.name, self._who(sid), via)

    def _on_status_message(self, sid: str, msg) -> None:
        """A message on a subscriber's data channel: Talk clients send
        {"type": "audioOn"|"audioOff"|...} there (the only place talk-ios and
        talk-android announce their mute state)."""
        try:
            kind = json.loads(msg).get("type") if isinstance(msg, (str, bytes)) else None
        except (ValueError, AttributeError):
            return
        if kind in ("audioOn", "audioOff"):
            self._set_audio(sid, kind == "audioOn", "data channel " + kind)

    def resubscribe(self, sid: str, why: str = "") -> bool:
        """Tear down the subscription to `sid` (if any) and ask Janus for a
        fresh offer. For audio that stopped without the peer connection
        saying so (an RTCP BYE ends aiortc's track silently; RTP can stop
        while ICE consent still passes). False when there is nothing to do."""
        p = self._parts.get(sid)
        if self._leaving or p is None or p.is_agent or self.sig is None:
            return False
        LOG.warning("%s: re-subscribing to %s%s", self.name, self._who(sid), f" ({why})" if why else "")
        pc = self._subs.pop(sid, None)       # popped first: its close is not a failure to recover
        self._remote_set.pop(sid, None)
        if pc is not None:
            self._spawn(_close_quietly(pc))
        self._requested[sid] = time.monotonic()
        self._spawn(self._request_until_offer(sid))
        return True

    async def _drop(self, sid: str) -> None:
        p = self._parts.pop(sid, None)
        task = self._state_tasks.pop(sid, None)
        if task is not None:
            task.cancel()
        pc = self._subs.pop(sid, None)
        self._remote_set.pop(sid, None)
        self._requested.pop(sid, None)
        self._audio_off.pop(sid, None)
        self._ended_muted.discard(sid)
        if pc is not None:
            await pc.close()
        if p is not None:
            await self._q.put(ParticipantLeft(p))

    async def _subscribe(self, sender: str, data: dict) -> None:
        assert self.sig is not None
        old = self._subs.pop(sender, None)
        if old is not None:
            await old.close()
        pc = self._pc_factory(self._cfg)
        self._subs[sender] = pc
        evt = self._remote_set.setdefault(sender, asyncio.Event())
        evt.clear()
        tim = self.sub_timing.setdefault(sender, {})
        tim["offer"] = time.monotonic()

        pc.on("connectionstatechange", lambda: self._on_sub_state(sender, pc))
        pc.on("iceconnectionstatechange",
              lambda: LOG.info("%s subscriber %s ice %s", self.name, self._who(sender), pc.iceConnectionState))

        @pc.on("track")
        def _track(tr):
            p = self._parts.get(sender) or Participant(pid=sender, user_id="")
            if tr.kind == "audio":
                self._q.put_nowait(AudioTrack(p, self._frames(tr, sender, pc)))
            elif tr.kind == "video" and self.receive_video:
                self._q.put_nowait(VideoTrack(p, self._video_frames(tr)))

        @pc.on("datachannel")
        def _datachannel(ch):
            ch.on("message", lambda m: self._on_status_message(sender, m))

        pc.addTransceiver("audio", direction="recvonly")
        if self.receive_video:
            pc.addTransceiver("video", direction="recvonly")
        await pc.setRemoteDescription(RTCSessionDescription(sdp=data["payload"]["sdp"], type="offer"))
        if not self.receive_video:
            # the voice agent never decodes anyone's camera: refuse the video m-line
            for t in pc.getTransceivers():
                if t.kind == "video":
                    t.direction = "inactive"
        evt.set()
        await pc.setLocalDescription(await pc.createAnswer())
        to = data.get("from", sender)
        await self.sig.send_sdp(to, "answer", data["sid"], pc.localDescription.sdp, nick=self.nick)
        for c in candidates_from_sdp(pc.localDescription.sdp):
            await self.sig.send_candidate(to, data["sid"], c)

    async def _frames(self, tr, sender: str, pc=None) -> AsyncIterator[np.ndarray]:
        first = True
        n = 0
        while True:
            try:
                f = await tr.recv()
            except Exception as e:
                # The track ended. When the peer connection is still ours and
                # the participant still here, nothing else will notice: an
                # RTCP BYE stops aiortc's decoder (the track ends) without any
                # connection state change, so no recovery runs and she is deaf.
                if (pc is not None and self._subs.get(sender) is pc and not self._leaving
                        and sender in self._parts and self.sig is not None):
                    if self.audio_muted(sender):
                        # Expected: a muted client may stop sending or end
                        # the track. Subscribe again when it unmutes.
                        LOG.info("%s: audio from %s ended after %d frames while muted; "
                                 "re-subscribing on unmute", self.name, self._who(sender), n)
                        self._ended_muted.add(sender)
                        return
                    LOG.warning("%s: audio from %s ended after %d frames while the subscription "
                                "is %s (%s)", self.name, self._who(sender), n,
                                getattr(pc, "connectionState", "?"), type(e).__name__)
                    self.resubscribe(sender, "track ended")
                return
            n += 1
            if first:
                first = False
                tim = self.sub_timing.get(sender, {})
                tim["first_frame"] = time.monotonic()
                if "connected" in tim:
                    LOG.info("%s: first audio from %s %.0f ms after connect (%.0f ms after requestoffer)",
                             self.name, self._who(sender), (tim["first_frame"] - tim["connected"]) * 1000,
                             (tim["first_frame"] - tim.get("requestoffer", tim["offer"])) * 1000)
            yield frame_to_mono(f)

    async def _video_frames(self, tr) -> AsyncIterator[tuple[float, np.ndarray]]:
        while True:
            try:
                f = await tr.recv()
            except Exception:
                return
            yield time.monotonic(), f.to_ndarray(format="rgb24")

    def _who(self, sid: str) -> str:
        p = self._parts.get(sid)
        return p.user_id if p else sid[:8]

    # ---------------------------------------------------------- recovery
    def _spawn(self, coro) -> asyncio.Task:
        task = asyncio.ensure_future(coro)
        self._bg.add(task)
        task.add_done_callback(self._bg.discard)
        return task

    def _on_pub_state(self, pub) -> None:
        state = pub.connectionState
        LOG.info("%s publisher %s", self.name, state)
        if state not in ("failed", "closed") or pub is not self._pub or self._leaving:
            return
        # Ours, and not closed by us (every close we make clears self._pub first).
        self._pub = None
        self._status_dc = None
        if not self._pub_recovering and not self._sig_recovering:
            self._pub_recovering = True      # set before the task runs: one recovery at a time
            self._spawn(self._recover_publisher(pub))

    def _on_sub_state(self, sender: str, pc) -> None:
        state = pc.connectionState
        tim = self.sub_timing.setdefault(sender, {})
        if state == "connected":
            tim["connected"] = time.monotonic()
        LOG.info("%s subscriber %s: %s", self.name, self._who(sender), state)
        if state not in ("failed", "closed") or self._subs.get(sender) is not pc or self._leaving:
            return
        # Ours, and not closed by us (_drop, _subscribe and leave pop it first).
        self._subs.pop(sender, None)
        self._remote_set.pop(sender, None)
        if state == "failed":
            self._spawn(_close_quietly(pc))
        if self._sig_recovering:
            return      # a signaling reset re-subscribes to everyone
        if sender not in self._parts:
            self._requested.pop(sender, None)
            return
        self._requested[sender] = time.monotonic()
        self._spawn(self._resubscribe(sender))

    async def _resubscribe(self, sender: str) -> None:
        delay = self._budget.take()
        if delay is None:
            await self._give_up(f"subscriber {self._who(sender)}")
            return
        await asyncio.sleep(delay)
        if self._leaving or sender in self._subs:
            return
        if not await self._still_in_call(sender):
            LOG.info("%s: %s left the call; not re-subscribing", self.name, self._who(sender))
            await self._drop(sender)
            return
        LOG.info("%s: re-subscribing to %s (attempt %d/%d)", self.name, self._who(sender),
                 self._budget.used, self._budget.max_attempts)
        await self._request_until_offer(sender)

    async def _still_in_call(self, sender: str) -> bool:
        """The participants update says so (sender in _parts) and, for a user,
        the OCS participant list agrees (inCall). OCS unreachable: trust _parts."""
        p = self._parts.get(sender)
        if p is None:
            return False
        if not p.user_id:
            return True
        try:
            parts = await self.ocs.participants(self.room)
        except Exception as e:
            LOG.debug("%s: participants check failed: %s", self.name, e)
            return True
        return any(x.get("actorType") == "users" and x.get("actorId") == p.user_id
                   and int(x.get("inCall") or 0) & 1 for x in parts)

    async def _wait_connected(self, pc, timeout: float = PC_CONNECT_TIMEOUT_S) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if pc.connectionState == "connected":
                return True
            if pc.connectionState in ("failed", "closed"):
                return False
            await asyncio.sleep(0.1)
        return pc.connectionState == "connected"

    async def _recover_publisher(self, old) -> None:
        """New offer on the same signaling session; if that does not connect,
        a fresh signaling session (what Talk's web client does: forceReconnect)."""
        self._pub_recovering = True
        try:
            await _close_quietly(old)
            fresh_session = False
            while not self._leaving:
                delay = self._budget.take()
                if delay is None:
                    await self._give_up("publisher")
                    return
                await asyncio.sleep(delay)
                if self._leaving or await self._end_if_call_over("publisher"):
                    return
                LOG.info("%s: re-publishing (attempt %d/%d%s)", self.name, self._budget.used,
                         self._budget.max_attempts, ", fresh signaling session" if fresh_session else "")
                try:
                    if self._pub is not None:
                        pass            # signaling recovery re-published meanwhile: check that one
                    elif fresh_session:
                        await self._reset_signaling(resume=False)
                    elif self.sig is None or self.sig.closed.is_set():
                        return          # the signaling watcher recovers everything
                    else:
                        await self._publish()
                    pub = self._pub
                    if pub is not None and await self._wait_connected(pub):
                        LOG.info("%s: publisher recovered", self.name)
                        return
                except Exception as e:
                    LOG.warning("%s: re-publish failed: %s", self.name, e)
                pub, self._pub, self._status_dc = self._pub, None, None
                if pub is not None:
                    await _close_quietly(pub)
                fresh_session = True
        finally:
            self._pub_recovering = False

    async def _watch_signaling(self, sig: SignalingClient) -> None:
        await sig.closed.wait()
        if sig is self.sig and not self._leaving:
            LOG.warning("%s: signaling websocket closed; reconnecting", self.name)
            self._start_signaling_recovery(resume=True)

    def _start_signaling_recovery(self, resume: bool) -> None:
        if not self._sig_recovering and not self._leaving:
            self._sig_recovering = True      # set before the task runs: one recovery at a time
            self._spawn(self._recover_signaling(resume))

    async def _recover_signaling(self, resume: bool) -> None:
        self._sig_recovering = True
        try:
            while not self._leaving:
                delay = self._budget.take()
                if delay is None:
                    await self._give_up("signaling")
                    return
                await asyncio.sleep(delay)
                if self._leaving or await self._end_if_call_over("signaling"):
                    return
                try:
                    await self._reset_signaling(resume=resume)
                    LOG.info("%s: signaling recovered", self.name)
                    return
                except Exception as e:
                    LOG.warning("%s: signaling reconnect failed: %s", self.name, e)
        finally:
            self._sig_recovering = False

    def _new_signaling(self) -> SignalingClient:
        backend = self.ocs.base_url + "/ocs/v2.php/apps/spreed/api/v3/signaling/backend"
        return SignalingClient(self.settings["server"], backend, self.settings["helloAuthParams"],
                               name=self.name)

    async def _reset_signaling(self, resume: bool) -> None:
        """New websocket. With a resume id the HPB hands back the same session
        (room and streams intact). Otherwise a fresh session: rejoin the room
        with our Nextcloud session, rejoin the call, re-publish; the HPB's
        participants update then drives a new subscription to everyone."""
        # Fresh signaling settings first: the hello v2.0 JWT in them lives
        # about 60 s, so the one from open() is long expired by now (2026-10-26:
        # every fresh-session attempt was refused with token_expired).
        self.settings = await self.ocs.signaling_settings(self.room)
        old = self.sig
        resume_id = old.resumeid if (resume and old is not None) else None
        self.sig = None
        if old is not None:
            with contextlib.suppress(Exception):
                await old.close()
        new = self._new_signaling()
        try:
            await new.connect()
            resumed = False
            if resume_id:
                try:
                    await new.resume(resume_id, version=old.hello_version or "2.0")
                    resumed = True
                except (SignalingError, asyncio.TimeoutError, KeyError) as e:
                    LOG.info("%s: resume refused (%s); new session", self.name, e)
            if not resumed:
                await new.hello()
            # Live before the room join: its participants update must find self.sig.
            self.sig = new
            self._own_sids.add(new.sessionid or "")
            new.on(self._on_msg)
            if not resumed:
                # The old session's "bye" made Nextcloud drop our Nextcloud
                # session from the room (system message call_left), so joining
                # the HPB room with it is refused ("The user is not invited to
                # this room"). Join the room again over REST for a new one;
                # force replaces any leftover session of ours.
                joined = await self.ocs.join_room(self.room, force=True)
                self.nc_session = joined.get("sessionId") or self.nc_session
                await new.join_room(self.room, self.nc_session)
        except BaseException:
            if self.sig is new:
                self.sig = None
            with contextlib.suppress(Exception):
                await new.close()
            raise
        self._spawn(self._watch_signaling(new))
        if resumed:
            LOG.info("%s: signaling session resumed", self.name)
            await self._resync_media()
            return
        # The old session's streams are gone with it.
        pcs = list(self._subs.values()) + ([self._pub] if self._pub is not None else [])
        self._subs.clear()
        self._pub = None
        self._status_dc = None
        self._remote_set.clear()
        self._requested.clear()
        for pc in pcs:
            await _close_quietly(pc)
        if self._in_call:
            flags = (7 if self.video_out is not None else 3) if self.can_speak else 1
            await self.ocs.join_call(self.room, flags=flags, silent=True)
        if self.can_speak:
            await self._publish()

    async def _resync_media(self) -> None:
        """After a resume: re-publish if our publisher died meanwhile, and ask
        for an offer from everyone in the call we have no subscription to."""
        if self._in_call and self.can_speak and self._pub is None and not self._pub_recovering:
            await self._publish()
        for sid in list(self._parts):
            if sid not in self._subs and sid not in self._requested:
                self._requested[sid] = time.monotonic()
                self._spawn(self._request_until_offer(sid))

    async def _call_is_over(self) -> bool:
        """True only on Talk's word: nobody but us (and other agents) is in the
        call any more, or the room no longer lets us in (403/404). An OCS that
        cannot be reached is no evidence either way: keep recovering."""
        try:
            parts = await self.ocs.participants(self.room)
        except OCSError as e:
            return e.status in (403, 404)
        except Exception as e:
            LOG.debug("%s: participants check failed: %s", self.name, e)
            return False
        ours = {self.self_user_id or getattr(self.ocs, "user", ""), *self.agent_users}
        return not any(int(p.get("inCall") or 0) & 1 for p in parts
                       if not (p.get("actorType") == "users" and p.get("actorId") in ours))

    async def _end_if_call_over(self, what: str) -> bool:
        """Before a recovery attempt: when the call itself ended, stop and
        leave quietly instead of spending the budget on a call that is gone."""
        if self.call_over:
            return True
        if not await self._call_is_over():
            return False
        if self.call_over or self.connection_lost or self._leaving:
            return True
        self.call_over = True
        LOG.info("%s: %s dropped because the call ended on the Talk side; leaving", self.name, what)
        await self._q.put(CallEnded(CALL_OVER_REASON))
        return True

    async def _give_up(self, what: str) -> None:
        if self.connection_lost or self._leaving or self.call_over:
            return
        if await self._end_if_call_over(what):
            return
        self.connection_lost = True
        LOG.warning("%s: %s recovery used %d attempts in %.0f s; leaving the call", self.name, what,
                    self._budget.used, self._budget.window_s)
        await self._q.put(CallEnded(LOST_REASON))

    def lost_connection_note(self) -> str:
        return lost_connection_note(self.self_user_id or getattr(self.ocs, "user", ""))

    # ---------------------------------------------------------- CallSession
    def participants(self) -> list[Participant]:
        return list(self._parts.values())

    async def events(self) -> AsyncIterator[CallEvent]:
        while True:
            ev = await self._q.get()
            yield ev
            if isinstance(ev, CallEnded):
                return

    async def post_chat(self, text: str) -> None:
        await self.ocs.post_chat(self.room, text)

    async def leave(self) -> None:
        """Best effort, every step independent: call, PCs, signaling, room."""
        self._leaving = True
        current = asyncio.current_task()
        for task in list(self._bg):
            if task is not current:
                task.cancel()
        self._bg.clear()
        for task in self._state_tasks.values():
            task.cancel()
        self._state_tasks.clear()
        pcs = list(self._subs.values()) + ([self._pub] if self._pub else [])
        self._subs.clear()
        self._pub = None
        for pc in pcs:
            try:
                await pc.close()
            except Exception:
                pass
        if self._in_call:
            try:
                await self.ocs.leave_call(self.room)
            except OCSError as e:
                if e.status == 404:     # our session is already out of the call
                    LOG.info("%s leave_call: already out of the call", self.name)
                else:
                    LOG.warning("%s leave_call: %s", self.name, e)
            except Exception as e:
                LOG.warning("%s leave_call: %s", self.name, e)
            self._in_call = False
        if self.sig:
            await self.sig.leave_room()
            await self.sig.close()
            self.sig = None
        if self._in_room:
            try:
                await self.ocs.leave_room(self.room)
            except Exception as e:
                LOG.warning("%s leave_room: %s", self.name, e)
            self._in_room = False
        if self._ocs_open:
            await self.ocs.__aexit__(None, None, None)
            self._ocs_open = False
        LOG.info("%s: left %s", self.name, self.room)
