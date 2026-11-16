"""skcall smoke: prove, after a start, that the services, Talk and the site's TURN work.

  skcall smoke [--agent A] [--tts] [--stt] [--lipsync] [--talk] [--turn]
               [--env-file PATH]...

Each check run prints exactly one line on stdout, `ok <check> <detail>` or
`FAIL <check> <detail>`, and the exit code is 0 only when none FAILed.

Which checks run, and what missing configuration means:
  - A check named by a flag always runs. If its configuration is missing it
    FAILs and names the variable.
  - No flag runs every check. Then tts, stt and lipsync each look at their
    service URL first (SKCALL_TTS_URL; SKCALL_STT_BACKENDS or SKCALL_STT_URL;
    SKCALL_LIPSYNC_URL): unset prints `ok <check> not configured` and does
    not fail, since a site may run without that service. Once the URL is set
    the check runs as if named, so a missing token still FAILs. talk and turn
    always run and always need their settings.

/v1/info: our services (sktts, skstt, sklipsync) answer GET /v1/info with
their service name and interface number, and an interface below
MIN_INTERFACES, or another service, FAILs. A server that answers 404 there
(F5-TTS, upstream Kokoro, whisper.cpp) is unversioned: the ok line says so,
and only the real request below decides. Any other /v1/info error FAILs.

  tts      /v1/info on every SKCALL_TTS_URL server, then one 1-word speech
           from EACH server through the client calls use
           (skcall.speech.F5TTS, with SKCALL_TTS_TOKEN); any server that
           returns no audio is named.
  stt      /v1/info on every SKCALL_STT_BACKENDS (or SKCALL_STT_URL)
           backend, then each backend (with SKCALL_STT_TOKEN) transcribes the
           TTS output of that word.
  lipsync  /v1/info on SKCALL_LIPSYNC_URL's host, then one session opened
           (token, hello, ready) and closed with the call's LipsyncClient.
  talk     OCS auth as the agent user (NEXTCLOUD_TALK_USERNAME), and the room
           list holds every SKCALL_ALLOWED_ROOMS token.
  turn     The TURN servers in Talk's signaling settings (for the first
           allowed room). Every host must equal SKCALL_SITE_TURN when that is
           set. Then one TURN allocation with Talk's credentials (aioice), on
           the server a call would use, and the relayed address is printed.

Read only: OCS GETs, one TURN allocation (deleted again), one lip-sync
session with no utterance, and speech requests. It never joins a room or a
call, and never posts.

Settings come from env files, read in order (a later file wins) and never
overriding a variable already in the environment. By default these are the
files the example join unit's EnvironmentFile lines name
(~/.config/skcall/skcall.env, then ~/.config/skcall/join.env), and a missing
default file is skipped. `--env-file PATH`, repeatable, replaces that list:
a unit template passes the same files, in the same order, as its own
EnvironmentFile lines (for example a secrets file). A file named with
--env-file that does not exist FAILs (`FAIL env-file <path> ...`) before any
check runs. Then cli.load_env runs as it does for a call: SKCALL_ENV_FILE
keeps its one meaning, the file the Talk login is read from (default
~/.hermes/.env). Tokens are sent, never printed.
"""
from __future__ import annotations

import asyncio
import logging
import os
import socket
import time
from collections.abc import Awaitable, Callable
from urllib.parse import urlsplit

import httpx
from aioice.turn import create_turn_endpoint

from skcall.agents import ConfigError, profile_for
from skcall.interfaces import MIN_INTERFACES
from skcall.talk.ocs import TalkOCS
from skcall.video.lipsync import LipsyncClient

CHECKS = ("tts", "stt", "lipsync", "talk", "turn")
WORD = "Hello."
CHECK_TIMEOUT_S = 120.0
LIPSYNC_READY_S = 15.0
TURN_TIMEOUT_S = 10.0

Result = tuple[bool, str]


class Missing(Exception):
    """A setting the check needs is not configured."""


# ------------------------------------------------------------------ env
def _read_env_file(path: str) -> dict[str, str]:
    path = os.path.expanduser(path)
    out: dict[str, str] = {}
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            k, sep, v = line.partition("=")
            if sep and k.strip():
                out[k.strip()] = v.strip().strip("\"'")
    return out


# The join unit's EnvironmentFile lines: %h/.config/skcall/skcall.env, then
# %h/.config/skcall/join.env (the per-room runtime file is per call, not site).
SITE_ENV_FILES = ("~/.config/skcall/skcall.env", "~/.config/skcall/join.env")


class EnvFileMissing(Exception):
    """An env file named with --env-file does not exist."""


def load_site_env(paths: list[str] | None = None) -> None:
    """Load the env files without overriding anything already set: `paths`
    (from --env-file) when given, each of which must exist, else the join
    unit's files (skcall.env, then join.env), skipped when absent."""
    if paths:
        gone = [p for p in paths if not os.path.isfile(os.path.expanduser(p))]
        if gone:
            raise EnvFileMissing(", ".join(gone))
    merged: dict[str, str] = {}
    for path in paths or SITE_ENV_FILES:
        merged.update(_read_env_file(path))
    for k, v in merged.items():
        if not os.environ.get(k):
            os.environ[k] = v


def _need(*names: str) -> None:
    for n in names:
        if not os.environ.get(n, "").strip():
            raise Missing(f"{n} is not set")


def _csv(v: str) -> list[str]:
    return [x.strip() for x in (v or "").split(",") if x.strip()]


# ------------------------------------------------------------------ http
def _transport() -> httpx.AsyncBaseTransport | None:
    """The httpx transport every request uses (None: the default; tests fake it)."""
    return None


def _http() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=CHECK_TIMEOUT_S, transport=_transport())


def _base(url: str) -> str:
    u = urlsplit(url)
    scheme = {"ws": "http", "wss": "https"}.get(u.scheme, u.scheme)
    return f"{scheme}://{u.netloc}"


def _host(url: str) -> str:
    return urlsplit(url).netloc or url


UNVERSIONED = "unversioned"


async def _info(http: httpx.AsyncClient, service: str, url: str) -> str:
    """'' when the server's /v1/info says it is `service` at a new enough
    interface, UNVERSIONED when it answers 404 (a server without /v1/info:
    the real request decides), else why not."""
    base = _base(url)
    try:
        r = await http.get(base + "/v1/info")
    except httpx.HTTPError as e:
        return f"{_host(url)}: /v1/info unreachable ({type(e).__name__}: {e or 'no detail'})"
    if r.status_code == 404:
        return UNVERSIONED
    if r.status_code != 200:
        return f"{_host(url)}: /v1/info HTTP {r.status_code}"
    try:
        info = r.json()
        got = int(info.get("interface"))
    except (ValueError, TypeError, AttributeError):
        return f"{_host(url)}: /v1/info is not a service info document"
    if info.get("service") != service:
        return f"{_host(url)}: /v1/info says service {info.get('service')!r}, not {service!r}"
    need = MIN_INTERFACES[service]
    if got < need:
        return f"{_host(url)}: interface {got}, this client needs at least {need}"
    return ""


async def _infos(service: str, urls: list[str]) -> tuple[list[str], str]:
    """(problems, note): the /v1/info problems that FAIL the check, and the
    ok line's opening, which names the unversioned servers."""
    async with _http() as http:
        got = [(u, await _info(http, service, u)) for u in urls]
    problems = [p for _, p in got if p and p != UNVERSIONED]
    plain = [_host(u) for u, p in got if p == UNVERSIONED]
    if not plain:
        return problems, "interface ok"
    note = f"{UNVERSIONED} ({', '.join(plain)}: no /v1/info, checked by the request only)"
    return problems, (note if len(plain) == len(urls) else "interface ok, " + note)


# ------------------------------------------------------------------ speech
def _tts_client(agent: str, url: str):
    from skcall.speech import F5TTS, tts_speed
    prof = profile_for(agent) if os.environ.get("SKCALL_NC_URL") else None
    voice = os.environ.get("SKCALL_TTS_VOICE") or (prof.tts_voice if prof else agent)
    model = os.environ.get("SKCALL_TTS_MODEL") or (prof.tts_model if prof else "f5-tts")
    return F5TTS(url=url, voice=voice, model=model,
                 nfe_step=int(os.environ.get("SKCALL_TTS_NFE", "16")) or None,
                 speed=tts_speed(), trim=False, timeout=CHECK_TIMEOUT_S, transport=_transport())


async def _speak(agent: str, url: str, text: str = WORD):
    """(pcm at CALL_RATE, ms) for `text` through the calls' TTS client, on
    `url` (one server, or the comma list a call uses)."""
    tts = _tts_client(agent, url)
    try:
        t = time.monotonic()
        pcm = await tts.synth(text)
        return pcm, (time.monotonic() - t) * 1000
    finally:
        await tts.aclose()


async def check_tts(agent: str) -> Result:
    _need("SKCALL_TTS_URL")
    urls = _csv(os.environ["SKCALL_TTS_URL"])
    problems, note = await _infos("tts", urls)
    if problems:
        return False, "; ".join(problems)
    from skcall.core.audio import CALL_RATE
    spoke, silent = [], []
    for u in urls:
        pcm, ms = await _speak(agent, u)
        if len(pcm):
            spoke.append(f"{_host(u)} {len(pcm) / CALL_RATE:.2f} s in {ms:.0f} ms")
        else:
            silent.append(_host(u))
    if silent:
        return False, f"no audio for {WORD!r} from {', '.join(silent)} (see the log)"
    return True, f"{note}, {WORD!r} -> " + ", ".join(spoke)


def _stt_specs() -> list[str]:
    specs = os.environ.get("SKCALL_STT_BACKENDS", "").strip()
    if not specs and os.environ.get("SKCALL_STT_URL", "").strip():
        specs = "cpp:" + os.environ["SKCALL_STT_URL"].strip()
    if not specs:
        raise Missing("SKCALL_STT_BACKENDS (or SKCALL_STT_URL) is not set")
    return _csv(specs)


async def check_stt(agent: str) -> Result:
    from skcall.core.audio import CALL_RATE, STT_RATE, resample
    from skcall.speech import stt_backend
    specs = _stt_specs()
    _need("SKCALL_TTS_URL")
    backends = [stt_backend(s, timeout=CHECK_TIMEOUT_S, transport=_transport()) for s in specs]
    try:
        problems, note = await _infos("stt", [b.url for b in backends])
        if problems:
            return False, "; ".join(problems)
        pcm, _ = await _speak(agent, os.environ["SKCALL_TTS_URL"])
        if not len(pcm):
            return False, f"no TTS audio for {WORD!r} to transcribe (SKCALL_TTS_URL, see the log)"
        pcm16 = resample(pcm, CALL_RATE, STT_RATE)
        heard, bad = [], []
        want = WORD.strip(".").lower()
        for b in backends:
            t = time.monotonic()
            try:
                text = await b.transcribe_or_raise(pcm16)
            except Exception as e:
                bad.append(f"{_host(b.url)}: {type(e).__name__}: {e or 'no detail'}")
                continue
            ms = (time.monotonic() - t) * 1000
            if want not in text.lower():
                bad.append(f"{_host(b.url)}: heard {text!r} for {WORD!r}")
            else:
                heard.append(f"{_host(b.url)} {text!r} {ms:.0f} ms")
        if bad:
            return False, "; ".join(bad)
        return True, f"{note}, " + ", ".join(heard)
    finally:
        for b in backends:
            await b.aclose()


# ------------------------------------------------------------------ lipsync
class _LastWarning(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.last = ""

    def emit(self, record: logging.LogRecord) -> None:
        self.last = record.getMessage()


async def check_lipsync(agent: str) -> Result:
    _need("SKCALL_LIPSYNC_URL", "SKCALL_LIPSYNC_TOKEN")
    from skcall.video import VideoConfig
    cfg = VideoConfig.from_env(agent)
    problems, note = await _infos("lipsync", [cfg.lipsync_url])
    if problems:
        return False, "; ".join(problems)
    seen = _LastWarning()
    log = logging.getLogger("skcall.lipsync")
    log.addHandler(seen)
    client = LipsyncClient(cfg.lipsync_url, cfg.lipsync_token, avatar=cfg.avatar, fps=int(cfg.fps),
                           retry_s=(LIPSYNC_READY_S, LIPSYNC_READY_S))
    try:
        client.start()
        ready = await client.wait_ready(LIPSYNC_READY_S)
    finally:
        await client.aclose()
        log.removeHandler(seen)
    if not ready:
        return False, (f"{_host(cfg.lipsync_url)}: no session for avatar {cfg.avatar!r} within "
                       f"{LIPSYNC_READY_S:.0f} s" + (f" ({seen.last})" if seen.last else ""))
    return True, f"{_host(cfg.lipsync_url)} {note}, session for avatar {cfg.avatar!r} opened and closed"


# ------------------------------------------------------------------ talk
TALK_VARS = ("SKCALL_NC_URL", "NEXTCLOUD_TALK_USERNAME", "NEXTCLOUD_TALK_APP_PASSWORD")


async def check_talk(agent: str) -> Result:
    _need(*TALK_VARS)
    rooms = profile_for(agent).rooms
    async with TalkOCS() as o:
        listed = {r.get("token") for r in await o.list_rooms() if isinstance(r, dict)}
        user = o.user
    missing = [t for t in rooms if t not in listed]
    if missing:
        return False, (f"as {user}: allowed room(s) {', '.join(missing)} not in the room list "
                       f"({len(listed)} rooms)")
    if not rooms:
        return True, f"as {user}, {len(listed)} rooms (SKCALL_ALLOWED_ROOMS is empty: no room to check)"
    return True, f"as {user}, all {len(rooms)} allowed rooms listed ({len(listed)} rooms)"


# ------------------------------------------------------------------ turn
def _turn_urls(settings: dict) -> list[str]:
    out: list[str] = []
    for t in settings.get("turnservers") or []:
        urls = t.get("urls") or []
        out.extend([urls] if isinstance(urls, str) else urls)
    return out


def _parse_turn(url: str) -> tuple[str, str, int, str]:
    """('turn'|'turns', host, port, 'udp'|'tcp') of a turn:/turns: URL."""
    scheme, _, rest = url.partition(":")
    hostport, _, query = rest.partition("?")
    if hostport.startswith("["):
        host, _, port = hostport[1:].partition("]")
        port = port.lstrip(":")
    else:
        host, _, port = hostport.rpartition(":") if hostport.count(":") == 1 else (hostport, "", "")
    transport = "udp"
    for kv in query.split("&"):
        k, _, v = kv.partition("=")
        if k == "transport" and v:
            transport = v.lower()
    if scheme == "turns":
        transport = "tcp"
    return scheme, host, int(port or (5349 if scheme == "turns" else 3478)), transport


def _site_host(v: str) -> str:
    v = v.strip()
    if ":" in v and v.startswith(("turn:", "turns:")):
        return _parse_turn(v)[1]
    return v.rsplit(":", 1)[0] if v.count(":") == 1 else v


async def _resolve(host: str, port: int) -> str:
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, family=socket.AF_INET,
                                                         type=socket.SOCK_DGRAM)
    return infos[0][4][0]


async def check_turn(agent: str) -> Result:
    _need(*TALK_VARS)
    rooms = profile_for(agent).rooms
    if not rooms:
        raise Missing("SKCALL_ALLOWED_ROOMS is not set (Talk's signaling settings need a room)")
    async with TalkOCS() as o:
        settings = await o.signaling_settings(rooms[0])
    site = _site_host(os.environ.get("SKCALL_SITE_TURN", ""))
    hosts = sorted({_parse_turn(u)[1] for u in _turn_urls(settings)})
    if not hosts:
        return False, "Talk offers no TURN server" + (f" (site TURN is {site})" if site else "")
    wrong = [h for h in hosts if site and h.lower() != site.lower()]
    if wrong:
        return False, f"Talk offers TURN {', '.join(wrong)}, the site's TURN (SKCALL_SITE_TURN) is {site}"
    from skcall.talk.media import ice_config
    cfg, used = ice_config(settings, stun="none")
    if not cfg.iceServers:
        return False, f"no TURN URL usable by a call among {', '.join(_turn_urls(settings))}"
    srv = cfg.iceServers[0]
    scheme, host, port, transport = _parse_turn(used[0])
    # UDP sends to the resolved address. TCP and TLS connect by host name, so
    # TLS verifies the relay's certificate against its name, not an IP literal.
    addr = (await _resolve(host, port) if transport == "udp" else host, port)
    before = asyncio.all_tasks()
    try:
        turn, _ = await asyncio.wait_for(
            create_turn_endpoint(asyncio.DatagramProtocol, addr, srv.username, srv.credential,
                                 transport=transport, ssl=True if scheme == "turns" else None),
            TURN_TIMEOUT_S)
    except asyncio.TimeoutError:
        return False, f"{host}:{port}/{transport}: TURN allocation timed out after {TURN_TIMEOUT_S:g} s"
    relayed = turn.get_extra_info("sockname")
    turn.close()                                   # deletes the allocation
    pending = asyncio.all_tasks() - before - {asyncio.current_task()}
    if pending:
        await asyncio.wait(pending, timeout=3)
    note = "" if site else " (SKCALL_SITE_TURN is not set: host not checked)"
    return True, f"{host}:{port}/{transport} relayed {relayed[0]}:{relayed[1]}{note}"


# ------------------------------------------------------------------ run
RUNNERS: dict[str, Callable[[str], Awaitable[Result]]] = {
    "tts": check_tts, "stt": check_stt, "lipsync": check_lipsync, "talk": check_talk, "turn": check_turn}

# With no flag, a service check whose URL (any of these) is unset is "not
# configured" rather than failed. talk and turn are not listed: they always run.
SERVICE_URLS: dict[str, tuple[str, ...]] = {
    "tts": ("SKCALL_TTS_URL",), "stt": ("SKCALL_STT_BACKENDS", "SKCALL_STT_URL"),
    "lipsync": ("SKCALL_LIPSYNC_URL",)}


def _configured(name: str) -> bool:
    urls = SERVICE_URLS.get(name)
    return urls is None or any(os.environ.get(v, "").strip() for v in urls)


async def _one(name: str, agent: str) -> Result:
    try:
        return await asyncio.wait_for(RUNNERS[name](agent), CHECK_TIMEOUT_S)
    except (Missing, ConfigError) as e:
        return False, str(e)
    except asyncio.TimeoutError:
        return False, f"timed out after {CHECK_TIMEOUT_S:.0f} s"
    except Exception as e:
        return False, f"{type(e).__name__}: {e or 'no detail'}"


async def run(checks: list[str], agent: str) -> int:
    """Run `checks` (named by flags: each runs and needs its settings), or,
    when empty, every check with unconfigured services reported ok."""
    failed = 0
    for name in checks or list(CHECKS):
        if not checks and not _configured(name):
            print(f"ok {name} not configured", flush=True)
            continue
        ok, detail = await _one(name, agent)
        failed += not ok
        print(f"{'ok' if ok else 'FAIL'} {name} {detail}", flush=True)
    return 0 if not failed else 1
