"""Per-agent settings: who the agent is in a call, and which estate it calls on.

The built-in profiles hold only who the agent is (name, spoken names,
pronoun, voice). Where the agent lives (the Nextcloud URL, the rooms it may
join, its operators) is site configuration and comes from the environment,
which a site config renders. With no SKCALL_NC_URL there is no profile
(ConfigError): skcall never falls back to a built-in estate.

Each field can be set from the environment:

  SKCALL_NICK            display name in Talk (default: the profile's)
  SKCALL_AGENT_NAMES     comma list of spoken names ("Jarvis, leave the call")
  SKCALL_PRONOUN         she | he | they (consent line)
  SKCALL_NC_URL          the Nextcloud the agent's user lives on (required)
  SKCALL_ALLOWED_ROOMS   comma list of room tokens skcall may use
  SKCALL_OPERATORS       comma list of Talk user ids treated as operators
  SKCALL_ALLOW_OPERATOR_DMS  1 | 0: also allow, without listing them in
                         SKCALL_ALLOWED_ROOMS, a one-to-one room (Talk type 1)
                         with an operator and a group room whose only members
                         are the agent and one operator (default on; any other
                         room must be listed; see skcall/talk/access.py)
  SKCALL_TTS_VOICE       TTS voice id
  SKCALL_TTS_MODEL       TTS model (for example f5-tts for F5, tts-1 for Kokoro)
  SKCALL_TTS_SPEED       speech rate sent as `speed` (1.0 default, clamped 0.5..2.0)
  SKCALL_TTS_URL         OpenAI-shaped /v1/audio/speech URL (required; see skcall.speech)
  SKCALL_TTS_TOKEN       bearer token sent to the TTS servers (unset: no header)
  SKCALL_STT_BACKENDS    STT backends, "kind:url" comma list (required, or SKCALL_STT_URL)
  SKCALL_STT_TOKEN       bearer token sent to every STT backend (unset: no header)
  SKCALL_SITE_TURN       the site's own TURN host; `skcall smoke --turn` fails when
                         Talk hands out a different one (see skcall.smoke)
  SKCALL_CONTEXT         1 | 0: load the agent's own memory (whisper.md, latest
                         reflection, journal tail) into the voice prompt once at
                         call start (default on; see skcall/brain/memory_context.py)

Host settings, not per agent:

  SKCALL_ICE_IFACES      comma list of IP prefixes to gather ICE host
                         candidates on, such as "192.0.2." (default: unset,
                         every interface). "*", "all" or an empty value also
                         means every interface. Use it to keep out an
                         interface STUN cannot cross: on a host with a VPN
                         interface (tailscale0, wg0) STUN through it may never
                         answer, which costs a 5 s gather timeout.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace


class ConfigError(RuntimeError):
    """A site value skcall needs is not configured."""


@dataclass(frozen=True)
class AgentProfile:
    agent: str
    nick: str
    names: tuple[str, ...]
    pronoun: str = "they"
    nc_url: str = ""
    rooms: tuple[str, ...] = ()
    operators: tuple[str, ...] = ()
    tts_voice: str = "lumina"
    tts_model: str = "f5-tts"
    tts_url: str = ""
    context: bool = True
    operator_dms: bool = True

    @property
    def consent_line(self) -> str:
        return (f"{self.nick} joined the call; {self.pronoun} can hear and transcribe "
                f"what is said here.")

    @property
    def listen_only_note(self) -> str:
        return f" ({self.pronoun.capitalize()} can listen but not talk in this room.)"


PROFILES: dict[str, AgentProfile] = {
    # Names, pronoun and voice only. Jarvis speaks through a Kokoro server
    # (model tts-1, British male voice bm_george); the site config gives its URL.
    "lumina": AgentProfile("lumina", "Lumina", ("lumina", "luna", "lumi"), "she",
                           tts_voice="lumina"),
    "jarvis": AgentProfile("jarvis", "Jarvis", ("jarvis",), "he",
                           tts_voice="bm_george", tts_model="tts-1"),
}


def _csv(v: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in v.split(",") if x.strip())


def context_enabled(env: Mapping[str, str] = os.environ) -> bool:
    """SKCALL_CONTEXT: load the agent's memory into the voice prompt (default on)."""
    return env.get("SKCALL_CONTEXT", "").strip().lower() not in ("0", "false", "no", "off")


def profile_for(agent: str, env: Mapping[str, str] = os.environ) -> AgentProfile:
    """The profile for `agent` (a generic one for unknown agents) with the
    SKCALL_* settings from `env` (default os.environ) applied. Raises
    ConfigError when no Nextcloud URL is configured."""
    agent = (agent or "lumina").strip().lower()
    p = PROFILES.get(agent) or AgentProfile(agent, agent.capitalize(), (agent,))
    over: dict = {}
    if env.get("SKCALL_NICK"):
        over["nick"] = env["SKCALL_NICK"].strip()
    if env.get("SKCALL_AGENT_NAMES"):
        over["names"] = tuple(n.lower() for n in _csv(env["SKCALL_AGENT_NAMES"]))
    if env.get("SKCALL_PRONOUN"):
        over["pronoun"] = env["SKCALL_PRONOUN"].strip().lower()
    if env.get("SKCALL_NC_URL"):
        over["nc_url"] = env["SKCALL_NC_URL"].strip().rstrip("/")
    if "SKCALL_ALLOWED_ROOMS" in env:
        over["rooms"] = _csv(env["SKCALL_ALLOWED_ROOMS"])
    if env.get("SKCALL_OPERATORS"):
        over["operators"] = _csv(env["SKCALL_OPERATORS"])
    if env.get("SKCALL_ALLOW_OPERATOR_DMS", "").strip():
        over["operator_dms"] = env["SKCALL_ALLOW_OPERATOR_DMS"].strip().lower() not in (
            "0", "false", "no", "off")
    if env.get("SKCALL_TTS_VOICE"):
        over["tts_voice"] = env["SKCALL_TTS_VOICE"].strip()
    if env.get("SKCALL_TTS_MODEL"):
        over["tts_model"] = env["SKCALL_TTS_MODEL"].strip()
    if env.get("SKCALL_TTS_URL"):
        over["tts_url"] = env["SKCALL_TTS_URL"].strip()
    if env.get("SKCALL_CONTEXT", "").strip():
        over["context"] = context_enabled(env)
    p = replace(p, **over) if over else p
    if not p.nc_url:
        raise ConfigError("SKCALL_NC_URL is not set: skcall has no built-in Nextcloud, "
                          "set it in the site config (join.env)")
    return p


def ice_ifaces(env: Mapping[str, str] | None = None) -> tuple[str, ...] | None:
    """IP prefixes for ICE host candidates; None means every interface."""
    env = os.environ if env is None else env
    v = env.get("SKCALL_ICE_IFACES", "").strip()
    if v in ("*", "all"):
        return None
    return _csv(v) or None
