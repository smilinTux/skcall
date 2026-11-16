"""What a call may leave behind in the logs.

  SKCALL_LOG_TRANSCRIPT   on (default) / off. Off: log lines keep timings and
                          lengths but never the words (what was heard, tool
                          arguments and results, Hermes requests and answers).
  SKCALL_HERMES           on (default) / off. Off: never escalate to Hermes,
                          the same as --no-hermes (Hermes keeps its own sessions).

Both are meant for a per-room settings file, for example a private room
(see systemd/examples/skcall-join@.service, room-<room>.env).
"""
from __future__ import annotations

import os

_OFF = ("off", "0", "no", "false")


def _on(name: str) -> bool:
    return os.environ.get(name, "on").strip().lower() not in _OFF


def transcript_logged() -> bool:
    return _on("SKCALL_LOG_TRANSCRIPT")


def hermes_enabled() -> bool:
    return _on("SKCALL_HERMES")


def said(text: object) -> str:
    """`text` for a log line, or "<N chars>" when transcripts are off."""
    s = "" if text is None else str(text)
    return s if transcript_logged() else f"<{len(s)} chars>"
