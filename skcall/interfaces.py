"""Service interfaces this client needs.

MIN_INTERFACES maps each service to the minimum `/v1/info` interface version
that this client needs from it.
"""

MIN_INTERFACES: dict[str, int] = {"lipsync": 1, "tts": 1, "stt": 1}
