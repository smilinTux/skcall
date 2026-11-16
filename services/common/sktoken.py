"""Per-client bearer tokens, shared by the skcall services.

Edit only this file. A service's sk/setup.sh copies it to <package>/_sktoken.py
(git-ignored) so an export installs alone; a plain checkout instead finds it by
path at services/common/ next to services/<svc>/ (see sklipsync/auth.py).

A tokens file has one `client=token` per line. Blank lines and lines
starting with `#` are skipped. Callers map token to client name.
"""
from __future__ import annotations

import hmac
from typing import Optional


def load_tokens(path: str) -> dict[str, str]:
    """Read a `client=token` file and return {token: client}."""
    tokens: dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for n, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            client, sep, token = line.partition("=")
            client, token = client.strip(), token.strip()
            if not sep or not client or not token:
                raise ValueError(f"{path}:{n}: expected client=token")
            if token in tokens:
                raise ValueError(f"{path}:{n}: token already assigned to client {tokens[token]!r}")
            tokens[token] = client
    return tokens


def client_for(header: Optional[str], tokens: dict[str, str]) -> Optional[str]:
    """Return the client for an `Authorization: Bearer <tok>` header, else None.

    Every known token is compared in constant time, with no early exit and
    no dict lookup on the presented value.
    """
    if not header:
        return None
    scheme, _, presented = header.partition(" ")
    if scheme.lower() != "bearer" or not presented.strip():
        return None
    given = presented.strip().encode()
    found: Optional[str] = None
    for token, client in tokens.items():
        if hmac.compare_digest(given, token.encode()):
            found = client
    return found
