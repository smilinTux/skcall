"""Minimal Talk OCS (REST) client, authenticated as a Nextcloud user with an app password.

Credentials come from the environment only:
  SKCALL_NC_URL, NEXTCLOUD_TALK_USERNAME, NEXTCLOUD_TALK_APP_PASSWORD
"""
from __future__ import annotations

import os
from typing import Any

import aiohttp

SPREED = "/ocs/v2.php/apps/spreed/api"


class OCSError(RuntimeError):
    def __init__(self, status: int, method: str, path: str, body: Any):
        super().__init__(f"{method} {path} -> HTTP {status}: {str(body)[:300]}")
        self.status = status
        self.body = body


class TalkOCS:
    """One instance = one HTTP cookie jar = one Nextcloud session.

    Two instances for the same user give two independent Talk sessions
    (that is how the loopback spike gets a publisher and a subscriber).
    """

    def __init__(self, base_url: str | None = None, user: str | None = None,
                 password: str | None = None):
        self.base_url = (base_url or os.environ["SKCALL_NC_URL"]).rstrip("/")
        self.user = user or os.environ["NEXTCLOUD_TALK_USERNAME"]
        password = password or os.environ["NEXTCLOUD_TALK_APP_PASSWORD"]
        self._auth = aiohttp.BasicAuth(self.user, password)
        self._http: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> "TalkOCS":
        self._http = aiohttp.ClientSession(
            auth=self._auth,
            cookie_jar=aiohttp.CookieJar(),
            headers={"OCS-APIRequest": "true", "Accept": "application/json"},
            timeout=aiohttp.ClientTimeout(total=20),
        )
        return self

    async def __aexit__(self, *exc) -> None:
        if self._http:
            await self._http.close()

    async def request(self, method: str, path: str, *, ok=(200, 201), **kw) -> Any:
        assert self._http is not None
        url = self.base_url + path
        async with self._http.request(method, url, **kw) as r:
            try:
                body = await r.json(content_type=None)
            except Exception:
                body = await r.text()
            if r.status not in ok:
                raise OCSError(r.status, method, path, body)
            if isinstance(body, dict) and "ocs" in body:
                return body["ocs"]["data"]
            return body

    # --- capabilities / settings ---------------------------------------
    async def capabilities(self) -> dict:
        return await self.request("GET", "/ocs/v2.php/cloud/capabilities")

    async def signaling_settings(self, token: str) -> dict:
        return await self.request("GET", f"{SPREED}/v3/signaling/settings",
                                  params={"token": token})

    # --- rooms ----------------------------------------------------------
    async def list_rooms(self) -> list:
        return await self.request("GET", f"{SPREED}/v4/room")

    async def get_room(self, token: str) -> dict:
        return await self.request("GET", f"{SPREED}/v4/room/{token}")

    async def create_group_room(self, name: str) -> dict:
        # roomType 2 = group conversation; no invite fields -> only the creator.
        return await self.request("POST", f"{SPREED}/v4/room",
                                  json={"roomType": 2, "roomName": name})

    async def participants(self, token: str) -> list:
        return await self.request("GET", f"{SPREED}/v4/room/{token}/participants",
                                  params={"includeStatus": "false"})

    async def join_room(self, token: str, force: bool = False) -> dict:
        """REST active-participant join. Returns the room incl. our Nextcloud sessionId.
        ``force`` replaces any other session of this user (recovery uses it to
        take over from our own dead session)."""
        return await self.request("POST", f"{SPREED}/v4/room/{token}/participants/active",
                                  json={"force": force})

    async def leave_room(self, token: str) -> None:
        await self.request("DELETE", f"{SPREED}/v4/room/{token}/participants/active")

    # --- call -----------------------------------------------------------
    async def join_call(self, token: str, flags: int = 3, silent: bool = True) -> Any:
        return await self.request("POST", f"{SPREED}/v4/call/{token}",
                                  json={"flags": flags, "silent": silent})

    async def update_call(self, token: str, flags: int) -> Any:
        return await self.request("PUT", f"{SPREED}/v4/call/{token}", json={"flags": flags})

    async def leave_call(self, token: str, all_: bool = False) -> Any:
        return await self.request("DELETE", f"{SPREED}/v4/call/{token}",
                                  json={"all": all_})

    async def call_peers(self, token: str) -> list:
        return await self.request("GET", f"{SPREED}/v4/call/{token}")

    # --- chat -----------------------------------------------------------
    async def post_chat(self, token: str, message: str) -> Any:
        return await self.request("POST", f"{SPREED}/v1/chat/{token}",
                                  json={"message": message})
