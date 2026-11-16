"""S0: read-only probe of the target server as lumina.

Prints Nextcloud/Talk versions, signaling capability config, HPB welcome and
the list of lumina's rooms (name + token only). Joins nothing.
"""
import asyncio
import json

import aiohttp

from skcall.talk.ocs import TalkOCS


async def main() -> None:
    async with TalkOCS() as ocs:
        caps = await ocs.capabilities()
        sp = caps["capabilities"]["spreed"]
        print("nextcloud", caps["version"]["string"], "talk", sp.get("version"))
        cfg = sp.get("config", {})
        print("signaling cfg:", json.dumps(cfg.get("signaling"), indent=1))
        print("call cfg:", json.dumps(cfg.get("call"), indent=1))
        print("features with 'call' or 'silent':",
              [f for f in sp.get("features", []) if "call" in f or "silent" in f])
        rooms = await ocs.list_rooms()
        for r in rooms:
            print(f"room type={r['type']} token={r['token']} name={r['displayName']!r}"
                  f" hasCall={r.get('hasCall')}")
        async with aiohttp.ClientSession() as s:
            async with s.get(ocs.base_url + "/standalone-signaling/api/v1/welcome") as r:
                print("HPB welcome:", await r.text())


asyncio.run(main())
