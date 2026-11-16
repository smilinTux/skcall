"""Create (or find) the private spike room "skcall spike (test, ignore)" as lumina.

Idempotent: if a room with that exact name already exists it is reused.
The room is a group conversation (roomType 2) created with no invitees, so it
contains only lumina. Prints the token.
"""
import asyncio

from skcall.talk.ocs import TalkOCS

NAME = "skcall spike (test, ignore)"


async def main() -> None:
    async with TalkOCS() as ocs:
        for r in await ocs.list_rooms():
            if r["displayName"] == NAME:
                print("exists", r["token"])
                break
        else:
            r = await ocs.create_group_room(NAME)
            print("created", r["token"])
        parts = await ocs.participants(r["token"])
        print("participants:", [(p["actorType"], p["actorId"]) for p in parts])


asyncio.run(main())
