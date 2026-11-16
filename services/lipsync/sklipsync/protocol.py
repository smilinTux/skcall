"""sklipsync wire protocol (fixed; the skcall client in skcall/video/lipsync.py
is built against the same text, so change both together and say so in the
README).

  WebSocket ws://<host>:<port>/v1/lipsync
  header    Authorization: Bearer <SKCALL_LIPSYNC_TOKEN>
  client -> {"type":"hello","avatar":"lumina","fps":25,"engine":"musetalk"}
  server -> {"type":"ready","fps":25,"crop":{"x":..,"y":..,"w":..,"h":..},"frame_size":[w,h]}
  per utterance:
    client -> {"type":"utterance","id":"<uuid>"}
    client -> binary 16 kHz mono s16le PCM chunks (any size)
    client -> {"type":"end","id":"<uuid>"}
    client -> {"type":"cancel","id":"<uuid>"}            (barge-in, any time)
    server -> binary: u32 BE frame index, u16 BE uuid length, uuid ascii, JPEG mouth crop
    server -> {"type":"done","id":"<uuid>","frames":N}
  keepalive: {"type":"ping"} / {"type":"pong"}, either side
  errors:   server -> {"type":"error","code":"...","message":"...","id":"<uuid, if any>"}

Frame i is the mouth at audio time i/fps from the start of the utterance.
The crop JPEG goes at `crop` on the avatar's full frame of size frame_size.
"""
from __future__ import annotations

import struct

PATH = "/v1/lipsync"
PCM_RATE = 16000


def pack_frame(index: int, utt_id: str, jpeg: bytes) -> bytes:
    u = utt_id.encode("ascii")
    return struct.pack(">IH", index, len(u)) + u + jpeg


def unpack_frame(data: bytes) -> tuple[int, str, bytes]:
    if len(data) < 6:
        raise ValueError("short frame message")
    index, n = struct.unpack_from(">IH", data, 0)
    if len(data) < 6 + n:
        raise ValueError("truncated utterance id")
    return index, data[6:6 + n].decode("ascii"), data[6 + n:]


def frames_for(samples: int, fps: int) -> int:
    """Frames that cover `samples` of 16 kHz audio (the last one may be partial)."""
    return -(-samples * fps // PCM_RATE)
