"""Avatar directory: one folder per avatar, picked up on the next hello.

  <avatars_dir>/<avatar_id>/portrait.png   (or .jpg/.jpeg/.webp)
  <avatars_dir>/<avatar_id>/avatar.json    (optional settings, see AvatarSource)

Engines cache their per-avatar preprocessing under
  <cache_dir>/<engine>/<avatar_id>/<key>/
where key hashes the portrait bytes and the settings, so replacing the
portrait (or changing avatar.json) re-prepares it automatically.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

PORTRAIT_NAMES = ("portrait.png", "portrait.jpg", "portrait.jpeg", "portrait.webp")
AVATAR_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class AvatarNotFound(LookupError):
    pass


@dataclass
class AvatarSource:
    id: str
    portrait: Path
    max_side: int = 720            # published frame's long edge
    bbox_shift: int = 0            # MuseTalk bbox_shift (move the mouth box up or down)
    extra_margin: int = 10         # pixels added below the chin
    parsing_mode: str = "jaw"
    cheek_width: int = 90
    extra: dict = field(default_factory=dict)

    def cache_key(self, engine: str) -> str:
        h = hashlib.sha256()
        h.update(self.portrait.read_bytes())
        settings = {k: getattr(self, k) for k in ("max_side", "bbox_shift", "extra_margin",
                                                  "parsing_mode", "cheek_width")}
        h.update(json.dumps([engine, settings, self.extra], sort_keys=True).encode())
        return h.hexdigest()[:16]


def load_avatar(avatars_dir: Path, avatar_id: str) -> AvatarSource:
    if not AVATAR_ID.match(avatar_id or ""):
        raise AvatarNotFound(f"bad avatar id {avatar_id!r}")
    d = avatars_dir / avatar_id
    portrait = next((d / n for n in PORTRAIT_NAMES if (d / n).is_file()), None)
    if portrait is None:
        raise AvatarNotFound(f"no portrait for {avatar_id!r} in {d} (expected one of {', '.join(PORTRAIT_NAMES)})")
    cfg = {}
    if (d / "avatar.json").is_file():
        cfg = json.loads((d / "avatar.json").read_text())
    known = {k: cfg.pop(k) for k in list(cfg) if k in AvatarSource.__dataclass_fields__ and k not in ("id", "portrait")}
    return AvatarSource(id=avatar_id, portrait=portrait, extra=cfg, **known)


def list_avatars(avatars_dir: Path) -> list[str]:
    if not avatars_dir.is_dir():
        return []
    out = []
    for d in sorted(avatars_dir.iterdir()):
        if d.is_dir() and AVATAR_ID.match(d.name) and any((d / n).is_file() for n in PORTRAIT_NAMES):
            out.append(d.name)
    return out
