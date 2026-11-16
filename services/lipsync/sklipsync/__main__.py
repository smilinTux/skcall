"""sklipsync CLI.

  python -m sklipsync serve     [--host H] [--port P] ...   (tokens from SKLIPSYNC_TOKENS_FILE)
  python -m sklipsync prepare   lumina [jarvis ...]         (build the avatar cache, print the crop)

Every flag also reads an environment variable (shown in --help), so the
systemd unit only needs an EnvironmentFile.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from aiohttp import web

from .avatars import list_avatars
from .engines import EngineUnavailable, create_engine
from .auth import load_tokens
from .server import LipsyncServer

LOG = logging.getLogger("sklipsync")


def _env(name, default):
    return os.environ.get(name, default)


def resolve_tokens() -> dict[str, str]:
    """token -> client.

    With SKLIPSYNC_TOKENS_FILE set, tokens come from that file (client=token lines).
    The old shared token is client "legacy" and works only while SKLIPSYNC_TOKEN is set
    explicitly (the transition switch); SKCALL_LIPSYNC_TOKEN is ignored then.
    With no tokens file, the legacy token is SKLIPSYNC_TOKEN, else SKCALL_LIPSYNC_TOKEN
    (what the shipped unit sets).
    """
    tokens: dict[str, str] = {}
    path = os.environ.get("SKLIPSYNC_TOKENS_FILE", "")
    if path:
        tokens.update(load_tokens(os.path.expanduser(path)))
        names = ("SKLIPSYNC_TOKEN",)
    else:
        names = ("SKLIPSYNC_TOKEN", "SKCALL_LIPSYNC_TOKEN")
    for name in names:
        legacy = os.environ.get(name, "")
        if legacy:
            LOG.warning("the single shared token (%s) is deprecated; use SKLIPSYNC_TOKENS_FILE", name)
            tokens.setdefault(legacy, "legacy")
            break
    return tokens


def build_engines(a) -> dict:
    engines = {}
    for name in [n.strip() for n in a.engines.split(",") if n.strip()]:
        eng = create_engine(name, models_dir=a.models, cache_dir=a.cache, musetalk_dir=a.musetalk_dir,
                            batch=a.batch, lookahead_s=a.lookahead, vram_gb=a.vram_gb or None)
        eng.load()
        engines[name] = eng
    return engines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sklipsync")
    ap.add_argument("command", choices=["serve", "prepare"])
    ap.add_argument("avatars", nargs="*", help="prepare: avatar ids (default: all in the avatar dir)")
    ap.add_argument("--host", default=_env("SKLIPSYNC_HOST", "127.0.0.1"), help="env SKLIPSYNC_HOST")
    ap.add_argument("--port", type=int, default=int(_env("SKLIPSYNC_PORT", "18850")), help="env SKLIPSYNC_PORT")
    ap.add_argument("--avatars-dir", default=_env("SKLIPSYNC_AVATARS", "~/skcall-lipsync/avatars"),
                    help="env SKLIPSYNC_AVATARS")
    ap.add_argument("--cache", default=_env("SKLIPSYNC_CACHE", "~/skcall-lipsync/cache"), help="env SKLIPSYNC_CACHE")
    ap.add_argument("--models", default=_env("MUSETALK_MODELS", "~/skcall-lipsync/models"),
                    help="MuseTalk models dir (musetalkV15, sd-vae, whisper, face-parse-bisent); env MUSETALK_MODELS")
    ap.add_argument("--musetalk-dir", default=_env("MUSETALK_DIR", ""),
                    help="MuseTalk checkout (default: the third_party submodule); env MUSETALK_DIR")
    ap.add_argument("--engines", default=_env("SKLIPSYNC_ENGINES", "musetalk"),
                    help="comma list of engines to load: musetalk, fake (ditto is a stub); env SKLIPSYNC_ENGINES")
    ap.add_argument("--batch", type=int, default=int(_env("SKLIPSYNC_BATCH", "8")), help="env SKLIPSYNC_BATCH")
    ap.add_argument("--lookahead", type=float, default=float(_env("SKLIPSYNC_LOOKAHEAD_S", "0.2")),
                    help="seconds of audio past a frame's own window before rendering it mid-stream; env SKLIPSYNC_LOOKAHEAD_S")
    ap.add_argument("--max-fps", type=int, default=int(_env("SKLIPSYNC_MAX_FPS", "25")), help="env SKLIPSYNC_MAX_FPS")
    ap.add_argument("--jpeg-quality", type=int, default=int(_env("SKLIPSYNC_JPEG_QUALITY", "85")),
                    help="env SKLIPSYNC_JPEG_QUALITY")
    ap.add_argument("--vram-gb", type=float, default=float(_env("SKLIPSYNC_VRAM_GB", "12")),
                    help="cap this process's GPU memory (0 = no cap); env SKLIPSYNC_VRAM_GB")
    ap.add_argument("--no-warm", action="store_true", help="serve: do not prepare every avatar at startup")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    a.avatars_dir = Path(a.avatars_dir).expanduser()
    a.cache = str(Path(a.cache).expanduser())
    a.models = str(Path(a.models).expanduser())
    a.musetalk_dir = a.musetalk_dir or None

    try:
        tokens = resolve_tokens()
    except (OSError, ValueError) as e:
        print(f"sklipsync: cannot read tokens: {e}", file=sys.stderr)
        return 2
    if a.command == "serve" and not tokens:
        LOG.warning("no tokens configured: set SKLIPSYNC_TOKENS_FILE (or the legacy SKLIPSYNC_TOKEN)")
        print("sklipsync: SKLIPSYNC_TOKENS_FILE (or the legacy SKLIPSYNC_TOKEN) is required", file=sys.stderr)
        return 2
    try:
        engines = build_engines(a)
    except EngineUnavailable as e:
        print(f"sklipsync: {e}", file=sys.stderr)
        return 3
    server = LipsyncServer(engines, a.avatars_dir, tokens or {"unused": "none"}, max_fps=a.max_fps,
                           jpeg_quality=a.jpeg_quality)
    ids = a.avatars or list_avatars(a.avatars_dir)
    if a.command == "prepare" or not a.no_warm:
        for avatar_id in ids:
            for name in engines:
                try:
                    h = server.prepare_sync(name, avatar_id)
                    LOG.info("avatar %s (%s): frame %s crop %s", avatar_id, name, h.frame_size, h.crop_json())
                except Exception as e:
                    LOG.error("avatar %s (%s) not prepared: %s", avatar_id, name, e)
                    if a.command == "prepare":
                        return 1
    if a.command == "prepare":
        return 0
    LOG.info("sklipsync serving ws://%s:%d/v1/lipsync (engines %s, avatars %s)", a.host, a.port,
             ",".join(engines), ",".join(ids) or "none")
    web.run_app(server.app(), host=a.host, port=a.port, print=None, access_log=None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
