"""python -m sktts: serve on SKTTS_HOST:SKTTS_PORT (default 127.0.0.1:18793)."""
from __future__ import annotations

import logging

import uvicorn

from .server import Settings, create_app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    settings = Settings.from_env()
    # access_log off: the app logs each request itself, naming the client and never the token.
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, workers=1, access_log=False)


if __name__ == "__main__":
    main()
