#!/usr/bin/env bash
# Run a script with the repo venv. skcall.cli.load_env() reads the Talk
# credentials (only NEXTCLOUD_TALK_USERNAME / NEXTCLOUD_TALK_APP_PASSWORD) from
# ~/.hermes/.env into the process environment; nothing echoes them.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
: "${SKCALL_NC_URL:?SKCALL_NC_URL is not set: skcall has no built-in Nextcloud, set it in the site config}"
export SKCALL_NC_URL
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
exec "$HERE/.venv/bin/python" "$@"
