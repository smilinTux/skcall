#!/usr/bin/env bash
# Run a spike with lumina's Talk credentials loaded from the Hermes env file.
# The app password is exported into the child environment only; never echoed.
set -euo pipefail
: "${SKCALL_NC_URL:?SKCALL_NC_URL is not set: skcall has no built-in Nextcloud, set it in the site config}"
export SKCALL_NC_URL
ENV_FILE="${SKCALL_ENV_FILE:-$HOME/.hermes/.env}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
NEXTCLOUD_TALK_USERNAME="$(grep -E '^NEXTCLOUD_TALK_USERNAME=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d "\"'")"
NEXTCLOUD_TALK_APP_PASSWORD="$(grep -E '^NEXTCLOUD_TALK_APP_PASSWORD=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d "\"'")"
export NEXTCLOUD_TALK_USERNAME NEXTCLOUD_TALK_APP_PASSWORD
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
exec "$HERE/.venv/bin/python" "$@"
