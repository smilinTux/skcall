#!/usr/bin/env bash
# Build skcall's own venv inside this export. Needs uv and PyPI, nothing else.
# Safe to re-run; run it from anywhere, it works at the export's final path.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

[ -x .venv/bin/python ] || uv venv .venv --python 3.12
uv pip install --python .venv/bin/python ".[voice]" -c sk/constraints.txt
.venv/bin/python -c "import skcall, skchat"
echo "skcall venv ready: $PWD/.venv"
