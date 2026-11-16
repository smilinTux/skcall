#!/usr/bin/env bash
# Build the STT service's own venv (server and dispatcher) inside this export.
# Needs uv and network (PyPI). Safe to re-run. Run from anywhere; it works at
# the export's final path.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Shared token helper: the export installs alone with its own copy (see skstt/auth.py).
cp ../common/sktoken.py skstt/_sktoken.py

[ -x .venv/bin/python ] || uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -c "import faster_whisper, ctranslate2, httpx, skstt.server, skstt.dispatcher; print('ctranslate2', ctranslate2.__version__, 'cuda devices', ctranslate2.get_cuda_device_count())"
echo "skstt venv ready: $PWD/.venv (run: .venv/bin/python -m skstt.server, or -m skstt.dispatcher)"
