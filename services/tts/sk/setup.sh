#!/usr/bin/env bash
# Build the TTS service's own venv inside this export. Needs uv and network
# (PyPI, the PyTorch cu128 index, GitHub for the spaCy English model). Safe to re-run.
# Run from anywhere; it works at the export's final path.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Shared token helper: the export installs alone with its own copy (see sktts/auth.py).
cp ../common/sktoken.py sktts/_sktoken.py

[ -x .venv/bin/python ] || uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt --index-strategy unsafe-best-match
.venv/bin/python -c "import torch, kokoro, soundfile, sktts.server; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "sktts venv ready: $PWD/.venv (run: .venv/bin/python -m sktts)"
