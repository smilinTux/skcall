#!/usr/bin/env bash
# Build the lip-sync service's own venv inside this export and fetch MuseTalk.
# Needs uv, git and network (PyPI, the PyTorch XPU index, GitHub). Safe to re-run.
# Run from anywhere; it works at the export's final path.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# git archive skips submodules, so MuseTalk is cloned at the pinned commit.
REF="$(tr -d '[:space:]' < MUSETALK_REF)"
DEST=third_party/MuseTalk
if [ "$(git -C "$DEST" rev-parse HEAD 2>/dev/null || true)" != "$REF" ]; then
  mkdir -p "$DEST"
  [ -e "$DEST/.git" ] || git -C "$DEST" init -q
  git -C "$DEST" remote remove origin 2>/dev/null || true
  git -C "$DEST" remote add origin https://github.com/smilinTux/MuseTalk.git
  git -C "$DEST" fetch -q --depth 1 origin "$REF"
  git -C "$DEST" checkout -q FETCH_HEAD
fi
test "$(git -C "$DEST" rev-parse HEAD)" = "$REF"

# Shared token helper: the export installs alone with its own copy (see sklipsync/auth.py).
cp ../common/sktoken.py sklipsync/_sktoken.py

[ -x .venv/bin/python ] || uv venv .venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt --index-strategy unsafe-best-match
.venv/bin/python -c "import torch, aiohttp, cv2, sklipsync.server"
echo "sklipsync venv ready: $PWD/.venv (MuseTalk at $REF)"
