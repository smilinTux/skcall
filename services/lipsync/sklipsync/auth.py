"""Bearer-token helpers for the service (load_tokens, client_for).

The code lives once, in services/common/sktoken.py. Import order:
  1. sklipsync/_sktoken.py, the copy sk/setup.sh makes so an export installs alone;
  2. else services/common/sktoken.py, found by path from this package. That needs
     the repo layout services/common/ next to services/lipsync/, so a plain
     checkout that never ran setup keeps working after a pull.
If neither exists the import fails loudly.
"""
from __future__ import annotations

try:
    from ._sktoken import client_for, load_tokens
except ImportError:
    import importlib.util
    from pathlib import Path

    _src = Path(__file__).resolve().parents[2] / "common" / "sktoken.py"
    if not _src.is_file():
        raise ImportError(f"sktoken not found: run sk/setup.sh (to create sklipsync/_sktoken.py) or keep {_src}")
    _spec = importlib.util.spec_from_file_location("sktoken", _src)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    client_for, load_tokens = _mod.client_for, _mod.load_tokens

__all__ = ["client_for", "load_tokens"]
