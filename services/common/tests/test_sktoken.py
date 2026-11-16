"""Unit tests for the shared token helper. Stdlib only, so any service venv runs them:

  cd services/common && ../tts/.venv/bin/python -m pytest -q
"""
from __future__ import annotations

import pytest

from sktoken import client_for, load_tokens


def _file(tmp_path, text):
    p = tmp_path / "tokens"
    p.write_text(text)
    return str(p)


def test_load_skips_comments_and_blanks(tmp_path):
    path = _file(tmp_path, "# clients\n\nalice = alice-secret-0001 \nbob=bob=with=equals\n")
    assert load_tokens(path) == {"alice-secret-0001": "alice", "bob=with=equals": "bob"}


def test_empty_file_gives_no_tokens(tmp_path):
    assert load_tokens(_file(tmp_path, "")) == {}


@pytest.mark.parametrize("line", ["just-a-token", "=tok", "alice=", "  =  "])
def test_malformed_line_names_the_line(tmp_path, line):
    path = _file(tmp_path, f"ok=tok-1\n{line}\n")
    with pytest.raises(ValueError, match=r":2: expected client=token"):
        load_tokens(path)


def test_duplicate_token_refused_without_echoing_it(tmp_path):
    path = _file(tmp_path, "alice=same-secret\nbob=same-secret\n")
    with pytest.raises(ValueError, match=r":2: token already assigned to client 'alice'") as e:
        load_tokens(path)
    assert "same-secret" not in str(e.value)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_tokens(str(tmp_path / "nope"))


TOKENS = {"alice-secret-0001": "alice", "bob-secret-0002": "bob"}


def test_bearer_maps_to_client():
    assert client_for("Bearer alice-secret-0001", TOKENS) == "alice"
    assert client_for("bearer  bob-secret-0002 ", TOKENS) == "bob"


@pytest.mark.parametrize("header", [None, "", "Bearer", "Bearer ", "Basic alice-secret-0001",
                                    "alice-secret-0001", "Token alice-secret-0001"])
def test_missing_or_non_bearer_header_is_none(header):
    assert client_for(header, TOKENS) is None


def test_unknown_token_is_none():
    assert client_for("Bearer old-shared-token", TOKENS) is None
    assert client_for("Bearer alice-secret-000", TOKENS) is None


def test_no_tokens_means_nobody():
    assert client_for("Bearer anything", {}) is None
