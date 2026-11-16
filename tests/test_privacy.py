"""Per-room privacy switches: SKCALL_LOG_TRANSCRIPT and SKCALL_HERMES."""
import logging

import pytest

from skcall import cli, privacy


@pytest.mark.parametrize("value, logged", [(None, True), ("on", True), ("off", False), ("0", False),
                                           (" Off ", False), ("no", False)])
def test_said_hides_words_only_when_transcripts_are_off(monkeypatch, value, logged):
    if value is None:
        monkeypatch.delenv("SKCALL_LOG_TRANSCRIPT", raising=False)
    else:
        monkeypatch.setenv("SKCALL_LOG_TRANSCRIPT", value)
    assert privacy.said("take my hand") == ("take my hand" if logged else "<12 chars>")
    assert privacy.said(None) == ("" if logged else "<0 chars>")


def test_heard_line_carries_no_words_when_off(monkeypatch, caplog):
    monkeypatch.setenv("SKCALL_LOG_TRANSCRIPT", "off")
    log = logging.getLogger("skcall.loop")
    with caplog.at_level(logging.INFO, logger="skcall.loop"):
        log.info("heard %s: %s", "alice", privacy.said("a secret sentence"))
    assert "secret" not in caplog.text and "<17 chars>" in caplog.text


@pytest.mark.parametrize("value, no_hermes", [(None, False), ("on", False), ("off", True)])
def test_skcall_hermes_off_sets_no_hermes(monkeypatch, value, no_hermes):
    if value is None:
        monkeypatch.delenv("SKCALL_HERMES", raising=False)
    else:
        monkeypatch.setenv("SKCALL_HERMES", value)
    seen = {}

    async def fake_join(a):
        seen["no_hermes"] = a.no_hermes
        return 0

    monkeypatch.setattr(cli, "load_env", lambda agent=None: None)
    monkeypatch.setattr(cli, "join", fake_join)
    assert cli.main(["join", "--room", "room0001"]) == 0
    assert seen["no_hermes"] is no_hermes


def test_example_unit_reads_the_room_file_last():
    from pathlib import Path
    unit = (Path(__file__).parent.parent / "systemd/examples/skcall-join@.service").read_text()
    files = [l.split("=", 1)[1] for l in unit.splitlines() if l.startswith("EnvironmentFile=")]
    assert files[-1] == "-%h/.config/skcall/room-%i.env"
