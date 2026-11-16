"""The example systemd units ship with the repo and carry no site values."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = sorted((ROOT / "systemd" / "examples").glob("*"))
IP_ADDRESS = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")


def test_example_units_have_no_site_values():
    assert EXAMPLES, "systemd/examples/ must hold the example units"
    for unit in EXAMPLES:
        text = unit.read_text()
        assert "https://" not in text, unit
        assert not IP_ADDRESS.search(text), unit
        assert "skhub" not in text, unit


def test_example_unit_reads_the_fixed_env_files_and_runs_from_the_export_venv():
    unit = ROOT / "systemd" / "examples" / "skcall-join@.service"
    lines = unit.read_text().splitlines()
    assert "EnvironmentFile=%h/.config/skcall/skcall.env" in lines
    assert "EnvironmentFile=-%h/.config/skcall/join.env" in lines
    assert "EnvironmentFile=-%t/skcall/join-%i.env" in lines
    assert any(line.startswith("ExecStart=%h/skcall/.venv/bin/python -m skcall") for line in lines)


def test_site_units_are_gone_from_the_repo():
    assert not (ROOT / "systemd" / "skcall-join@.service").exists()
    assert not list((ROOT / "systemd").glob("skcall-join-*@.service"))   # no per-site units
