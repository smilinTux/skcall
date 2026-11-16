"""_add_extra_site must expose EDITABLE installs in ~/.skenv (skmemory, lumina_creative).

Editable installs live outside site-packages and are found through a .pth finder;
a bare sys.path.append never runs .pth files, so `import skmemory` failed in every
call and the memory context silently came up empty.
"""
import importlib
import sys

from skcall.brain import voice_engine


def _fake_editable_site(tmp_path, name):
    """A site-packages dir whose only route to `name` is a .pth line, like an editable install."""
    src = tmp_path / "src"
    (src / name).mkdir(parents=True)
    (src / name / "__init__.py").write_text("OK = True\n")
    site = tmp_path / "site-packages"
    site.mkdir()
    (site / f"__editable__.{name}.pth").write_text(f"{src}\n")
    return site


def test_editable_install_in_extra_site_is_importable(tmp_path, monkeypatch):
    name = "skcall_fake_editable_pkg"
    site = _fake_editable_site(tmp_path, name)
    monkeypatch.setattr(voice_engine, "EXTRA_SITE", str(site))
    monkeypatch.setattr(sys, "path", list(sys.path))
    sys.modules.pop(name, None)
    voice_engine._add_extra_site()
    importlib.invalidate_caches()
    assert importlib.import_module(name).OK is True
    sys.modules.pop(name, None)


def test_extra_site_goes_after_own_venv(tmp_path, monkeypatch):
    site = _fake_editable_site(tmp_path, "skcall_fake_order_pkg")
    monkeypatch.setattr(voice_engine, "EXTRA_SITE", str(site))
    before = list(sys.path)
    monkeypatch.setattr(sys, "path", before.copy())
    voice_engine._add_extra_site()
    assert sys.path[: len(before)] == before  # nothing inserted ahead of this venv
    assert str(site) in sys.path[len(before):]


def test_missing_extra_site_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.setattr(voice_engine, "EXTRA_SITE", str(tmp_path / "nope" / "site-packages"))
    monkeypatch.setattr(sys, "path", list(sys.path))
    before = list(sys.path)
    voice_engine._add_extra_site()
    assert sys.path == before


def test_empty_extra_site_disables(monkeypatch):
    monkeypatch.setattr(voice_engine, "EXTRA_SITE", "")
    monkeypatch.setattr(sys, "path", list(sys.path))
    before = list(sys.path)
    voice_engine._add_extra_site()
    assert sys.path == before
