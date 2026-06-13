"""Report-path resolution tests — filesystem only, no GPU or report parsing.

resolve_report just stats paths and mtimes, so empty placeholder files named
*.ncu-rep exercise every branch without a real (large) report.
"""

import os

import pytest

from perfdigest.report_store.discovery import resolve_report


def _touch(path, mtime=None):
    path.write_bytes(b"")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def test_resolve_exact_file(tmp_path):
    f = _touch(tmp_path / "report.ncu-rep")
    assert resolve_report(str(f)) == f


def test_resolve_name_without_suffix(tmp_path):
    _touch(tmp_path / "report.ncu-rep")
    resolved = resolve_report(str(tmp_path / "report"))
    assert resolved.name == "report.ncu-rep"


def test_resolve_directory_picks_newest(tmp_path):
    _touch(tmp_path / "old.ncu-rep", mtime=1_000_000)
    _touch(tmp_path / "new.ncu-rep", mtime=2_000_000)
    assert resolve_report(str(tmp_path)).name == "new.ncu-rep"


def test_missing_report_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_report(str(tmp_path / "does_not_exist"))


def test_empty_directory_raises(tmp_path):
    empty = tmp_path / "debug_runs"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_report(str(empty))


def test_directory_ignores_non_reports(tmp_path):
    _touch(tmp_path / "notes.txt")
    _touch(tmp_path / "data.csv")
    with pytest.raises(FileNotFoundError):
        resolve_report(str(tmp_path))
