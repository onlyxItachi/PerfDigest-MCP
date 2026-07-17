"""git numstat backend (RepoState) — session-retrospective change digests.

Two fixture families, both genuinely produced (nothing hand-written):

* ``repo_session_early_sample.numstat`` / ``repo_session_sample.numstat`` —
  REAL captures of `git diff --numstat -M` run on THIS repository for the
  v1.2 lane's own footprint (meta: the sessions that built these backends):
  ``v1.1.1..499fb2c`` (after the gha_log merge, 27 files) and
  ``v1.1.1..a82e7fd`` (after the previous-green merge, 46 files). Paths and
  counts only; scanned before committing.
* ``repo_lab_renames_sample.numstat`` — a REAL `git diff --numstat -M
  state-a..state-b` capture from a throwaway lab repository (git 2.53) built
  with real commits specifically to exhibit every numstat shape at once:
  binary files (guaranteed non-text bytes), a whole-path rename with an edit
  (``NOTES => TASKS.md``), braced renames with a common prefix and/or suffix
  (``src/{parser.py => parser_v2.py}``, ``{src => docs}/util.py``,
  ``docs/{ => api}/config.py`` — empty brace side), a no-edit rename (the
  genuine-zero case), a path with a space, and a non-ASCII path (git prints
  it C-quoted; the reader keeps it verbatim). The capture command and repo
  script are in the reader's docstring lineage; no line was invented or
  edited after capture.

Also real (observed while building the lab, kept out of the fixture): git
only pairs a rename when similarity stays above its threshold — a heavily
edited move splits into an add+delete pair, and that split IS the correct
digest of what git reported.
"""

from __future__ import annotations

import json

import pytest

from perfdigest.adapters import registry
from perfdigest.core.digest import NOT_AVAILABLE
from perfdigest.server import tools

FMT = "git-numstat"

LAB = "repo_lab_renames_sample"
EARLY = "repo_session_early_sample"
FINAL = "repo_session_sample"


def _fx(fixtures_dir, name):
    p = fixtures_dir / f"{name}.numstat"
    assert p.exists(), f"committed fixture missing: {p}"
    return str(p)


# ---------------------------------------------------------------------------
# Registry dispatch
# ---------------------------------------------------------------------------


def test_registered_formats_domain_platforms():
    b = registry.get_backend(FMT)
    assert b.name == "git_numstat"
    assert b.domain == "repo_change"
    assert registry.get_backend("numstat").name == "git_numstat"
    # A change digest is plain text: digesting is universal.
    assert b.platforms == frozenset({"linux", "darwin", "win32"})
    assert b.suffixes == (".numstat",)


# ---------------------------------------------------------------------------
# Real session fixture: the lane's own footprint
# ---------------------------------------------------------------------------


def test_real_session_fixture_parses_in_file_order(fixtures_dir):
    units = tools.list_kernels(_fx(fixtures_dir, FINAL), FMT)
    assert len(units) == 46  # files the v1.2 lane touched by a82e7fd
    assert [u["index"] for u in units] == list(range(46))
    assert all(u["domain"] == "repo_change" for u in units)
    # No time dimension in a change digest: every duration honestly absent.
    assert all(u["duration_us"] is None for u in units)


def test_real_session_file_metrics(fixtures_dir):
    m = tools.get_metrics(
        _fx(fixtures_dir, FINAL), FMT, "src/perfdigest/adapters/gha_log/gha_log_reader.py"
    )["metrics"]
    assert m["lines_added"] == 204.0  # the task-1 reader, exactly as merged
    assert m["lines_deleted"] == 0.0


def test_summarize_falls_back_to_file_order_and_carries_line_counts(fixtures_dir):
    # Durations are all None -> no duration ranking exists; the digest states
    # file order and hands the agent lines_added as the fact to sort by
    # (ranking is the model's job — facts, not verdicts).
    s = tools.summarize_report(
        _fx(fixtures_dir, FINAL), FMT, top_n=3, metrics=["lines_added", "lines_deleted"]
    )
    assert s["domain"] == "repo_change"
    assert s["sorted_by"] == "file_order"
    assert "coverage_pct_of_total_duration" not in s  # nothing to fabricate
    assert all("lines_added" in u["metrics"] for u in s["units"])


def test_footprint_evolution_across_two_real_snapshots(fixtures_dir):
    # The usage-prompt flow: same session base, two moments. gha_log/backend.py
    # really grew by 14 lines between the two merges (task 2's usage-prompt
    # extension) — the numstat pair carries that as a plain fact.
    cmp = tools.compare_metrics(
        _fx(fixtures_dir, EARLY),
        _fx(fixtures_dir, FINAL),
        FMT,
        "src/perfdigest/adapters/gha_log/backend.py",
    )
    assert cmp["metrics"]["lines_added"]["a"] == 96.0
    assert cmp["metrics"]["lines_added"]["b"] == 110.0
    assert cmp["metrics"]["lines_added"]["delta"] == 14.0
    assert cmp["metrics"]["lines_deleted"]["delta"] == 0.0


# ---------------------------------------------------------------------------
# Lab fixture: every real numstat path shape
# ---------------------------------------------------------------------------


def test_lab_fixture_full_unit_listing(fixtures_dir):
    units = tools.list_kernels(_fx(fixtures_dir, LAB), FMT)
    assert [u["name"] for u in units] == [
        "TASKS.md",                                # NOTES => TASKS.md
        "data/assets.dat",                         # new binary
        "data/logo.bin",                           # modified binary
        "docs/api/config.py",                      # docs/{ => api}/config.py
        "docs/read me.txt",                        # space kept verbatim
        "docs/util.py",                            # {src => docs}/util.py
        '"docs/\\303\\266l\\303\\247\\303\\274m.txt"',  # C-quoted by git, verbatim
        "src/engine.py",
        "src/legacy.py",
        "src/new_module.py",
        "src/parser_v2.py",                        # src/{parser.py => parser_v2.py}
    ]


def test_binary_file_is_honestly_absent_end_to_end(fixtures_dir):
    # THE headline honesty case, through the full tool path: numstat prints
    # '-\t-' for a binary — that is "no line counts exist", never zero.
    m = tools.get_metrics(_fx(fixtures_dir, LAB), FMT, "data/logo.bin")["metrics"]
    assert m["lines_added"] == NOT_AVAILABLE
    assert m["lines_deleted"] == NOT_AVAILABLE


def test_pure_rename_zero_is_measured_not_absent(fixtures_dir):
    # The contrast case that keeps the dash honest: a no-edit rename really
    # was measured at 0 added / 0 deleted — a genuine 0.0, not the sentinel.
    m = tools.get_metrics(_fx(fixtures_dir, LAB), FMT, "docs/util.py")["metrics"]
    assert m["lines_added"] == 0.0
    assert m["lines_deleted"] == 0.0


def test_rename_units_are_named_by_new_path_with_old_path_in_expand(fixtures_dir):
    p = _fx(fixtures_dir, LAB)
    for new_path, old_path in [
        ("TASKS.md", "NOTES"),                     # whole-path form
        ("src/parser_v2.py", "src/parser.py"),     # common-prefix brace
        ("docs/util.py", "src/util.py"),           # common-suffix brace
        ("docs/api/config.py", "docs/config.py"),  # empty-old brace side
    ]:
        raw = tools.expand(p, FMT, new_path, "all")["metrics"]
        assert raw["renamed"] is True
        assert raw["old_path"] == old_path
    # ...and an unrenamed unit says so.
    raw = tools.expand(p, FMT, "src/engine.py", "all")["metrics"]
    assert raw["renamed"] is False and raw["old_path"] is None


def test_expand_marks_binary_and_keeps_the_printed_path(fixtures_dir):
    raw = tools.expand(_fx(fixtures_dir, LAB), FMT, "assets", "all")["metrics"]
    assert raw["is_binary"] is True
    assert raw["path_as_printed"] == "data/assets.dat"
    renamed = tools.expand(_fx(fixtures_dir, LAB), FMT, "TASKS", "all")["metrics"]
    assert renamed["path_as_printed"] == "NOTES => TASKS.md"


def test_mixed_added_and_deleted_counts(fixtures_dir):
    m = tools.get_metrics(_fx(fixtures_dir, LAB), FMT, "src/engine.py")["metrics"]
    assert m["lines_added"] == 4.0
    assert m["lines_deleted"] == 1.0


# ---------------------------------------------------------------------------
# Non-numstat input: loud, named errors
# ---------------------------------------------------------------------------


def test_non_numstat_input_raises_actionable_error(tmp_path):
    prose = tmp_path / "notes.numstat"
    prose.write_text("session summary:\nwe changed some files today\n")
    with pytest.raises(ValueError, match="git diff --numstat"):
        tools.list_kernels(str(prose), FMT)

    js = tmp_path / "config.numstat"
    js.write_text(json.dumps({"files": 3}))
    with pytest.raises(ValueError, match="git diff --numstat"):
        tools.list_kernels(str(js), FMT)

    empty = tmp_path / "empty.numstat"
    empty.write_text("")
    # An empty diff writes an empty file; nothing to digest is said out loud.
    with pytest.raises(ValueError, match="empty"):
        tools.list_kernels(str(empty), FMT)


def test_mixed_content_is_loud_not_silently_partial(tmp_path, fixtures_dir):
    # A numstat pasted into a bigger log digests SOME lines — refusing quietly
    # dropping the rest is the same honesty rule wearing its parser hat.
    real = (fixtures_dir / f"{LAB}.numstat").read_text()
    polluted = tmp_path / "polluted.numstat"
    polluted.write_text("$ git diff --numstat -M state-a..\n" + real)
    with pytest.raises(ValueError, match="not a clean"):
        tools.list_kernels(str(polluted), FMT)


# ---------------------------------------------------------------------------
# Capture advisory
# ---------------------------------------------------------------------------


def test_capture_command_is_range_based_and_file_bound():
    from perfdigest.adapters.git_numstat import backend as ns_backend
    from perfdigest.platform.detect import PlatformInfo

    info = PlatformInfo(
        os="linux", is_wsl=False, shell="posix", machine_model=None,
        cpu=None, gpu_vendors=(), profilers_on_path={},
    )
    cmd = ns_backend._capture_command("v1.1.1", info)
    assert cmd.startswith("git diff --numstat -M v1.1.1.. > session.numstat")
    assert "uncommitted-only" in cmd  # the second capture shape is advertised
