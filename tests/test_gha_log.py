"""GitHub Actions saved-log backend (CIDigest) — against a REAL captured run.

The committed fixture (tests/fixtures/ci_digest_macos_sample.gha.log) is the
verbatim output of:

    gh run view 29192019451 --repo onlyxItachi/PerfDigest-MCP-Bench --log \
        --job 86648312008

(the smallest job — "digest on macos-latest (reports captured on Linux)",
12s wall — of the public onlyxItachi/PerfDigest-MCP-Bench "capture matrix"
run). Only the redaction GitHub itself already applies (``token: ***``)
touches the content; no line was hand-edited.

Real-report lesson (verified against this fixture, exactly the trap the repo
CLAUDE.md warns about): a CURRENT ``gh`` cannot always attribute a log line to
a specific step on a modern Actions runner (the server-side zip changed from
per-step files to one flat per-job file with ``##[group]`` sections) and
prints the literal step name ``"UNKNOWN STEP"`` for every line of every job in
this run. That is not a bug in this reader — it is gh's real, honest output.
Every unit in this fixture is therefore one ``<job> / UNKNOWN STEP`` per job.
Tests that need real multi-step-within-a-job ordering use hand-written
SYNTHETIC lines (clearly marked below), since no real export we could source
demonstrates it.
"""

from __future__ import annotations

import json

import pytest

from perfdigest.adapters import registry
from perfdigest.core.digest import NOT_AVAILABLE
from perfdigest.server import tools

FMT = "gha-log"


def _path(fixtures_dir):
    p = fixtures_dir / "ci_digest_macos_sample.gha.log"
    assert p.exists(), f"committed fixture missing: {p}"
    return str(p)


# ---------------------------------------------------------------------------
# Registry dispatch
# ---------------------------------------------------------------------------


def test_registered_and_formats_and_domain():
    b = registry.get_backend(FMT)
    assert b.name == "gha_log"
    assert b.domain == "ci_step"
    assert registry.get_backend("gh-run-log").name == "gha_log"  # second format alias


def test_registered_for_expected_platforms():
    # Digesting is universal — CI logs are plain text, same on every host.
    b = registry.get_backend(FMT)
    assert b.platforms == frozenset({"linux", "darwin", "win32"})


# ---------------------------------------------------------------------------
# Real fixture: parsing, ordering, honest duration/annotation counts
# ---------------------------------------------------------------------------


def test_real_fixture_parses_into_ordered_step_unit(fixtures_dir):
    units = tools.list_kernels(_path(fixtures_dir), FMT)
    # This real run's gh cannot resolve per-step attribution (see module
    # docstring) -> the whole job collapses into one honestly-named unit.
    assert len(units) == 1
    unit = units[0]
    assert unit["name"] == "digest on macos-latest (reports captured on Linux) / UNKNOWN STEP"
    assert unit["index"] == 0
    assert unit["domain"] == "ci_step"


def test_real_fixture_duration_is_elapsed_from_log_lines(fixtures_dir):
    units = tools.list_kernels(_path(fixtures_dir), FMT)
    # First line 2026-07-12T12:07:34.9804200Z, last 2026-07-12T12:07:44.8419550Z
    # -> ~9.8615 s elapsed between first and last log line (NOT the job's full
    # wall time, which also includes runner provisioning before the first log
    # line and cleanup after the last one — an approximation, by design).
    assert units[0]["duration_us"] == pytest.approx(9_861_535.0, rel=1e-6)


def test_real_fixture_metrics_are_honest_and_real(fixtures_dir):
    m = tools.get_metrics(_path(fixtures_dir), FMT, "UNKNOWN STEP")["metrics"]
    assert m["duration_s"] == pytest.approx(9.861535, rel=1e-6)
    assert m["log_lines"] == 333.0
    # This run is genuinely all-green: 0 real '##[error]' markers.
    assert m["error_annotations"] == 0.0
    # ...but it DOES carry one real Node.js-deprecation '##[warning]' line —
    # a true positive, not a fabricated one.
    assert m["warning_annotations"] == 1.0


def test_expand_exposes_timestamps_and_the_real_warning_line(fixtures_dir):
    raw = tools.expand(_path(fixtures_dir), FMT, "0", "all")["metrics"]
    assert raw["job"] == "digest on macos-latest (reports captured on Linux)"
    assert raw["step"] == "UNKNOWN STEP"
    assert raw["first_timestamp"].startswith("2026-07-12T12:07:34.980420")
    assert raw["last_timestamp"].startswith("2026-07-12T12:07:44.841955")
    assert raw["error_lines"] == []
    assert len(raw["warning_lines"]) == 1
    assert "Node.js 20 is deprecated" in raw["warning_lines"][0]


def test_expand_section_filter_narrows_to_matching_keys(fixtures_dir):
    raw = tools.expand(_path(fixtures_dir), FMT, "0", "warning")["metrics"]
    assert set(raw) == {"warning_annotations", "warning_lines"}


# ---------------------------------------------------------------------------
# Synthetic multi-job / multi-step fixtures — clearly marked, used only to
# exercise generic grouping/ordering/honesty behavior the real (single-job,
# 'UNKNOWN STEP') fixture cannot demonstrate on its own.
# ---------------------------------------------------------------------------

# SYNTHETIC — hand-written to look like a `gh run view --log` export where gh
# WAS able to resolve real per-step names (unlike the real committed fixture).
# Not sourced from any actual run.
_SYNTHETIC_TWO_JOB_LOG = (
    "build (ubuntu)\tCheckout\t2026-07-01T10:00:00.0000000Z Cloning repo\n"
    "build (ubuntu)\tCheckout\t2026-07-01T10:00:02.5000000Z Clone complete\n"
    "build (ubuntu)\tRun tests\t2026-07-01T10:00:03.0000000Z Running pytest\n"
    "build (ubuntu)\tRun tests\t2026-07-01T10:00:09.0000000Z 42 passed\n"
    "build (windows)\tCheckout\t2026-07-01T10:05:00.0000000Z Cloning repo\n"
    "build (windows)\tCheckout\t2026-07-01T10:05:04.0000000Z Clone complete\n"
)


def test_synthetic_multi_step_file_order_and_durations(tmp_path):
    p = tmp_path / "two_job.gha.log"
    p.write_text(_SYNTHETIC_TWO_JOB_LOG)
    units = tools.list_kernels(str(p), FMT)
    names = [u["name"] for u in units]
    # File order = index order; the same step NAME ("Checkout") in two
    # different jobs stays distinguishable via the job prefix, no collision.
    assert names == [
        "build (ubuntu) / Checkout",
        "build (ubuntu) / Run tests",
        "build (windows) / Checkout",
    ]
    by_name = {u["name"]: u for u in units}
    assert by_name["build (ubuntu) / Checkout"]["duration_us"] == pytest.approx(2_500_000.0)
    assert by_name["build (ubuntu) / Run tests"]["duration_us"] == pytest.approx(6_000_000.0)
    assert by_name["build (windows) / Checkout"]["duration_us"] == pytest.approx(4_000_000.0)


def test_synthetic_summarize_report_ranks_by_duration(tmp_path):
    p = tmp_path / "two_job.gha.log"
    p.write_text(_SYNTHETIC_TWO_JOB_LOG)
    s = tools.summarize_report(str(p), FMT, top_n=1)
    assert s["sorted_by"] == "duration_us"
    assert s["units"][0]["name"] == "build (ubuntu) / Run tests"  # the 6s step


def test_step_with_no_timestamped_lines_gets_duration_none(tmp_path):
    # SYNTHETIC: a step whose lines carry no timestamp at all must not crash,
    # and its duration must be honestly None — never a fabricated 0.0.
    p = tmp_path / "no_ts.gha.log"
    p.write_text(
        "deploy\tPrepare\tsome tool printed a raw line with no timestamp at all\n"
        "deploy\tPrepare\tnor does this continuation line\n"
    )
    units = tools.list_kernels(str(p), FMT)
    assert len(units) == 1
    assert units[0]["duration_us"] is None
    m = tools.get_metrics(str(p), FMT, "0")["metrics"]
    assert m["duration_s"] == NOT_AVAILABLE
    assert m["log_lines"] == 2.0  # still counted, even without a timestamp


def test_mixed_timestamped_and_untimestamped_lines_dont_crash(tmp_path):
    # SYNTHETIC: a step mixing timestamped and untimestamped lines (e.g. a
    # tool's raw multi-line stdout blob spliced into the log). Duration comes
    # only from the timestamped lines; every line still counts toward log_lines.
    p = tmp_path / "mixed.gha.log"
    p.write_text(
        "deploy\tPrepare\t2026-07-01T09:00:00.0000000Z start\n"
        "deploy\tPrepare\tsome raw untimestamped continuation text\n"
        "deploy\tPrepare\t2026-07-01T09:00:05.0000000Z end\n"
    )
    units = tools.list_kernels(str(p), FMT)
    assert units[0]["duration_us"] == pytest.approx(5_000_000.0)
    m = tools.get_metrics(str(p), FMT, "0")["metrics"]
    assert m["log_lines"] == 3.0


def test_single_timestamped_line_is_a_genuine_zero_not_absent(tmp_path):
    # SYNTHETIC: exactly one timestamped line -> elapsed time between a line
    # and itself is a TRUE 0.0 (measured), distinct from the "no timestamps at
    # all" case above, which stays honestly None.
    p = tmp_path / "single.gha.log"
    p.write_text("deploy\tPrepare\t2026-07-01T09:00:00.0000000Z only line\n")
    units = tools.list_kernels(str(p), FMT)
    assert units[0]["duration_us"] == 0.0
    m = tools.get_metrics(str(p), FMT, "0")["metrics"]
    assert m["duration_s"] == 0.0


# SYNTHETIC — not sourced from any real gh export. A REAL single-`##[error]`
# fixture now exists (ci_test_failed_sample.gha.log, exercised in
# test_ci_prev_green.py); these fabricated lines remain only for what real
# data still does not show: MULTIPLE error/warning markers within one step.
_SYNTHETIC_ANNOTATIONS_LOG = (
    "test\tRun\t2026-07-01T09:00:00.0000000Z starting\n"
    "test\tRun\t2026-07-01T09:00:01.0000000Z ##[error]assertion failed in test_foo\n"
    "test\tRun\t2026-07-01T09:00:02.0000000Z ##[warning]deprecated API used\n"
    "test\tRun\t2026-07-01T09:00:03.0000000Z ##[warning]flaky retry triggered\n"
    "test\tRun\t2026-07-01T09:00:04.0000000Z ##[error]second failure\n"
)


def test_synthetic_error_and_warning_annotation_counting(tmp_path):
    p = tmp_path / "annotations.gha.log"
    p.write_text(_SYNTHETIC_ANNOTATIONS_LOG)
    m = tools.get_metrics(str(p), FMT, "0")["metrics"]
    assert m["error_annotations"] == 2.0
    assert m["warning_annotations"] == 2.0


# ---------------------------------------------------------------------------
# compare_metrics: previous-green analysis (measure -> edit -> measure)
# ---------------------------------------------------------------------------


def test_compare_metrics_against_a_previous_green_run(tmp_path):
    green = tmp_path / "green.gha.log"
    green.write_text(
        "test\tRun\t2026-07-01T09:00:00.0000000Z starting\n"
        "test\tRun\t2026-07-01T09:00:10.0000000Z 42 passed\n"
    )
    red = tmp_path / "red.gha.log"
    red.write_text(
        "test\tRun\t2026-07-02T09:00:00.0000000Z starting\n"
        "test\tRun\t2026-07-02T09:00:01.0000000Z ##[error]assertion failed\n"
        "test\tRun\t2026-07-02T09:00:15.0000000Z 41 passed, 1 failed\n"
    )
    cmp = tools.compare_metrics(str(green), str(red), FMT, "0")
    assert cmp["metrics"]["error_annotations"]["delta"] == 1.0
    assert cmp["metrics"]["duration_s"]["delta"] == pytest.approx(5.0)  # 10s -> 15s


# ---------------------------------------------------------------------------
# Non-log input: loud, named error
# ---------------------------------------------------------------------------


def test_non_log_input_raises_actionable_error(tmp_path):
    random_text = tmp_path / "notes.gha.log"
    random_text.write_text("just some random text\nwith multiple lines\nno tabs at all\n")
    with pytest.raises(ValueError, match="gh run view"):
        tools.list_kernels(str(random_text), FMT)

    json_input = tmp_path / "config.gha.log"
    json_input.write_text(json.dumps({"a": 1, "b": [1, 2, 3]}))
    with pytest.raises(ValueError, match="gh run view"):
        tools.list_kernels(str(json_input), FMT)


def test_empty_file_raises_actionable_error(tmp_path):
    empty = tmp_path / "empty.gha.log"
    empty.write_text("")
    with pytest.raises(ValueError, match="gh run view"):
        tools.list_kernels(str(empty), FMT)


# ---------------------------------------------------------------------------
# Probe: gh on PATH AND `gh auth status` succeeding — capture-gate only
# ---------------------------------------------------------------------------


def test_probe_distinguishes_missing_tool_from_missing_auth(monkeypatch):
    from perfdigest.adapters.gha_log import backend as gha_backend
    from perfdigest.platform.detect import PlatformInfo

    def _info(profilers):
        return PlatformInfo(
            os="linux", is_wsl=False, shell="posix", machine_model=None,
            cpu=None, gpu_vendors=(), profilers_on_path=profilers,
        )

    # gh missing entirely -> refuses without ever touching subprocess.
    monkeypatch.setattr(gha_backend, "detect", lambda: _info({}))
    report = gha_backend._probe()
    assert report.available is False
    assert "PATH" in report.reason

    # gh present but not authenticated -> distinct reason mentioning auth.
    monkeypatch.setattr(gha_backend, "detect", lambda: _info({"gha_log": "/usr/bin/gh"}))

    class _Unauthed:
        returncode = 1

    monkeypatch.setattr(gha_backend.subprocess, "run", lambda *a, **k: _Unauthed())
    gha_backend._clear_auth_memo()  # TTL memo (finding 5) must not leak stages
    report = gha_backend._probe()
    assert report.available is False
    assert "auth" in report.reason.lower()

    # gh present AND authenticated -> available.
    class _Authed:
        returncode = 0

    monkeypatch.setattr(gha_backend.subprocess, "run", lambda *a, **k: _Authed())
    gha_backend._clear_auth_memo()  # TTL memo (finding 5) must not leak stages
    report = gha_backend._probe()
    assert report.available is True
    assert report.tool == "/usr/bin/gh"


def test_capture_command_redirects_to_a_file():
    from perfdigest.adapters.gha_log import backend as gha_backend
    from perfdigest.platform.detect import PlatformInfo

    info = PlatformInfo(
        os="linux", is_wsl=False, shell="posix", machine_model=None,
        cpu=None, gpu_vendors=(), profilers_on_path={},
    )
    cmd = gha_backend._capture_command("29192019451", info)
    assert cmd == "gh run view 29192019451 --log > run.gha.log"
