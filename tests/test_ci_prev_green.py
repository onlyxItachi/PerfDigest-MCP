"""Previous-green comparison for CI runs — REAL two-run artifacts end to end.

The schematic CIDigest loop: a CI run fails (or slows down), the agent diffs
it against the last GREEN run of the same workflow using two SAVED logs +
``compare_metrics`` — never re-reading the raw failed log, never touching the
live GitHub API. Every fixture here is a verbatim `gh run view --log --job`
capture (gh 2.96.0; the shipped 2.45.0 silently emits empty output for
``--log`` against modern runs):

* ``ci_perf_green_runA_sample.gha.log`` / ``ci_perf_green_runB_sample.gha.log``
  — the SAME job ("capture linux perf (degraded-PMU honesty test)") from two
  consecutive green "capture matrix" runs of the public
  onlyxItachi/PerfDigest-MCP-Bench repo (runs 29192019451 / job 86648181000
  and 29192140809 / job 86648506836, the latter triggered by commit 319a17c).
  Genuinely different hosts: run A landed on an AVX2 box in northcentralus,
  run B on an AVX-512 box in westus3 — same green workflow, real duration
  drift.
* ``ci_test_green_sample.gha.log`` / ``ci_test_failed_sample.gha.log`` — the
  SAME job ("test (ubuntu-latest, py3.11)") from a real red/green pair on THIS
  repo's own CI history (branch fix/v1.1.1-contract-hardening: failed run
  29168978670 / job 86586895834, then green run 29169039455 / job 86587053615
  two minutes later). The failed log carries a REAL
  ``##[error]Process completed with exit code 1.`` annotation — no synthetic
  error markers needed.

All four exports show the "UNKNOWN STEP" degradation documented in
test_gha_log.py (gh could not attribute lines to steps on these runs), so each
job digests as one ``<job> / UNKNOWN STEP`` unit — which is exactly what makes
the job-level comparison meaningful.
"""

from __future__ import annotations

import pytest

from perfdigest.core.compare import UNDEFINED_PCT
from perfdigest.core.digest import NOT_AVAILABLE
from perfdigest.server import tools

FMT = "gha-log"

PERF_JOB = "capture linux perf (degraded-PMU honesty test)"
PERF_UNIT = f"{PERF_JOB} / UNKNOWN STEP"
TEST_JOB = "test (ubuntu-latest, py3.11)"
TEST_UNIT = f"{TEST_JOB} / UNKNOWN STEP"


def _fx(fixtures_dir, name):
    p = fixtures_dir / f"{name}.gha.log"
    assert p.exists(), f"committed fixture missing: {p}"
    return str(p)


# ---------------------------------------------------------------------------
# The two same-job green runs digest independently, with real distinct values
# ---------------------------------------------------------------------------


def test_both_green_runs_digest_into_the_same_named_unit(fixtures_dir):
    for name, dur_s in [
        ("ci_perf_green_runA_sample", 41.084044),
        ("ci_perf_green_runB_sample", 47.865172),
    ]:
        units = tools.list_kernels(_fx(fixtures_dir, name), FMT)
        assert len(units) == 1
        assert units[0]["name"] == PERF_UNIT  # stable across the two exports
        assert units[0]["domain"] == "ci_step"
        assert units[0]["duration_us"] == pytest.approx(dur_s * 1e6)


# ---------------------------------------------------------------------------
# Green-to-green drift: same job, two runs, delta = b - a
# ---------------------------------------------------------------------------


def test_same_job_across_two_green_runs_duration_drift(fixtures_dir):
    cmp = tools.compare_metrics(
        _fx(fixtures_dir, "ci_perf_green_runA_sample"),
        _fx(fixtures_dir, "ci_perf_green_runB_sample"),
        FMT,
        PERF_JOB,
    )
    assert cmp["sign_convention"] == "delta = b - a"
    assert cmp["kernel_a"] == cmp["kernel_b"] == PERF_UNIT
    m = cmp["metrics"]
    # Run B really was ~6.8s slower (different Azure host); positive delta.
    assert m["duration_s"]["a"] == pytest.approx(41.084044)
    assert m["duration_s"]["b"] == pytest.approx(47.865172)
    assert m["duration_s"]["delta"] == pytest.approx(6.781128)
    # Both runs happen to emit 431 lines: a genuine measured-zero delta —
    # measured on both sides, NOT the absence sentinel.
    assert m["log_lines"]["delta"] == 0.0
    assert m["log_lines"]["delta_pct"] == 0.0
    # Green on both sides: error delta is a measured 0, and pct over the
    # 0-error baseline is the UNDEFINED sentinel (measured, ratio undefined) —
    # never NOT_AVAILABLE, which would falsely claim "not measured".
    assert m["error_annotations"]["delta"] == 0.0
    assert m["error_annotations"]["delta_pct"] == UNDEFINED_PCT


# ---------------------------------------------------------------------------
# Previous-green analysis proper: A = last green, B = the failed run
# ---------------------------------------------------------------------------


def test_failed_vs_previous_green_error_annotations_delta(fixtures_dir):
    cmp = tools.compare_metrics(
        _fx(fixtures_dir, "ci_test_green_sample"),
        _fx(fixtures_dir, "ci_test_failed_sample"),
        FMT,
        TEST_JOB,
    )
    m = cmp["metrics"]
    # THE failure scent, from real logs: green measured 0, failed measured 1.
    assert m["error_annotations"]["a"] == 0.0
    assert m["error_annotations"]["b"] == 1.0
    assert m["error_annotations"]["delta"] == 1.0
    assert m["error_annotations"]["delta_pct"] == UNDEFINED_PCT
    # The real failed run finished FASTER than green (it died early): the
    # duration delta is NEGATIVE. A speedup plus a positive error delta is a
    # failure signature, not an improvement — exactly why the usage prompt
    # says to read error_annotations next to duration.
    assert m["duration_s"]["delta"] == pytest.approx(-0.833697)
    assert m["log_lines"]["delta"] == 24.0  # 244 -> 268: the failure chatter


def test_failed_fixture_expand_reaches_the_real_error_line(fixtures_dir):
    raw = tools.expand(_fx(fixtures_dir, "ci_test_failed_sample"), FMT, "0", "error")[
        "metrics"
    ]
    assert raw["error_annotations"] == 1
    assert len(raw["error_lines"]) == 1
    assert "Process completed with exit code 1" in raw["error_lines"][0]


def test_summarize_surfaces_the_failure_scent_without_grepping(fixtures_dir):
    # The usage prompt's "start with summarize_report" claim, proven: the
    # default core set already carries error_annotations for every unit.
    s = tools.summarize_report(_fx(fixtures_dir, "ci_test_failed_sample"), FMT)
    assert s["domain"] == "ci_step"
    assert s["units"][0]["metrics"]["error_annotations"] == 1.0


# ---------------------------------------------------------------------------
# kernel_b: the job was renamed between the two runs
# ---------------------------------------------------------------------------


RENAMED_JOB = "capture linux perf v2 (degraded-PMU honesty test)"


def _renamed_run_b(fixtures_dir, tmp_path) -> str:
    """Run B's REAL fixture with the job name changed mid-string (perf -> perf v2).

    Renaming real data is the documented, acceptable transform here (the test
    needs a name change, which no real export pair we could source exhibits);
    no log LINE is invented — only the job column is rewritten, tab boundary
    included, so the payload text stays byte-identical. The rename goes
    MID-name deliberately: a pure suffix rename would leave the old name a
    substring of the new one, and _find_unit's substring matching would then
    resolve it without kernel_b at all — which is fine when it happens, but
    not the case this test is about.
    """
    real = (fixtures_dir / "ci_perf_green_runB_sample.gha.log").read_text()
    renamed = real.replace(f"{PERF_JOB}\t", f"{RENAMED_JOB}\t")
    p = tmp_path / "runB_renamed.gha.log"
    p.write_text(renamed)
    return str(p)


def test_renamed_job_needs_kernel_b_and_yields_identical_deltas(
    fixtures_dir, tmp_path
):
    run_a = _fx(fixtures_dir, "ci_perf_green_runA_sample")
    run_b_renamed = _renamed_run_b(fixtures_dir, tmp_path)

    # Without kernel_b the A-side name no longer resolves in B: loud, not a
    # silent empty comparison.
    with pytest.raises(ValueError, match="not found"):
        tools.compare_metrics(run_a, run_b_renamed, FMT, PERF_JOB)

    # kernel_b targets the new name; the numbers must be exactly the ones the
    # un-renamed comparison produced (the rename changed identity, not data).
    cmp = tools.compare_metrics(
        run_a, run_b_renamed, FMT, PERF_JOB, kernel_b=RENAMED_JOB
    )
    assert cmp["kernel_a"] == PERF_UNIT
    assert cmp["kernel_b"] == f"{RENAMED_JOB} / UNKNOWN STEP"
    assert cmp["metrics"]["duration_s"]["delta"] == pytest.approx(6.781128)
    assert cmp["metrics"]["log_lines"]["delta"] == 0.0


# ---------------------------------------------------------------------------
# Comparison honesty when one side cannot derive a duration
# ---------------------------------------------------------------------------


def test_comparison_with_untimestamped_side_keeps_duration_absent(
    fixtures_dir, tmp_path
):
    # SYNTHETIC B side (clearly marked; a no-timestamp export cannot be
    # sourced from any real gh capture we have): the same job name, but lines
    # carrying no parseable timestamp. duration_s must compare as
    # NOT_AVAILABLE on the B side AND in the delta — while log_lines, which
    # IS measured on both sides, still yields a real delta.
    p = tmp_path / "no_ts.gha.log"
    p.write_text(
        f"{PERF_JOB}\tUNKNOWN STEP\ttool output with no timestamp\n"
        f"{PERF_JOB}\tUNKNOWN STEP\tanother untimestamped line\n"
    )
    cmp = tools.compare_metrics(
        _fx(fixtures_dir, "ci_perf_green_runA_sample"), str(p), FMT, PERF_JOB
    )
    m = cmp["metrics"]
    assert m["duration_s"]["a"] == pytest.approx(41.084044)  # A side measured
    assert m["duration_s"]["b"] == NOT_AVAILABLE
    assert m["duration_s"]["delta"] == NOT_AVAILABLE
    assert m["duration_s"]["delta_pct"] == NOT_AVAILABLE
    assert m["log_lines"]["delta"] == pytest.approx(2.0 - 431.0)
