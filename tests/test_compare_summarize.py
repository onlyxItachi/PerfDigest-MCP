"""compare_metrics + summarize_report — the v1.1 agent-loop tools, on fixtures."""

import pytest

from perfdigest.core.digest import NOT_AVAILABLE
from perfdigest.server import tools


# ---------------------------------------------------------------- compare

def test_compare_before_after_cpu_stat(fixtures_dir):
    before = str(fixtures_dir / "cpu_stat_sample.perf-stat.json")
    after = str(fixtures_dir / "cpu_stat_improved_sample.perf-stat.json")
    cmp = tools.compare_metrics(before, after, "perf-stat-json", "0")

    assert cmp["sign_convention"] == "delta = b - a"
    ipc = cmp["metrics"]["ipc"]
    assert ipc["a"] == pytest.approx(2.0)
    assert ipc["b"] == pytest.approx(2.5)
    assert ipc["delta"] == pytest.approx(0.5)
    assert ipc["delta_pct"] == pytest.approx(25.0)

    wall = cmp["metrics"]["wall_time_ms"]
    assert wall["delta"] == pytest.approx(-20.5)  # negative = B faster


def test_compare_absence_is_asymmetric_and_honest(fixtures_dir):
    # A measured LLC on side A but no LLC events on side B: both raw values are
    # reported as-is, and the delta is NOT_AVAILABLE — never a fake 0.0.
    before = str(fixtures_dir / "cpu_stat_sample.perf-stat.json")
    after = str(fixtures_dir / "cpu_stat_improved_sample.perf-stat.json")
    llc = tools.compare_metrics(before, after, "perf-stat-json", "0")["metrics"]["llc_miss_rate"]
    assert llc["a"] == pytest.approx(5.0)
    assert llc["b"] == NOT_AVAILABLE
    assert llc["delta"] == NOT_AVAILABLE
    assert llc["delta_pct"] == NOT_AVAILABLE


def test_compare_two_kernels_of_one_report(fixtures_dir):
    # kernel_b overrides the B side: comparing two kernels of the SAME report.
    path = str(fixtures_dir / "nvidia_sample.ncu-csv")
    cmp = tools.compare_metrics(path, path, "ncu-csv", "add_kernel", kernel_b="matmul_kernel")
    dur = cmp["metrics"]["duration_us"]
    assert dur["a"] == pytest.approx(5.0)
    assert dur["b"] == pytest.approx(20.0)
    assert dur["delta"] == pytest.approx(15.0)
    assert dur["delta_pct"] == pytest.approx(300.0)
    assert cmp["kernel_a"] == "add_kernel" and cmp["kernel_b"] == "matmul_kernel"


# ---------------------------------------------------------------- summarize

def test_summarize_gpu_by_duration_with_coverage(fixtures_dir):
    path = str(fixtures_dir / "nvidia_sample.ncu-csv")
    s = tools.summarize_report(path, "ncu-csv", top_n=1)
    assert s["sorted_by"] == "duration_us"
    assert s["total_units"] == 2 and s["returned"] == 1
    top = s["units"][0]
    assert top["name"] == "matmul_kernel"  # 20us > 5us
    assert top["metrics"]["dram_pct_peak"] == pytest.approx(40.0)
    # matmul covers 20 of 25 total us.
    assert s["coverage_pct_of_total_duration"] == pytest.approx(80.0)


def test_summarize_cpu_sampler_falls_back_to_self_pct(fixtures_dir):
    path = str(fixtures_dir / "cpu_report_sample.perf-report.txt")
    s = tools.summarize_report(path, "perf-report", top_n=2)
    assert s["sorted_by"] == "self_pct"
    assert s["units"][0]["name"] == "compute_heavy"
    assert "coverage_pct_of_total_duration" not in s  # no durations: no fabrication


def test_summarize_codegen_falls_back_to_file_order(fixtures_dir):
    path = str(fixtures_dir / "gpu_codegen_sample.ptxas.txt")
    s = tools.summarize_report(path, "ptxas-verbose", top_n=2)
    assert s["sorted_by"] == "file_order"
    assert s["units"][0]["name"] == "_Z17true_spill_kernelPKfPfi [sm_89]"


def test_summarize_top_n_zero_and_typo_hint(fixtures_dir):
    path = str(fixtures_dir / "nvidia_sample.ncu-csv")
    s = tools.summarize_report(path, "ncu-csv", top_n=0)
    assert s["returned"] == 0 and s["units"] == []
    assert s["total_units"] == 2  # the report itself is still described

    # A typo'd metric must be flagged as out-of-vocabulary, not silently
    # uniform NOT_AVAILABLE (indistinguishable from a sparse export).
    s = tools.summarize_report(path, "ncu-csv", metrics=["dram_pct_peakk"])
    assert "unknown_metrics" in s
    assert s["unknown_metrics"]["requested_but_not_in_vocabulary"] == ["dram_pct_peakk"]


def test_compare_delta_pct_over_zero_baseline_is_undefined_not_absent():
    # A measured 0.0 baseline (e.g. branch_efficiency of a branchless kernel):
    # the delta exists; the PERCENTAGE is mathematically undefined — and that
    # must NOT reuse NOT_AVAILABLE, which means "not measured in this export".
    from perfdigest.core.compare import UNDEFINED_PCT, build_comparison
    from perfdigest.core.metrics import NormalizedUnit

    a = NormalizedUnit(name="k", index=0, duration_us=None, raw_ref="a",
                       metrics={"x": 0.0})
    b = NormalizedUnit(name="k", index=0, duration_us=None, raw_ref="b",
                       metrics={"x": 5.0})
    entry = build_comparison(a, b, "any", ["x"])["metrics"]["x"]
    assert entry["delta"] == 5.0  # arithmetic still fine
    assert entry["delta_pct"] == UNDEFINED_PCT
    assert entry["delta_pct"] != "not_available_in_this_export"
