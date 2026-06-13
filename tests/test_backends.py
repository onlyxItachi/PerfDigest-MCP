"""Per-backend parsing against committed text/JSON fixtures — no hardware, no GPU.

This is the tier-1 universality proof encoded as tests: every backend's reader
turns a report (captured anywhere) into a standard-term digest on ANY OS. The
same file runs on the Linux/macOS/Windows CI runners — a macOS job digesting the
NVIDIA fixture IS the "CUDA-dev-on-a-Mac-via-CI" workflow.
"""

import pytest

from perfdigest.core.digest import NOT_AVAILABLE
from perfdigest.server import tools

# (fixture filename, format, expected backend domain)
CASES = {
    "nvidia": ("nvidia_sample.ncu-csv", "ncu-csv", "gpu_kernel"),
    "amd": ("amd_sample.rocprof-csv", "rocprof-csv", "gpu_kernel"),
    "cpu_stat": ("cpu_stat_sample.perf-stat.json", "perf-stat-json", "cpu_function"),
    "cpu_report": ("cpu_report_sample.perf-report.txt", "perf-report", "cpu_function"),
    "metal": ("metal_sample.metal-json", "metal-json", "gpu_pass"),
}


def _path(fixtures_dir, key):
    name, fmt, _ = CASES[key]
    p = fixtures_dir / name
    assert p.exists(), f"committed fixture missing: {p}"
    return str(p), fmt


def test_nvidia_csv_digest_without_a_gpu(fixtures_dir):
    path, fmt = _path(fixtures_dir, "nvidia")
    units = tools.list_kernels(path, fmt)
    assert [u["name"] for u in units] == ["add_kernel", "matmul_kernel"]
    assert all(u["domain"] == "gpu_kernel" for u in units)
    # ns -> us conversion (5,000 ns -> 5.0 us)
    assert units[0]["duration_us"] == pytest.approx(5.0)

    add = tools.get_metrics(path, fmt, "add_kernel")["metrics"]
    mm = tools.get_metrics(path, fmt, "matmul_kernel")["metrics"]
    # Distinct profiles: add is DRAM-bound, matmul is compute-bound.
    assert add["dram_pct_peak"] == 95.0 and add["compute_pct_peak"] == 30.0
    assert mm["compute_pct_peak"] == 85.0 and mm["dram_pct_peak"] == 40.0


def test_amd_rocprof_csv(fixtures_dir):
    path, fmt = _path(fixtures_dir, "amd")
    units = tools.list_kernels(path, fmt)
    assert {u["name"] for u in units} == {"add_kernel", "gemm_kernel"}
    gemm = tools.get_metrics(path, fmt, "gemm_kernel")["metrics"]
    assert gemm["compute_pct_peak"] == 88.0  # VALUUtilization -> standard term
    assert gemm["achieved_occupancy"] == 40.0  # Occupancy -> standard term
    assert tools.list_kernels(path, fmt)[0]["duration_us"] == pytest.approx(6.0)


def test_cpu_perf_stat_derived_ratios(fixtures_dir):
    path, fmt = _path(fixtures_dir, "cpu_stat")
    units = tools.list_kernels(path, fmt)
    assert len(units) == 1 and units[0]["domain"] == "cpu_function"
    m = tools.get_metrics(path, fmt, "0")["metrics"]
    assert m["ipc"] == pytest.approx(2.0)  # 3,000,000 / 1,500,000
    assert m["cache_miss_rate"] == pytest.approx(10.0)  # 200/2000
    assert m["llc_miss_rate"] == pytest.approx(5.0)  # 30/600
    assert m["branch_mispredict_rate"] == pytest.approx(1.0)  # 50/5000
    assert m["wall_time_ms"] == pytest.approx(120.5)
    # A program-level unit has no per-symbol self%: honest absence, not zero.
    assert m["self_pct"] == NOT_AVAILABLE


def test_cpu_perf_report_hot_symbols(fixtures_dir):
    path, fmt = _path(fixtures_dir, "cpu_report")
    units = tools.list_kernels(path, fmt)
    assert [u["name"] for u in units][:1] == ["compute_heavy"]
    top = tools.get_metrics(path, fmt, "compute_heavy")["metrics"]
    assert top["self_pct"] == 62.5 and top["samples"] == 1250.0
    # Counters are not in a sampling report: absent, never fabricated.
    assert top["ipc"] == NOT_AVAILABLE


def test_metal_passes(fixtures_dir):
    path, fmt = _path(fixtures_dir, "metal")
    units = tools.list_kernels(path, fmt)
    assert all(u["domain"] == "gpu_pass" for u in units)
    mm = tools.get_metrics(path, fmt, "matmul")["metrics"]
    assert mm["compute_pct_peak"] == 82.0  # ALU Utilization
    assert mm["achieved_occupancy"] == 70.0  # Occupancy
    raw = tools.expand(path, fmt, "matmul", "all")["metrics"]
    assert "ALU Utilization" in raw  # expand exposes the raw Apple names


def test_absence_is_never_zero_across_backends(fixtures_dir):
    # Request a GPU register metric the NVIDIA CSV export does not carry.
    path, fmt = _path(fixtures_dir, "nvidia")
    m = tools.get_metrics(path, fmt, "add_kernel", ["registers_per_thread"])["metrics"]
    assert m["registers_per_thread"] == NOT_AVAILABLE
