"""Chrome-trace (torch/Kineto) backend — against a REAL torch 2.12 CUDA trace.

The committed fixture (tests/fixtures/torch_sample.torch-trace.json) is a real
``torch.profiler`` export (torch 2.12.1+cu132, RTX 4060): 4x relu(a @ b) + sum
on CUDA — aten ops, cudaLaunchKernel runtime calls, and the actual sgemm /
elementwise / reduce kernels. Only the ``host_name`` metadata was redacted.
"""

import json

import pytest

from perfdigest.adapters import registry
from perfdigest.core.digest import NOT_AVAILABLE
from perfdigest.server import tools

FMT = "torch-trace"


def _path(fixtures_dir):
    p = fixtures_dir / "torch_sample.torch-trace.json"
    assert p.exists(), f"committed fixture missing: {p}"
    return str(p)


def test_registered_and_formats():
    assert registry.get_backend(FMT).name == "chrome_trace"
    assert registry.get_backend("chrome-trace").name == "chrome_trace"


def test_aggregates_ops_and_kernels_in_one_report(fixtures_dir):
    units = tools.list_kernels(_path(fixtures_dir), FMT)
    by_name = {u["name"]: u for u in units}

    # Framework side: 4 loop iterations -> one aggregated aten::mm unit.
    assert "aten::mm" in by_name
    assert by_name["aten::mm"]["domain"] == "framework_op"

    # Device side, same report: the real sgemm kernel, domain gpu_kernel.
    sgemm = [u for u in units if "ampere_sgemm" in u["name"]]
    assert len(sgemm) == 1 and sgemm[0]["domain"] == "gpu_kernel"

    # Aggregated units always carry a duration (sum of spans) — sortable.
    assert all(u["duration_us"] is not None for u in units)


def test_call_count_and_derived_timing(fixtures_dir):
    m = tools.get_metrics(_path(fixtures_dir), FMT, "aten::mm")["metrics"]
    assert m["calls"] == 4.0  # 4 profiled loop iterations
    assert m["total_time_us"] > 0
    assert m["avg_time_us"] == pytest.approx(m["total_time_us"] / 4.0)
    assert m["max_time_us"] >= m["avg_time_us"]


def test_hardware_metrics_are_honestly_absent(fixtures_dir):
    # A trace has spans, not counters: occupancy/DRAM stay absent AND the
    # vocabulary hint fires (they are not chrome-trace terms at all).
    d = tools.get_metrics(_path(fixtures_dir), FMT, "aten::mm",
                          ["achieved_occupancy", "dram_pct_peak"])
    assert d["metrics"]["achieved_occupancy"] == NOT_AVAILABLE
    assert d["metrics"]["dram_pct_peak"] == NOT_AVAILABLE
    assert "unknown_metrics" in d


def test_summarize_ranks_by_aggregate_time(fixtures_dir):
    s = tools.summarize_report(_path(fixtures_dir), FMT, top_n=5)
    assert s["sorted_by"] == "duration_us"
    assert s["total_units"] >= 10  # ops + runtime calls + kernels
    assert len(s["units"]) == 5


def test_expand_exposes_cat_and_args(fixtures_dir):
    raw = tools.expand(_path(fixtures_dir), FMT, "aten::mm", "all")["metrics"]
    assert raw["cat"] == "cpu_op"
    assert raw["calls"] == 4


def test_legacy_bare_array_trace(tmp_path):
    # Older exporters emit a bare JSON array; non-X and dur-less events skipped.
    trace = tmp_path / "legacy.trace.json"
    trace.write_text(json.dumps([
        {"ph": "X", "cat": "cpu_op", "name": "work", "ts": 10, "dur": 5.0},
        {"ph": "X", "cat": "cpu_op", "name": "work", "ts": 20, "dur": 7.0},
        {"ph": "B", "cat": "cpu_op", "name": "open-span", "ts": 0},
        {"ph": "X", "cat": "cpu_op", "name": "no-dur", "ts": 30},
    ]))
    units = tools.list_kernels(str(trace), FMT)
    assert [u["name"] for u in units] == ["work"]
    m = tools.get_metrics(str(trace), FMT, "work")["metrics"]
    assert m["calls"] == 2.0 and m["total_time_us"] == pytest.approx(12.0)


def test_cross_category_name_collision_gets_tagged(tmp_path):
    # The same name in two categories must yield two distinguishable units.
    trace = tmp_path / "collide.trace.json"
    trace.write_text(json.dumps({"traceEvents": [
        {"ph": "X", "cat": "cpu_op", "name": "memcpy", "ts": 0, "dur": 3.0},
        {"ph": "X", "cat": "kernel", "name": "memcpy", "ts": 5, "dur": 9.0},
    ]}))
    units = tools.list_kernels(str(trace), FMT)
    names = sorted(u["name"] for u in units)
    assert names == ["memcpy [cpu_op]", "memcpy [kernel]"]
    domains = {u["name"]: u["domain"] for u in units}
    assert domains["memcpy [kernel]"] == "gpu_kernel"
    assert domains["memcpy [cpu_op]"] == "framework_op"
