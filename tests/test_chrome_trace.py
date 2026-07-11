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
    # Every summary unit carries its own domain (mixed-domain backend).
    assert all("domain" in u for u in s["units"])


def test_bookkeeping_spans_are_tagged_in_every_payload(fixtures_dir):
    # The capture-window span ('Trace') and profiler overhead rank among the
    # hottest units on a real trace. They stay digestible (never dropped) but
    # their names must carry the [cat] tag so no payload can pass them off as
    # workload — the summarize flow especially.
    units = tools.list_kernels(_path(fixtures_dir), FMT)
    trace_units = [u for u in units if "[Trace]" in u["name"]]
    overhead_units = [u for u in units if "[overhead]" in u["name"]]
    assert len(trace_units) == 1  # 'PyTorch Profiler (0) [Trace]'
    assert trace_units[0]["name"].startswith("PyTorch Profiler")
    assert len(overhead_units) >= 1
    # ...and no untagged bookkeeping name leaks anywhere.
    assert not any(u["name"] == "PyTorch Profiler (0)" for u in units)


def test_tagged_unit_resolves_by_name(fixtures_dir):
    m = tools.get_metrics(_path(fixtures_dir), FMT, "PyTorch Profiler (0) [Trace]")
    assert m["metrics"]["calls"] == 1.0


def test_expand_and_list_indices_agree(fixtures_dir):
    # load_units and raw_metrics call _grouped independently; their orderings
    # must never drift.
    p = _path(fixtures_dir)
    for u in tools.list_kernels(p, FMT):
        raw = tools.expand(p, FMT, str(u["index"]), "all")
        assert raw["kernel"] == u["name"]


def test_expand_reaches_bounded_per_call_durations(fixtures_dir):
    raw = tools.expand(_path(fixtures_dir), FMT, "aten::mm", "durs")["metrics"]
    assert len(raw["durs_us"]) == 4  # one span per profiled call
    assert all(d > 0 for d in raw["durs_us"])


def test_compare_differing_capture_lengths(tmp_path):
    # 2 calls vs 3 calls of the same op: total misleads (+87.5%), avg is the
    # calls-normalized signal — exactly what the usage prompt tells agents.
    a = tmp_path / "a.trace.json"
    b = tmp_path / "b.trace.json"
    a.write_text(json.dumps({"traceEvents": [
        {"ph": "X", "cat": "cpu_op", "name": "op", "ts": 0, "dur": 4.0},
        {"ph": "X", "cat": "cpu_op", "name": "op", "ts": 10, "dur": 4.0},
    ]}))
    b.write_text(json.dumps({"traceEvents": [
        {"ph": "X", "cat": "cpu_op", "name": "op", "ts": 0, "dur": 5.0},
        {"ph": "X", "cat": "cpu_op", "name": "op", "ts": 10, "dur": 5.0},
        {"ph": "X", "cat": "cpu_op", "name": "op", "ts": 20, "dur": 5.0},
    ]}))
    cmp = tools.compare_metrics(str(a), str(b), FMT, "op")
    assert cmp["metrics"]["calls"]["delta"] == 1.0
    assert cmp["metrics"]["total_time_us"]["delta_pct"] == pytest.approx(87.5)
    assert cmp["metrics"]["avg_time_us"]["delta_pct"] == pytest.approx(25.0)


def test_wrong_json_shapes_raise_actionable_errors(tmp_path):
    # A JSON object that is not a trace (e.g. some tool's config grabbed from a
    # cluttered dir) must be a loud, named error — never a silent empty digest.
    not_trace = tmp_path / "config.trace.json"
    not_trace.write_text(json.dumps({"version": 1, "settings": {}}))
    with pytest.raises(Exception, match="traceEvents"):
        tools.list_kernels(str(not_trace), FMT)

    # JSON-lines (a perf-stat export) is not valid JSON as a whole.
    jsonl = tmp_path / "stat.trace.json"
    jsonl.write_text('{"a": 1}\n{"b": 2}\n')
    with pytest.raises(Exception, match="Chrome trace"):
        tools.list_kernels(str(jsonl), FMT)

    # Scalar top level: named error, not a raw TypeError.
    scalar = tmp_path / "scalar.trace.json"
    scalar.write_text("42")
    with pytest.raises(Exception, match="top-level"):
        tools.list_kernels(str(scalar), FMT)


def test_bool_dur_is_not_a_measurement(tmp_path):
    # "dur": true from a broken emitter must not fabricate a 1.0us span.
    trace = tmp_path / "bool.trace.json"
    trace.write_text(json.dumps({"traceEvents": [
        {"ph": "X", "cat": "cpu_op", "name": "real", "ts": 0, "dur": 2.0},
        {"ph": "X", "cat": "cpu_op", "name": "fake", "ts": 1, "dur": True},
    ]}))
    units = tools.list_kernels(str(trace), FMT)
    assert [u["name"] for u in units] == ["real"]


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
