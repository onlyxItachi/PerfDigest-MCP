"""clang -ftime-trace (BuildDigest) backend — against a REAL clang 22 capture.

The committed fixtures are a real ``clang++ -ftime-trace`` export:
``tests/fixtures/template_heavy.cpp`` (deliberately template-heavy: compile-time
Fibonacci, variadic-template pack expansion, a generic Box<T> instantiated over
several types, STL containers) compiled with
``clang++-22 -std=c++17 -ftime-trace -c template_heavy.cpp`` producing
``tests/fixtures/template_heavy.ftime-trace.json``. Nothing host-identifying
was present to redact (checked: no absolute paths, hostname, or username in
the export — only the relative source filename, toolchain paths under
/usr/lib/gcc..., and ephemeral pid/tid/timestamp values).
"""

import json

import pytest

from perfdigest.adapters import registry
from perfdigest.core.digest import NOT_AVAILABLE
from perfdigest.server import tools

FMT = "clang-time-trace"
ALIAS = "ftime-trace"


def _path(fixtures_dir):
    p = fixtures_dir / "template_heavy.ftime-trace.json"
    assert p.exists(), f"committed fixture missing: {p}"
    return str(p)


def test_registered_and_formats():
    assert registry.get_backend(FMT).name == "clang_time_trace"
    assert registry.get_backend(ALIAS).name == "clang_time_trace"


def test_units_are_build_phase_domain(fixtures_dir):
    units = tools.list_kernels(_path(fixtures_dir), FMT)
    assert len(units) >= 40  # real compile: dozens of distinct phases
    assert all(u["domain"] == "build_phase" for u in units)
    assert all(u["duration_us"] is not None for u in units)
    names = {u["name"] for u in units}
    # Real payload phases from an actual template-heavy compile.
    assert "Frontend" in names
    assert "InstantiateFunction" in names
    assert "ParseClass" in names


def test_total_events_are_tagged_and_siblings_are_not(fixtures_dir):
    units = tools.list_kernels(_path(fixtures_dir), FMT)
    by_name = {u["name"]: u for u in units}

    # 'Total Frontend' is clang's bookkeeping aggregate over the 2 individual
    # Frontend spans in this report — must carry the [total] tag.
    assert "Total Frontend [total]" in by_name
    # ...and the untagged aggregate name must never leak into the payload.
    assert "Total Frontend" not in by_name
    # The payload sibling stays untagged.
    assert "Frontend" in by_name
    assert "Frontend [total]" not in by_name

    # Every 'Total *' unit in the whole report carries the tag — no exceptions.
    total_units = [u for u in units if u["name"].startswith("Total ")]
    assert len(total_units) >= 20  # this compile emits dozens of Total X events
    assert all(u["name"].endswith(" [total]") for u in total_units)

    # ...and no untagged 'Total X' name leaks anywhere (mirrors chrome_trace's
    # bookkeeping-leak guard for [Trace]/[overhead]).
    assert not any(
        u["name"].startswith("Total ") and not u["name"].endswith(" [total]")
        for u in units
    )


def test_total_frontend_matches_its_two_payload_siblings(fixtures_dir):
    # Verified against the real capture: 'Total Frontend' aggregates exactly
    # the 2 individual 'Frontend' spans elsewhere in the same report (a
    # coarse-grained phase clang always emits individually) — the concrete
    # proof that '[total]' really is a re-measurement of visible payload, not
    # independent signal, and must be discounted when summing by hand.
    p = _path(fixtures_dir)
    frontend = tools.get_metrics(p, FMT, "Frontend")["metrics"]
    total_frontend = tools.get_metrics(p, FMT, "Total Frontend [total]")["metrics"]
    assert frontend["calls"] == 2.0
    assert total_frontend["calls"] == 1.0  # one bookkeeping event, not 2
    # Within clang's own internal rounding (its own recorded values differ by
    # ~1us between the two independently-taken aggregates).
    assert total_frontend["total_time_us"] == pytest.approx(
        frontend["total_time_us"], abs=5.0
    )


def test_total_event_survives_as_the_sole_measurement_for_fine_grained_phases(
    fixtures_dir,
):
    # clang's -ftime-trace-granularity floor (default 500us) drops most short
    # per-instance spans from the file entirely; only 1 of 'RunPass's 18428
    # real instances is coarse enough to appear as its own X event. The
    # 'Total RunPass [total]' unit is therefore the ONLY complete measurement
    # of that phase's real cost — never drop it, and 'arg:count' (clang's own
    # instance count) is reachable via expand, not fabricated by this reader.
    p = _path(fixtures_dir)
    raw = tools.expand(p, FMT, "Total RunPass [total]", "all")["metrics"]
    assert raw["arg:count"] == 18428
    assert raw["calls"] == 1  # one bookkeeping event aggregates all 18428


def test_call_count_and_derived_timing(fixtures_dir):
    m = tools.get_metrics(_path(fixtures_dir), FMT, "InstantiateFunction")["metrics"]
    assert m["calls"] == 54.0
    assert m["total_time_us"] > 0
    assert m["avg_time_us"] == pytest.approx(m["total_time_us"] / 54.0)
    assert m["max_time_us"] >= m["avg_time_us"]


def test_hardware_metrics_are_honestly_absent(fixtures_dir):
    # A compiler phase trace has spans, not hardware counters: occupancy/DRAM
    # stay absent AND the vocabulary hint fires (not clang-time-trace terms).
    d = tools.get_metrics(
        _path(fixtures_dir), FMT, "Frontend", ["achieved_occupancy", "dram_pct_peak"]
    )
    assert d["metrics"]["achieved_occupancy"] == NOT_AVAILABLE
    assert d["metrics"]["dram_pct_peak"] == NOT_AVAILABLE
    assert "unknown_metrics" in d


def test_summarize_ranks_by_aggregate_time(fixtures_dir):
    s = tools.summarize_report(_path(fixtures_dir), FMT, top_n=5)
    assert s["sorted_by"] == "duration_us"
    assert s["total_units"] >= 40
    assert len(s["units"]) == 5
    assert all(u["domain"] == "build_phase" for u in s["units"])
    # ExecuteCompiler (the whole-compile span) is the single hottest unit.
    assert s["units"][0]["name"] == "ExecuteCompiler"


def test_expand_and_list_indices_agree(fixtures_dir):
    # load_units and raw_metrics both call the shared _grouped(); orderings
    # (and the [total] tagging) must never drift between the two entry points.
    p = _path(fixtures_dir)
    for u in tools.list_kernels(p, FMT):
        raw = tools.expand(p, FMT, str(u["index"]), "all")
        assert raw["kernel"] == u["name"]


def test_expand_reaches_bounded_per_call_durations(fixtures_dir):
    raw = tools.expand(_path(fixtures_dir), FMT, "InstantiateFunction", "durs")["metrics"]
    assert len(raw["durs_us"]) == 54  # one span per instantiation
    assert all(d > 0 for d in raw["durs_us"])


def test_compare_metrics_delta_is_absence_honest(tmp_path):
    a = tmp_path / "a.ftime-trace.json"
    b = tmp_path / "b.ftime-trace.json"
    a.write_text(json.dumps({"traceEvents": [
        {"ph": "X", "name": "ParseClass", "ts": 0, "dur": 100.0},
        {"ph": "X", "name": "ParseClass", "ts": 10, "dur": 100.0},
    ]}))
    b.write_text(json.dumps({"traceEvents": [
        {"ph": "X", "name": "ParseClass", "ts": 0, "dur": 150.0},
        {"ph": "X", "name": "ParseClass", "ts": 10, "dur": 150.0},
        {"ph": "X", "name": "ParseClass", "ts": 20, "dur": 150.0},
    ]}))
    cmp = tools.compare_metrics(str(a), str(b), FMT, "ParseClass")
    assert cmp["metrics"]["calls"]["delta"] == 1.0
    assert cmp["metrics"]["total_time_us"]["delta_pct"] == pytest.approx(125.0)
    assert cmp["metrics"]["avg_time_us"]["delta"] == pytest.approx(50.0)


def test_wrong_json_shapes_raise_actionable_errors(tmp_path):
    # Inherited straight from chrome_trace's _complete_events — must stay loud.
    not_trace = tmp_path / "config.json"
    not_trace.write_text(json.dumps({"version": 1, "settings": {}}))
    with pytest.raises(Exception, match="traceEvents"):
        tools.list_kernels(str(not_trace), FMT)

    scalar = tmp_path / "scalar.json"
    scalar.write_text("42")
    with pytest.raises(Exception, match="top-level"):
        tools.list_kernels(str(scalar), FMT)

    jsonl = tmp_path / "stat.json"
    jsonl.write_text('{"a": 1}\n{"b": 2}\n')
    with pytest.raises(Exception, match="Chrome trace"):
        tools.list_kernels(str(jsonl), FMT)


def test_bool_dur_is_not_a_measurement(tmp_path):
    trace = tmp_path / "bool.json"
    trace.write_text(json.dumps({"traceEvents": [
        {"ph": "X", "name": "ParseClass", "ts": 0, "dur": 2.0},
        {"ph": "X", "name": "Total ParseClass", "ts": 1, "dur": True},
    ]}))
    units = tools.list_kernels(str(trace), FMT)
    assert [u["name"] for u in units] == ["ParseClass"]


def test_registry_dispatch_on_both_format_names(fixtures_dir):
    p = _path(fixtures_dir)
    a = tools.list_kernels(p, FMT)
    b = tools.list_kernels(p, ALIAS)
    assert [u["name"] for u in a] == [u["name"] for u in b]
