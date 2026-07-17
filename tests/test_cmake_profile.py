"""cmake configure-profile backend — parsing a REAL cmake 4.3.2 trace.

The committed fixture (tests/fixtures/cmake_profile_sample.cmake-profile.json)
is a real ``cmake --profiling-format=google-trace --profiling-output=...``
capture of configuring the small project committed alongside it
(tests/fixtures/cmake_profile_sample/: two targets, find_package(Threads), an
include()d helper defining a user function, a foreach of messages). Provenance:

  * run 1 (cold configure) produced a 1.5MB trace dominated by compiler
    detection; run 2 — a warm RE-configure of the same tree, the everyday
    "why is my configure slow" artifact — is the committed one (1598 events).
  * cmake emits ``ph=='B'``/``'E'`` PAIRS (no ``X`` events at all; the E side
    carries only ph/pid/tid/ts) — the real-report finding that drove the
    ``fold_be_pairs`` hook in chrome_trace's ``_grouped``.
  * 86 ``args.location`` values pointing into the temp build/source dirs were
    relativized to ``proj/``/``build/`` before committing (host-path hygiene);
    verified programmatically that every field OUTSIDE ``args`` — names, ph,
    ts, pid, tid — is byte-identical to the raw capture, and /usr/share/cmake
    module locations are verbatim. Timings are untouched real data.

Expected values below are read straight from the committed trace.
"""

import pytest

from perfdigest.adapters import registry
from perfdigest.core.digest import NOT_AVAILABLE
from perfdigest.server import tools

FMT = "cmake-profile"

CONFIGURE_US = 10166.0  # cmake's whole-configure frame
INCLUDE_CALLS = 28.0
INCLUDE_TOTAL_US = 6132.0
INCLUDE_MAX_US = 1091.0
FIND_PACKAGE_US = 1680.0
GENERATE_US = 2015.0
TOTAL_UNITS = 50


def _path(fixtures_dir):
    p = fixtures_dir / "cmake_profile_sample.cmake-profile.json"
    assert p.exists(), f"committed fixture missing: {p}"
    return str(p)


def test_registered_and_domain():
    b = registry.get_backend(FMT)
    assert b.name == "cmake_profile"
    assert b.domain == "build_phase"  # a configure frame is a build phase
    assert registry.get_backend("cmake-trace").name == "cmake_profile"
    assert registry.get_backend("CMAKE-PROFILE").name == "cmake_profile"  # case-insensitive


def test_parses_real_trace_from_be_pairs(fixtures_dir):
    # The trace contains ZERO ph=='X' events — every unit below exists only
    # because B/E pairs were stack-folded. A naive chrome-trace read would
    # return [] here.
    units = tools.list_kernels(_path(fixtures_dir), FMT)
    assert len(units) == TOTAL_UNITS
    names = {u["name"] for u in units}
    # cmake's own step frames + built-ins + the user function from helpers.cmake
    assert {"configure", "generate", "project", "find_package", "include",
            "add_library", "add_executable", "add_tagged_executable",
            "foreach", "message"} <= names
    assert all(u["domain"] == "build_phase" for u in units)


def test_name_aggregation_across_call_sites(fixtures_dir):
    # 28 include() executions (CMakeLists + cmake's own module chains) fold
    # into ONE unit — the (cat, name) rule shared with the trace siblings.
    m = tools.get_metrics(_path(fixtures_dir), FMT, "include")["metrics"]
    assert m["calls"] == INCLUDE_CALLS
    assert m["total_time_us"] == pytest.approx(INCLUDE_TOTAL_US)
    assert m["avg_time_us"] == pytest.approx(INCLUDE_TOTAL_US / INCLUDE_CALLS)
    assert m["max_time_us"] == pytest.approx(INCLUDE_MAX_US)


def test_ranking_by_total_time(fixtures_dir):
    summary = tools.summarize_report(_path(fixtures_dir), FMT, top_n=3)
    assert summary["sorted_by"] == "duration_us"
    assert [u["name"] for u in summary["units"]] == ["configure", "include", "project"]
    assert summary["units"][0]["duration_us"] == pytest.approx(CONFIGURE_US)
    assert summary["total_units"] == TOTAL_UNITS


def test_nested_overlap_totals_are_not_additive(fixtures_dir):
    # 'configure' is cmake's whole-step frame: everything else during configure
    # nests inside it, so summing units GROSSLY exceeds the step itself —
    # the documented overlap caveat, provable on real data.
    units = tools.list_kernels(_path(fixtures_dir), FMT)
    by = {u["name"]: u["duration_us"] for u in units}
    assert by["configure"] == pytest.approx(CONFIGURE_US)
    assert by["project"] < by["configure"]  # project() ran inside configure
    assert by["find_package"] == pytest.approx(FIND_PACKAGE_US)
    total = sum(u["duration_us"] for u in units)
    assert total > by["configure"] + by["generate"]  # overlap, not additivity


def test_expand_reaches_call_site_detail(fixtures_dir):
    # Unit names are aggregated by design; the file:line and argument detail
    # must stay reachable per-unit through expand.
    raw = tools.expand(_path(fixtures_dir), FMT, "find_package", "all")["metrics"]
    assert raw["arg:functionArgs"] == "Threads REQUIRED"
    assert raw["arg:location"] == "proj/CMakeLists.txt:6"  # relativized, no host path
    assert raw["cat"] == "script"
    assert raw["durs_us"] == [pytest.approx(FIND_PACKAGE_US)]


def test_truncated_trace_drops_unpaired_b_never_fabricates(tmp_path):
    # A trace cut off mid-configure leaves an open B with no E: that span has
    # no measurable duration and must vanish — not appear with a made-up dur.
    trace = tmp_path / "truncated.cmake-profile.json"
    trace.write_text(
        '[{"ph":"B","name":"include","cat":"script","ts":100,"pid":1,"tid":0},'
        '{"ph":"E","ts":250,"pid":1,"tid":0},'
        '{"ph":"B","name":"project","cat":"script","ts":300,"pid":1,"tid":0}]'
    )
    units = tools.list_kernels(str(trace), FMT)
    assert [u["name"] for u in units] == ["include"]
    assert units[0]["duration_us"] == pytest.approx(150.0)


def test_loud_errors_on_non_trace_input(tmp_path):
    # Inherited from the shared chrome-trace machinery: junk must raise, never
    # digest to a silent [].
    not_json = tmp_path / "junk.cmake-profile.json"
    not_json.write_text("this is a build log, not json")
    with pytest.raises(ValueError) as exc:
        tools.list_kernels(str(not_json), FMT)
    assert "not valid JSON" in str(exc.value)

    wrong_shape = tmp_path / "wrong.cmake-profile.json"
    wrong_shape.write_text('{"benchmarks": []}')
    with pytest.raises(ValueError) as exc:
        tools.list_kernels(str(wrong_shape), FMT)
    assert "traceEvents" in str(exc.value)


def test_absence_honesty_for_a_metric_the_trace_lacks(fixtures_dir):
    d = tools.get_metrics(_path(fixtures_dir), FMT, "configure", ["dram_pct_peak", "calls"])
    assert d["metrics"]["dram_pct_peak"] == NOT_AVAILABLE
    assert d["metrics"]["calls"] == 1.0
    assert "unknown_metrics" in d


def test_chrome_trace_default_path_unaffected_by_fold_hook(fixtures_dir):
    # The torch fixture digested through chrome_trace (fold default OFF) must
    # be byte-for-byte the behavior the fold hook found: B/E pairs in an
    # ORDINARY chrome trace stay un-folded (Kineto uses X events; any stray
    # B/E there is not this backend's business).
    path = str(fixtures_dir / "torch_sample.torch-trace.json")
    units = tools.list_kernels(path, "torch-trace")
    assert len(units) > 0  # existing behavior intact (full assertions live in
    # test_chrome_trace.py; this is the cross-backend smoke check)
