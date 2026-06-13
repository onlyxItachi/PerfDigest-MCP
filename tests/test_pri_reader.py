"""Reader + tool tests against the REAL fixture (skipped if it's absent).

These are the spec's Phase-2 gate: input = a real .ncu-rep, output = a
NormalizedKernel / digest that holds together.
"""

import pytest

from perfdigest.adapters.nsight import pri_reader
from perfdigest.adapters.nsight.mapping import DEFAULT_CORE_SET
from perfdigest.core.digest import NOT_AVAILABLE
from perfdigest.server import tools

# the benchmark launches gafime_global_continuous_kernel at arity 1/2/3/5
EXPECTED_KERNELS = 4


def test_loads_all_kernels(report_path):
    kernels = pri_reader.load_kernels(report_path)
    assert len(kernels) == EXPECTED_KERNELS
    assert all(k.name for k in kernels)
    assert all(k.duration_us and k.duration_us > 0 for k in kernels)


def test_list_kernels_tool_shape(report_path):
    rows = tools.list_kernels(report_path, "ncu-rep")
    assert len(rows) == EXPECTED_KERNELS
    assert set(rows[0]) == {"name", "index", "duration_us"}


def test_default_core_set_is_populated(report_path):
    digest = tools.get_metrics(report_path, "ncu-rep", "0")
    assert set(digest["metrics"]) == set(DEFAULT_CORE_SET)
    # this DRAM-streaming kernel must report real compute AND dram %-of-peak —
    # if dram_pct_peak is NOT_AVAILABLE the mapping name regressed again.
    assert isinstance(digest["metrics"]["compute_pct_peak"], float)
    assert isinstance(digest["metrics"]["dram_pct_peak"], float)
    assert digest["metrics"]["dram_pct_peak"] > 50.0  # known memory-bound


def test_format_is_mandatory_and_validated(report_path):
    with pytest.raises(ValueError):
        tools.get_metrics(report_path, "csv", "0")


def test_kernel_selection_by_index_and_substring(report_path):
    by_index = tools.get_metrics(report_path, "ncu-rep", "1")
    by_substr = tools.get_metrics(report_path, "ncu-rep", "int)2")
    assert by_index["index"] == 1
    assert "(int)2" in by_substr["kernel"]


def test_explicit_metric_subset(report_path):
    digest = tools.get_metrics(report_path, "ncu-rep", "0", metrics=["duration_us", "l2_hit_rate"])
    assert set(digest["metrics"]) == {"duration_us", "l2_hit_rate"}


def test_unknown_requested_metric_flagged_not_zeroed(report_path):
    digest = tools.get_metrics(report_path, "ncu-rep", "0", metrics=["made_up_metric"])
    assert digest["metrics"]["made_up_metric"] == NOT_AVAILABLE
    assert "unknown_metrics" in digest


def test_expand_returns_raw_vendor_metrics(report_path):
    ex = tools.expand(report_path, "ncu-rep", "0", "dram")
    assert ex["metric_count"] > 0
    assert all("dram" in name.lower() for name in ex["metrics"])


def test_unknown_kernel_raises_with_available_listing(report_path):
    with pytest.raises(ValueError) as exc:
        tools.get_metrics(report_path, "ncu-rep", "no_such_kernel")
    # the error must help the agent recover, not just fail
    assert "gafime_global_continuous_kernel" in str(exc.value)


def test_out_of_range_index_raises(report_path):
    with pytest.raises(ValueError):
        tools.get_metrics(report_path, "ncu-rep", "999")


def test_expand_all_is_superset_of_filtered(report_path):
    every = tools.expand(report_path, "ncu-rep", "0", "all")
    dram = tools.expand(report_path, "ncu-rep", "0", "dram")
    assert every["metric_count"] >= dram["metric_count"]
    assert every["metric_count"] > 100  # --set full carries a large raw set


def test_durations_descend_when_sorted(report_path):
    rows = tools.list_kernels(report_path, "ncu-rep")
    hottest = max(rows, key=lambda r: r["duration_us"])
    # the arity-5 kernel does the most work and is the known hot spot
    assert "(int)5" in hottest["name"]


def test_kernel_names_are_demangled(report_path):
    # raw mangled names look like _Z..; demangled carry the readable signature
    rows = tools.list_kernels(report_path, "ncu-rep")
    assert all("gafime_global_continuous_kernel" in r["name"] for r in rows)


def test_get_metrics_raw_ref_points_back_to_report(report_path):
    digest = tools.get_metrics(report_path, "ncu-rep", "0")
    assert digest["raw_ref"].endswith("gafime.ncu-rep")


def test_every_kernel_satisfies_core_invariants(report_path):
    # Per-launch coverage that holds for ANY kernel in ANY --set full report:
    # a launch always has a name, a positive duration, and the universally
    # collected SpeedOfLight/occupancy metrics as real floats (not absence).
    kernels = pri_reader.load_kernels(report_path)
    assert kernels, "report has no kernels"
    for k in kernels:
        digest = tools.get_metrics(report_path, "ncu-rep", str(k.index))
        assert k.name, f"kernel #{k.index} has no name"
        assert isinstance(k.duration_us, float) and k.duration_us > 0
        assert set(digest["metrics"]) == set(DEFAULT_CORE_SET)
        for always_present in ("duration_us", "compute_pct_peak", "achieved_occupancy"):
            assert isinstance(digest["metrics"][always_present], float), (
                f"kernel #{k.index} {k.name}: {always_present} should be measured"
            )


def test_kernel_indices_are_contiguous(report_path):
    indices = [k.index for k in pri_reader.load_kernels(report_path)]
    assert indices == list(range(len(indices)))
