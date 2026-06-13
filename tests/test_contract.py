"""Contract tests — the absence convention, with no profiler or GPU needed."""

import dataclasses

import pytest

from perfdigest.core.digest import NOT_AVAILABLE, build_digest
from perfdigest.core.metrics import NormalizedKernel


def _kernel(**metrics):
    return NormalizedKernel(
        name="k", index=0, duration_us=12.5, raw_ref="x.ncu-rep", metrics=metrics
    )


def test_none_metric_becomes_not_available_never_zero():
    k = _kernel(dram_pct_peak=None, l2_hit_rate=80.0)
    digest = build_digest(k, "ncu-rep", ["dram_pct_peak", "l2_hit_rate"])
    assert digest.metrics["dram_pct_peak"] == NOT_AVAILABLE
    assert digest.metrics["dram_pct_peak"] != 0.0
    assert digest.metrics["l2_hit_rate"] == 80.0


def test_unmapped_metric_is_absent_not_zero():
    k = _kernel(l2_hit_rate=80.0)
    digest = build_digest(k, "ncu-rep", ["never_measured"])
    assert digest.metrics["never_measured"] == NOT_AVAILABLE


def test_zero_is_preserved_as_real_measurement():
    # A genuine 0.0 (e.g. zero branch divergence) must NOT be confused with absence.
    k = _kernel(branch_efficiency=0.0)
    digest = build_digest(k, "ncu-rep", ["branch_efficiency"])
    assert digest.metrics["branch_efficiency"] == 0.0


def test_duration_is_a_first_class_term():
    k = _kernel()
    digest = build_digest(k, "ncu-rep", ["duration_us"])
    assert digest.metrics["duration_us"] == 12.5


def test_duration_none_is_not_available():
    k = NormalizedKernel(name="k", index=0, duration_us=None, raw_ref="x.ncu-rep")
    digest = build_digest(k, "ncu-rep", ["duration_us"])
    assert digest.metrics["duration_us"] == NOT_AVAILABLE


def test_digest_preserves_kernel_identity():
    k = NormalizedKernel(
        name="myKernel", index=3, duration_us=9.0, raw_ref="run.ncu-rep",
        metrics={"l2_hit_rate": 50.0},
    )
    digest = build_digest(k, "ncu-rep", ["l2_hit_rate"])
    assert (digest.kernel, digest.index, digest.format, digest.raw_ref) == (
        "myKernel", 3, "ncu-rep", "run.ncu-rep",
    )
    assert digest.to_dict()["metrics"] == {"l2_hit_rate": 50.0}


def test_empty_request_yields_empty_metrics():
    digest = build_digest(_kernel(l2_hit_rate=1.0), "ncu-rep", [])
    assert digest.metrics == {}


def test_normalized_kernel_metric_lookup():
    k = _kernel(l2_hit_rate=42.0)
    assert k.metric("l2_hit_rate") == 42.0
    assert k.metric("absent") is None  # absent, not a KeyError, not 0.0


def test_normalized_kernel_is_frozen():
    k = _kernel()
    with pytest.raises(dataclasses.FrozenInstanceError):
        k.name = "mutated"  # type: ignore[misc]
