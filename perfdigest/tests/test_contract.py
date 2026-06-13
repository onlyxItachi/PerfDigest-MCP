"""Contract tests — the absence convention, with no profiler or GPU needed."""

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
