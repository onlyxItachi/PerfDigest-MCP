"""rocprof counter/column names -> standard GPU terms.

AMD HIP kernels share the GPU vocabulary with NVIDIA (compute/dram %-of-peak,
cache hit rates, occupancy) — the same standard terms, different vendor counter
names — so an agent reads an AMD digest with no new vocabulary to learn.
rocprof emits *wide* CSV (one row per kernel dispatch, counters as columns),
unlike ncu's long form.
"""

from __future__ import annotations

# rocprof column / derived-counter name -> standard term
METRIC_MAP: dict[str, str] = {
    "DurationNs": "duration_us",
    "VALUUtilization": "compute_pct_peak",
    "VALUBusy": "compute_pct_peak",
    "MemUnitBusy": "dram_pct_peak",
    "L2CacheHit": "l2_hit_rate",
    "Occupancy": "achieved_occupancy",
    "GPUBusy": "gpu_busy_pct",
    "Wavefronts": "wavefronts",
    "SGPRsUsed": "sgprs_per_wave",
    "VGPRsUsed": "vgprs_per_thread",
}

STANDARD_TO_VENDOR: dict[str, str] = {v: k for k, v in METRIC_MAP.items()}

DEFAULT_CORE_SET: list[str] = [
    "duration_us",
    "compute_pct_peak",
    "dram_pct_peak",
    "l2_hit_rate",
    "achieved_occupancy",
    "gpu_busy_pct",
    "wavefronts",
    "vgprs_per_thread",
    "sgprs_per_wave",
]


def ns_to_us(value: float) -> float:
    return value / 1e3


CONVERTERS = {"duration_us": ns_to_us}
