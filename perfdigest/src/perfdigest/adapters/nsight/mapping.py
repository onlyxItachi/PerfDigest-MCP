"""ncu vendor metric name -> standard hardware term (the most-changed file).

The one rule: vendor jargon maps to a KNOWN standard term the model already
understands (``l2_hit_rate``), never to an invented abstract name
(``low_level_cache_usage``). Extending this dict is the expected way to grow
perfdigest's vocabulary.

Default core set confirmed with the user 2026-06-11 ("Extended 13"):
core diagnosis loop + warp/branch signal + raw memory throughput.
"""

from __future__ import annotations

# vendor (Nsight Compute) -> standard term
METRIC_MAP: dict[str, str] = {
    "gpu__time_duration.sum": "duration_us",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "compute_pct_peak",
    # NOTE: the headline DRAM %-of-peak (SpeedOfLight "Memory Throughput") is
    # gpu__dram_throughput on ncu 2026.1, NOT dram__throughput (which the init
    # prompt's example listed — that name is absent and would silently read as
    # not_available). Verified against a real RTX 4060 --set full report.
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "dram_pct_peak",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved_occupancy",
    "sm__maximum_warps_per_active_cycle_pct": "theoretical_occupancy",
    "l1tex__t_sector_hit_rate.pct": "l1_hit_rate",
    "lts__t_sector_hit_rate.pct": "l2_hit_rate",
    "launch__registers_per_thread": "registers_per_thread",
    "launch__block_size": "block_size",
    "launch__grid_size": "grid_size",
    "dram__bytes.sum.per_second": "mem_throughput_gbps",
    "smsp__sass_average_branch_targets_threads_uniform.pct": "branch_efficiency",
    "smsp__average_warp_latency_per_inst_issued.ratio": "warp_cycles_per_issue",
    "launch__shared_mem_per_block_static": "shared_mem_per_block",
}

STANDARD_TO_VENDOR: dict[str, str] = {v: k for k, v in METRIC_MAP.items()}

# What get_metrics(metrics=None) returns — a sufficient first answer for the
# memory-vs-compute-bound diagnosis loop (~200 dense tokens).
DEFAULT_CORE_SET: list[str] = [
    "duration_us",
    "compute_pct_peak",
    "dram_pct_peak",
    "achieved_occupancy",
    "theoretical_occupancy",
    "l1_hit_rate",
    "l2_hit_rate",
    "registers_per_thread",
    "block_size",
    "grid_size",
    "mem_throughput_gbps",
    "branch_efficiency",
    "warp_cycles_per_issue",
    "shared_mem_per_block",
]


def to_microseconds(value: float, unit: str | None) -> float:
    """Normalize a duration to microseconds based on the reported unit."""
    unit = (unit or "nsecond").lower()
    if unit.startswith(("nsecond", "ns", "nano")):
        return value / 1e3
    if unit.startswith(("usecond", "us", "micro")):
        return value
    if unit.startswith(("msecond", "ms", "milli")):
        return value * 1e3
    if unit in ("second", "s") or unit.startswith("second"):
        return value * 1e6
    return value / 1e3  # PRI's canonical duration unit is nanoseconds


def to_gbps(value: float, unit: str | None) -> float:
    """Normalize a byte/second throughput to GB/s based on the reported unit."""
    unit = (unit or "byte/second").lower()
    if unit.startswith("gbyte"):
        return value
    if unit.startswith("mbyte"):
        return value / 1e3
    if unit.startswith("kbyte"):
        return value / 1e6
    return value / 1e9  # plain byte/second


# standard term -> unit normalizer applied by readers after extraction
CONVERTERS = {
    "duration_us": to_microseconds,
    "mem_throughput_gbps": to_gbps,
}
