"""perf event names -> standard CPU terms (the most-changed file for this adapter).

The GPU vocabulary (occupancy, dram_pct_peak, registers) is meaningless on a CPU
sampler, so this backend introduces a CPU vocabulary. Several terms are *derived*
ratios (IPC, miss rates) the reader computes from raw events — the dict below is
the expand-hint surface, mapping each standard term to the perf event(s) behind it.
"""

from __future__ import annotations

# standard CPU term -> representative perf event(s) (for the `expand` hint)
STANDARD_TO_VENDOR: dict[str, str] = {
    "wall_time_ms": "task-clock",
    "instructions": "instructions",
    "cycles": "cycles",
    "ipc": "instructions/cycles",
    "cache_miss_rate": "cache-misses/cache-references",
    "llc_miss_rate": "LLC-load-misses/LLC-loads",
    "branch_mispredict_rate": "branch-misses/branches",
    "self_pct": "overhead",
    "samples": "samples",
}

# What get_metrics(metrics=None) returns. Union of program-level (perf stat) and
# per-symbol (perf report) terms; whichever the unit lacks is honestly absent.
DEFAULT_CORE_SET: list[str] = [
    "self_pct",
    "samples",
    "wall_time_ms",
    "ipc",
    "instructions",
    "cycles",
    "cache_miss_rate",
    "llc_miss_rate",
    "branch_mispredict_rate",
]

# perf stat event name -> raw term we keep verbatim (others feed derived ratios).
EVENT_TO_TERM: dict[str, str] = {
    "instructions": "instructions",
    "cycles": "cycles",
    "cpu-cycles": "cycles",
}
