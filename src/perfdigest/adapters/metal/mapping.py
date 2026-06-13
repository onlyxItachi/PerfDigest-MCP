"""Apple GPU counter names -> standard terms.

Metal exposes compute/memory utilization and cache hit rates under Apple's own
counter names; we map them onto the shared GPU vocabulary so a Metal digest reads
like an NVIDIA/AMD one. A Metal "unit" is an encoder/compute pass (gpu_pass), not
a kernel launch.
"""

from __future__ import annotations

# Apple counter name -> standard term
METRIC_MAP: dict[str, str] = {
    "ALU Utilization": "compute_pct_peak",
    "ALU Limiter": "compute_pct_peak",
    "Device Memory Utilization": "dram_pct_peak",
    "Memory Utilization": "dram_pct_peak",
    "L1 Hit Rate": "l1_hit_rate",
    "L2 Hit Rate": "l2_hit_rate",
    "Occupancy": "achieved_occupancy",
    "Threadgroup Memory": "shared_mem_per_block",
}

STANDARD_TO_VENDOR: dict[str, str] = {v: k for k, v in METRIC_MAP.items()}

DEFAULT_CORE_SET: list[str] = [
    "duration_us",
    "compute_pct_peak",
    "dram_pct_peak",
    "l1_hit_rate",
    "l2_hit_rate",
    "achieved_occupancy",
    "shared_mem_per_block",
]
