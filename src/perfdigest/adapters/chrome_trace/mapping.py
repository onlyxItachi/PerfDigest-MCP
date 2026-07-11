"""Chrome-trace event fields -> standard timing terms.

A trace exporter reports spans, not hardware counters, so the vocabulary is
deliberately small and generic timing: it applies equally to a torch/Kineto
trace, a JAX trace, or a clang ``-ftime-trace``. Hardware truth about a kernel
seen here (occupancy, DRAM) lives one layer down — profile it with ncu and read
the SAME kernel name through the nsight backend.
"""

from __future__ import annotations

# standard term -> the trace-event arithmetic behind it (the `expand` hint)
STANDARD_TO_VENDOR: dict[str, str] = {
    "calls": "count of complete events (ph=='X') with this (cat, name)",
    "total_time_us": "sum of dur over those events",
    "avg_time_us": "total_time_us / calls",
    "max_time_us": "max(dur) over those events",
}

# What get_metrics(metrics=None) returns for a chrome-trace unit.
DEFAULT_CORE_SET: list[str] = [
    "calls",
    "total_time_us",
    "avg_time_us",
    "max_time_us",
]
