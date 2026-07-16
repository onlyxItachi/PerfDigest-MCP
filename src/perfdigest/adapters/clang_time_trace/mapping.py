"""clang ``-ftime-trace`` event fields -> standard timing terms.

``-ftime-trace`` reports compiler-phase spans, not hardware counters, so the
vocabulary is the same deliberately small, generic timing shape as
chrome_trace: it applies equally to ``ParseClass``, ``InstantiateFunction``,
``CodeGen Function``, or a ``Total <Phase>`` aggregate. There is no
build-specific hardware truth one layer down here (unlike a GPU kernel, a
compiler phase has no occupancy/DRAM counterpart) — this backend's ceiling is
wall-clock timing.
"""

from __future__ import annotations

# standard term -> the trace-event arithmetic behind it (the `expand` hint)
STANDARD_TO_VENDOR: dict[str, str] = {
    "calls": "count of complete events (ph=='X') with this (cat, name) — for a "
    "'Total <Phase>' aggregate this equals clang's own args.count",
    "total_time_us": "sum of dur over those events — for 'Total <Phase>' this IS "
    "the aggregate clang already computed, not a re-sum by this reader",
    "avg_time_us": "total_time_us / calls",
    "max_time_us": "max(dur) over those events",
}

# What get_metrics(metrics=None) returns for a clang-time-trace unit.
DEFAULT_CORE_SET: list[str] = [
    "calls",
    "total_time_us",
    "avg_time_us",
    "max_time_us",
]
