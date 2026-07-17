"""cmake profile event fields -> standard timing terms.

A cmake configure profile reports script-command spans, not hardware counters,
so the vocabulary is the same deliberately small, generic timing shape as
chrome_trace/clang_time_trace: it applies equally to ``find_package``,
``try_compile``, a user ``function()``, or cmake's whole-step ``configure``
frame. The per-call detail (which file/line, which arguments) is expand-level
data (``arg:location``, ``arg:functionArgs``), not a metric.
"""

from __future__ import annotations

# standard term -> the trace-event arithmetic behind it (the `expand` hint)
STANDARD_TO_VENDOR: dict[str, str] = {
    "calls": "count of folded B/E spans with this (cat, name) — how many times "
    "the command/macro/function ran during configure",
    "total_time_us": "sum of (E.ts - B.ts) over those spans",
    "avg_time_us": "total_time_us / calls",
    "max_time_us": "max span duration — one slow call vs death-by-thousand-calls",
}

# What get_metrics(metrics=None) returns for a cmake-profile unit.
DEFAULT_CORE_SET: list[str] = [
    "calls",
    "total_time_us",
    "avg_time_us",
    "max_time_us",
]
