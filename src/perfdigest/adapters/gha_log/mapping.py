"""GitHub Actions log fields -> standard CI-step terms.

A saved ``gh run view --log`` export has no hardware counters at all — the
vocabulary is deliberately small: how long a step ran (as approximated from
log-line timestamps), how big its log is, and whether it screamed. The dict
below is the expand-hint surface, mapping each standard term to the log
arithmetic behind it.
"""

from __future__ import annotations

# standard CI term -> the log arithmetic behind it (the `expand` hint)
STANDARD_TO_VENDOR: dict[str, str] = {
    "duration_s": "max(timestamp) - min(timestamp) over this step's log lines",
    "log_lines": "count of raw log lines attributed to this (job, step)",
    "error_annotations": "count of '##[error]' annotation markers in this step",
    "warning_annotations": "count of '##[warning]' annotation markers in this step",
}

# What get_metrics(metrics=None) returns for a gha-log unit.
DEFAULT_CORE_SET: list[str] = [
    "duration_s",
    "log_lines",
    "error_annotations",
    "warning_annotations",
]
