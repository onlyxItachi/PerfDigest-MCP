"""git numstat columns -> standard repo-change terms.

A numstat export has exactly two numbers per file — the vocabulary is
deliberately the smallest in the project. It carries FACTS about a session's
footprint (which files, how many lines), never verdicts: whether a 400-line
delta is "too big to release" is the model's conclusion, not this backend's.
"""

from __future__ import annotations

# standard term -> the numstat column behind it (the `expand` hint)
STANDARD_TO_VENDOR: dict[str, str] = {
    "lines_added": "numstat column 1 ('-' for binary: not a line count, stays absent)",
    "lines_deleted": "numstat column 2 ('-' for binary: not a line count, stays absent)",
}

# What get_metrics(metrics=None) returns for a repo-change unit.
DEFAULT_CORE_SET: list[str] = [
    "lines_added",
    "lines_deleted",
]
