"""ninja log fields -> standard build-step terms (the most-changed file).

The vocabulary is deliberately small and honest: ``.ninja_log`` carries exactly
three per-edge numbers worth calling metrics. ``restat_mtime`` (a filesystem
mtime ninja uses for its own restat optimization) and ``cmd_hash`` (a hash of
the command line, used to detect when a rule changed) are ninja BOOKKEEPING,
not performance signal — they are deliberately excluded from this dict so they
never appear as a "metric" the model might try to rank or diff. Both are still
reachable raw via ``expand``.
"""

from __future__ import annotations

# standard build-step term -> ninja log field (for the `expand` hint)
STANDARD_TO_VENDOR: dict[str, str] = {
    "duration_ms": "end_ms - start_ms",
    "start_ms": "start_ms",
    "end_ms": "end_ms",
}

# What get_metrics(metrics=None) returns.
DEFAULT_CORE_SET: list[str] = [
    "duration_ms",
    "start_ms",
    "end_ms",
]
