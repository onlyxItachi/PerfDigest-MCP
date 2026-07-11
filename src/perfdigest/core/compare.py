"""Comparison digest — the agent-loop primitive (measure -> edit -> measure).

An optimizing agent otherwise holds TWO full digests in context and diffs them
mentally; this collapses that into one delta table. Still a translator, not a
judge: deltas are arithmetic, and deciding whether a delta is *good* stays the
model's job (a smaller duration is better; smaller spill bytes usually are; a
register count has no universal direction).

Sign convention: ``delta = b - a`` (report B relative to report A), so with
A = before and B = after, a negative ``duration_us`` delta means B is faster.

Absence rule, compare edition: a delta only exists when BOTH sides were
measured. One missing side keeps its NOT_AVAILABLE sentinel and the delta
becomes NOT_AVAILABLE too — never a fake 0.0. ``delta_pct`` over a measured
``a == 0`` baseline is a DIFFERENT case: the data was measured, the ratio is
just undefined — it gets its own sentinel so NOT_AVAILABLE keeps meaning
exactly "not measured in this export".
"""

from __future__ import annotations

from typing import Sequence

from perfdigest.core.digest import NOT_AVAILABLE
from perfdigest.core.metrics import NormalizedUnit

# Measured on both sides, but a == 0 so a percentage is mathematically undefined.
# Deliberately distinct from NOT_AVAILABLE (= "not measured in this export").
UNDEFINED_PCT = "undefined_baseline_is_zero"


def _value(unit: NormalizedUnit, term: str) -> float | None:
    return unit.duration_us if term == "duration_us" else unit.metric(term)


def build_comparison(
    unit_a: NormalizedUnit,
    unit_b: NormalizedUnit,
    format: str,
    requested: Sequence[str],
) -> dict:
    """Delta digest for one unit across two reports of the same format."""
    metrics: dict[str, dict] = {}
    for term in requested:
        va, vb = _value(unit_a, term), _value(unit_b, term)
        entry: dict[str, float | str] = {
            "a": va if va is not None else NOT_AVAILABLE,
            "b": vb if vb is not None else NOT_AVAILABLE,
        }
        if va is not None and vb is not None:
            entry["delta"] = vb - va
            entry["delta_pct"] = (vb - va) / va * 100.0 if va != 0 else UNDEFINED_PCT
        else:
            entry["delta"] = NOT_AVAILABLE
            entry["delta_pct"] = NOT_AVAILABLE
        metrics[term] = entry
    return {
        "kernel_a": unit_a.name,
        "kernel_b": unit_b.name,
        "index_a": unit_a.index,
        "index_b": unit_b.index,
        "format": format,
        "domain": unit_a.domain,
        "raw_ref_a": unit_a.raw_ref,
        "raw_ref_b": unit_b.raw_ref,
        "sign_convention": "delta = b - a",
        "metrics": metrics,
    }
