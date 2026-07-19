"""KernelDigest — the JSON response schema and the absence convention.

``get_metrics`` must surface missing data transparently: a requested metric
that the export does not contain is returned as the literal string
``"not_available_in_this_export"`` — never silently dropped, never ``0.0``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from perfdigest.core.metrics import NormalizedKernel, NormalizedUnit

NOT_AVAILABLE = "not_available_in_this_export"


@dataclass(frozen=True)
class UnitDigest:
    """The compact, structured answer returned by ``get_metrics``.

    ``kernel`` keeps its v1 name (the field every client already reads); for a
    CPU function or Metal pass it is simply that unit's name. ``domain`` tells
    the model which kind of unit it is so it reads the vocabulary correctly.
    """

    kernel: str
    index: int
    format: str
    raw_ref: str
    metrics: dict[str, float | str]  # float, or the NOT_AVAILABLE sentinel
    domain: str = "gpu_kernel"

    def to_dict(self) -> dict:
        return {
            "kernel": self.kernel,
            "index": self.index,
            "format": self.format,
            "domain": self.domain,
            "raw_ref": self.raw_ref,
            "metrics": self.metrics,
        }


# Back-compat alias: v1 called this KernelDigest.
KernelDigest = UnitDigest


def build_digest(
    kernel: NormalizedKernel,
    format: str,
    requested: Sequence[str],
) -> UnitDigest:
    """Filter a NormalizedUnit down to the requested standard terms.

    Honesty rule: a requested term whose value is ``None`` (or that the
    reader did not map at all) becomes the NOT_AVAILABLE sentinel.
    """
    metrics: dict[str, float | str] = {}
    for term in requested:
        value = kernel.duration_us if term == "duration_us" else kernel.metric(term)
        metrics[term] = value if value is not None else NOT_AVAILABLE
    return UnitDigest(
        kernel=kernel.name,
        index=kernel.index,
        format=format,
        raw_ref=kernel.raw_ref,
        metrics=metrics,
        domain=getattr(kernel, "domain", "gpu_kernel"),
    )


def build_summary(
    units: Sequence[NormalizedUnit],
    format: str,
    requested: Sequence[str],
    top_n: int,
) -> dict:
    """The N hottest units with their core metrics — one call, one payload.

    Replaces the list_kernels + N x get_metrics round trips an agent spends to
    orient itself in an unfamiliar report. Hotness falls back honestly by domain:
    ``duration_us`` (GPU kernels/passes) -> ``self_pct`` (CPU sampler symbols) ->
    file order (e.g. codegen units, which have no runtime cost at all).
    """
    def _by_duration(u: NormalizedUnit) -> float:
        return u.duration_us if u.duration_us is not None else float("-inf")

    def _by_self_pct(u: NormalizedUnit) -> float:
        v = u.metric("self_pct")
        return v if v is not None else float("-inf")

    if any(u.duration_us is not None for u in units):
        sorted_by = "duration_us"
        ordered = sorted(units, key=_by_duration, reverse=True)
    elif any(u.metric("self_pct") is not None for u in units):
        sorted_by = "self_pct"
        ordered = sorted(units, key=_by_self_pct, reverse=True)
    else:
        sorted_by = "file_order"
        ordered = list(units)

    top = ordered[: max(top_n, 0)]
    # ``total_units`` counts UNITS, and a report can hold many units per name
    # (repeated kernel launches, repeated build edges). Top-N can therefore drop
    # a whole name — a variant that lost a near-tie disappears entirely — and a
    # payload that looks complete would imply the report holds fewer distinct
    # units than it does. The counts below make that visible.
    distinct_names = {u.name for u in units}
    omitted_names = distinct_names - {u.name for u in top}
    summary: dict = {
        "format": format,
        "domain": units[0].domain if units else None,
        "raw_ref": units[0].raw_ref if units else None,
        "total_units": len(units),
        "distinct_names_total": len(distinct_names),
        "returned": len(top),
        "sorted_by": sorted_by,
        "units": [
            {
                "name": u.name,
                "index": u.index,
                "domain": u.domain,  # mixed-domain backends (chrome_trace) need this per unit
                "duration_us": u.duration_us,
                "metrics": build_digest(u, format, requested).metrics,
            }
            for u in top
        ],
    }
    if omitted_names:  # additive: absent when top-N shows every name
        summary["distinct_names_omitted"] = len(omitted_names)
    # Coverage is only meaningful over measured durations (never fabricate);
    # an all-zero total makes the ratio undefined, so the key is simply omitted
    # rather than overloading the NOT_AVAILABLE (= "not measured") sentinel.
    #
    # fsum, not sum: it is exactly rounded, so the digest of one report is
    # byte-identical on every interpreter. The builtin sum() changed its float
    # accumulation in CPython 3.12 (Neumaier), which made this the only value in
    # the whole surface that differed across supported Pythons — measured, not
    # theoretical. Stability matters here because it means a diff between two
    # digests always says the REPORT changed, never the runtime.
    durations = [u.duration_us for u in units if u.duration_us is not None]
    if durations and sorted_by == "duration_us":
        total = math.fsum(durations)
        summary["total_duration_us"] = total
        if total > 0:
            covered = math.fsum(u.duration_us for u in top if u.duration_us is not None)
            summary["coverage_pct_of_total_duration"] = covered / total * 100.0
    return summary
