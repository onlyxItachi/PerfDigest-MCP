"""KernelDigest — the JSON response schema and the absence convention.

``get_metrics`` must surface missing data transparently: a requested metric
that the export does not contain is returned as the literal string
``"not_available_in_this_export"`` — never silently dropped, never ``0.0``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from perfdigest.core.metrics import NormalizedKernel

NOT_AVAILABLE = "not_available_in_this_export"


@dataclass(frozen=True)
class KernelDigest:
    """The compact, structured answer returned by ``get_metrics``."""

    kernel: str
    index: int
    format: str
    raw_ref: str
    metrics: dict[str, float | str]  # float, or the NOT_AVAILABLE sentinel

    def to_dict(self) -> dict:
        return {
            "kernel": self.kernel,
            "index": self.index,
            "format": self.format,
            "raw_ref": self.raw_ref,
            "metrics": self.metrics,
        }


def build_digest(
    kernel: NormalizedKernel,
    format: str,
    requested: Sequence[str],
) -> KernelDigest:
    """Filter a NormalizedKernel down to the requested standard terms.

    Honesty rule: a requested term whose value is ``None`` (or that the
    reader did not map at all) becomes the NOT_AVAILABLE sentinel.
    """
    metrics: dict[str, float | str] = {}
    for term in requested:
        value = kernel.duration_us if term == "duration_us" else kernel.metric(term)
        metrics[term] = value if value is not None else NOT_AVAILABLE
    return KernelDigest(
        kernel=kernel.name,
        index=kernel.index,
        format=format,
        raw_ref=kernel.raw_ref,
        metrics=metrics,
    )
