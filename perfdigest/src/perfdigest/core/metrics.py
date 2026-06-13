"""NormalizedKernel — the neutral, profiler-agnostic contract.

Every reader (PRI, CSV, future perf/cargo adapters) parses its native format
into this shape. core/ never sees vendor metric names; keys in ``metrics`` are
standard hardware terms (``dram_pct_peak``, ``achieved_occupancy``, ...).

THE absence rule (the single most dangerous failure mode if violated):

    A metric value of ``None`` means "not measured in this export format".
    It NEVER means zero. Readers must NEVER substitute ``0.0`` for a metric
    the export does not contain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class NormalizedKernel:
    """One profiled kernel launch, normalized to standard terms.

    Attributes:
        name: Demangled kernel function name as reported by the profiler.
        index: Launch index within the report (disambiguates same-name launches).
        duration_us: Kernel duration in microseconds, or ``None`` if the
            export does not carry a duration (never 0.0 as a filler).
        metrics: standard term -> value. A key mapped to ``None`` (or absent
            entirely) means the metric was not measured in this export.
        raw_ref: Path back to the raw report this kernel was read from, so
            the digest can always point the agent at lazy expansion.
    """

    name: str
    index: int
    duration_us: float | None
    raw_ref: str
    metrics: Mapping[str, float | None] = field(default_factory=dict)

    def metric(self, standard_term: str) -> float | None:
        """Value for a standard term; ``None`` if not measured in this export."""
        return self.metrics.get(standard_term)
