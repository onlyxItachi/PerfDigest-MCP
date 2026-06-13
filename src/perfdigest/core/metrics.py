"""NormalizedUnit — the neutral, profiler-agnostic contract.

Every reader (Nsight PRI, Linux ``perf``, ROCm, Metal, future Go/Java adapters)
parses its native format into this one shape. ``core/`` never sees a vendor
metric name; keys in ``metrics`` are standard hardware/runtime terms
(``dram_pct_peak``, ``achieved_occupancy``, ``ipc``, ``llc_miss_rate`` ...).

A "unit" is one profiled unit of work, whatever the backend's domain is:

    * ``gpu_kernel``   — a CUDA / HIP kernel launch (NVIDIA, AMD)
    * ``cpu_function`` — a hot symbol/function from a CPU sampler (``perf``)
    * ``gpu_pass``     — a Metal encoder/compute pass (Apple)

THE absence rule (the single most dangerous failure mode if violated):

    A metric value of ``None`` means "not measured in this export format".
    It NEVER means zero. Readers must NEVER substitute ``0.0`` for a metric
    the export does not contain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

# Recognized unit domains. A backend declares which one its units carry so the
# server can label digests and so vocabulary stays domain-appropriate.
DOMAIN_GPU_KERNEL = "gpu_kernel"
DOMAIN_CPU_FUNCTION = "cpu_function"
DOMAIN_GPU_PASS = "gpu_pass"


@dataclass(frozen=True)
class NormalizedUnit:
    """One profiled unit of work, normalized to standard terms.

    Attributes:
        name: Demangled symbol/kernel/pass name as reported by the profiler.
        index: Index within the report (disambiguates same-name units).
        duration_us: Wall/device duration in microseconds, or ``None`` if the
            export does not carry a duration (never 0.0 as a filler).
        raw_ref: Path back to the raw report this unit was read from, so the
            digest can always point the agent at lazy expansion.
        metrics: standard term -> value. A key mapped to ``None`` (or absent
            entirely) means the metric was not measured in this export.
        domain: which kind of unit this is (see ``DOMAIN_*``); defaults to
            ``gpu_kernel`` so the original NVIDIA path is unchanged.
    """

    name: str
    index: int
    duration_us: float | None
    raw_ref: str
    metrics: Mapping[str, float | None] = field(default_factory=dict)
    domain: str = DOMAIN_GPU_KERNEL

    def metric(self, standard_term: str) -> float | None:
        """Value for a standard term; ``None`` if not measured in this export."""
        return self.metrics.get(standard_term)


# Back-compat alias: the original v1 name. The NVIDIA path and existing tests
# refer to NormalizedKernel; it is exactly a NormalizedUnit (domain gpu_kernel).
NormalizedKernel = NormalizedUnit
