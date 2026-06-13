"""Backend descriptor — the neutral plug each profiler adapter registers.

A ``Backend`` is pure data + callables. It is the *only* thing the server shell
knows about: given a ``format`` string the server resolves a Backend and calls
its reader. ``core/`` stays free of any vendor metric name; the vendor mapping
lives inside each adapter and is exposed here only as ``standard_to_vendor`` for
the ``expand`` hint surface.

Two independent axes (see ``platform/capabilities.py``):

  * **readable** — can we *parse* a report of this format on this host? Limited
    only by whether the reader's dependency imports. NEVER gated by OS/GPU, so a
    Mac can digest an NVIDIA report produced on a CI GPU runner.
  * **capturable** — can we *run* this profiler on this host right now? Gated by
    ``platforms`` + ``probe()``. This is the only platform-restricted axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

from perfdigest.core.metrics import NormalizedUnit

if TYPE_CHECKING:  # avoid a runtime import cycle; only used for typing
    from perfdigest.platform.detect import PlatformInfo


@dataclass(frozen=True)
class CapabilityReport:
    """Result of a backend's capture probe (tier-2 only)."""

    available: bool          # can this backend CAPTURE on this host right now?
    reason: str              # human-readable why / why-not
    tool: str | None = None  # resolved profiler executable, if found
    notes: tuple[str, ...] = ()  # caveats, e.g. "WSL2: HW PMU counters absent"

    def to_dict(self) -> dict:
        return {
            "available": self.available,
            "reason": self.reason,
            "tool": self.tool,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class Backend:
    """One profiler adapter, registered once at import time.

    Attributes:
        name: stable identifier ("nsight", "linux_perf", "rocm", "metal").
        formats: ``format`` values this backend reads (e.g. {"ncu-rep"}).
        suffixes: report file suffixes, used by report discovery.
        domain: the unit domain its reader emits (see ``core.metrics``).
        platforms: ``sys.platform`` values where *capture* is possible.
        standard_to_vendor: standard term -> vendor counter (for expand hints).
        default_core_set: terms returned when ``get_metrics(metrics=None)``.
        usage_prompt: backend-specific convention text composed into the prompt.
        load_units: report path -> list[NormalizedUnit].
        raw_metrics: (path, unit_index, name_filter) -> raw vendor metrics dict.
        probe: capture-availability check (tool present, platform ok).
        reader_available: can the reader's dependency import here? (tier-1 gate).
        capture_command: optional advisor — build the correct, platform-aware
            profile invocation string for a target.
    """

    name: str
    formats: frozenset[str]
    suffixes: tuple[str, ...]
    domain: str
    platforms: frozenset[str]
    standard_to_vendor: dict[str, str]
    default_core_set: tuple[str, ...]
    usage_prompt: str
    load_units: Callable[[str], list[NormalizedUnit]]
    raw_metrics: Callable[[str, int, str], dict]
    probe: Callable[[], CapabilityReport]
    reader_available: Callable[[], bool] = lambda: True
    capture_command: Callable[[str, "PlatformInfo"], str] | None = None
    known_terms: tuple[str, ...] = field(default=())

    def __post_init__(self):
        # known_terms defaults to the mapping's standard terms + duration.
        if not self.known_terms:
            object.__setattr__(
                self,
                "known_terms",
                tuple(sorted({*self.standard_to_vendor, "duration_us"})),
            )
