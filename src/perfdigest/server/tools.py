"""The MCP tools — a thin shell over registry + core + platform, no business logic.

Two tiers (see ``server/prompts.py``):
  * Tier 1 — read/digest: ``list_kernels`` / ``get_metrics`` / ``expand`` work for
    ANY registered ``format`` on ANY host. A report's origin is irrelevant.
  * Tier 2 — capture advisory: ``platform_capabilities`` / ``suggest_profile_command``
    are platform-verified and refuse a capture that cannot run here.

``format`` is MANDATORY: the agent passes what it produced; a path says where a
file is, not what format it is. ``get_backend(format)`` does the dispatch — the
server never imports a concrete reader.
"""

from __future__ import annotations

from perfdigest.adapters import registry
from perfdigest.core.backend import Backend
from perfdigest.core.compare import build_comparison
from perfdigest.core.digest import build_digest, build_summary
from perfdigest.core.metrics import NormalizedUnit
from perfdigest.platform import capabilities
from perfdigest.platform.detect import detect
from perfdigest.report_store.cache import cached_units
from perfdigest.report_store.discovery import resolve_report
from perfdigest.server.app import mcp


def _units(backend: Backend, path: str) -> list[NormalizedUnit]:
    """Load units through the parse-once cache (reports are immutable on disk)."""
    return cached_units(backend.name, backend.load_units, path)


def _vocabulary_hint(backend: Backend, requested: list[str]) -> dict | None:
    """Flag requested terms outside the backend's vocabulary (typos vs absence)."""
    unknown = [
        t for t in requested if t != "duration_us" and t not in backend.standard_to_vendor
    ]
    if not unknown:
        return None
    return {
        "requested_but_not_in_vocabulary": unknown,
        "known_terms": list(backend.known_terms),
        "hint": "use expand(section=...) for raw vendor counters",
    }


def _find_unit(units: list[NormalizedUnit], kernel: str) -> NormalizedUnit:
    """Match by exact name, then '#3' explicit index, then plain-numeric index,
    then unique substring.

    Exact-name matching comes FIRST so a unit literally named '3' (arbitrary
    emitters produce such names — chrome traces especially) stays addressable;
    '#3' is the unambiguous way to say index 3 (issue #2). Plain numerics keep
    working as an index fallback for units whose name doesn't shadow them.
    """
    k = kernel.strip()

    exact = [u for u in units if u.name == k]
    if len(exact) == 1:
        return exact[0]

    def _by_index(index: int) -> NormalizedUnit | None:
        for u in units:
            if u.index == index:
                return u
        return None

    if k.startswith("#") and k[1:].isdigit():
        unit = _by_index(int(k[1:]))
        if unit is not None:
            return unit
        available = ", ".join(f"#{u.index} {u.name}" for u in units)
        raise ValueError(f"index {k} not present. Available: {available}")

    if not exact and k.isdigit():
        unit = _by_index(int(k))
        if unit is not None:
            return unit

    partial = [u for u in units if k in u.name]
    if len(partial) == 1:
        return partial[0]
    available = ", ".join(f"#{u.index} {u.name}" for u in units)
    if not exact and not partial:
        raise ValueError(f"unit {kernel!r} not found. Available: {available}")
    raise ValueError(
        f"unit {kernel!r} is ambiguous ({len(exact) or len(partial)} matches). "
        f"Disambiguate with '#<index>' from list_kernels. Available: {available}"
    )


def _resolve(report_ref: str, format: str) -> tuple[Backend, str]:
    backend = registry.get_backend(format)
    path = resolve_report(report_ref, backend.suffixes)
    return backend, str(path)


@mcp.tool()
def list_kernels(report_ref: str, format: str) -> list[dict]:
    """List every profiled unit in a report (name, index, duration_us, domain).

    Works for any backend's ``format`` on any host (tier-1, never platform-gated):
    'ncu-rep'/'ncu-csv' (NVIDIA kernels), 'rocprof-csv' (AMD kernels),
    'perf-stat-json'/'perf-report' (CPU functions), 'metal-trace' (Metal passes).
    Call this first to pick the hottest unit by duration_us.
    """
    backend, path = _resolve(report_ref, format)
    return [
        {"name": u.name, "index": u.index, "duration_us": u.duration_us, "domain": u.domain}
        for u in _units(backend, path)
    ]


@mcp.tool()
def get_metrics(
    report_ref: str,
    format: str,
    kernel: str,
    metrics: list[str] | None = None,
) -> dict:
    """Compact numeric digest for one unit, in standard hardware/runtime terms.

    ``metrics=None`` returns the backend's default core set. A metric the export
    does not contain is returned as 'not_available_in_this_export' — honest
    absence, never zero. ``kernel`` accepts an exact name, a unique substring, or
    an index. The vocabulary is domain-appropriate: GPU terms (dram_pct_peak,
    achieved_occupancy) for kernels/passes, CPU terms (ipc, llc_miss_rate,
    self_pct) for cpu_function units.
    """
    backend, path = _resolve(report_ref, format)
    target = _find_unit(_units(backend, path), kernel)

    # Only None means "give me the defaults"; an explicit [] means an empty
    # metrics object (issue #2 — the two were previously conflated).
    requested = list(metrics) if metrics is not None else list(backend.default_core_set)
    digest = build_digest(target, format, requested).to_dict()

    hint = _vocabulary_hint(backend, requested)
    if hint:
        digest["unknown_metrics"] = hint
    return digest


@mcp.tool()
def expand(report_ref: str, format: str, kernel: str, section: str) -> dict:
    """Lazy raw access — the safety valve when the digest is not enough.

    Returns raw vendor metrics for one unit, filtered by a case-insensitive
    substring over vendor metric names (e.g. 'dram', 'stall', 'cache').
    ``section='all'`` returns everything (large). Use this instead of ever
    reading the report file directly.
    """
    backend, path = _resolve(report_ref, format)
    target = _find_unit(_units(backend, path), kernel)
    raw = backend.raw_metrics(path, target.index, section)
    return {
        "kernel": target.name,
        "index": target.index,
        "domain": target.domain,
        "section_filter": section,
        "metric_count": len(raw),
        "raw_ref": path,
        "metrics": raw,
    }


@mcp.tool()
def summarize_report(
    report_ref: str,
    format: str,
    top_n: int = 5,
    metrics: list[str] | None = None,
) -> dict:
    """The N hottest units + their core metrics in ONE call — start here.

    Replaces the list_kernels + N x get_metrics round trips when orienting in an
    unfamiliar report. Units are ranked by duration_us, falling back to self_pct
    (CPU sampler symbols carry no wall time), then file order. When durations
    exist, ``coverage_pct_of_total_duration`` says how much of the report's total
    time the returned units account for. Same honesty rule as get_metrics:
    'not_available_in_this_export' is NOT MEASURED, never zero.
    """
    backend, path = _resolve(report_ref, format)
    units = _units(backend, path)
    requested = list(metrics) if metrics is not None else list(backend.default_core_set)
    summary = build_summary(units, format, requested, top_n)
    hint = _vocabulary_hint(backend, requested)
    if hint:
        summary["unknown_metrics"] = hint
    return summary


@mcp.tool()
def compare_metrics(
    report_a: str,
    report_b: str,
    format: str,
    kernel: str,
    kernel_b: str | None = None,
    metrics: list[str] | None = None,
) -> dict:
    """Delta digest of one unit across two reports — the measure->edit->measure tool.

    After changing code and re-profiling, call this with A = the before report
    and B = the after report: each metric comes back as {a, b, delta, delta_pct}
    with ``delta = b - a`` (negative duration_us delta = B is faster). ``kernel``
    matches in BOTH reports unless ``kernel_b`` overrides the B-side (renamed or
    restructured kernels). Comparing two DIFFERENT kernels of one report (pass
    the same path twice) is also legitimate. A metric missing on either side
    yields delta = 'not_available_in_this_export' — never a fake 0.0.
    """
    backend = registry.get_backend(format)
    path_a = str(resolve_report(report_a, backend.suffixes))
    path_b = str(resolve_report(report_b, backend.suffixes))
    unit_a = _find_unit(_units(backend, path_a), kernel)
    unit_b = _find_unit(_units(backend, path_b), kernel_b if kernel_b else kernel)
    requested = list(metrics) if metrics is not None else list(backend.default_core_set)
    comparison = build_comparison(unit_a, unit_b, format, requested)
    hint = _vocabulary_hint(backend, requested)
    if hint:
        comparison["unknown_metrics"] = hint
    return comparison


@mcp.tool()
def platform_capabilities() -> dict:
    """What this host can DIGEST (everything) vs CAPTURE (gated) — call before profiling.

    Returns the machine identity plus two lists: ``can_digest`` (universal — any
    backend with an installed reader, so a GPU-less host is still a first-class
    digester of reports captured elsewhere) and ``can_capture_here`` (gated by
    OS + tool presence). Use it to avoid spending context on a capture that cannot
    run here — e.g. CUDA on a Mac is not capturable but is fully digestable.
    """
    return capabilities.capability_summary()


@mcp.tool()
def suggest_profile_command(backend: str, target: str) -> dict:
    """The correct, platform-aware profiler invocation for ``target`` — or a refusal.

    ``backend`` is a registered name ('nsight','rocm','linux_perf','metal'). If
    capturable here, returns the exact command (POSIX/PowerShell/WSL-aware). If
    not (wrong OS / missing tool), refuses and redirects to the tier-1 path:
    capture on a host that supports it, then digest the report here.
    """
    b = registry.get_by_name(backend)
    report = capabilities.capture_report(b)
    if not report.available:
        return {
            "ok": False,
            "reason": report.reason,
            "alternative": (
                f"Capture on a host where {backend} runs (or a CI/remote runner), "
                f"then digest the report here with get_metrics(format=...)."
            ),
        }
    info = detect()
    command = b.capture_command(target, info) if b.capture_command else None
    return {
        "ok": True,
        "backend": b.name,
        "command": command,
        "shell": info.shell,
        "notes": list(report.notes),
        "next": "Then: list_kernels -> get_metrics on the hottest unit.",
    }
