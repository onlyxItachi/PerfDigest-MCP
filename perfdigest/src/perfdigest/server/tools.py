"""The 3 MCP tools — a thin shell over core + adapters, no business logic.

``format`` is MANDATORY in v1: the agent passes what it produced. There is
no auto-detect; a path says where a file is, not what format it is.
"""

from __future__ import annotations

from perfdigest.adapters.nsight import pri_reader
from perfdigest.adapters.nsight.mapping import DEFAULT_CORE_SET, STANDARD_TO_VENDOR
from perfdigest.core.digest import build_digest
from perfdigest.core.metrics import NormalizedKernel
from perfdigest.report_store.discovery import resolve_report
from perfdigest.server.app import mcp

_PRI_FORMATS = {"ncu-rep", "ncurep", "pri"}


def _check_format(format: str) -> str:
    normalized = format.lower().lstrip(".").strip()
    if normalized in _PRI_FORMATS:
        return "ncu-rep"
    raise ValueError(
        f"unsupported format {format!r}: v1 reads native Nsight Compute reports "
        "(format='ncu-rep', produced by `ncu --set full -o report.ncu-rep <app>`). "
        "The CSV fallback reader is not wired up yet."
    )


def _find_kernel(kernels: list[NormalizedKernel], kernel: str) -> NormalizedKernel:
    """Match by exact name, unique substring, or numeric launch index."""
    if kernel.strip().lstrip("#").isdigit():
        index = int(kernel.strip().lstrip("#"))
        for k in kernels:
            if k.index == index:
                return k
    exact = [k for k in kernels if k.name == kernel]
    if len(exact) == 1:
        return exact[0]
    partial = [k for k in kernels if kernel in k.name]
    if len(partial) == 1:
        return partial[0]
    available = ", ".join(f"#{k.index} {k.name}" for k in kernels)
    if not exact and not partial:
        raise ValueError(f"kernel {kernel!r} not found. Available: {available}")
    raise ValueError(
        f"kernel {kernel!r} is ambiguous ({len(exact) or len(partial)} matches). "
        f"Disambiguate with the launch index from list_kernels. Available: {available}"
    )


@mcp.tool()
def list_kernels(report_ref: str, format: str) -> list[dict]:
    """List every kernel launch in a profiler report (name, index, duration_us).

    Call this first to pick the hot kernel by duration. ``report_ref`` may be
    a .ncu-rep path or a directory (newest report inside is used). ``format``
    is what you produced — 'ncu-rep' for `ncu -o` output.
    """
    _check_format(format)
    path = resolve_report(report_ref)
    return [
        {"name": k.name, "index": k.index, "duration_us": k.duration_us}
        for k in pri_reader.load_kernels(str(path))
    ]


@mcp.tool()
def get_metrics(
    report_ref: str,
    format: str,
    kernel: str,
    metrics: list[str] | None = None,
) -> dict:
    """Compact numeric digest for one kernel, in standard hardware terms.

    ``metrics=None`` returns the default core set (duration, compute/dram %
    of peak, occupancy, cache hit rates, launch config, throughput, warp
    stall signal). Pass ``metrics=[...]`` to select specific standard terms.
    A metric the export does not contain is returned as
    'not_available_in_this_export' — that is honest absence, not zero.
    ``kernel`` accepts an exact name, a unique substring, or a launch index.
    """
    norm_format = _check_format(format)
    path = resolve_report(report_ref)
    kernels = pri_reader.load_kernels(str(path))
    target = _find_kernel(kernels, kernel)

    requested = list(metrics) if metrics else list(DEFAULT_CORE_SET)
    digest = build_digest(target, norm_format, requested).to_dict()

    unknown = [
        t for t in requested if t != "duration_us" and t not in STANDARD_TO_VENDOR
    ]
    if unknown:
        digest["unknown_metrics"] = {
            "requested_but_not_in_vocabulary": unknown,
            "known_terms": sorted({*STANDARD_TO_VENDOR, "duration_us"}),
            "hint": "use expand(section=...) for raw vendor counters",
        }
    return digest


@mcp.tool()
def expand(report_ref: str, format: str, kernel: str, section: str) -> dict:
    """Lazy raw access — the safety valve when the digest is not enough.

    Returns raw vendor metrics for one kernel, filtered by a case-insensitive
    substring over vendor metric names (e.g. 'dram', 'stall', 'occupancy').
    ``section='all'`` returns every metric in the export (large). Use this
    instead of ever reading the report file directly.
    """
    _check_format(format)
    path = resolve_report(report_ref)
    kernels = pri_reader.load_kernels(str(path))
    target = _find_kernel(kernels, kernel)
    raw = pri_reader.raw_metrics(str(path), target.index, section)
    return {
        "kernel": target.name,
        "index": target.index,
        "section_filter": section,
        "metric_count": len(raw),
        "raw_ref": str(path),
        "metrics": raw,
    }
