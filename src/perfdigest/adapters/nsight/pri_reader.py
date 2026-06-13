"""PRIMARY reader: native .ncu-rep -> NormalizedKernel via NVIDIA's PRI.

``ncu_report`` (NVIDIA Python Report Interface) reads the binary report
structurally — we never guess byte layout. It is imported LAZILY inside the
functions that need it so CSV-only code paths never load the C-backed module.

Honesty rule enforced here: a metric the report does not contain stays
``None`` in NormalizedKernel.metrics — never 0.0.
"""

from __future__ import annotations

import os
from typing import Any

from perfdigest.adapters.nsight.mapping import CONVERTERS, METRIC_MAP
from perfdigest.core.metrics import NormalizedKernel

# (path, mtime) -> loaded report context; the stdio server is long-lived and
# reports are immutable on disk, so a tiny cache avoids re-parsing per call.
_REPORT_CACHE: dict[tuple[str, float], Any] = {}


def _load_report(report_path: str):
    import ncu_report  # lazy: CUDA-only dependency

    key = (report_path, os.path.getmtime(report_path))
    if key not in _REPORT_CACHE:
        _REPORT_CACHE.clear()  # keep at most one report resident
        _REPORT_CACHE[key] = ncu_report.load_report(report_path)
    return _REPORT_CACHE[key]


def _iter_actions(report):
    for r in range(report.num_ranges()):
        rng = report.range_by_idx(r)
        for a in range(rng.num_actions()):
            yield rng.action_by_idx(a)


def _action_name(action) -> str:
    import ncu_report  # lazy

    try:
        return action.name(ncu_report.IAction.NameBase_DEMANGLED)
    except Exception:
        return action.name()


def _metric_value(action, vendor_name: str) -> tuple[float | None, str | None]:
    """(numeric value, unit) for a vendor metric; (None, None) if absent."""
    metric = action.metric_by_name(vendor_name)
    if metric is None:
        return None, None
    try:
        value = metric.value()
    except Exception:
        return None, None
    if value is None or isinstance(value, str):
        return None, None
    unit = None
    try:
        unit = metric.unit()
    except Exception:
        pass
    return float(value), unit


def load_kernels(report_path: str) -> list[NormalizedKernel]:
    """Parse every kernel launch in the report into NormalizedKernel."""
    report = _load_report(report_path)
    kernels: list[NormalizedKernel] = []
    for index, action in enumerate(_iter_actions(report)):
        metrics: dict[str, float | None] = {}
        for vendor, term in METRIC_MAP.items():
            value, unit = _metric_value(action, vendor)
            if value is not None and term in CONVERTERS:
                value = CONVERTERS[term](value, unit)
            metrics[term] = value
        kernels.append(
            NormalizedKernel(
                name=_action_name(action),
                index=index,
                duration_us=metrics.get("duration_us"),
                raw_ref=report_path,
                metrics=metrics,
            )
        )
    return kernels


def raw_metrics(report_path: str, kernel_index: int, name_filter: str) -> dict[str, Any]:
    """The expand safety valve: raw vendor metrics straight from the report.

    ``name_filter`` is a case-insensitive substring over vendor metric names;
    pass ``"all"`` for every metric in the export (large — the agent's call).
    """
    report = _load_report(report_path)
    for index, action in enumerate(_iter_actions(report)):
        if index != kernel_index:
            continue
        wanted = None if name_filter.lower() == "all" else name_filter.lower()
        out: dict[str, Any] = {}
        for vendor in action.metric_names():
            if wanted is not None and wanted not in vendor.lower():
                continue
            metric = action.metric_by_name(vendor)
            if metric is None:
                continue
            try:
                out[vendor] = metric.value()
            except Exception:
                continue
        return out
    raise IndexError(f"kernel index {kernel_index} not present in {report_path}")
