"""FALLBACK reader: ``ncu --csv`` export -> NormalizedUnit (pure Python).

This is the GPU-free, PRI-wheel-free path that keeps tier-1 digesting universal:
an `.ncu-rep` captured on a CI/remote GPU runner can be exported to CSV there and
digested anywhere (a Mac, a CI Windows job). ncu emits long-form CSV — one row
per (kernel launch, metric) — so we group rows by launch ID and pivot.

Honesty rule enforced here too: a metric absent from the export stays ``None``
in NormalizedUnit.metrics — never 0.0.
"""

from __future__ import annotations

import csv
from typing import Any

from perfdigest.adapters.nsight.mapping import CONVERTERS, METRIC_MAP
from perfdigest.core.metrics import DOMAIN_GPU_KERNEL, NormalizedUnit

# Column-header aliases ncu has used across versions.
_ID_COLS = ("ID", "Launch Index", "Kernel ID")
_NAME_COLS = ("Kernel Name", "Demangled Name", "Function Name", "Name")
_METRIC_NAME_COLS = ("Metric Name",)
_METRIC_VALUE_COLS = ("Metric Value",)
_METRIC_UNIT_COLS = ("Metric Unit",)


def _pick(row: dict, cols) -> str | None:
    for c in cols:
        if c in row and row[c] not in (None, ""):
            return row[c]
    return None


def _to_float(raw: str | None) -> float | None:
    if raw is None:
        return None
    s = raw.replace(",", "").strip()  # ncu thousands separators
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _grouped(report_path: str) -> "list[tuple[str, str, dict[str, tuple[float | None, str | None]]]]":
    """-> [(launch_id, kernel_name, {vendor_metric: (value, unit)})] in file order."""
    order: list[str] = []
    names: dict[str, str] = {}
    raw: dict[str, dict[str, tuple[float | None, str | None]]] = {}
    with open(report_path, newline="", encoding="utf-8", errors="ignore") as fh:
        for row in csv.DictReader(fh):
            launch = _pick(row, _ID_COLS) or "0"
            if launch not in raw:
                raw[launch] = {}
                order.append(launch)
                names[launch] = _pick(row, _NAME_COLS) or f"launch_{launch}"
            mname = _pick(row, _METRIC_NAME_COLS)
            if mname is None:
                continue
            value = _to_float(_pick(row, _METRIC_VALUE_COLS))
            unit = _pick(row, _METRIC_UNIT_COLS)
            raw[launch][mname] = (value, unit)
    return [(lid, names[lid], raw[lid]) for lid in order]


def load_units(report_path: str) -> list[NormalizedUnit]:
    units: list[NormalizedUnit] = []
    for index, (_lid, name, vendor_metrics) in enumerate(_grouped(report_path)):
        metrics: dict[str, float | None] = {}
        for vendor, term in METRIC_MAP.items():
            value, unit = vendor_metrics.get(vendor, (None, None))
            if value is not None and term in CONVERTERS:
                value = CONVERTERS[term](value, unit)
            metrics[term] = value
        units.append(
            NormalizedUnit(
                name=name,
                index=index,
                duration_us=metrics.get("duration_us"),
                raw_ref=report_path,
                metrics=metrics,
                domain=DOMAIN_GPU_KERNEL,
            )
        )
    return units


def raw_metrics(report_path: str, kernel_index: int, name_filter: str) -> dict[str, Any]:
    """The expand safety valve over the CSV export's raw vendor metrics."""
    groups = _grouped(report_path)
    if kernel_index >= len(groups):
        raise IndexError(f"kernel index {kernel_index} not present in {report_path}")
    _lid, _name, vendor_metrics = groups[kernel_index]
    wanted = None if name_filter.lower() == "all" else name_filter.lower()
    out: dict[str, Any] = {}
    for vendor, (value, _unit) in vendor_metrics.items():
        if wanted is not None and wanted not in vendor.lower():
            continue
        out[vendor] = value
    return out
