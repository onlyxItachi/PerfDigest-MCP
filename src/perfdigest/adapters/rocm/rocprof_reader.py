"""rocprof CSV (wide) -> NormalizedUnit (gpu_kernel). Pure Python, no deps.

Each CSV row is one kernel dispatch; counter values are columns. We pivot each
row's known columns into standard terms; an absent column stays ``None`` (never
0.0). Works on any host with no AMD GPU present — tier-1 digesting is universal.
"""

from __future__ import annotations

import csv
from typing import Any

from perfdigest.adapters.rocm.mapping import CONVERTERS, METRIC_MAP
from perfdigest.core.metrics import DOMAIN_GPU_KERNEL, NormalizedUnit

_NAME_COLS = ("KernelName", "Kernel_Name", "Name")


def _to_float(raw: str | None) -> float | None:
    if raw in (None, ""):
        return None
    try:
        return float(str(raw).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _rows(report_path: str) -> list[dict]:
    with open(report_path, newline="", encoding="utf-8", errors="ignore") as fh:
        return list(csv.DictReader(fh))


def _name(row: dict, index: int) -> str:
    for c in _NAME_COLS:
        if row.get(c):
            return row[c]
    return f"dispatch_{index}"


def load_units(report_path: str) -> list[NormalizedUnit]:
    units: list[NormalizedUnit] = []
    for index, row in enumerate(_rows(report_path)):
        metrics: dict[str, float | None] = {}
        for col, term in METRIC_MAP.items():
            value = _to_float(row.get(col))
            if value is not None and term in CONVERTERS:
                value = CONVERTERS[term](value)
            # keep first non-None mapping for terms with multiple source columns
            if metrics.get(term) is None:
                metrics[term] = value
        units.append(
            NormalizedUnit(
                name=_name(row, index),
                index=index,
                duration_us=metrics.get("duration_us"),
                raw_ref=report_path,
                metrics=metrics,
                domain=DOMAIN_GPU_KERNEL,
            )
        )
    return units


def raw_metrics(report_path: str, kernel_index: int, name_filter: str) -> dict[str, Any]:
    rows = _rows(report_path)
    if kernel_index >= len(rows):
        raise IndexError(f"kernel index {kernel_index} not present in {report_path}")
    row = rows[kernel_index]
    wanted = None if name_filter.lower() == "all" else name_filter.lower()
    out: dict[str, Any] = {}
    for col, raw in row.items():
        if wanted is not None and wanted not in col.lower():
            continue
        val = _to_float(raw)
        out[col] = val if val is not None else raw
    return out
