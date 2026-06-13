"""Metal trace export (JSON) -> NormalizedUnit (gpu_pass). Pure Python, no deps.

Raw `.trace` bundles are an opaque Instruments format, so the capture step exports
a small JSON intermediate that this reader consumes (see the backend's usage text):

    {
      "device": "Apple M5",
      "passes": [
        {"name": "compute:matmul", "duration_us": 1234.5,
         "metrics": {"ALU Utilization": 78.2, "L2 Hit Rate": 91.0}}
      ]
    }

Parsing the committed JSON is what makes the Metal path testable on a macOS CI
runner without a real GPU capture. Absent counter -> ``None`` (never 0.0).
"""

from __future__ import annotations

import json
from typing import Any

from perfdigest.adapters.metal.mapping import METRIC_MAP
from perfdigest.core.metrics import DOMAIN_GPU_PASS, NormalizedUnit


def _load(report_path: str) -> dict:
    with open(report_path, "r", encoding="utf-8", errors="ignore") as fh:
        return json.load(fh)


def _passes(doc: dict) -> list[dict]:
    if isinstance(doc, dict):
        return doc.get("passes") or doc.get("encoders") or []
    if isinstance(doc, list):
        return doc
    return []


def load_units(report_path: str) -> list[NormalizedUnit]:
    doc = _load(report_path)
    units: list[NormalizedUnit] = []
    for index, p in enumerate(_passes(doc)):
        vendor = p.get("metrics", {}) or {}
        metrics: dict[str, float | None] = {}
        for apple_name, value in vendor.items():
            term = METRIC_MAP.get(apple_name)
            if term is None:
                continue
            try:
                metrics[term] = float(value) if value is not None else None
            except (TypeError, ValueError):
                metrics[term] = None
        duration = p.get("duration_us")
        metrics.setdefault("duration_us", float(duration) if duration is not None else None)
        units.append(
            NormalizedUnit(
                name=p.get("name", f"pass_{index}"),
                index=index,
                duration_us=metrics.get("duration_us"),
                raw_ref=report_path,
                metrics=metrics,
                domain=DOMAIN_GPU_PASS,
            )
        )
    return units


def raw_metrics(report_path: str, kernel_index: int, name_filter: str) -> dict[str, Any]:
    passes = _passes(_load(report_path))
    if kernel_index >= len(passes):
        raise IndexError(f"pass index {kernel_index} not present in {report_path}")
    vendor = passes[kernel_index].get("metrics", {}) or {}
    wanted = None if name_filter.lower() == "all" else name_filter.lower()
    return {
        k: v for k, v in vendor.items() if wanted is None or wanted in k.lower()
    }
