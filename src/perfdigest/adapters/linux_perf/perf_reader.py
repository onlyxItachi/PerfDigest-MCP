"""Read ``perf`` output -> NormalizedUnit (cpu_function). Pure Python, no deps.

Two export shapes, sniffed by content (the agent already passed format=perf-*):

  * ``perf stat -j`` JSON-lines -> ONE program-level unit with derived IPC and
    miss/mispredict rates (the high-signal C++/Rust counters).
  * ``perf report --stdio -n`` table -> one cpu_function unit per hot symbol
    (self overhead %, sample count).

Honesty rule: a counter the export lacks stays ``None`` — never 0.0. A genuine
zero (e.g. zero branch-misses) is preserved as 0.0.
"""

from __future__ import annotations

import json
import re
from typing import Any

from perfdigest.core.metrics import DOMAIN_CPU_FUNCTION, NormalizedUnit

_REPORT_ROW = re.compile(r"^\s*(\d+(?:\.\d+)?)%\s+(\d+)\s+(.*)$")
_SYM_MARKER = re.compile(r"\[[.kgGuU]\]\s+")


def _ratio(num: float | None, den: float | None, scale: float = 100.0) -> float | None:
    if num is None or den in (None, 0):
        return None
    return num / den * scale


def _parse_stat_json(text: str, path: str) -> NormalizedUnit:
    events: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        event = obj.get("event")
        raw = obj.get("counter-value")
        if event is None or raw in (None, "", "<not counted>", "<not supported>"):
            continue
        try:
            events[event] = float(str(raw).replace(",", ""))
        except ValueError:
            continue

    def ev(*names: str) -> float | None:
        for n in names:
            if n in events:
                return events[n]
        return None

    instructions = ev("instructions")
    cycles = ev("cycles", "cpu-cycles")
    metrics: dict[str, float | None] = {
        "wall_time_ms": ev("task-clock"),
        "instructions": instructions,
        "cycles": cycles,
        "ipc": instructions / cycles if instructions is not None and cycles else None,
        "cache_miss_rate": _ratio(ev("cache-misses"), ev("cache-references")),
        "llc_miss_rate": _ratio(ev("LLC-load-misses"), ev("LLC-loads")),
        "branch_mispredict_rate": _ratio(ev("branch-misses"), ev("branches", "branch-instructions")),
    }
    wall = metrics["wall_time_ms"]
    return NormalizedUnit(
        name="<program>",
        index=0,
        duration_us=(wall * 1e3) if wall is not None else None,
        raw_ref=path,
        metrics=metrics,
        domain=DOMAIN_CPU_FUNCTION,
    )


def _parse_report(text: str, path: str) -> list[NormalizedUnit]:
    units: list[NormalizedUnit] = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _REPORT_ROW.match(line)
        if not m:
            continue
        overhead, samples, rest = float(m.group(1)), int(m.group(2)), m.group(3)
        parts = _SYM_MARKER.split(rest, maxsplit=1)
        symbol = parts[-1].strip() if len(parts) > 1 else rest.split()[-1]
        units.append(
            NormalizedUnit(
                name=symbol,
                index=len(units),
                duration_us=None,  # a sampler has no per-symbol wall time
                raw_ref=path,
                metrics={"self_pct": overhead, "samples": float(samples)},
                domain=DOMAIN_CPU_FUNCTION,
            )
        )
    return units


def load_units(report_path: str) -> list[NormalizedUnit]:
    with open(report_path, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read()
    if text.lstrip().startswith("{"):
        return [_parse_stat_json(text, report_path)]
    return _parse_report(text, report_path)


def raw_metrics(report_path: str, kernel_index: int, name_filter: str) -> dict[str, Any]:
    """perf's 'raw' is the parsed counters; expose them filtered for ``expand``."""
    units = load_units(report_path)
    if kernel_index >= len(units):
        raise IndexError(f"unit index {kernel_index} not present in {report_path}")
    wanted = None if name_filter.lower() == "all" else name_filter.lower()
    out: dict[str, Any] = {}
    for term, value in units[kernel_index].metrics.items():
        if value is None:
            continue
        if wanted is not None and wanted not in term.lower():
            continue
        out[term] = value
    return out
