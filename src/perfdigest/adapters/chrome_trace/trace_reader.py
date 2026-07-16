"""Read Chrome-trace JSON (torch/Kineto & friends) -> NormalizedUnit.

The FRAMEWORK layer. ``torch.profiler`` (Kineto) exports the Chrome Trace
Format — and so do JAX, clang ``-ftime-trace``, Bazel and others — so this one
pure-Python reader digests them all. We bind to the exported ARTIFACT, never to
the framework's profiler API: the format has stayed stable across torch
releases while the internals churned (fragility stays on the other side of the
file boundary).

Two top-level shapes, both handled:

  * dict with a ``traceEvents`` list (Kineto, ``-ftime-trace``)
  * bare JSON array of events (legacy exporters)

Only complete events (``"ph": "X"``) with a numeric ``dur`` are aggregated —
one unit per (category, name), e.g. 4 calls of ``aten::mm`` become one
``framework_op`` unit with ``calls=4``. Device-side ``"cat": "kernel"`` events
become ``gpu_kernel`` units, so a torch trace digests into ops AND the CUDA
kernels they launched, under one report. A name appearing in several
categories gets a ``[cat]`` tag to stay unique and name-addressable.

Honesty notes:
  * ``total_time_us`` sums nested spans as the exporter reported them
    (``aten::matmul`` CONTAINS ``aten::mm``), so totals across units overlap —
    that is how framework profilers report hierarchies, and summarize coverage
    over this backend is indicative, not additive.
  * An event without ``dur`` cannot be aggregated and is skipped; a metric this
    export does not carry stays ``None`` — never 0.0.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from perfdigest.core.metrics import (
    DOMAIN_FRAMEWORK_OP,
    DOMAIN_GPU_KERNEL,
    NormalizedUnit,
)

# Device-side categories Kineto emits (verified against a real capture with
# H2D/D2H transfers); everything else is host/framework side.
_KERNEL_CATS = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})

# Profiler bookkeeping, not workload: 'Trace' is the whole capture window,
# 'overhead' is the profiler's own cost. They are real data (kept, digestible)
# but always name-tagged so every payload carries the discount signal.
_BOOKKEEPING_CATS = frozenset({"Trace", "overhead"})

# Per-call spans kept per unit for expand — bounded so a million-event trace
# cannot balloon the raw payload.
_MAX_RAW_DURS = 64


def _dur(e: dict) -> float | None:
    d = e.get("dur")
    # bool is an int subclass; a broken emitter's "dur": true must not become 1.0us
    if isinstance(d, bool) or not isinstance(d, (int, float)):
        return None
    return float(d)


def _complete_events(report_path: str) -> list[dict]:
    with open(report_path, "r", encoding="utf-8", errors="ignore") as fh:
        try:
            raw = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{report_path} is not valid JSON ({exc.msg} at line {exc.lineno}) — "
                "is this really a Chrome trace? A perf-stat export is JSON-lines "
                "(format perf-stat-json), not a trace."
            ) from None
    if isinstance(raw, dict):
        if not isinstance(raw.get("traceEvents"), list):
            raise ValueError(
                f"{report_path} is a JSON object without a 'traceEvents' list — "
                "not a Chrome trace. Check the file and the format argument."
            )
        events = raw["traceEvents"]
    elif isinstance(raw, list):
        events = raw
    else:
        raise ValueError(
            f"{report_path} top-level JSON is {type(raw).__name__}, expected a "
            "trace object or event array — not a Chrome trace."
        )
    return [
        e
        for e in events
        if isinstance(e, dict)
        and e.get("ph") == "X"
        and _dur(e) is not None
        # explicit None/empty check: an emitter's numeric name 0 is still a name
        and e.get("name") is not None
        and str(e["name"]).strip() != ""
    ]


def _grouped(
    report_path: str,
    *,
    tag_override: "Callable[[dict[str, Any]], str | None] | None" = None,
) -> list[dict[str, Any]]:
    """-> one record per (cat, name), ordered by first appearance (ts).

    ``tag_override(rec)``, if given, is consulted per group AHEAD of the
    default cat-collision / bookkeeping-cat rule below; a truthy return forces
    ``rec['unit_name']`` to ``f"{name} [{tag_override(rec)}]"`` regardless of
    cat. This is the surgical hook ``clang_time_trace`` reuses to tag
    ``-ftime-trace``'s ``Total <Phase>`` aggregates (a name-prefix rule, not a
    cat collision — those events carry no ``cat`` at all). Default ``None``
    leaves chrome_trace's own behavior byte-for-byte unchanged.
    """
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for e in _complete_events(report_path):
        cat = str(e.get("cat", ""))
        key = (cat, str(e["name"]))
        dur = _dur(e)
        assert dur is not None  # filtered in _complete_events
        rec = groups.get(key)
        if rec is None:
            groups[key] = {
                "cat": cat,
                "name": str(e["name"]),
                "calls": 1,
                "total": dur,
                "min": dur,
                "max": dur,
                "durs": [dur],
                "first_ts": float(e.get("ts", 0.0)),
                "first_args": e.get("args") if isinstance(e.get("args"), dict) else {},
            }
        else:
            rec["calls"] += 1
            rec["total"] += dur
            rec["min"] = min(rec["min"], dur)
            rec["max"] = max(rec["max"], dur)
            if len(rec["durs"]) < _MAX_RAW_DURS:
                rec["durs"].append(dur)

    ordered = sorted(groups.values(), key=lambda r: r["first_ts"])

    # A name used in >1 category gets tagged so every unit stays unique;
    # bookkeeping cats are ALWAYS tagged so the discount signal travels in
    # every payload ('PyTorch Profiler (0) [Trace]' cannot pass as workload).
    cats_per_name: dict[str, set[str]] = {}
    for rec in ordered:
        cats_per_name.setdefault(rec["name"], set()).add(rec["cat"])
    for rec in ordered:
        forced = tag_override(rec) if tag_override is not None else None
        collide = len(cats_per_name[rec["name"]]) > 1 and rec["cat"]
        if forced:
            rec["unit_name"] = f"{rec['name']} [{forced}]"
        elif collide or rec["cat"] in _BOOKKEEPING_CATS:
            rec["unit_name"] = f"{rec['name']} [{rec['cat']}]"
        else:
            rec["unit_name"] = rec["name"]
    return ordered


def load_units(report_path: str) -> list[NormalizedUnit]:
    units: list[NormalizedUnit] = []
    for index, rec in enumerate(_grouped(report_path)):
        metrics: dict[str, float | None] = {
            "calls": float(rec["calls"]),
            "total_time_us": rec["total"],
            "avg_time_us": rec["total"] / rec["calls"],
            "max_time_us": rec["max"],
        }
        units.append(
            NormalizedUnit(
                name=rec["unit_name"],
                index=index,
                duration_us=rec["total"],  # aggregate time: the ranking signal
                raw_ref=report_path,
                metrics=metrics,
                domain=DOMAIN_GPU_KERNEL if rec["cat"] in _KERNEL_CATS else DOMAIN_FRAMEWORK_OP,
            )
        )
    return units


def raw_metrics(report_path: str, kernel_index: int, name_filter: str) -> dict[str, Any]:
    """Aggregate stats + the first event's args, for ``expand``."""
    records = _grouped(report_path)
    if kernel_index >= len(records):
        raise IndexError(f"unit index {kernel_index} not present in {report_path}")
    rec = records[kernel_index]
    out: dict[str, Any] = {
        "cat": rec["cat"],
        "calls": rec["calls"],
        "total_time_us": rec["total"],
        "avg_time_us": rec["total"] / rec["calls"],
        "min_time_us": rec["min"],
        "max_time_us": rec["max"],
        # the safety valve reaches per-call spans too (bounded, file order)
        "durs_us": list(rec["durs"]),
    }
    for k, v in rec["first_args"].items():
        if isinstance(v, (int, float, str, bool)):
            out[f"arg:{k}"] = v
    wanted = None if name_filter.lower() == "all" else name_filter.lower()
    if wanted is not None:
        out = {k: v for k, v in out.items() if wanted in k.lower()}
    return out
