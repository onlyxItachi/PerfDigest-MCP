"""Read Chrome-trace JSON (torch/Kineto & friends) -> NormalizedUnit.

The FRAMEWORK layer. ``torch.profiler`` (Kineto) exports the Chrome Trace
Format — and so do JAX, Bazel and others — so this one pure-Python reader
digests them all. We bind to the exported ARTIFACT, never to the framework's
profiler API: the format has stayed stable across torch releases while the
internals churned (fragility stays on the other side of the file boundary).
Compiler/build traces that share the format have DEDICATED backends built on
these same helpers — clang ``-ftime-trace`` -> ``clang_time_trace`` (adds
``[total]`` aggregate tagging), cmake profiling -> ``cmake_profile`` (adds
B/E pair folding) — route those formats there, not here.

Two top-level shapes, both handled:

  * dict with a ``traceEvents`` list (Kineto and friends)
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

# torch.compile's own dispatch machinery is the SAME kind of envelope — each of
# these spans exists to wrap the compiled work and therefore contains its time —
# but the cat rule above cannot see them: verified against a real compiled
# training capture (torch 2.12), every one arrives with cat 'cpu_op', exactly
# like a genuine `aten::` op. So they are matched by name instead, in the same
# spirit as clang's `Total ` rule. Untagged, they outrank real compute in a
# top-N summary and an agent double-counts the work they merely wrap.
_DISPATCH_NAME_PREFIXES = (
    "Torch-Compiled Region",    # Dynamo's compiled-region envelope
    "TorchDynamo ",             # e.g. 'TorchDynamo Cache Lookup'
    "Pregraph bytecode",        # Dynamo, pre-graph interpretation
    "AOTDispatcher ",           # e.g. 'AOTDispatcher Runtime Wrapper Prologue'
    "CompiledFunction",         # AOTAutograd node ('CompiledFunctionBackward')
    "## Call CompiledFxGraph",  # Inductor's graph-call marker
)
_DISPATCH_TAG = "dispatch"


def _is_compile_dispatch(name: str) -> bool:
    """True for a torch.compile dispatch/envelope span (see above).

    ``CompiledFunction`` is matched anywhere in the name, not just at the start:
    the autograd engine wraps the compiled backward in an envelope of its own
    ('autograd::engine::evaluate_function: CompiledFunctionBackward'), and a
    prefix-only rule would leave that one — often a top-3 span — untagged.
    Eager autograd envelopes ('...: MseLossBackward0') are deliberately NOT
    tagged here: they are the engine's normal structure rather than compile
    machinery, and their overlap is documented in the usage prompt instead.
    """
    return name.startswith(_DISPATCH_NAME_PREFIXES) or "CompiledFunction" in name

# Per-call spans kept per unit for expand — bounded so a million-event trace
# cannot balloon the raw payload.
_MAX_RAW_DURS = 64


def _dur(e: dict) -> float | None:
    d = e.get("dur")
    # bool is an int subclass; a broken emitter's "dur": true must not become 1.0us
    if isinstance(d, bool) or not isinstance(d, (int, float)):
        return None
    return float(d)


def _fold_begin_end_pairs(events: list) -> list:
    """Fold ``ph=='B'``/``'E'`` pairs into synthetic complete events.

    Some Chrome-trace emitters (cmake ``--profiling-format=google-trace``) use
    begin/end pairs instead of complete ``X`` events. Per the format, an ``E``
    closes the MOST RECENT open ``B`` on the same (pid, tid) — a stack. The
    folded event takes name/cat/args/ts from the B side (cmake's E events carry
    only ph/pid/tid/ts, verified on a real capture) and ``dur`` = E.ts - B.ts.

    Honesty: a leftover B (truncated trace) has no measurable duration and is
    dropped — never a fabricated ``dur``; a stray E has nothing to close and is
    likewise dropped. Non-B/E events (including real X events) pass through
    untouched.
    """
    folded: list = []
    stacks: dict[tuple, list[dict]] = {}
    for e in events:
        if not isinstance(e, dict):
            folded.append(e)
            continue
        ph = e.get("ph")
        if ph == "B":
            stacks.setdefault((e.get("pid"), e.get("tid")), []).append(e)
        elif ph == "E":
            stack = stacks.get((e.get("pid"), e.get("tid")))
            if stack:
                b = stack.pop()
                try:
                    dur = float(e["ts"]) - float(b["ts"])
                except (KeyError, TypeError, ValueError):
                    continue  # unpaired timing info: skip, never fabricate
                if dur < 0:
                    # E before its own B on one thread is emitter/clock
                    # corruption — a negative elapsed is physically impossible,
                    # so drop the pair like an unparseable one.
                    continue
                out = dict(b)
                out["ph"] = "X"
                out["dur"] = dur
                folded.append(out)
        else:
            folded.append(e)
    return folded


def _complete_events(report_path: str, *, fold_be_pairs: bool = False) -> list[dict]:
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
    if fold_be_pairs:
        events = _fold_begin_end_pairs(events)
    complete = [
        e
        for e in events
        if isinstance(e, dict)
        and e.get("ph") == "X"
        and _dur(e) is not None
        # explicit None/empty check: an emitter's numeric name 0 is still a name
        and e.get("name") is not None
        and str(e["name"]).strip() != ""
    ]
    if not complete and events:
        # A valid trace with zero digestible events must not become a silent
        # empty report ("the run did nothing" is a conclusion, not a default).
        phases = {e.get("ph") for e in events if isinstance(e, dict)}
        if not fold_be_pairs and ("B" in phases or "E" in phases):
            raise ValueError(
                f"{report_path} contains only begin/end (ph B/E) pair events and "
                "no complete (ph 'X') events — this looks like a cmake "
                "--profiling-format=google-trace capture; digest it with format "
                "cmake-profile, which folds the pairs."
            )
        raise ValueError(
            f"{report_path} is a valid trace container but has no complete "
            "(ph=='X') events with a numeric dur — nothing to digest. Check the "
            "emitter and the format argument."
        )
    return complete


def _grouped(
    report_path: str,
    *,
    tag_override: "Callable[[dict[str, Any]], str | None] | None" = None,
    fold_be_pairs: bool = False,
) -> list[dict[str, Any]]:
    """-> one record per (cat, name), ordered by first appearance (ts).

    Tagging precedence: an explicit ``tag_override`` wins, then the name-based
    ``[dispatch]`` rule for torch.compile envelopes, then the cat-based
    collision / bookkeeping rule. A forced tag never suppresses a cat-collision
    tag — both are appended, so unit names stay unique either way.

    ``tag_override(rec)``, if given, is consulted per group AHEAD of the
    default cat-collision / bookkeeping-cat rule below; a truthy return forces
    ``rec['unit_name']`` to ``f"{name} [{tag_override(rec)}]"`` regardless of
    cat. This is the surgical hook ``clang_time_trace`` reuses to tag
    ``-ftime-trace``'s ``Total <Phase>`` aggregates (a name-prefix rule, not a
    cat collision — those events carry no ``cat`` at all). Default ``None``
    leaves chrome_trace's own behavior byte-for-byte unchanged.

    ``fold_be_pairs=True`` is the same kind of surgical hook for
    ``cmake_profile``: cmake's profiler emits ``B``/``E`` pairs instead of
    ``X`` events, so they are folded first (see ``_fold_begin_end_pairs``).
    Default ``False`` — chrome_trace's own inputs are untouched.
    """
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for e in _complete_events(report_path, fold_be_pairs=fold_be_pairs):
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
        if forced is None and _is_compile_dispatch(rec["name"]):
            forced = _DISPATCH_TAG
        collide = len(cats_per_name[rec["name"]]) > 1 and rec["cat"]
        if forced:
            # A forced tag must not defeat the uniqueness invariant: if the
            # name ALSO collides across categories, keep the cat tag too.
            rec["unit_name"] = (
                f"{rec['name']} [{forced}] [{rec['cat']}]"
                if collide
                else f"{rec['name']} [{forced}]"
            )
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
