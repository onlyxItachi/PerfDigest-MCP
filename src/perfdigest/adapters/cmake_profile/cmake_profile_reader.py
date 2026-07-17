"""Read cmake ``--profiling-format=google-trace`` JSON -> NormalizedUnit.

The CONFIGURE layer of BuildDigest: ninja_log says which build EDGES are slow,
clang -ftime-trace says which compiler PHASES are slow — this says where the
``cmake -S ... -B ...`` configure step itself spends time (find_package probes,
try_compile, include chains, user macros/functions).

cmake >= 3.18 emits Chrome-trace JSON, so this reader REUSES chrome_trace's
grouping mechanics (``trace_reader._grouped``) exactly like ``clang_time_trace``
does. One real-report lesson made that reuse conditional: cmake does NOT emit
complete ``ph=='X'`` events — a real cmake 4.3.2 capture is a bare JSON array
of ``ph=='B'``/``'E'`` BEGIN/END PAIRS (4523 of each in the verification run;
zero ``X`` events), where the ``E`` side carries only ``ph/pid/tid/ts``. Naive
reuse would digest every cmake trace to zero units. Hence the
``fold_be_pairs=True`` hook (added to ``_grouped`` for this backend, same
surgical-hook pattern as ``tag_override`` was for clang_time_trace): B/E pairs
are stack-folded into complete events first, then everything downstream is
shared.

Grouping — decided from the real trace, not assumed: every B event carries
``cat: "script"`` and ``name`` = the cmake command/macro/function that ran
(``include``, ``find_package``, ``add_executable``, a user ``function()`` name,
plus cmake's own whole-step ``configure``/``generate`` frames). The per-call
detail lives in ``args`` (``functionArgs``, ``location``). Aggregating by
(cat, name) — the sibling backends' rule — turned 4523 spans into 83 units on
the verification capture, with the hot ones directly actionable
(``try_compile`` 99ms, ``find_package`` 55ms of a 223ms cold configure).
Per-call-site grouping (by ``args.location``) was rejected: it multiplies unit
count by ~10 and pushes host paths into unit NAMES; the location detail stays
reachable per-unit via ``expand`` (``arg:location``).

Every unit is domain ``build_phase`` — a configure frame is a build phase; the
real data gave no reason for a new domain.

Honesty notes:
  * Frames NEST (``configure`` contains everything; ``find_package`` contains
    the ``include``/``try_compile`` it triggers), so totals overlap across
    units — not additive, same caveat family as every Chrome-trace sibling.
  * A leftover ``B`` in a truncated trace has no measurable duration and is
    dropped by the fold — never a fabricated ``dur`` (see
    ``_fold_begin_end_pairs``).
"""

from __future__ import annotations

from typing import Any

from perfdigest.adapters.chrome_trace.trace_reader import _grouped
from perfdigest.core.metrics import DOMAIN_BUILD_PHASE, NormalizedUnit


def load_units(report_path: str) -> list[NormalizedUnit]:
    units: list[NormalizedUnit] = []
    for index, rec in enumerate(_grouped(report_path, fold_be_pairs=True)):
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
                domain=DOMAIN_BUILD_PHASE,
            )
        )
    return units


def raw_metrics(report_path: str, kernel_index: int, name_filter: str) -> dict[str, Any]:
    """Aggregate stats + the first event's args, for ``expand``.

    ``_grouped`` is called with the SAME ``fold_be_pairs`` as ``load_units`` so
    indices agree byte-for-byte between the two entry points (the invariant the
    sibling backends keep). ``arg:location``/``arg:functionArgs`` carry the
    call-site detail the grouped unit name intentionally leaves out.
    """
    records = _grouped(report_path, fold_be_pairs=True)
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
