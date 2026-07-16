"""Read clang ``-ftime-trace`` JSON (Chrome Trace Format) -> NormalizedUnit.

``-ftime-trace`` emits the exact Chrome Trace Format ``chrome_trace`` already
digests (torch/Kineto, JAX, Bazel...): a ``traceEvents`` list of ``ph=='X'``
complete spans. So this reader REUSES chrome_trace's grouping mechanics
(``trace_reader._grouped``, which itself calls ``_complete_events``) instead of
re-implementing event parsing — the artifact shape is identical; only the
vocabulary and one honesty nuance differ, both handled here:

  clang additionally emits a ``Total <Phase>`` aggregate event per phase
  (e.g. ``Total Frontend``, ``args.count`` == N, ``dur`` == sum of the N
  per-instance ``Frontend`` spans elsewhere in the same trace — verified
  against a real capture: ``Total Frontend`` dur 187194us == the two
  individual ``Frontend`` spans' 177948us + 9247us). These are bookkeeping
  roll-ups, NOT a second measurement of payload work — naively summing every
  unit's ``total_time_us`` would double-count whatever a ``Total X`` already
  aggregates. They are never dropped (still fully digestible, addressable,
  expandable — bookkeeping is real data) but ALWAYS name-tagged
  ``<Phase> [total]`` so ranking/summarize and any agent-side arithmetic can
  discount them on sight — exactly how chrome_trace always tags its
  ``_BOOKKEEPING_CATS`` (``[Trace]``/``[overhead]``), just keyed off a name
  prefix here instead of a ``cat`` (clang's phase events carry no ``cat`` at
  all, so the cat-collision tagging path never fires on this input).

Every unit here is domain ``build_phase`` (see ``core/metrics.py``): a
compiler phase, not a GPU kernel or a framework op. ``-ftime-trace`` carries no
hardware counters, only nested wall-clock spans (see ``mapping.py``).

Honesty notes (inherited from chrome_trace's mechanics, still true here):
  * spans nest (``ExecuteCompiler`` contains ``Frontend`` contains
    ``ParseClass``/``InstantiateFunction``/...), so totals across payload
    units overlap — coverage is indicative, not additive, same as any
    Chrome-trace hierarchy.
  * clang separately emits per-include-file ``Source`` spans as ``ph``
    ``"b"``/``"e"`` (async begin/end) pairs, not ``ph=='X'`` complete events;
    those fall outside this reader's (and chrome_trace's) event shape and stay
    absent from the digest — only the ``Total Source`` aggregate surfaces.
    Never fabricated, never zero.
  * An event without ``dur`` is skipped by the shared ``_complete_events``
    filter; a metric this export does not carry stays ``None``.
"""

from __future__ import annotations

from typing import Any

from perfdigest.adapters.chrome_trace.trace_reader import _grouped
from perfdigest.core.metrics import DOMAIN_BUILD_PHASE, NormalizedUnit


def _total_tag(rec: dict[str, Any]) -> str | None:
    """``Total <Phase>`` aggregates get tagged ``[total]``; siblings don't."""
    return "total" if str(rec["name"]).startswith("Total ") else None


def load_units(report_path: str) -> list[NormalizedUnit]:
    units: list[NormalizedUnit] = []
    for index, rec in enumerate(_grouped(report_path, tag_override=_total_tag)):
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

    ``_grouped`` is called with the SAME ``tag_override`` as ``load_units`` so
    indices agree byte-for-byte between the two entry points (a chrome_trace
    invariant this backend preserves).
    """
    records = _grouped(report_path, tag_override=_total_tag)
    if kernel_index >= len(records):
        raise IndexError(f"unit index {kernel_index} not present in {report_path}")
    rec = records[kernel_index]
    out: dict[str, Any] = {
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
