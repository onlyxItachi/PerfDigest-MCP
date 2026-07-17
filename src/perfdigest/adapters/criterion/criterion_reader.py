"""Read criterion's output tree -> NormalizedUnit (benchmark). Pure Python, no deps.

Criterion (the Rust benchmarking crate) writes results under ``target/criterion``
after ``cargo bench``. The report_ref for this backend is that DIRECTORY — the
report is a tree, not a single file::

    target/criterion/
      <bench>/new/estimates.json              # plain benchmark
      <group>/<bench>/new/estimates.json      # benchmark_group: one level deeper
      report/ ...                             # criterion's own HTML (skipped)

``new/`` is the latest run; ``base/`` (previous run) and ``change/`` (criterion's
own delta estimates) exist too, but perfdigest reads only ``new`` — cross-run
deltas are compare_metrics' job on two report trees, not a re-serving of
criterion's. Each ``estimates.json`` keys estimators ("mean", "median",
"std_dev", "median_abs_dev", "slope") to ``{point_estimate, standard_error,
confidence_interval}``, all in NANOSECONDS.

Unit name = the benchmark's directory path relative to the root (``fib_plain``,
``fib/fib_20``); index = deterministic sorted walk order.

Honesty rule, criterion edition: an estimator a given run did not produce is
serialized by criterion as JSON ``null`` (it is a Rust ``Option``) — observed on
a real flat-sampling-mode run, where ``slope`` is genuinely not estimated
(slope regression needs linear sampling's varying iteration counts). Both
``null`` and a missing key surface as ``None`` -> not measured, never 0.0.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from perfdigest.core.metrics import DOMAIN_BENCHMARK, NormalizedUnit

# estimator keys in estimates.json; standard terms are these + "_ns" (mapping.py)
_ESTIMATORS = ("mean", "median", "std_dev", "median_abs_dev", "slope")


def _bench_dirs(root: Path) -> list[tuple[str, Path]]:
    """-> sorted (relative bench name, estimates.json path); loud if none."""
    if not root.is_dir():
        raise ValueError(
            f"{str(root)!r} is not a directory — the criterion report_ref is the "
            "output ROOT directory cargo bench writes (e.g. target/criterion), "
            "not an individual file inside it."
        )
    found: list[tuple[str, Path]] = []
    for est in root.glob("**/new/estimates.json"):
        rel_parts = est.parent.parent.relative_to(root).parts
        # criterion's own HTML bookkeeping lives under report/ dirs, but those
        # never contain a new/estimates.json, so the glob alone excludes them.
        # No name-based guard on top: a user CAN name a bench "report" (it
        # collides with criterion's HTML dir — criterion's wart, not ours), and
        # suppressing its genuinely-written estimates would fabricate absence.
        found.append(("/".join(rel_parts), est))
    if not found:
        raise ValueError(
            f"no */new/estimates.json under {str(root)!r} — not a criterion "
            "output directory? Pass the root cargo bench writes, e.g. "
            "target/criterion."
        )
    found.sort(key=lambda item: item[0])
    return found


def _load_estimates(est_path: Path) -> dict[str, Any]:
    with open(est_path, "r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed estimates.json at {est_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"malformed estimates.json at {est_path}: expected a JSON object of "
            f"estimators, found {type(data).__name__}"
        )
    return data


def _point_estimate(data: dict[str, Any], estimator: str) -> float | None:
    node = data.get(estimator)
    if not isinstance(node, dict):  # missing key, or criterion's `null` Option
        return None
    value = node.get("point_estimate")
    return float(value) if value is not None else None


def load_units(report_path: str) -> list[NormalizedUnit]:
    root = Path(report_path)
    units: list[NormalizedUnit] = []
    for index, (name, est_path) in enumerate(_bench_dirs(root)):
        data = _load_estimates(est_path)
        metrics: dict[str, float | None] = {
            f"{estimator}_ns": _point_estimate(data, estimator)
            for estimator in _ESTIMATORS
        }
        mean_ns = metrics["mean_ns"]
        units.append(
            NormalizedUnit(
                name=name,
                index=index,
                duration_us=(mean_ns / 1000.0) if mean_ns is not None else None,
                raw_ref=report_path,
                metrics=metrics,
                domain=DOMAIN_BENCHMARK,
            )
        )
    return units


def raw_metrics(report_path: str, kernel_index: int, name_filter: str) -> dict[str, Any]:
    """Raw estimator fields, dot-flattened (``mean.confidence_interval.lower_bound``)."""
    benches = _bench_dirs(Path(report_path))
    if kernel_index >= len(benches):
        raise IndexError(f"unit index {kernel_index} not present in {report_path!r}")
    data = _load_estimates(benches[kernel_index][1])

    flat: dict[str, Any] = {}

    def _flatten(node: Any, prefix: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                _flatten(value, f"{prefix}.{key}" if prefix else key)
        elif node is not None:  # a null estimator was not measured: omit, never 0.0
            flat[prefix] = node

    _flatten(data, "")
    wanted = None if name_filter.lower() == "all" else name_filter.lower()
    return {k: v for k, v in flat.items() if wanted is None or wanted in k.lower()}
