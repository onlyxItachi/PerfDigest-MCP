"""Read ninja's ``.ninja_log`` -> NormalizedUnit (build_step). Pure Python, no deps.

``.ninja_log`` is a byproduct ninja writes after every build (not something you
ask it to export). Shape::

    # ninja log v5
    start_ms<TAB>end_ms<TAB>restat_mtime<TAB>output_path<TAB>cmd_hash

One line per build EDGE executed. Ninja keys its in-memory log by
``output_path``, so a target rebuilt across multiple invocations (incremental
builds append, they never rewrite the file) appears more than once — the LAST
occurrence's timings are what ninja itself trusts, so we supersede on load
exactly the way ``BuildLog::Load`` does: later lines overwrite earlier ones for
the same output, keeping the FIRST-seen position as the report's file order.

Honesty rule: ``restat_mtime`` and ``cmd_hash`` are ninja bookkeeping, not
performance metrics — they never enter ``metrics``/``mapping.py``, though they
are still exposed verbatim through ``expand`` (raw access), same treatment
ptxas gives its non-metric ``arch`` tag.

Version note: the spec instructions this reader was built against name v5/v6
as the current formats ("v6 in ninja >=1.12 — same tab-separated fields"). The
REAL fixture here was captured with the installed ninja 1.13.2, which already
writes v7 (a further bump past v6, same 5-field shape — verified by capturing
it). All three are accepted; anything else is named and rejected loudly rather
than silently returning nothing.
"""

from __future__ import annotations

from typing import Any

from perfdigest.core.metrics import DOMAIN_BUILD_STEP, NormalizedUnit

_HEADER_PREFIX = "# ninja log v"
_SUPPORTED_VERSIONS = {"5", "6", "7"}
_FIELD_NAMES = ("start_ms", "end_ms", "restat_mtime", "output_path", "cmd_hash")


def _check_header(header: str, report_path: str) -> None:
    if not header.startswith(_HEADER_PREFIX):
        raise ValueError(
            f"unrecognized ninja log header in {report_path!r}: expected a line "
            f"starting with {_HEADER_PREFIX!r}, found {header!r}"
        )
    version = header[len(_HEADER_PREFIX) :].strip()
    if version not in _SUPPORTED_VERSIONS:
        raise ValueError(
            f"unsupported ninja log version {version!r} in {report_path!r} "
            f"(header {header!r}); perfdigest reads v{'/v'.join(sorted(_SUPPORTED_VERSIONS))} "
            "(tab-separated start_ms/end_ms/restat_mtime/output_path/cmd_hash)"
        )


def _parse_records(report_path: str) -> list[dict[str, Any]]:
    """Parse -> one record per ``output_path``, LAST occurrence wins.

    Position in the returned list is the output's FIRST-seen line (matching
    ninja's own hash-map-keyed-by-path load behavior); the VALUES at that
    position are from whichever line for that path came last in the file.
    """
    with open(report_path, "r", encoding="utf-8", errors="ignore") as fh:
        lines = fh.read().splitlines()

    if not lines:
        raise ValueError(f"empty ninja log: {report_path!r} (no header line found)")

    _check_header(lines[0].strip(), report_path)

    order: list[str] = []  # output_path, first-seen order
    by_output: dict[str, dict[str, Any]] = {}

    for lineno, raw_line in enumerate(lines[1:], start=2):
        if not raw_line.strip():
            continue  # ninja does not emit blank edge lines, but tolerate a trailing one
        fields = raw_line.split("\t")
        if len(fields) != 5:
            raise ValueError(
                f"malformed ninja log line {lineno} in {report_path!r}: expected 5 "
                f"tab-separated fields ({', '.join(_FIELD_NAMES)}), found "
                f"{len(fields)}: {raw_line!r}"
            )
        start_raw, end_raw, restat_mtime, output_path, cmd_hash = fields
        try:
            start_ms = float(start_raw)
            end_ms = float(end_raw)
        except ValueError as exc:
            raise ValueError(
                f"malformed ninja log line {lineno} in {report_path!r}: start_ms/"
                f"end_ms must be numeric, found {raw_line!r}"
            ) from exc

        if output_path not in by_output:
            order.append(output_path)
        by_output[output_path] = {
            "output_path": output_path,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "restat_mtime": restat_mtime,
            "cmd_hash": cmd_hash,
        }

    return [by_output[path] for path in order]


def load_units(report_path: str) -> list[NormalizedUnit]:
    units: list[NormalizedUnit] = []
    for index, rec in enumerate(_parse_records(report_path)):
        duration_ms = rec["end_ms"] - rec["start_ms"]
        metrics: dict[str, float | None] = {
            "duration_ms": duration_ms,
            "start_ms": rec["start_ms"],
            "end_ms": rec["end_ms"],
        }
        units.append(
            NormalizedUnit(
                name=rec["output_path"],
                index=index,
                duration_us=duration_ms * 1000.0,
                raw_ref=report_path,
                metrics=metrics,
                domain=DOMAIN_BUILD_STEP,
            )
        )
    return units


def raw_metrics(report_path: str, kernel_index: int, name_filter: str) -> dict[str, Any]:
    """Raw parsed fields (including non-metric ``restat_mtime``/``cmd_hash``)."""
    records = _parse_records(report_path)
    if kernel_index >= len(records):
        raise IndexError(f"unit index {kernel_index} not present in {report_path!r}")
    rec = records[kernel_index]
    duration_ms = rec["end_ms"] - rec["start_ms"]
    raw = {
        "start_ms": rec["start_ms"],
        "end_ms": rec["end_ms"],
        "duration_ms": duration_ms,
        "restat_mtime": rec["restat_mtime"],
        "cmd_hash": rec["cmd_hash"],
    }
    wanted = None if name_filter.lower() == "all" else name_filter.lower()
    return {k: v for k, v in raw.items() if wanted is None or wanted in k.lower()}
