"""Read a saved ``gh run view --log`` export -> NormalizedUnit (ci_step). Pure
Python, no deps.

Artifact-first (owner directive): we bind to a SAVED log FILE, never to the
live GitHub API. Produce it with::

    gh run view <run-id> --log > run.gha.log          # every job, every step
    gh run view <run-id> --log-failed > run.gha.log   # failed steps only

Line shape, verified against a REAL fixture (onlyxItachi/PerfDigest-MCP-Bench
run 29192019451, job "digest on macos-latest ..."; ``tests/fixtures/
ci_digest_macos_sample.gha.log``)::

    <job name>\\t<step name>\\t<ISO-8601 timestamp> <text>

Real-report lesson (the init prompt's real-report warning, again): on a
current GitHub Actions runner (server-side log format has moved to one flat
per-job file with ``##[group]``/``##[endgroup]`` sections instead of legacy
per-step files), ``gh`` cannot always attribute a line to a specific step and
falls back to the literal step name ``"UNKNOWN STEP"`` for EVERY line of a
job. That is not a parsing bug here — it is the real, honest granularity gh
handed back for this run. This reader groups by whatever (job, step) pairs
actually appear, so it transparently gets real per-step units on exports
where gh COULD resolve steps, and degrades to one unit per job (step =
"UNKNOWN STEP") where it could not — never fabricating a step boundary that
was not in the log.

Honesty rule: a step whose log lines carry no parseable timestamp at all gets
``duration_us=None`` (and the ``duration_s`` metric ``None``) — never 0.0. A
step with exactly one timestamped line gets a genuine 0.0 (elapsed time
between one line and itself is truly zero, not "unmeasured").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from perfdigest.core.metrics import DOMAIN_CI_STEP, NormalizedUnit

# job / step / rest, tab-separated. Job and step names are free text (anything
# but a literal tab); `rest` is everything after the second tab, unparsed here.
_LINE = re.compile(r"^(?P<job>[^\t]+)\t(?P<step>[^\t]+)\t(?P<rest>.*)$")

# GitHub Actions timestamps are RFC3339 UTC with a variable-precision
# fractional-second field (observed: 7 digits / 100ns ticks, e.g.
# '2026-07-12T12:07:34.9804200Z') that stdlib `datetime.fromisoformat` does not
# accept directly, so the fractional part is parsed manually and truncated to
# microseconds. A stray leading BOM (real fixture: the very first line of each
# job's original raw log carries one) is tolerated, not treated as malformed.
_TIMESTAMP = re.compile(
    r"^﻿?(?P<y>\d{4})-(?P<mo>\d{2})-(?P<d>\d{2})T"
    r"(?P<h>\d{2}):(?P<mi>\d{2}):(?P<s>\d{2})(?:\.(?P<frac>\d+))?Z"
)

_ERROR_MARKER = "##[error]"
_WARNING_MARKER = "##[warning]"

# expand() bounds how many offending lines it echoes back, so a log with
# thousands of warnings cannot balloon the raw payload.
_MAX_RAW_ANNOTATION_LINES = 64


@dataclass
class _LogLine:
    job: str
    step: str
    timestamp: datetime | None
    text: str  # content after the timestamp (or the whole `rest` if none)


def _parse_timestamp(rest: str) -> tuple[datetime | None, str]:
    """(timestamp or None, text-after-timestamp). Never raises."""
    m = _TIMESTAMP.match(rest)
    if not m:
        return None, rest
    frac = (m.group("frac") or "0") + "000000"
    micros = int(frac[:6])
    try:
        ts = datetime(
            int(m["y"]), int(m["mo"]), int(m["d"]),
            int(m["h"]), int(m["mi"]), int(m["s"]),
            micros, tzinfo=timezone.utc,
        )
    except ValueError:
        return None, rest
    return ts, rest[m.end():].lstrip()


def _iter_lines(text: str) -> list[_LogLine]:
    out: list[_LogLine] = []
    for raw_line in text.splitlines():
        if not raw_line:
            continue
        m = _LINE.match(raw_line)
        if not m:
            continue  # doesn't even have the job\tstep\t shape — not a log line
        ts, body = _parse_timestamp(m.group("rest"))
        out.append(_LogLine(job=m.group("job"), step=m.group("step"), timestamp=ts, text=body))
    return out


def _grouped(report_path: str) -> list[dict[str, Any]]:
    """-> one record per (job, step), in order of first appearance in the file."""
    with open(report_path, "r", encoding="utf-8", errors="ignore") as fh:
        text = fh.read()

    lines = _iter_lines(text)
    if not lines:
        raise ValueError(
            f"{report_path} does not look like a `gh run view --log` export: no "
            "line matched the expected '<job>\\t<step>\\t<ISO-8601 timestamp> "
            "<text>' shape (job and step are tab-separated prefixes on every "
            "line). Produce this artifact with: "
            "gh run view <run-id> --log > run.gha.log"
        )

    groups: dict[tuple[str, str], dict[str, Any]] = {}
    order: list[tuple[str, str]] = []
    for ln in lines:
        key = (ln.job, ln.step)
        rec = groups.get(key)
        if rec is None:
            rec = {
                "job": ln.job,
                "step": ln.step,
                "log_lines": 0,
                "timestamps": [],
                "error_lines": [],
                "warning_lines": [],
            }
            groups[key] = rec
            order.append(key)
        rec["log_lines"] += 1
        if ln.timestamp is not None:
            rec["timestamps"].append(ln.timestamp)
        if _ERROR_MARKER in ln.text:
            rec["error_lines"].append(ln.text)
        if _WARNING_MARKER in ln.text:
            rec["warning_lines"].append(ln.text)

    return [groups[key] for key in order]


def _duration_us(rec: dict[str, Any]) -> float | None:
    timestamps: list[datetime] = rec["timestamps"]
    if not timestamps:
        return None  # unmeasurable — honestly absent, never 0.0
    delta: timedelta = max(timestamps) - min(timestamps)
    return delta.total_seconds() * 1e6  # a genuine 0.0 if only one timestamp seen


def load_units(report_path: str) -> list[NormalizedUnit]:
    units: list[NormalizedUnit] = []
    for index, rec in enumerate(_grouped(report_path)):
        duration_us = _duration_us(rec)
        error_count = sum(text.count(_ERROR_MARKER) for text in rec["error_lines"])
        warning_count = sum(text.count(_WARNING_MARKER) for text in rec["warning_lines"])
        metrics: dict[str, float | None] = {
            "duration_s": (duration_us / 1e6) if duration_us is not None else None,
            "log_lines": float(rec["log_lines"]),
            "error_annotations": float(error_count),
            "warning_annotations": float(warning_count),
        }
        units.append(
            NormalizedUnit(
                name=f"{rec['job']} / {rec['step']}",
                index=index,
                duration_us=duration_us,
                raw_ref=report_path,
                metrics=metrics,
                domain=DOMAIN_CI_STEP,
            )
        )
    return units


def raw_metrics(report_path: str, kernel_index: int, name_filter: str) -> dict[str, Any]:
    """The parsed grouping detail behind a step's digest, for ``expand``."""
    records = _grouped(report_path)
    if kernel_index >= len(records):
        raise IndexError(f"unit index {kernel_index} not present in {report_path}")
    rec = records[kernel_index]
    timestamps = rec["timestamps"]
    duration_us = _duration_us(rec)
    out: dict[str, Any] = {
        "job": rec["job"],
        "step": rec["step"],
        "log_lines": rec["log_lines"],
        "duration_s": (duration_us / 1e6) if duration_us is not None else None,
        "first_timestamp": min(timestamps).isoformat() if timestamps else None,
        "last_timestamp": max(timestamps).isoformat() if timestamps else None,
        "error_annotations": sum(t.count(_ERROR_MARKER) for t in rec["error_lines"]),
        "warning_annotations": sum(t.count(_WARNING_MARKER) for t in rec["warning_lines"]),
        # bounded so a step with thousands of warnings cannot balloon the payload
        "error_lines": rec["error_lines"][:_MAX_RAW_ANNOTATION_LINES],
        "warning_lines": rec["warning_lines"][:_MAX_RAW_ANNOTATION_LINES],
    }
    wanted = None if name_filter.lower() == "all" else name_filter.lower()
    if wanted is not None:
        out = {k: v for k, v in out.items() if wanted in k.lower()}
    return out
