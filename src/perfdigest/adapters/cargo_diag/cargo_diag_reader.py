"""Read saved ``cargo build --message-format=json`` output -> NormalizedUnit.

The BuildDigest DIAGNOSTICS layer: where do a Rust build's errors/warnings
concentrate, per crate. The artifact is NDJSON (one JSON object per line,
saved via ``cargo build --message-format=json > build.cargo-diag.jsonl``;
messages go to stdout, human progress to stderr). Records this reader uses:

  * ``"reason": "compiler-message"``  — one diagnostic; carries ``package_id``
    and a ``message`` object with ``level``/``code``/``spans``.
  * ``"reason": "compiler-artifact"`` — one target's compile completed;
    carries ``package_id`` (+ ``fresh``: served from cache).
  * ``"reason": "build-finished"``    — report-level ``success`` bool.

A unit is a CRATE: one per distinct ``package_id`` seen in compiler-artifact
and/or compiler-message records (a crate that failed to compile emits messages
but never an artifact — it must still be a unit), in first-appearance order.
Other reasons (``build-script-executed``, future ones) are tolerated and
skipped: they prove the stream is cargo's but carry no diagnostic counts.

``package_id`` shape (verified against a REAL cargo 1.94 capture — it changed
across cargo versions):

  * modern (cargo >= 1.77, the Package ID Spec URL):
    ``path+file:///ws/mathlib#0.1.0`` — fragment is the bare version when the
    crate is named like the directory, else ``name@version``
    (``registry+https://github.com/rust-lang/crates.io-index#serde@1.0.219``).
  * legacy (older cargo): ``name version (source-url)``.

Both are parsed to a short ``name@version`` unit name; an unrecognized shape
falls back to the raw ``package_id`` string (stays addressable, never crashes,
never invents a name).

THE honesty story of this backend: cargo's JSON stream carries NO timing at
all, so ``duration_us`` is ``None`` for EVERY unit — never a fabricated 0.0.
(Build timing lives in other backends: ninja-log for step timings,
clang -ftime-trace for compiler phases.) The diagnostic COUNTS, by contrast,
are genuine 0.0 when a crate compiled clean: like the ``ptxas`` ``Used ...``
line, the message stream is a complete enumeration of the diagnostics this
build emitted, so absence of messages for a crate that appears in the stream
IS a measured zero, not a gap.
"""

from __future__ import annotations

import json
import re
from typing import Any

from perfdigest.core.metrics import DOMAIN_BUILD_DIAG, NormalizedUnit

# Legacy package_id: "name version (source-url)" (cargo < 1.77).
_LEGACY_PACKAGE_ID = re.compile(r"^(\S+) (\S+) \(.+\)$")

# Diagnostics with no primary span (e.g. rustc's final 'aborting due to N
# previous errors' failure-note) are counted under this explicit bucket in the
# per-file concentration keys — never silently merged into a real file's count.
_NO_SPAN = "(no-span)"


def _parse_package_id(package_id: str) -> tuple[str, str | None]:
    """-> (short crate name, version or None). Never raises on a weird shape."""
    m = _LEGACY_PACKAGE_ID.match(package_id)
    if m:
        return m.group(1), m.group(2)
    if "#" in package_id:
        url, frag = package_id.rsplit("#", 1)
        if "@" in frag:
            name, _, version = frag.rpartition("@")
            if name and version:
                return name, version
        elif frag:
            # bare-version fragment: the crate is named like the last URL path
            # segment (query params like ?rev=... stripped, trailing .git too)
            path = url.split("?", 1)[0].rstrip("/")
            name = path.rsplit("/", 1)[-1]
            if name.endswith(".git"):
                name = name[: -len(".git")]
            if name:
                return name, frag
    return package_id, None  # unknown shape: raw id stays the addressable name


def _unit_name(package_id: str) -> str:
    name, version = _parse_package_id(package_id)
    return f"{name}@{version}" if version else name


def _primary_file(message: dict) -> str:
    spans = message.get("spans")
    if isinstance(spans, list):
        for span in spans:
            if isinstance(span, dict) and span.get("is_primary"):
                file_name = span.get("file_name")
                if isinstance(file_name, str) and file_name:
                    return file_name
    return _NO_SPAN


def _parse_stream(report_path: str) -> tuple[list[dict[str, Any]], bool | None]:
    """-> (one record per crate, first-appearance order; build-finished success).

    Loud on non-cargo input: a line that is not JSON, not an object, or an
    object without cargo's ``reason`` key raises a named ValueError — never a
    silent empty digest (a perf-stat export is ALSO JSON-lines; the error text
    redirects it to its own format).
    """
    crates: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    build_finished: bool | None = None

    with open(report_path, "r", encoding="utf-8", errors="ignore") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{report_path} line {lineno} is not valid JSON ({exc.msg}) — "
                    "is this really saved `cargo build --message-format=json` "
                    "output (NDJSON, one JSON object per line)? A Chrome trace "
                    "is ONE multi-line JSON document (format chrome-trace), not "
                    "JSON-lines."
                ) from None
            if not isinstance(rec, dict) or "reason" not in rec:
                raise ValueError(
                    f"{report_path} line {lineno} is JSON but has no 'reason' "
                    "key — not a cargo --message-format=json stream. (A "
                    "perf-stat export is also JSON-lines; that is format "
                    "perf-stat-json.)"
                )

            reason = rec["reason"]
            if reason == "build-finished":
                success = rec.get("success")
                if isinstance(success, bool):
                    build_finished = success
                continue

            package_id = rec.get("package_id")
            if not isinstance(package_id, str) or not package_id:
                continue  # a report-level reason we don't know; nothing to attribute

            crate = crates.get(package_id)
            if crate is None:
                crate = {
                    "package_id": package_id,
                    "name": _unit_name(package_id),
                    "errors": 0,
                    "warnings": 0,
                    "notes_helps": 0,
                    "artifacts": 0,
                    "fresh_artifacts": 0,
                    "by_level_file": {},  # 'level@file' -> count
                    "by_code": {},  # 'code:<lint or E-code>' -> count
                }
                crates[package_id] = crate
                order.append(package_id)

            if reason == "compiler-artifact":
                crate["artifacts"] += 1
                if rec.get("fresh") is True:
                    crate["fresh_artifacts"] += 1
            elif reason == "compiler-message":
                message = rec.get("message")
                if not isinstance(message, dict):
                    continue
                level = str(message.get("level", ""))
                if level.startswith("error"):
                    # 'error' and 'error: internal compiler error' both count
                    crate["errors"] += 1
                elif level == "warning":
                    crate["warnings"] += 1
                else:
                    # note / help / failure-note ('aborting due to ...')
                    crate["notes_helps"] += 1
                where = f"{level}@{_primary_file(message)}"
                crate["by_level_file"][where] = crate["by_level_file"].get(where, 0) + 1
                code = message.get("code")
                code_str = code.get("code") if isinstance(code, dict) else None
                if isinstance(code_str, str) and code_str:
                    key = f"code:{code_str}"
                    crate["by_code"][key] = crate["by_code"].get(key, 0) + 1
            # other reasons: skipped (see module docstring)

    return [crates[pkg] for pkg in order], build_finished


def load_units(report_path: str) -> list[NormalizedUnit]:
    records, _build_finished = _parse_stream(report_path)
    units: list[NormalizedUnit] = []
    for index, rec in enumerate(records):
        metrics: dict[str, float | None] = {
            # genuine 0.0 when a crate compiled clean: the stream is a complete
            # enumeration of this build's diagnostics (ptxas 'Used ...' rule)
            "errors": float(rec["errors"]),
            "warnings": float(rec["warnings"]),
            "notes_helps": float(rec["notes_helps"]),
        }
        units.append(
            NormalizedUnit(
                name=rec["name"],
                index=index,
                duration_us=None,  # cargo's stream carries NO timing — honest None
                raw_ref=report_path,
                metrics=metrics,
                domain=DOMAIN_BUILD_DIAG,
            )
        )
    return units


def raw_metrics(report_path: str, kernel_index: int, name_filter: str) -> dict[str, Any]:
    """Counts + per-level-per-file concentration keys, for ``expand``.

    'warning@mathlib/src/lib.rs' -> 2.0 says WHERE the diagnostics concentrate
    without shipping their text (reading diagnostic text is the agent's
    editor/compiler job, not perfdigest's). ``build_finished_success`` is the
    report-level ``build-finished`` outcome (bool), repeated on every unit
    because the digest schema has no report-level slot — absent entirely if
    the capture was truncated before build-finished (never fabricated).
    """
    records, build_finished = _parse_stream(report_path)
    if kernel_index >= len(records):
        raise IndexError(f"unit index {kernel_index} not present in {report_path!r}")
    rec = records[kernel_index]
    out: dict[str, Any] = {
        "package_id": rec["package_id"],
        "errors": rec["errors"],
        "warnings": rec["warnings"],
        "notes_helps": rec["notes_helps"],
        "artifacts": rec["artifacts"],
        "fresh_artifacts": rec["fresh_artifacts"],
    }
    for key, count in rec["by_level_file"].items():
        out[key] = float(count)
    for key, count in rec["by_code"].items():
        out[key] = float(count)
    if build_finished is not None:
        out["build_finished_success"] = build_finished
    wanted = None if name_filter.lower() == "all" else name_filter.lower()
    if wanted is not None:
        out = {k: v for k, v in out.items() if wanted in k.lower()}
    return out
