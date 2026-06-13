"""Report path resolution — avoid reading stale data from a cluttered dir.

A debug directory accumulates reports over time; ``resolve_report`` turns the
agent's reference (exact file, extensionless name, or a directory meaning
"the newest report in here") into one concrete path or a clear error.
"""

from __future__ import annotations

from pathlib import Path

REPORT_SUFFIX = ".ncu-rep"


def resolve_report(report_ref: str) -> Path:
    """Resolve a report reference to an existing .ncu-rep file.

    Accepts: a path to a report file, a path missing the ``.ncu-rep``
    suffix, or a directory (resolves to the newest report inside it).
    """
    p = Path(report_ref).expanduser()

    if p.is_dir():
        candidates = sorted(
            p.glob(f"*{REPORT_SUFFIX}"), key=lambda f: f.stat().st_mtime, reverse=True
        )
        if not candidates:
            raise FileNotFoundError(
                f"no {REPORT_SUFFIX} files in directory {p} — profile first with: "
                "ncu --set full -o report.ncu-rep <app>"
            )
        return candidates[0]

    if p.is_file():
        return p

    with_suffix = p.with_name(p.name + REPORT_SUFFIX)
    if with_suffix.is_file():
        return with_suffix

    raise FileNotFoundError(
        f"report not found: {report_ref} (also tried {with_suffix}). "
        "Pass the path produced by: ncu --set full -o <file> <app>"
    )
