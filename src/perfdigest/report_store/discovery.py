"""Report path resolution — avoid reading stale data from a cluttered dir.

A debug directory accumulates reports over time; ``resolve_report`` turns the
agent's reference (exact file, extensionless name, or a directory meaning "the
newest report in here") into one concrete path or a clear error. The set of
report suffixes is no longer NVIDIA-specific — it comes from the resolved
backend (``.ncu-rep``, ``.csv``, ``.perf-stat.json``, ``.json`` ...).
"""

from __future__ import annotations

from pathlib import Path

# Default kept for back-compat with v1 callers that pass no suffixes.
DEFAULT_SUFFIXES: tuple[str, ...] = (".ncu-rep",)


def resolve_report(
    report_ref: str, suffixes: tuple[str, ...] = DEFAULT_SUFFIXES
) -> Path:
    """Resolve a report reference to an existing report file.

    Accepts: a path to a report file, a path missing the backend's suffix, or a
    directory (resolves to the newest matching report inside it).
    """
    suffixes = tuple(suffixes) or DEFAULT_SUFFIXES
    p = Path(report_ref).expanduser()

    if p.is_dir():
        candidates = [
            f for suf in suffixes for f in p.glob(f"*{suf}")
        ]
        if not candidates:
            shown = " / ".join(suffixes)
            raise FileNotFoundError(
                f"no {shown} files in directory {p} — capture first "
                "(see suggest_profile_command for the right invocation)."
            )
        return max(candidates, key=lambda f: f.stat().st_mtime)

    if p.is_file():
        return p

    for suf in suffixes:
        with_suffix = p.with_name(p.name + suf)
        if with_suffix.is_file():
            return with_suffix

    tried = ", ".join(str(p.with_name(p.name + s)) for s in suffixes)
    raise FileNotFoundError(
        f"report not found: {report_ref} (also tried {tried})."
    )
