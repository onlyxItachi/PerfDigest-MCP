"""git numstat backend registration (RepoState — session-retrospective).

RepoState is NOT GitHub/API monitoring: it is a backward look at what a
working session did to a repo, read from a SAVED artifact. It is also the
provenance axis — a numstat snapshot pins the perf/build/CI reports captured
next to it to a concrete code state. Facts only, never verdicts: the backend
reports which files changed and by how many lines; whether that footprint is
"ready" is the model's conclusion.
"""

from __future__ import annotations

from perfdigest.adapters import registry
from perfdigest.adapters.git_numstat import mapping, numstat_reader
from perfdigest.core.backend import Backend, CapabilityReport
from perfdigest.platform.detect import PlatformInfo, detect

_PLATFORMS = frozenset({"linux", "darwin", "win32"})  # git runs everywhere

GIT_NUMSTAT_USAGE = (
    "git numstat digest (RepoState — the session-retrospective change digest). "
    "Artifact-first; the raw diff is the token sink, numstat is its digest:\n"
    "  git diff --numstat -M <base>.. > session.numstat   (the session's range, "
    "format git-numstat)\n"
    "  git diff --numstat -M > session.numstat            (uncommitted work only)\n"
    "Units are changed files named by path — a renamed file is named by its "
    "NEW path (where it lives now); the old path is in expand. Use this to "
    "REORIENT after compaction or a session hand-off instead of re-reading raw "
    "diffs: summarize_report lists the footprint, lines_added ranks it (units "
    "carry no duration — a change digest has no time dimension — so ordering "
    "is file order and size ranking comes from reading the metrics). BINARY "
    "files have no line counts: lines_added/deleted are honestly "
    "'not_available_in_this_export', never 0.0 — while a pure no-edit rename "
    "is a genuine measured 0. Pair two snapshots with compare_metrics "
    "(report_a=earlier, report_b=later, kernel=<path>) to see how the "
    "session's footprint evolved. Keep snapshots next to captured perf/build/"
    "CI reports as cheap provenance anchors: the numstat pins those reports "
    "to the code state that produced them."
)


def _probe() -> CapabilityReport:
    info = detect()
    exe = info.profilers_on_path.get("git_numstat")
    if not exe:
        return CapabilityReport(False, "git not found on PATH", None)
    return CapabilityReport(
        True,
        "git present (capture runs inside the repo being digested)",
        exe,
        notes=(
            "Add -M so renames digest as one moved file instead of a fake "
            "add+delete pair; note git only pairs a rename when similarity "
            "stays above its threshold — a heavily rewritten move still "
            "splits, and that split is git's real answer, not a parse bug.",
        ),
    )


def _capture_command(target: str, info: PlatformInfo) -> str:
    # `target` is the base ref of the session (a tag, branch, or commit).
    return (
        f"git diff --numstat -M {target}.. > session.numstat"
        "  # uncommitted-only: git diff --numstat -M > session.numstat"
    )


registry.register(
    Backend(
        name="git_numstat",
        formats=frozenset({"git-numstat", "numstat"}),
        suffixes=(".numstat",),
        domain="repo_change",
        platforms=_PLATFORMS,
        standard_to_vendor=mapping.STANDARD_TO_VENDOR,
        default_core_set=tuple(mapping.DEFAULT_CORE_SET),
        usage_prompt=GIT_NUMSTAT_USAGE,
        load_units=numstat_reader.load_units,
        raw_metrics=numstat_reader.raw_metrics,
        probe=_probe,
        reader_available=lambda: True,  # pure Python
        capture_command=_capture_command,
    )
)
