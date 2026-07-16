"""GitHub Actions saved-log backend registration (CIDigest).

Core design principle (owner directive): bind to ARTIFACTS, never APIs. We
never call the GitHub API ourselves — the reader only ever opens a log FILE
the agent already saved with ``gh run view --log``. Digesting that file is
universal (any host, any OS); only *capturing* it (running ``gh`` against a
real run) needs the tool on PATH AND an authenticated session, which is why
this is the only backend whose probe checks two independent conditions.
"""

from __future__ import annotations

import subprocess

from perfdigest.adapters import registry
from perfdigest.adapters.gha_log import gha_log_reader, mapping
from perfdigest.core.backend import Backend, CapabilityReport
from perfdigest.platform import shell
from perfdigest.platform.detect import PlatformInfo, detect

# Digesting a saved log is pure text parsing — universal, same as chrome_trace.
_PLATFORMS = frozenset({"linux", "darwin", "win32"})

GHA_LOG_USAGE = (
    "GitHub Actions CI log digest (CIDigest). Capture is artifact-first — bind "
    "to a SAVED log FILE, never to the live GitHub API:\n"
    "  gh run view <run-id> --log > run.gha.log          (every job/step, format gha-log)\n"
    "  gh run view <run-id> --log-failed > run.gha.log   (failed steps only)\n"
    "Units are CI steps named '<job> / <step>', in file order. Real-report "
    "caveat (verified against a real fixture): on current GitHub Actions "
    "runners gh cannot always attribute a log line to a specific step (no "
    "per-step files server-side any more) and falls back to the literal step "
    "name 'UNKNOWN STEP' for the whole job — that unit then covers the WHOLE "
    "job, which is still a real, honest signal, just coarser-grained. "
    "duration_s is elapsed time between a step's first and last log line — an "
    "approximation from log timestamps, not a measured wall clock; a step with "
    "no parseable timestamps gets duration None (never 0.0). Start with "
    "summarize_report to find the slowest/failed steps; error_annotations > 0 "
    "is the failure scent — no grepping needed. Compare a run against a "
    "previous GREEN run's saved log with compare_metrics for previous-green "
    "analysis. Capture needs `gh` on PATH AND `gh auth status` to succeed; "
    "digesting an already-saved .gha.log needs neither."
)


def _probe() -> CapabilityReport:
    info = detect()
    exe = info.profilers_on_path.get("gha_log")
    if not exe:
        return CapabilityReport(False, "gh (GitHub CLI) not found on PATH", None)
    # `gh` present is necessary but not sufficient for CAPTURE: pulling a real
    # run's log also needs an authenticated session. Digesting an already-saved
    # log needs neither check — that split is called out in the usage prompt.
    try:
        result = subprocess.run(
            [exe, "auth", "status"], capture_output=True, text=True, timeout=10
        )
    except Exception as exc:  # noqa: BLE001 — surfaced as a diagnosable reason
        return CapabilityReport(
            False, f"gh present but 'gh auth status' failed to run: {exc}", exe
        )
    if result.returncode != 0:
        return CapabilityReport(
            False,
            "gh present but not authenticated (gh auth status failed); run "
            "`gh auth login` to enable capture. Digesting an already-saved "
            ".gha.log never needs authentication.",
            exe,
        )
    return CapabilityReport(True, "gh present and authenticated", exe)


def _capture_command(target: str, info: PlatformInfo) -> str:
    # `target` is the run-id (or 'owner/repo#run-id' style reference the agent
    # already has); the log always goes to a FILE, never to stdout in-context.
    view = shell.join(["gh", "run", "view", target, "--log"], info.shell)
    return f"{view} > run.gha.log"


registry.register(
    Backend(
        name="gha_log",
        formats=frozenset({"gha-log", "gh-run-log"}),
        suffixes=(".gha.log", ".log"),
        domain="ci_step",
        platforms=_PLATFORMS,
        standard_to_vendor=mapping.STANDARD_TO_VENDOR,
        default_core_set=tuple(mapping.DEFAULT_CORE_SET),
        usage_prompt=GHA_LOG_USAGE,
        load_units=gha_log_reader.load_units,
        raw_metrics=gha_log_reader.raw_metrics,
        probe=_probe,
        reader_available=lambda: True,  # pure Python
        capture_command=_capture_command,
    )
)
