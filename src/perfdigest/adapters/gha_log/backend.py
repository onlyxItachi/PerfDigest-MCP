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
import time

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
    "is the failure scent — no grepping needed.\n"
    "Previous-green flow (the CI failure/slowdown loop): when a run is green, "
    "save its log ONCE and keep it as the baseline artifact — cheap insurance "
    "against the day the workflow breaks:\n"
    "  gh run view <green-run-id> --log > baseline-green.gha.log\n"
    "When a later run fails or slows down, save its log too and diff the same "
    "job with compare_metrics(report_a=baseline-green, report_b=failed, "
    "kernel='<job name>'). The delta table (delta = b - a) replaces re-reading "
    "the raw failed log: error_annotations delta > 0 says WHERE it screamed, "
    "duration_s/log_lines deltas say where it slowed or ballooned. Caveats "
    "proven on real runs: a FAILED run is often FASTER than green (it died "
    "early) — a negative duration delta is not an improvement when "
    "error_annotations rose; and error_annotations delta_pct over a clean "
    "green baseline (0 errors) is 'undefined_baseline_is_zero' by the compare "
    "contract — read the delta, not the pct. If the job was RENAMED between "
    "the two runs, pass kernel_b='<new job name>' for the B side. Capture "
    "needs `gh` on PATH AND `gh auth status` to succeed; digesting an "
    "already-saved .gha.log needs neither."
)


# `gh auth status` is the fleet's only probe doing I/O beyond shutil.which —
# it can hit the network. capability_summary() probes every backend, so the
# auth result is memoized per process with a TTL: platform_capabilities stays
# fast/offline-safe, while a mid-session `gh auth login` is still picked up
# within a few minutes (pre-release review finding 5).
_AUTH_TTL_S = 300.0
_auth_memo: tuple[float, CapabilityReport] | None = None


def _clear_auth_memo() -> None:
    """Testing hook: forget the memoized auth probe (mirrors cache.clear())."""
    global _auth_memo
    _auth_memo = None


def _auth_probe(exe: str) -> CapabilityReport:
    global _auth_memo
    now = time.monotonic()
    if _auth_memo is not None and now - _auth_memo[0] < _AUTH_TTL_S:
        return _auth_memo[1]
    try:
        result = subprocess.run(
            [exe, "auth", "status"], capture_output=True, text=True, timeout=10
        )
    except Exception as exc:  # noqa: BLE001 — surfaced as a diagnosable reason
        report = CapabilityReport(
            False, f"gh present but 'gh auth status' failed to run: {exc}", exe
        )
    else:
        if result.returncode != 0:
            report = CapabilityReport(
                False,
                "gh present but not authenticated (gh auth status failed); run "
                "`gh auth login` to enable capture. Digesting an already-saved "
                ".gha.log never needs authentication.",
                exe,
            )
        else:
            report = CapabilityReport(True, "gh present and authenticated", exe)
    _auth_memo = (now, report)
    return report


def _probe() -> CapabilityReport:
    info = detect()
    exe = info.profilers_on_path.get("gha_log")
    if not exe:
        return CapabilityReport(False, "gh (GitHub CLI) not found on PATH", None)
    # `gh` present is necessary but not sufficient for CAPTURE: pulling a real
    # run's log also needs an authenticated session. Digesting an already-saved
    # log needs neither check — that split is called out in the usage prompt.
    return _auth_probe(exe)


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
