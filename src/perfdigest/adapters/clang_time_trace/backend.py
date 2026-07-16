"""clang ``-ftime-trace`` backend registration (BuildDigest: compiler phase profile).

Digest is pure Python (it reuses chrome_trace's Chrome-trace parsing, see
``time_trace_reader.py``) and universal — a report produced by any clang on any
CI box is readable anywhere. "Capture" is a compiler flag, not a separate
tool: any clang new enough to support ``-ftime-trace`` (3.9+) can produce one.
"""

from __future__ import annotations

import shutil

from perfdigest.adapters import registry
from perfdigest.adapters.clang_time_trace import mapping, time_trace_reader
from perfdigest.core.backend import Backend, CapabilityReport
from perfdigest.platform.detect import PlatformInfo, detect

_PLATFORMS = frozenset({"linux", "darwin", "win32"})  # clang runs everywhere

CLANG_TIME_TRACE_USAGE = (
    "clang -ftime-trace digest (compiler phase profile — where BUILD time goes, "
    "not runtime). Capture: the JSON lands next to the object file, no separate "
    "profiler tool:\n"
    "  clang++ -ftime-trace -c file.cpp   (emits file.json next to file.o)"
    "   (format clang-time-trace, alias ftime-trace)\n"
    "Units aggregate by (cat, name) exactly like chrome_trace: one unit per "
    "compiler phase (Frontend, ParseClass, InstantiateFunction, CodeGen "
    "Function...), domain build_phase. TWO honesty nuances specific to this "
    "backend:\n"
    "  1. clang ALSO emits a 'Total <Phase>' aggregate per phase (e.g. 'Total "
    "Frontend', tagged '[total]') whose total_time_us already sums every "
    "per-instance 'Frontend' sibling in the same report. This is bookkeeping, "
    "not extra payload — do NOT add '[total]' units to their own siblings when "
    "estimating where build time goes, or you will double-count. They stay "
    "digestible (never dropped) precisely so an agent can see and discount "
    "them.\n"
    "  2. Phases NEST (ExecuteCompiler contains Frontend contains ParseClass/"
    "InstantiateFunction/...), so span totals overlap across units — "
    "total_time_us is NOT additive across a report and summarize's "
    "coverage_pct_of_total_duration is indicative only, same as chrome_trace.\n"
    "For a hot InstantiateFunction/ParseClass, cross-reference the "
    "'arg:detail' field from expand() — it names the template/file."
)


def _probe() -> CapabilityReport:
    info = detect()
    exe = info.profilers_on_path.get("clang_time_trace") or shutil.which("clang")
    if not exe:
        return CapabilityReport(
            False, "clang/clang++ not found on PATH", None
        )
    return CapabilityReport(
        True,
        "clang/clang++ present — -ftime-trace is a compiler flag, no extra tool",
        exe,
        notes=(
            "-ftime-trace needs clang (not gcc). -ftime-trace-granularity=<us> "
            "(default 500) raises the floor to drop tiny sub-events if a report "
            "is too large.",
        ),
    )


def _capture_command(target: str, info: PlatformInfo) -> str:
    return (
        f"clang++ -ftime-trace -c {target}  # JSON report lands next to the "
        "object file (file.json for file.cpp -> file.o)"
    )


registry.register(
    Backend(
        name="clang_time_trace",
        formats=frozenset({"clang-time-trace", "ftime-trace"}),
        suffixes=(".json",),
        domain="build_phase",
        platforms=_PLATFORMS,
        standard_to_vendor=mapping.STANDARD_TO_VENDOR,
        default_core_set=tuple(mapping.DEFAULT_CORE_SET),
        usage_prompt=CLANG_TIME_TRACE_USAGE,
        load_units=time_trace_reader.load_units,
        raw_metrics=time_trace_reader.raw_metrics,
        probe=_probe,
        reader_available=lambda: True,  # pure Python
        capture_command=_capture_command,
    )
)
