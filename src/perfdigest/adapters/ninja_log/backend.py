"""ninja backend registration (build-step timings from ``.ninja_log``)."""

from __future__ import annotations

from perfdigest.adapters import registry
from perfdigest.adapters.ninja_log import mapping, ninja_log_reader
from perfdigest.core.backend import Backend, CapabilityReport
from perfdigest.platform import shell
from perfdigest.platform.detect import PlatformInfo, detect

_PLATFORMS = frozenset({"linux", "darwin", "win32"})  # ninja runs everywhere

NINJA_LOG_USAGE = (
    "ninja build-step digest. .ninja_log is a BYPRODUCT, not something you ask "
    "ninja to export — it is written next to the build directory after any "
    "`ninja -C <build-dir>` invocation, at <build-dir>/.ninja_log (format "
    "ninja-log). Units are build EDGES named by their output path (an object "
    "file, a binary, ...). IMPORTANT: under a parallel build (-j > 1) edges "
    "overlap in wall-clock time, so summed edge duration is NOT wall-clock "
    "time — summarize_report's coverage_pct_of_total_duration is coverage over "
    "SUMMED edge time, not build wall-clock. A target rebuilt across multiple "
    "invocations appears once per rebuild in the raw file; this reader keeps "
    "only the LAST timing for each output (the earlier ones are stale)."
)


def _probe() -> CapabilityReport:
    info = detect()
    exe = info.profilers_on_path.get("ninja_log")
    if not exe:
        return CapabilityReport(False, "ninja not found on PATH", None)
    return CapabilityReport(
        True,
        "ninja present (.ninja_log is written as a side effect of a normal build)",
        exe,
        notes=(
            "This is a byproduct capture: run your normal build through ninja, "
            "then digest the .ninja_log it leaves behind — there is no separate "
            "profiling flag to pass.",
        ),
    )


def _capture_command(target: str, info: PlatformInfo) -> str:
    build_cmd = shell.join(["ninja", "-C", target], info.shell)
    return f"{build_cmd}  # writes {target}/.ninja_log as a byproduct"


registry.register(
    Backend(
        name="ninja_log",
        formats=frozenset({"ninja-log", "ninja_log"}),
        suffixes=(".ninja_log",),
        domain="build_step",
        platforms=_PLATFORMS,
        standard_to_vendor=mapping.STANDARD_TO_VENDOR,
        default_core_set=tuple(mapping.DEFAULT_CORE_SET),
        usage_prompt=NINJA_LOG_USAGE,
        load_units=ninja_log_reader.load_units,
        raw_metrics=ninja_log_reader.raw_metrics,
        probe=_probe,
        reader_available=lambda: True,  # pure Python
        capture_command=_capture_command,
    )
)
