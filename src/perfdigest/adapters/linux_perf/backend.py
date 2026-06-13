"""Linux ``perf`` backend registration (CPU C++/Rust hot functions)."""

from __future__ import annotations

from perfdigest.adapters import registry
from perfdigest.adapters.linux_perf import mapping, perf_reader
from perfdigest.core.backend import Backend, CapabilityReport
from perfdigest.platform import shell
from perfdigest.platform.detect import PlatformInfo, detect

_PLATFORMS = frozenset({"linux"})  # perf is a Linux tool

PERF_USAGE = (
    "Linux perf (CPU hot functions for C++/Rust). Two capture shapes:\n"
    "  program counters: perf stat -j -o report.json -- <app>   (format perf-stat-json)\n"
    "  hot symbols:      perf record -g -- <app> && perf report --stdio -n > report.txt"
    "   (format perf-report)\n"
    "Counters (IPC, cache/branch miss rates) need HW PMU access — unavailable on "
    "most WSL2 kernels; check platform_capabilities first."
)


def _probe() -> CapabilityReport:
    info = detect()
    exe = info.profilers_on_path.get("linux_perf")
    if not exe:
        return CapabilityReport(False, "perf not found on PATH", None)
    return CapabilityReport(True, "perf present", exe)


def _capture_command(target: str, info: PlatformInfo) -> str:
    return shell.join(
        ["perf", "stat", "-j", "-o", "report.json", "--", target], info.shell
    )


registry.register(
    Backend(
        name="linux_perf",
        formats=frozenset({"perf-stat-json", "perf-report", "perf"}),
        suffixes=(".perf-stat.json", ".perf.txt"),
        domain="cpu_function",
        platforms=_PLATFORMS,
        standard_to_vendor=mapping.STANDARD_TO_VENDOR,
        default_core_set=tuple(mapping.DEFAULT_CORE_SET),
        usage_prompt=PERF_USAGE,
        load_units=perf_reader.load_units,
        raw_metrics=perf_reader.raw_metrics,
        probe=_probe,
        reader_available=lambda: True,  # pure Python
        capture_command=_capture_command,
    )
)
