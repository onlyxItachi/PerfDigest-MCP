"""AMD ROCm / HIP backend registration."""

from __future__ import annotations

from perfdigest.adapters import registry
from perfdigest.adapters.rocm import mapping, rocprof_reader
from perfdigest.core.backend import Backend, CapabilityReport
from perfdigest.platform import shell
from perfdigest.platform.detect import PlatformInfo, detect

_PLATFORMS = frozenset({"linux", "win32"})  # ROCm/HIP SDK

ROCM_USAGE = (
    "AMD ROCm/HIP (GPU kernels). Capture per-kernel stats + counters to CSV:\n"
    "  rocprof --stats --basenames on -o results.csv <app>\n"
    "Then list_kernels -> get_metrics on the hottest dispatch by duration_us. "
    "Standard GPU terms (compute_pct_peak, l2_hit_rate, achieved_occupancy) carry "
    "over from NVIDIA; AMD-specific counters via expand()."
)


def _probe() -> CapabilityReport:
    info = detect()
    exe = info.profilers_on_path.get("rocm")
    if not exe:
        return CapabilityReport(False, "rocprof not found on PATH", None)
    return CapabilityReport(True, "rocprof present", exe)


def _capture_command(target: str, info: PlatformInfo) -> str:
    return shell.join(
        ["rocprof", "--stats", "--basenames", "on", "-o", "results.csv", target],
        info.shell,
    )


registry.register(
    Backend(
        name="rocm",
        formats=frozenset({"rocprof-csv", "rocprof", "rocm-csv"}),
        suffixes=(".csv",),
        domain="gpu_kernel",
        platforms=_PLATFORMS,
        standard_to_vendor=mapping.STANDARD_TO_VENDOR,
        default_core_set=tuple(mapping.DEFAULT_CORE_SET),
        usage_prompt=ROCM_USAGE,
        load_units=rocprof_reader.load_units,
        raw_metrics=rocprof_reader.raw_metrics,
        probe=_probe,
        reader_available=lambda: True,  # pure Python
        capture_command=_capture_command,
    )
)
