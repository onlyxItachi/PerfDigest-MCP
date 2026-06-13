"""NVIDIA Nsight Compute backend — wraps the existing PRI/CSV readers.

Two formats, same vendor mapping:
  * ``ncu-rep`` — native binary via ``ncu_report`` (rich; needs the PRI wheel).
  * ``ncu-csv`` — ``ncu --csv`` export, parsed in pure Python (NO GPU, NO PRI
    wheel). This is the tier-1 universal path: digest an NVIDIA capture on a Mac
    that produced its report on a CI/remote GPU runner.
"""

from __future__ import annotations

import importlib.util

from perfdigest.adapters import registry
from perfdigest.adapters.nsight import csv_reader, mapping, pri_reader
from perfdigest.core.backend import Backend, CapabilityReport
from perfdigest.platform import shell
from perfdigest.platform.detect import PlatformInfo, detect

_PLATFORMS = frozenset({"linux", "win32"})  # where you can RUN ncu

NSIGHT_USAGE = (
    "NVIDIA Nsight Compute (CUDA kernels). Capture ONCE, wide, to a file:\n"
    "  ncu --set full -o report.ncu-rep <app>\n"
    "NEVER run ncu without -o (its stdout table floods your context). Then "
    "list_kernels -> get_metrics on the hottest kernel by duration_us."
)


def _probe() -> CapabilityReport:
    info = detect()
    exe = info.profilers_on_path.get("nsight")
    if not exe:
        return CapabilityReport(False, "ncu not found on PATH", None)
    return CapabilityReport(True, "ncu present", exe)


def _pri_reader_available() -> bool:
    return importlib.util.find_spec("ncu_report") is not None


def _capture_command(target: str, info: PlatformInfo) -> str:
    out = "report.ncu-rep"
    return shell.join(["ncu", "--set", "full", "-o", out, target], info.shell)


def _build() -> None:
    common = dict(
        suffixes=(".ncu-rep",),
        domain="gpu_kernel",
        platforms=_PLATFORMS,
        standard_to_vendor=mapping.STANDARD_TO_VENDOR,
        default_core_set=tuple(mapping.DEFAULT_CORE_SET),
        usage_prompt=NSIGHT_USAGE,
        raw_metrics=pri_reader.raw_metrics,
        probe=_probe,
        capture_command=_capture_command,
    )
    # Native PRI binary reader.
    registry.register(
        Backend(
            name="nsight",
            formats=frozenset({"ncu-rep", "ncu_rep", "pri"}),
            load_units=pri_reader.load_kernels,
            reader_available=_pri_reader_available,
            **common,
        )
    )
    # Pure-Python CSV reader — always readable (no GPU / no PRI wheel).
    registry.register(
        Backend(
            name="nsight_csv",
            formats=frozenset({"ncu-csv", "ncu_csv"}),
            suffixes=(".csv",),
            domain="gpu_kernel",
            platforms=_PLATFORMS,
            standard_to_vendor=mapping.STANDARD_TO_VENDOR,
            default_core_set=tuple(mapping.DEFAULT_CORE_SET),
            usage_prompt=(
                "NVIDIA ncu CSV export (pure-Python, GPU-free digest). Produce "
                "with: ncu --csv --page raw --import report.ncu-rep > report.csv"
            ),
            load_units=csv_reader.load_units,
            raw_metrics=csv_reader.raw_metrics,
            probe=_probe,
            reader_available=lambda: True,
            capture_command=_capture_command,
        )
    )


_build()
