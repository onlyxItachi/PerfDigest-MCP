"""ptxas backend registration (NVIDIA compile-time codegen layer).

Capture needs ``nvcc`` but NO GPU — resource usage is a compiler fact. Digest is
pure Python, so a ptxas report from any CI compile job is readable anywhere.
"""

from __future__ import annotations

from perfdigest.adapters import registry
from perfdigest.adapters.ptxas import mapping, ptxas_reader
from perfdigest.core.backend import Backend, CapabilityReport
from perfdigest.platform import shell
from perfdigest.platform.detect import PlatformInfo, detect

_PLATFORMS = frozenset({"linux", "win32"})  # CUDA toolchain; no macOS since CUDA 11

PTXAS_USAGE = (
    "ptxas codegen digest (compile-time; capture needs nvcc but NO GPU). ptxas "
    "prints to STDERR — redirect it:\n"
    "  nvcc -O3 -arch=<sm> -Xptxas -v -c kernel.cu -o /dev/null 2> report.ptxas.txt"
    "   (format ptxas-verbose)\n"
    "Unit names are the mangled name tagged with the target arch "
    "('_Z9my_kernelPf... [sm_89]') — the plain kernel name is a substring, so "
    "kernel='my_kernel' matches; on a multi-gencode report add the tag "
    "(kernel='my_kernel...[sm_80]') or use the index. Nonzero spill_stores_bytes/"
    "spill_loads_bytes with capped registers is the classic occupancy-vs-spill "
    "trade-off signal to read next to an ncu digest of the same kernel."
)


def _probe() -> CapabilityReport:
    info = detect()
    exe = info.profilers_on_path.get("ptxas")
    if not exe:
        return CapabilityReport(False, "nvcc not found on PATH (CUDA toolkit needed)", None)
    return CapabilityReport(
        True,
        "nvcc present (no GPU needed — codegen is a compiler fact)",
        exe,
        notes=(
            "Add -arch=sm_XX for your DEPLOYMENT target — without it nvcc compiles "
            "for its default arch and the register/spill numbers describe that "
            "default, not your GPU. Multi-arch (-gencode ...) works too: each "
            "target becomes its own '[sm_XX]'-tagged unit.",
        ),
    )


def _capture_command(target: str, info: PlatformInfo) -> str:
    compile_part = shell.join(
        ["nvcc", "-O3", "-Xptxas", "-v", "-c", target, "-o",
         "NUL" if info.os == "win32" else "/dev/null"],
        info.shell,
    )
    return f"{compile_part} 2> report.ptxas.txt"


registry.register(
    Backend(
        name="ptxas",
        formats=frozenset({"ptxas-verbose", "ptxas"}),
        suffixes=(".ptxas.txt",),
        domain="kernel_codegen",
        platforms=_PLATFORMS,
        standard_to_vendor=mapping.STANDARD_TO_VENDOR,
        default_core_set=tuple(mapping.DEFAULT_CORE_SET),
        usage_prompt=PTXAS_USAGE,
        load_units=ptxas_reader.load_units,
        raw_metrics=ptxas_reader.raw_metrics,
        probe=_probe,
        reader_available=lambda: True,  # pure Python
        capture_command=_capture_command,
    )
)
