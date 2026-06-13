"""Host detection — what OS/shell/GPUs/profilers are present here.

Cheap and side-effect-free: tool presence is ``shutil.which`` (PATH only), and
machine identity is read from cheap files / one-shot ``sysctl``. We never run a
profiler here. Results feed the capture-gating in ``capabilities.py`` and the
``platform_capabilities`` MCP tool.
"""

from __future__ import annotations

import os
import platform as stdplatform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache

# Profiler executables we know how to advise/parse, by backend name.
PROFILER_TOOLS: dict[str, str] = {
    "nsight": "ncu",
    "linux_perf": "perf",
    "rocm": "rocprof",
    "metal": "xctrace",
}


@dataclass(frozen=True)
class PlatformInfo:
    """A snapshot of the host relevant to profiling."""

    os: str                       # sys.platform: "linux" | "darwin" | "win32"
    is_wsl: bool                  # Linux kernel running under WSL
    shell: str                    # "posix" | "powershell"
    machine_model: str | None     # e.g. "MacBookPro18,3", DMI product name
    cpu: str | None               # CPU brand string
    gpu_vendors: tuple[str, ...]  # subset of {"nvidia","amd","apple"}
    profilers_on_path: dict[str, str | None] = field(default_factory=dict)

    def has_tool(self, backend_name: str) -> bool:
        return bool(self.profilers_on_path.get(backend_name))

    def to_dict(self) -> dict:
        return {
            "os": self.os,
            "is_wsl": self.is_wsl,
            "shell": self.shell,
            "machine_model": self.machine_model,
            "cpu": self.cpu,
            "gpu_vendors": list(self.gpu_vendors),
            "profilers_on_path": dict(self.profilers_on_path),
        }


def _is_wsl() -> bool:
    if sys.platform != "linux":
        return False
    # WSL kernels carry "microsoft" in /proc/version and /proc/sys/kernel/osrelease.
    for path in ("/proc/version", "/proc/sys/kernel/osrelease"):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                if "microsoft" in fh.read().lower():
                    return True
        except OSError:
            continue
    return False


def _detect_shell() -> str:
    if sys.platform == "win32":
        # PowerShell sets PSModulePath; cmd.exe does not in the same way.
        if os.environ.get("PSModulePath") or os.environ.get("POWERSHELL_DISTRIBUTION_CHANNEL"):
            return "powershell"
        return "powershell"  # default advice on Windows is PowerShell
    return "posix"


def _machine_model() -> str | None:
    try:
        if sys.platform == "darwin":
            return _sysctl("hw.model")
        if sys.platform == "linux":
            for path in (
                "/sys/devices/virtual/dmi/id/product_name",
                "/sys/class/dmi/id/product_name",
            ):
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                        val = fh.read().strip()
                        if val:
                            return val
                except OSError:
                    continue
    except Exception:
        pass
    return None


def _cpu_brand() -> str | None:
    try:
        if sys.platform == "darwin":
            return _sysctl("machdep.cpu.brand_string")
        if sys.platform == "linux":
            try:
                with open("/proc/cpuinfo", "r", encoding="utf-8", errors="ignore") as fh:
                    for line in fh:
                        if line.lower().startswith("model name"):
                            return line.split(":", 1)[1].strip()
            except OSError:
                pass
    except Exception:
        pass
    return stdplatform.processor() or None


def _sysctl(key: str) -> str | None:
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=2
        )
        val = out.stdout.strip()
        return val or None
    except Exception:
        return None


def _gpu_vendors() -> tuple[str, ...]:
    vendors: list[str] = []
    if shutil.which("nvidia-smi") or os.path.exists("/proc/driver/nvidia"):
        vendors.append("nvidia")
    if shutil.which("rocminfo") or os.path.isdir("/sys/module/amdgpu"):
        vendors.append("amd")
    if sys.platform == "darwin":
        vendors.append("apple")  # Metal is present on every modern Mac
    return tuple(vendors)


@lru_cache(maxsize=1)
def detect() -> PlatformInfo:
    """Detect the host once; cached for the life of the (long-lived) server."""
    profilers = {name: shutil.which(exe) for name, exe in PROFILER_TOOLS.items()}
    return PlatformInfo(
        os=sys.platform,
        is_wsl=_is_wsl(),
        shell=_detect_shell(),
        machine_model=_machine_model(),
        cpu=_cpu_brand(),
        gpu_vendors=_gpu_vendors(),
        profilers_on_path=profilers,
    )
