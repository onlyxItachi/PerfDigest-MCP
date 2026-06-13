"""Platform detection, capture gating, and the read/capture split. No hardware.

These run identically on the Linux/macOS/Windows CI runners and assert the
load-bearing rule: digesting is universal, only capturing is platform-gated.
"""

import sys

from perfdigest.adapters import registry
from perfdigest.core.backend import Backend, CapabilityReport
from perfdigest.platform import capabilities, shell
from perfdigest.platform.detect import PlatformInfo, detect


def _info(os_name="linux", is_wsl=False, profilers=None) -> PlatformInfo:
    return PlatformInfo(
        os=os_name,
        is_wsl=is_wsl,
        shell="powershell" if os_name == "win32" else "posix",
        machine_model="TestBox",
        cpu="TestCPU",
        gpu_vendors=(),
        profilers_on_path=profilers or {},
    )


def test_detect_reports_this_host_os():
    assert detect().os == sys.platform


def test_capability_summary_shape_and_digest_universality():
    summ = capabilities.capability_summary()
    assert set(summ) == {"host", "can_digest", "can_capture_here"}
    digest_backends = {d["backend"] for d in summ["can_digest"]}
    # Every pure-Python backend is digestable here regardless of local hardware.
    assert {"nsight_csv", "rocm", "metal"} <= digest_backends


def test_os_gate_refuses_capture_on_wrong_platform():
    metal = registry.get_by_name("metal")
    report = capabilities.capture_report(metal, _info(os_name="linux"))
    assert report.available is False
    assert "darwin" in report.reason  # tells you where it CAN capture
    # ...but it is still digestable (universality), independent of capture.
    assert metal in registry.readable_backends()


def test_wsl_attaches_pmu_caveat_when_perf_is_capturable():
    # Construct a linux_perf-named backend whose probe is forced available, so
    # the WSL note logic is exercised deterministically on any CI host.
    fake = Backend(
        name="linux_perf",
        formats=frozenset({"perf-x"}),
        suffixes=(".txt",),
        domain="cpu_function",
        platforms=frozenset({"linux"}),
        standard_to_vendor={"ipc": "instructions/cycles"},
        default_core_set=("ipc",),
        usage_prompt="",
        load_units=lambda p: [],
        raw_metrics=lambda p, i, s: {},
        probe=lambda: CapabilityReport(True, "perf present", "/usr/bin/perf"),
    )
    report = capabilities.capture_report(fake, _info(os_name="linux", is_wsl=True))
    assert report.available is True
    assert any("WSL2" in n for n in report.notes)


def test_shell_quoting_posix_vs_powershell():
    assert shell.quote("a b", "posix") == "'a b'"
    assert shell.quote("a b", "powershell") == "'a b'"
    assert shell.quote("plain", "powershell") == "plain"
    assert shell.join(["ncu", "-o", "r r.rep", "./app"], "posix") == "ncu -o 'r r.rep' ./app"


def test_wsl_path_translation_roundtrip():
    assert shell.wsl_to_windows_path("/mnt/c/Users/x") == "C:\\Users\\x"
    assert shell.windows_to_wsl_path("C:\\Users\\x") == "/mnt/c/Users/x"
