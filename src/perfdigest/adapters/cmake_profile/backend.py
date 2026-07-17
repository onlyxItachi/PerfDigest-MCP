"""cmake configure-profile backend registration (BuildDigest: configure step).

Digest is pure Python (it reuses chrome_trace's parsing with the B/E-pair fold,
see ``cmake_profile_reader.py``) and universal. "Capture" is two cmake flags on
the configure invocation (cmake >= 3.18), no separate profiler tool.
"""

from __future__ import annotations

from perfdigest.adapters import registry
from perfdigest.adapters.cmake_profile import cmake_profile_reader, mapping
from perfdigest.core.backend import Backend, CapabilityReport
from perfdigest.platform import shell
from perfdigest.platform.detect import PlatformInfo, detect

_PLATFORMS = frozenset({"linux", "darwin", "win32"})  # cmake runs everywhere

CMAKE_PROFILE_USAGE = (
    "cmake configure-profile digest (where CONFIGURE time goes — find_package "
    "probes, try_compile, include chains, user macros — not compile time; see "
    "clang-time-trace/ninja-log for the build itself). Capture (cmake >= 3.18):\n"
    "  cmake --profiling-format=google-trace --profiling-output=cmake-profile.json"
    " -S <src> -B <build>   (format cmake-profile)\n"
    "Units aggregate by command/macro/function NAME across every call site "
    "('include' calls=28 = every include() that ran); the per-call file:line and "
    "arguments are expand() data (arg:location, arg:functionArgs). Frames NEST "
    "('configure' and 'generate' are cmake's whole-step frames containing "
    "everything else; find_package contains the includes/try_compiles it "
    "triggers), so total_time_us OVERLAPS across units — never additive, and "
    "summarize coverage is indicative only, same as every Chrome-trace backend. "
    "Expect 'configure' to rank first: it IS the whole step, not a hotspot. A "
    "cold first configure is dominated by compiler-detection try_compile; "
    "re-configure profiles (cached checks) show your own CMakeLists costs."
)


def _probe() -> CapabilityReport:
    info = detect()
    exe = info.profilers_on_path.get("cmake_profile")
    if not exe:
        return CapabilityReport(False, "cmake not found on PATH", None)
    return CapabilityReport(
        True,
        "cmake present — profiling is built into the configure step, no extra tool",
        exe,
        notes=(
            "--profiling-format=google-trace needs cmake >= 3.18 (an older "
            "cmake rejects the flag loudly — nothing silently missing). Profile "
            "a RE-configure of a warm build dir to see your own CMakeLists "
            "costs; a cold first configure is dominated by compiler detection.",
        ),
    )


def _capture_command(target: str, info: PlatformInfo) -> str:
    return shell.join(
        [
            "cmake",
            "--profiling-format=google-trace",
            "--profiling-output=cmake-profile.json",
            "-S", target,
            "-B", f"{target}/build",
        ],
        info.shell,
    )


registry.register(
    Backend(
        name="cmake_profile",
        formats=frozenset({"cmake-profile", "cmake-trace"}),
        suffixes=(".json",),
        domain="build_phase",
        platforms=_PLATFORMS,
        standard_to_vendor=mapping.STANDARD_TO_VENDOR,
        default_core_set=tuple(mapping.DEFAULT_CORE_SET),
        usage_prompt=CMAKE_PROFILE_USAGE,
        load_units=cmake_profile_reader.load_units,
        raw_metrics=cmake_profile_reader.raw_metrics,
        probe=_probe,
        reader_available=lambda: True,  # pure Python
        capture_command=_capture_command,
    )
)
