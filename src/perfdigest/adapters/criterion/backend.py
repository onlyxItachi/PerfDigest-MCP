"""criterion backend registration (Rust benchmark wall-time statistics)."""

from __future__ import annotations

from perfdigest.adapters import registry
from perfdigest.adapters.criterion import criterion_reader, mapping
from perfdigest.core.backend import Backend, CapabilityReport
from perfdigest.platform import shell
from perfdigest.platform.detect import PlatformInfo, detect

_PLATFORMS = frozenset({"linux", "darwin", "win32"})  # cargo runs everywhere

CRITERION_USAGE = (
    "criterion digest (Rust benchmarks). report_ref is the output DIRECTORY "
    "cargo bench writes — target/criterion — not a file (format criterion). "
    "Units are benchmarks named by their directory path relative to that root; "
    "a benchmark_group nests one level ('fib/fib_20' vs 'fib_plain'). Numbers "
    "are wall-time statistics in NANOSECONDS from criterion's own estimator "
    "(mean/median/std_dev/median_abs_dev/slope point estimates), not raw "
    "samples. slope is honestly absent on flat-sampling-mode runs. To compare "
    "runs, use compare_metrics with the SAME bench names — but copy the "
    "criterion directory aside before re-running, because cargo bench "
    "overwrites target/criterion in place."
)


def _probe() -> CapabilityReport:
    info = detect()
    exe = info.profilers_on_path.get("criterion")
    if not exe:
        return CapabilityReport(
            False, "cargo not found on PATH (Rust toolchain needed)", None
        )
    return CapabilityReport(
        True,
        "cargo present (criterion must be a dev-dependency of the crate under bench)",
        exe,
        notes=(
            "Results land in target/criterion/<bench>/new/estimates.json; pass "
            "the target/criterion DIRECTORY as report_ref. For a before/after "
            "loop, copy that directory aside between runs — cargo bench "
            "overwrites it in place.",
        ),
    )


def _capture_command(target: str, info: PlatformInfo) -> str:
    bench_cmd = shell.join(
        ["cargo", "bench", "--manifest-path", f"{target}/Cargo.toml"], info.shell
    )
    return (
        f"{bench_cmd}  # writes {target}/target/criterion — "
        "pass that directory as report_ref (format criterion)"
    )


registry.register(
    Backend(
        name="criterion",
        formats=frozenset({"criterion", "criterion-json"}),
        # The report_ref is a directory tree, so file-suffix completion never
        # applies (resolution returns the directory as-is); no suffixes claimed.
        suffixes=(),
        domain="benchmark",
        platforms=_PLATFORMS,
        standard_to_vendor=mapping.STANDARD_TO_VENDOR,
        default_core_set=tuple(mapping.DEFAULT_CORE_SET),
        usage_prompt=CRITERION_USAGE,
        load_units=criterion_reader.load_units,
        raw_metrics=criterion_reader.raw_metrics,
        probe=_probe,
        reader_available=lambda: True,  # pure Python
        capture_command=_capture_command,
        report_is_directory=True,
    )
)
