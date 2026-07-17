"""cargo build-diagnostics backend registration (BuildDigest: rustc/cargo).

Digest is pure Python NDJSON parsing — universal, any host reads a stream
saved on any CI box. Capture is a cargo flag, not a separate profiler:
``cargo build --message-format=json`` prints one JSON object per line to
stdout (human progress goes to stderr), so the capture is just a redirect.
"""

from __future__ import annotations

from perfdigest.adapters import registry
from perfdigest.adapters.cargo_diag import cargo_diag_reader, mapping
from perfdigest.core.backend import Backend, CapabilityReport
from perfdigest.platform.detect import PlatformInfo, detect

_PLATFORMS = frozenset({"linux", "darwin", "win32"})  # cargo runs everywhere

CARGO_DIAG_USAGE = (
    "cargo build-diagnostics digest (rustc errors/warnings per crate — where "
    "the diagnostics ARE, not how long the build took). Capture is a "
    "redirect, no profiler tool:\n"
    "  cargo build --message-format=json > build.cargo-diag.jsonl 2>/dev/null"
    "   (format cargo-diag, alias cargo-json; messages go to STDOUT)\n"
    "Units are CRATES named name@version (parsed from cargo's package_id), "
    "one per package seen in the stream; a crate that FAILED to compile "
    "still appears (it has messages, just no artifact). HONESTY NUANCES:\n"
    "  1. duration_us is None for EVERY unit — cargo's JSON stream carries "
    "no timing at all, so summarize_report ranks in file order here; for "
    "build TIMING use format ninja-log (step timings) or clang-time-trace "
    "(compiler phases) instead.\n"
    "  2. errors/warnings/notes_helps of 0.0 are GENUINE zeros (the stream "
    "is a complete enumeration of this build's diagnostics, like the ptxas "
    "'Used ...' line), not absent data.\n"
    "The digest tells you WHERE diagnostics concentrate: expand() adds "
    "per-level-per-file counts ('warning@src/lib.rs' -> 2.0), per-lint/"
    "E-code counts ('code:E0308' -> 1.0), and the report-level "
    "build_finished_success bool. Reading actual diagnostic TEXT is your "
    "editor/compiler's job, not perfdigest's — run `cargo build` without "
    "--message-format for the rendered messages once you know where to look."
)


def _probe() -> CapabilityReport:
    info = detect()
    exe = info.profilers_on_path.get("cargo_diag")
    if not exe:
        return CapabilityReport(False, "cargo not found on PATH (Rust toolchain needed)", None)
    return CapabilityReport(
        True,
        "cargo present — --message-format=json is a cargo flag, no extra tool",
        exe,
        notes=(
            "The JSON stream is on STDOUT and human progress on STDERR — "
            "redirect stdout to the report file; a clean (fully cached) "
            "rebuild replays cached warnings, so the stream stays a complete "
            "diagnostic enumeration even when nothing recompiles.",
        ),
    )


def _capture_command(target: str, info: PlatformInfo) -> str:
    err_sink = "NUL" if info.os == "win32" else "/dev/null"
    return (
        f"cargo build --message-format=json > {target} 2> {err_sink}"
        "  # run from the workspace root so span file_names stay relative"
    )


registry.register(
    Backend(
        name="cargo_diag",
        formats=frozenset({"cargo-diag", "cargo-json"}),
        suffixes=(".cargo-diag.jsonl", ".jsonl"),
        domain="build_diag",
        platforms=_PLATFORMS,
        standard_to_vendor=mapping.STANDARD_TO_VENDOR,
        default_core_set=tuple(mapping.DEFAULT_CORE_SET),
        usage_prompt=CARGO_DIAG_USAGE,
        load_units=cargo_diag_reader.load_units,
        raw_metrics=cargo_diag_reader.raw_metrics,
        probe=_probe,
        reader_available=lambda: True,  # pure Python
        capture_command=_capture_command,
    )
)
