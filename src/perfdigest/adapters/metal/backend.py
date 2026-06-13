"""Apple Metal backend registration (macOS only for capture; digest anywhere)."""

from __future__ import annotations

from perfdigest.adapters import registry
from perfdigest.adapters.metal import mapping, trace_reader
from perfdigest.core.backend import Backend, CapabilityReport
from perfdigest.platform import shell
from perfdigest.platform.detect import PlatformInfo, detect

_PLATFORMS = frozenset({"darwin"})  # Metal capture is macOS-only

METAL_USAGE = (
    "Apple Metal (GPU compute/render passes). Capture a Metal System Trace, then "
    "export the small JSON intermediate perfdigest reads:\n"
    "  xctrace record --template 'Metal System Trace' --launch -- <app>\n"
    "  (then export passes+counters to report.json — see docs)\n"
    "Standard GPU terms carry over; a 'unit' here is an encoder/compute pass."
)


def _probe() -> CapabilityReport:
    info = detect()
    exe = info.profilers_on_path.get("metal")
    if not exe:
        return CapabilityReport(False, "xctrace not found on PATH", None)
    return CapabilityReport(True, "xctrace present", exe)


def _capture_command(target: str, info: PlatformInfo) -> str:
    return shell.join(
        ["xctrace", "record", "--template", "Metal System Trace",
         "--launch", "--", target],
        info.shell,
    )


registry.register(
    Backend(
        name="metal",
        formats=frozenset({"metal-trace", "metal-json", "metal"}),
        suffixes=(".json", ".trace"),
        domain="gpu_pass",
        platforms=_PLATFORMS,
        standard_to_vendor=mapping.STANDARD_TO_VENDOR,
        default_core_set=tuple(mapping.DEFAULT_CORE_SET),
        usage_prompt=METAL_USAGE,
        load_units=trace_reader.load_units,
        raw_metrics=trace_reader.raw_metrics,
        probe=_probe,
        reader_available=lambda: True,  # pure Python JSON
        capture_command=_capture_command,
    )
)
