"""Chrome-trace backend registration (torch/Kineto + any Chrome-trace emitter).

Digest is pure Python and universal. "Capture" here is in-code, not a CLI: the
advisory tells the agent how to wrap the hot loop with ``torch.profiler`` and
export the artifact. We bind to the exported file, never to torch's API.
"""

from __future__ import annotations

import importlib.util

from perfdigest.adapters import registry
from perfdigest.adapters.chrome_trace import mapping, trace_reader
from perfdigest.core.backend import Backend, CapabilityReport
from perfdigest.platform.detect import PlatformInfo

_PLATFORMS = frozenset({"linux", "darwin", "win32"})  # torch captures everywhere

CHROME_TRACE_USAGE = (
    "Chrome-trace digest (torch/Kineto, JAX, clang -ftime-trace...). Capture in "
    "code, exporting to a FILE (never print the profiler table):\n"
    "  with torch.profiler.profile(activities=[ProfilerActivity.CPU, "
    "ProfilerActivity.CUDA]) as prof:\n"
    "      <hot loop>\n"
    "  prof.export_chrome_trace('report.torch-trace.json')   (format torch-trace)\n"
    "Units aggregate by (cat, name): framework ops (aten::mm, calls/total/avg) AND "
    "device-side kernels/memcpys (domain gpu_kernel) in one report. Totals overlap "
    "across nested spans (aten::matmul contains aten::mm) — framework hierarchies "
    "report that way, so per-unit totals are NOT additive and coverage% is "
    "indicative only. Units tagged [Trace] (the whole capture window) or "
    "[overhead] are profiler BOOKKEEPING, not workload — discount them when "
    "ranking. When comparing two traces whose capture lengths differ, compare "
    "avg_time_us (calls-normalized), not total_time_us. For hardware counters on "
    "a kernel seen here, profile it with ncu and digest via format ncu-rep."
)


def _probe() -> CapabilityReport:
    if importlib.util.find_spec("torch") is None:
        return CapabilityReport(
            False,
            "torch not importable in this environment (any Chrome-trace emitter "
            "works — torch is just the usual producer); a trace produced "
            "elsewhere is still digestible here",
            None,
        )
    return CapabilityReport(
        True,
        "torch importable — capture happens in-code via torch.profiler",
        "torch",
        notes=(
            "Keep traces small: profile a few steps (schedule/warmup), not the "
            "whole run; with_stack=False unless you need Python stacks.",
        ),
    )


def _capture_command(target: str, info: PlatformInfo) -> str:
    return (
        f"python {target}  # wrap the hot loop: "
        "with torch.profiler.profile(activities=[ProfilerActivity.CPU, "
        "ProfilerActivity.CUDA]) as prof: ... ; "
        "prof.export_chrome_trace('report.torch-trace.json')"
    )


registry.register(
    Backend(
        name="chrome_trace",
        formats=frozenset({"torch-trace", "chrome-trace"}),
        suffixes=(".torch-trace.json", ".trace.json", ".json"),
        domain="framework_op",  # per-unit: device kernels come back as gpu_kernel
        platforms=_PLATFORMS,
        standard_to_vendor=mapping.STANDARD_TO_VENDOR,
        default_core_set=tuple(mapping.DEFAULT_CORE_SET),
        usage_prompt=CHROME_TRACE_USAGE,
        load_units=trace_reader.load_units,
        raw_metrics=trace_reader.raw_metrics,
        probe=_probe,
        reader_available=lambda: True,  # pure Python
        capture_command=_capture_command,
    )
)
