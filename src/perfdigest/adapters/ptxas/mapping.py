"""ptxas resource-usage phrasing -> standard codegen terms.

``registers_per_thread`` deliberately REUSES the nsight standard term, so an
agent can hold an ncu digest and a ptxas digest of the same kernel side by side
and the register number reads identically. The rest is a small codegen-specific
vocabulary: spill traffic and static shared memory are compile-time facts that
no runtime profiler reports directly.
"""

from __future__ import annotations

# standard term -> the ptxas phrasing behind it (the `expand` hint surface)
STANDARD_TO_VENDOR: dict[str, str] = {
    "registers_per_thread": "Used N registers",
    "spill_stores_bytes": "N bytes spill stores",
    "spill_loads_bytes": "N bytes spill loads",
    "stack_frame_bytes": "N bytes stack frame",
    "static_smem_bytes": "N bytes smem (omitted by ptxas when zero)",
    "const_mem_bytes": "sum of N bytes cmem[k] banks",
    "barriers": "used N barriers (absent on older toolkits)",
    "compile_time_ms": "Compile time = N ms",
}

# What get_metrics(metrics=None) returns for a kernel_codegen unit.
DEFAULT_CORE_SET: list[str] = [
    "registers_per_thread",
    "spill_stores_bytes",
    "spill_loads_bytes",
    "stack_frame_bytes",
    "static_smem_bytes",
    "const_mem_bytes",
    "barriers",
    "compile_time_ms",
]
