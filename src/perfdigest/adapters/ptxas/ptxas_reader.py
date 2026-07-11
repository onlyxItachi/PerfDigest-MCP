"""Read ``ptxas -v`` resource-usage text -> NormalizedUnit (kernel_codegen).

The CODEGEN layer: runtime counters (ncu) say WHAT is slow; this artifact says
WHY the code became what it is — registers, spills, static shared memory. It is
produced at COMPILE time (``nvcc -Xptxas -v``, output on stderr), so it needs no
GPU at all: a pure static digest that works in any CI job.

Shape parsed (CUDA 12/13 ``ptxas info`` lines; tolerant of the older variants
that lack ``barriers`` / ``cumulative stack size``):

    ptxas info : Compiling entry function '_Z4addPfl' for 'sm_89'
    ptxas info : Function properties for _Z4addPfl
        312 bytes stack frame, 660 bytes spill stores, 656 bytes spill loads
    ptxas info : Used 24 registers, used 0 barriers, 4224 bytes smem, 372 bytes cmem[0]
    ptxas info : Compile time = 21.988 ms

Names stay MANGLED as ptxas reports them ('_Z17true_spill_kernelPKfPfi') — the
plain kernel name is a substring, so ``kernel='true_spill_kernel'`` still matches.

Honesty rule, ptxas edition: the ``Used ...`` line is a complete enumeration of
nonzero resources — ptxas OMITS a component that is zero (no ``bytes smem`` means
zero static smem). So within a parsed ``Used`` line an omitted component is a
GENUINE 0.0, while a kernel whose ``Used`` line is missing entirely keeps every
resource ``None`` (not measured). ``barriers`` is the exception: old toolkits
never printed it even when nonzero, so its absence stays ``None``.
"""

from __future__ import annotations

import re
from typing import Any

from perfdigest.core.metrics import DOMAIN_KERNEL_CODEGEN, NormalizedUnit

_ENTRY = re.compile(r"Compiling entry function '([^']+)' for '([^']+)'")
_PROPS = re.compile(r"Function properties for (\S+)")
_FRAME = re.compile(
    r"(\d+)\s+bytes stack frame,\s*(\d+)\s+bytes spill stores,\s*(\d+)\s+bytes spill loads"
)
_USED = re.compile(r"Used (\d+) registers")
_BARRIERS = re.compile(r"used (\d+) barriers")
_SMEM = re.compile(r"(\d+)\s+bytes smem")
_CMEM = re.compile(r"(\d+)\s+bytes cmem\[\d+\]")
_COMPILE_TIME = re.compile(r"Compile time = ([\d.]+) ms")


def _parse(report_path: str) -> list[dict[str, Any]]:
    """-> one raw record per entry function, in file order."""
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    props_for: str | None = None

    with open(report_path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            m = _ENTRY.search(line)
            if m:
                current = {"name": m.group(1), "arch": m.group(2)}
                records.append(current)
                props_for = None
                continue
            if current is None:
                continue  # preamble: 'Overriding ...', '0 bytes gmem'

            m = _PROPS.search(line)
            if m:
                props_for = m.group(1)
                continue

            m = _FRAME.search(line)
            if m and props_for == current["name"]:
                current["stack_frame_bytes"] = float(m.group(1))
                current["spill_stores_bytes"] = float(m.group(2))
                current["spill_loads_bytes"] = float(m.group(3))
                continue

            m = _USED.search(line)
            if m:
                current["registers"] = float(m.group(1))
                # Complete enumeration: an omitted component IS zero here.
                mb = _BARRIERS.search(line)
                if mb:  # old toolkits never print barriers: absent stays None
                    current["barriers"] = float(mb.group(1))
                ms = _SMEM.search(line)
                current["static_smem_bytes"] = float(ms.group(1)) if ms else 0.0
                cmem = [float(v) for v in _CMEM.findall(line)]
                current["const_mem_bytes"] = sum(cmem) if cmem else 0.0
                continue

            m = _COMPILE_TIME.search(line)
            if m:
                current["compile_time_ms"] = float(m.group(1))
    return records


def load_units(report_path: str) -> list[NormalizedUnit]:
    units: list[NormalizedUnit] = []
    for index, rec in enumerate(_parse(report_path)):
        metrics: dict[str, float | None] = {
            "registers_per_thread": rec.get("registers"),
            "spill_stores_bytes": rec.get("spill_stores_bytes"),
            "spill_loads_bytes": rec.get("spill_loads_bytes"),
            "stack_frame_bytes": rec.get("stack_frame_bytes"),
            "static_smem_bytes": rec.get("static_smem_bytes"),
            "const_mem_bytes": rec.get("const_mem_bytes"),
            "barriers": rec.get("barriers"),
            "compile_time_ms": rec.get("compile_time_ms"),
        }
        units.append(
            NormalizedUnit(
                name=rec["name"],
                index=index,
                duration_us=None,  # a compile artifact has no runtime duration
                raw_ref=report_path,
                metrics=metrics,
                domain=DOMAIN_KERNEL_CODEGEN,
            )
        )
    return units


def raw_metrics(report_path: str, kernel_index: int, name_filter: str) -> dict[str, Any]:
    """Raw parsed fields (including the non-numeric ``arch``) for ``expand``."""
    records = _parse(report_path)
    if kernel_index >= len(records):
        raise IndexError(f"unit index {kernel_index} not present in {report_path}")
    wanted = None if name_filter.lower() == "all" else name_filter.lower()
    out: dict[str, Any] = {}
    for key, value in records[kernel_index].items():
        if value is None:
            continue
        if wanted is not None and wanted not in key.lower():
            continue
        out[key] = value
    return out
