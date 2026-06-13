"""The usage convention (MCP prompt primitive) — behavioral, not sandboxed.

The value of perfdigest comes from a well-behaved agent following this
convention; the OS does not enforce it. Wording confirmed with the user
2026-06-11 ("strict workflow").
"""

from __future__ import annotations

from perfdigest.adapters.nsight.mapping import DEFAULT_CORE_SET
from perfdigest.server.app import mcp

USAGE_CONVENTION = f"""\
You have perfdigest: token-efficient access to NVIDIA Nsight Compute reports.
Follow this workflow when profiling GPU kernels — do not deviate:

1. Profile ONCE, wide, writing to a file so no profiler output enters your
   context:  ncu --set full -o report.ncu-rep <app>
   NEVER run ncu without -o; its stdout table would flood your context.
2. list_kernels(report_ref, format='ncu-rep') -> pick the hot kernel by
   duration_us.
3. get_metrics(report_ref, format='ncu-rep', kernel=...) -> reason on the
   digest. 'not_available_in_this_export' means the metric was not measured
   — it is NOT zero.
4. Need a specific counter? get_metrics(metrics=[...]) with standard terms:
   {", ".join(DEFAULT_CORE_SET)}.
5. Only if the digest is insufficient: expand(section='dram'|'stall'|...) for
   raw vendor counters. NEVER read or cat the report file or CSV exports
   directly — expand is strictly cheaper and structured.

Interpretation (memory-bound? occupancy-limited?) is YOUR job; the tools only
return clean numbers. One wide profile answers all follow-up questions — do
not re-profile per question.
"""


@mcp.prompt()
def perfdigest_usage() -> str:
    """The perfdigest usage convention to inject at session start."""
    return USAGE_CONVENTION
