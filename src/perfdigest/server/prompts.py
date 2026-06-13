"""The usage convention (MCP prompt primitive) — behavioral, not sandboxed.

The value comes from a well-behaved agent following this convention; the OS does
not enforce it. The convention now states the **two operations explicitly** —
digesting is always available, capturing must be verified first — and composes
each registered backend's own one-liner so the agent learns the whole surface.
"""

from __future__ import annotations

from perfdigest.adapters import registry
from perfdigest.server.app import mcp

_HEADER = """\
You have perfdigest: token-efficient access to performance-profiler reports across
backends (NVIDIA, AMD HIP, CPU perf, Apple Metal). It is a translator, not a judge —
it returns clean numbers; deciding "memory-bound?"/"occupancy-limited?" is YOUR job.

perfdigest has TWO operations — keep them separate:

A) DIGEST (read a report) — ALWAYS available on every machine, regardless of local
   hardware. A report captured anywhere (a CI/remote GPU runner, a teammate) can be
   digested here. This is how you do e.g. CUDA work on a Mac: capture on a GPU
   runner, digest the report locally.
     1. list_kernels(report_ref, format) -> pick the hottest unit by duration_us.
     2. get_metrics(report_ref, format, kernel=...) -> reason on the digest.
        'not_available_in_this_export' means NOT MEASURED — it is NOT zero.
     3. Only if insufficient: expand(section=...) for raw vendor counters. NEVER
        read/cat a report file directly — expand is strictly cheaper and structured.

B) CAPTURE (produce a report) — must be VERIFIED for this host first.
     1. platform_capabilities() -> see can_capture_here (gated) vs can_digest (all).
     2. suggest_profile_command(backend, target) -> the correct, platform-aware
        invocation, or a refusal that tells you to capture elsewhere then digest here.
   Always capture ONCE, wide, writing to a file (never let profiler stdout flood
   your context). One wide capture answers all follow-up get_metrics questions.

Per-backend conventions (pass `format` = what you produced):"""


def build_usage_convention() -> str:
    lines = [_HEADER]
    for b in registry.readable_backends():
        fmts = ", ".join(sorted(b.formats))
        lines.append(f"\n• {b.name} ({b.domain}) — formats: {fmts}\n  {b.usage_prompt}")
    return "\n".join(lines)


@mcp.prompt()
def perfdigest_usage() -> str:
    """The perfdigest usage convention to inject at session start."""
    return build_usage_convention()
