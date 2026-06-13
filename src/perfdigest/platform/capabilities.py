"""Capture gating (tier-2 only) + the capability summary surface.

The one rule this layer enforces: a backend is *capturable here* only if the
host OS is in ``backend.platforms`` AND the backend's own ``probe()`` passes
(tool present, etc.). Reading/digesting is never gated here — ``can_digest`` is
just ``registry.readable_backends()``.

This is what hard-restricts impossible combinations (running ``ncu`` on a Mac,
Metal on Linux, hardware PMU counters under WSL2) so the agent never spends
context on a capture that cannot run, while still being able to digest a report
captured anywhere else.
"""

from __future__ import annotations

from perfdigest.adapters import registry
from perfdigest.core.backend import Backend, CapabilityReport
from perfdigest.platform.detect import PlatformInfo, detect


def capture_report(backend: Backend, info: PlatformInfo | None = None) -> CapabilityReport:
    """Whether ``backend`` can *capture* on this host, with a clear reason."""
    info = info or detect()

    if info.os not in backend.platforms:
        plats = ", ".join(sorted(backend.platforms))
        return CapabilityReport(
            available=False,
            reason=(
                f"{backend.name} captures only on [{plats}]; this host is "
                f"{info.os}. You can still DIGEST a {backend.name} report "
                "produced elsewhere (e.g. a CI/remote GPU runner)."
            ),
            tool=None,
        )

    report = backend.probe()

    # WSL nuance: hardware PMU counters are typically unavailable under WSL2.
    if backend.name == "linux_perf" and info.is_wsl and report.available:
        report = CapabilityReport(
            available=report.available,
            reason=report.reason,
            tool=report.tool,
            notes=report.notes
            + (
                "WSL2: hardware PMU counters are usually unavailable; expect "
                "software-only events (task-clock, page-faults) unless a custom "
                "kernel exposes the PMU.",
            ),
        )
    return report


def capturable_backends(
    info: PlatformInfo | None = None,
) -> list[tuple[Backend, CapabilityReport]]:
    """Backends that can capture on this host, each with its report."""
    info = info or detect()
    out = []
    for backend in registry.all_backends():
        report = capture_report(backend, info)
        if report.available:
            out.append((backend, report))
    return out


def capability_summary(info: PlatformInfo | None = None) -> dict:
    """The payload behind the ``platform_capabilities`` MCP tool.

    ``can_digest`` is universal (any backend with an installed reader);
    ``can_capture_here`` is the gated tier-2 list. The split is the whole point:
    a host with no GPU can still be a first-class *digester*.
    """
    info = info or detect()
    digest = [
        {"backend": b.name, "domain": b.domain, "formats": sorted(b.formats)}
        for b in registry.readable_backends()
    ]
    capture = []
    for backend in registry.all_backends():
        report = capture_report(backend, info)
        capture.append(
            {
                "backend": backend.name,
                "domain": backend.domain,
                "capturable": report.available,
                "reason": report.reason,
                "tool": report.tool,
                "notes": list(report.notes),
            }
        )
    return {
        "host": info.to_dict(),
        "can_digest": digest,  # tier-1, universal
        "can_capture_here": capture,  # tier-2, gated
    }
