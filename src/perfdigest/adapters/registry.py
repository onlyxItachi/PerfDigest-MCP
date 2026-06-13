"""Backend registry — format -> Backend dispatch, replacing hardcoded NVIDIA.

Adapters call ``register()`` at import time (a side effect, like the FastMCP
tool decorators). The server resolves ``get_backend(format)`` and never imports a
concrete reader. ``readable_backends()`` is the tier-1 (digest) view and is NOT
platform-gated; capture gating lives in ``platform.capabilities``.
"""

from __future__ import annotations

from perfdigest.core.backend import Backend

# name -> Backend, and format -> Backend (a format resolves to exactly one).
_BY_NAME: dict[str, Backend] = {}
_BY_FORMAT: dict[str, Backend] = {}


def register(backend: Backend) -> Backend:
    """Register a backend and index every format it reads. Idempotent by name."""
    _BY_NAME[backend.name] = backend
    for fmt in backend.formats:
        existing = _BY_FORMAT.get(fmt)
        if existing is not None and existing.name != backend.name:
            raise ValueError(
                f"format {fmt!r} already claimed by backend {existing.name!r}; "
                f"{backend.name!r} cannot also claim it"
            )
        _BY_FORMAT[_norm(fmt)] = backend
    return backend


def _norm(fmt: str) -> str:
    return fmt.lower().lstrip(".").strip()


def get_backend(format: str) -> Backend:
    """Resolve a ``format`` to its Backend, or raise with the known formats."""
    backend = _BY_FORMAT.get(_norm(format))
    if backend is None:
        known = ", ".join(sorted(_BY_FORMAT)) or "(none registered)"
        raise ValueError(
            f"unsupported format {format!r}. Known formats: {known}. "
            "Pass the format you produced (e.g. 'ncu-rep' for `ncu -o`, "
            "'perf-script' for `perf script`, 'rocprof-csv', 'metal-trace')."
        )
    return backend


def get_by_name(name: str) -> Backend:
    """Resolve a backend by its stable name, or raise with the known names."""
    backend = _BY_NAME.get(name)
    if backend is None:
        known = ", ".join(sorted(_BY_NAME)) or "(none registered)"
        raise ValueError(f"unknown backend {name!r}. Known backends: {known}.")
    return backend


def all_backends() -> list[Backend]:
    """Every registered backend, sorted by name (deterministic)."""
    return [_BY_NAME[n] for n in sorted(_BY_NAME)]


def readable_backends() -> list[Backend]:
    """Tier-1: backends whose reader dependency imports here. NOT OS-gated."""
    return [b for b in all_backends() if _safe(b.reader_available)]


def _safe(fn) -> bool:
    try:
        return bool(fn())
    except Exception:
        return False
