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
    """Register a backend and index every format it reads. Idempotent by name.

    Collision detection and storage BOTH use the normalized key — checking the
    raw string while storing the normalized one would let 'NCU-REP' silently
    overwrite the backend that registered 'ncu-rep' (issue #2).
    """
    # Validate EVERY format key first, then commit name + formats together —
    # a rejected backend must leave no trace (no ghost in _BY_NAME, no partial
    # format claims dependent on frozenset iteration order).
    keys = []
    for fmt in backend.formats:
        key = _norm(fmt)
        existing = _BY_FORMAT.get(key)
        if existing is not None and existing.name != backend.name:
            raise ValueError(
                f"format {fmt!r} (normalized {key!r}) already claimed by backend "
                f"{existing.name!r}; {backend.name!r} cannot also claim it"
            )
        keys.append(key)
    _BY_NAME[backend.name] = backend
    for key in keys:
        _BY_FORMAT[key] = backend
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
    return [b for b in all_backends() if reader_status(b)[0]]


def reader_status(backend: Backend) -> tuple[bool, str | None]:
    """(readable, diagnostic) — a False is no longer silent (issue #4).

    Distinguishes 'reader dependency not importable / probe said no' (False,
    reason) from 'probe itself crashed' (False, the exception) so a broken
    optional wheel is diagnosable instead of looking identical to 'not
    installed'. True always pairs with None.
    """
    try:
        if bool(backend.reader_available()):
            return True, None
        return False, "reader_available() returned False (optional dependency not installed?)"
    except Exception as exc:  # noqa: BLE001 — the point is to surface it
        return False, f"reader_available() raised {type(exc).__name__}: {exc}"


def reader_issues() -> list[dict]:
    """Backends that cannot digest here, each with its diagnostic."""
    out = []
    for b in all_backends():
        ok, why = reader_status(b)
        if not ok:
            out.append({"backend": b.name, "formats": sorted(b.formats), "issue": why})
    return out
