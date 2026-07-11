"""Parse-once cache — a report on disk is immutable, so never parse it twice.

Keyed by (backend name, absolute path, mtime_ns, size): a rewritten report gets
a new key and the stale entry for the same path is dropped immediately, so the
cache can never serve data from a file that changed underneath it. Bounded LRU —
the agent loop touches a handful of reports, not thousands.

This is a latency/IO optimization only; it must be BEHAVIORALLY invisible.
Callers get a fresh list object each hit (units themselves are frozen
dataclasses, safe to share).
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Callable

from perfdigest.core.metrics import NormalizedUnit

_MAX_ENTRIES = 32
_CACHE: "OrderedDict[tuple, list[NormalizedUnit]]" = OrderedDict()


def cached_units(
    backend_name: str,
    loader: Callable[[str], list[NormalizedUnit]],
    path: str,
) -> list[NormalizedUnit]:
    """``loader(path)``, memoized on the file's identity + content version."""
    abspath = os.path.abspath(path)
    st = os.stat(abspath)
    key = (backend_name, abspath, st.st_mtime_ns, st.st_size)

    hit = _CACHE.get(key)
    if hit is not None:
        _CACHE.move_to_end(key)
        return list(hit)

    units = loader(path)

    # Drop stale versions of the same file before inserting the fresh one.
    for stale in [k for k in _CACHE if k[0] == backend_name and k[1] == abspath]:
        del _CACHE[stale]
    _CACHE[key] = list(units)
    while len(_CACHE) > _MAX_ENTRIES:
        _CACHE.popitem(last=False)
    return list(units)


def clear() -> None:
    """Testing hook: forget everything."""
    _CACHE.clear()
