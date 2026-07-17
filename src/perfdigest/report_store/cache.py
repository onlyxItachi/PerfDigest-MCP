"""Parse-once cache — a report on disk is written once, so rarely parse it twice.

Keyed by (backend name, absolute path, mtime_ns, size): a rewritten report gets
a new key and the stale entry for the same path is dropped immediately. Honest
limit: a rewrite that preserves BOTH byte size and mtime at the filesystem's
timestamp granularity (FAT/exFAT ~2s, some network mounts) would be served
stale — profiler reports are written once and replaced whole, not edited in
place, so the window is negligible in practice, but it is not zero. Bounded
LRU — the agent loop touches a handful of reports, not thousands.

This is a latency/IO optimization only; it must be BEHAVIORALLY invisible.
Callers get a fresh list each hit, and each cached unit's ``metrics`` is a
read-only mapping proxy, so no caller can mutate shared state (units are
frozen dataclasses, but a plain metrics dict would still be writable).
"""

from __future__ import annotations

import os
import stat
import threading
from collections import OrderedDict
from dataclasses import replace
from types import MappingProxyType
from typing import Callable

from perfdigest.core.metrics import NormalizedUnit

_MAX_ENTRIES = 32
_CACHE: "OrderedDict[tuple, list[NormalizedUnit]]" = OrderedDict()
# FastMCP currently runs sync tools inline on the event loop; the lock is
# future-proofing for an SDK that moves them to a threadpool.
_LOCK = threading.Lock()


def cached_units(
    backend_name: str,
    loader: Callable[[str], list[NormalizedUnit]],
    path: str,
) -> list[NormalizedUnit]:
    """``loader(path)``, memoized on the file's identity + content version."""
    abspath = os.path.abspath(path)
    st = os.stat(abspath)
    if stat.S_ISDIR(st.st_mode):
        # Directory-tree reports (criterion's output root): a directory's mtime
        # does NOT change when a nested file is rewritten in place (cargo bench
        # overwrites <bench>/new/estimates.json), so the key below cannot see a
        # re-run and would serve stale units. Caching must stay behaviorally
        # invisible, so directory refs are parsed fresh every time (the trees
        # are small JSON files; the IO saving is not worth the staleness risk).
        # Same mutation-proofing as the cached path — the module invariant
        # ("no caller can mutate shared state") holds for every return.
        return [
            replace(u, metrics=MappingProxyType(dict(u.metrics)))
            for u in loader(path)
        ]
    key = (backend_name, abspath, st.st_mtime_ns, st.st_size)

    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            _CACHE.move_to_end(key)
            return list(hit)

    frozen = [
        replace(u, metrics=MappingProxyType(dict(u.metrics)))
        for u in loader(path)
    ]

    with _LOCK:
        # Drop stale versions of the same file before inserting the fresh one.
        for stale in [k for k in _CACHE if k[0] == backend_name and k[1] == abspath]:
            del _CACHE[stale]
        _CACHE[key] = frozen
        while len(_CACHE) > _MAX_ENTRIES:
            _CACHE.popitem(last=False)
    return list(frozen)


def clear() -> None:
    """Testing hook: forget everything."""
    with _LOCK:
        _CACHE.clear()
