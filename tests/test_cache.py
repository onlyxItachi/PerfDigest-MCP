"""Parse-once cache — behaviorally invisible, invalidated by file identity."""

import os

from perfdigest.core.metrics import NormalizedUnit
from perfdigest.report_store import cache


def _unit(name: str) -> NormalizedUnit:
    return NormalizedUnit(name=name, index=0, duration_us=1.0, raw_ref="x")


def test_second_load_of_unchanged_file_is_cached(tmp_path):
    cache.clear()
    f = tmp_path / "r.txt"
    f.write_text("v1")
    calls = []

    def loader(path):
        calls.append(path)
        return [_unit("k")]

    a = cache.cached_units("b1", loader, str(f))
    b = cache.cached_units("b1", loader, str(f))
    assert len(calls) == 1
    assert a[0].name == b[0].name
    assert a is not b  # callers get fresh list objects, never a shared one


def test_rewritten_file_is_reparsed(tmp_path):
    cache.clear()
    f = tmp_path / "r.txt"
    f.write_text("v1")
    calls = []

    def loader(path):
        calls.append(path)
        return [_unit(f"k{len(calls)}")]

    cache.cached_units("b1", loader, str(f))
    f.write_text("v2-longer")  # new size + mtime
    os.utime(f, (1, 1))  # force a distinct mtime even on coarse filesystems
    out = cache.cached_units("b1", loader, str(f))
    assert len(calls) == 2
    assert out[0].name == "k2"


def test_cached_units_are_mutation_proof(tmp_path):
    # A caller mutating a returned unit's metrics must NOT poison later hits:
    # cached units carry read-only mapping proxies.
    import pytest

    cache.clear()
    f = tmp_path / "r.txt"
    f.write_text("v1")

    def loader(path):
        return [NormalizedUnit(name="k", index=0, duration_us=1.0, raw_ref="x",
                               metrics={"m": 1.0})]

    first = cache.cached_units("b1", loader, str(f))
    with pytest.raises(TypeError):
        first[0].metrics["m"] = 999.0  # read-only proxy
    second = cache.cached_units("b1", loader, str(f))
    assert second[0].metrics["m"] == 1.0


def test_backends_do_not_share_entries(tmp_path):
    cache.clear()
    f = tmp_path / "r.txt"
    f.write_text("v1")
    calls = []

    def loader(path):
        calls.append(path)
        return [_unit("k")]

    cache.cached_units("nsight", loader, str(f))
    cache.cached_units("rocm", loader, str(f))
    assert len(calls) == 2  # same path, different backend name -> different parse
