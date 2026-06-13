"""Registry dispatch — format -> backend, the thin-shell contract. No hardware."""

import pytest

from perfdigest.adapters import registry


def test_all_four_v1_backends_registered():
    names = {b.name for b in registry.all_backends()}
    # nsight (native) + nsight_csv (pure-Python NVIDIA) + the 3 new backends.
    assert {"nsight", "nsight_csv", "linux_perf", "rocm", "metal"} <= names


@pytest.mark.parametrize(
    "fmt,expected",
    [
        ("ncu-rep", "nsight"),
        ("ncu-csv", "nsight_csv"),
        ("perf-stat-json", "linux_perf"),
        ("PERF-REPORT", "linux_perf"),  # case-insensitive
        ("rocprof-csv", "rocm"),
        ("metal-trace", "metal"),
        (".metal-json", "metal"),  # leading-dot tolerated
    ],
)
def test_format_resolves_to_backend(fmt, expected):
    assert registry.get_backend(fmt).name == expected


def test_unknown_format_lists_known_formats():
    with pytest.raises(ValueError) as exc:
        registry.get_backend("vtune-xml")
    msg = str(exc.value)
    assert "vtune-xml" in msg and "ncu-rep" in msg  # error names known formats


def test_get_by_name_and_unknown():
    assert registry.get_by_name("rocm").domain == "gpu_kernel"
    with pytest.raises(ValueError):
        registry.get_by_name("no_such_backend")


def test_domains_are_correct():
    by = {b.name: b.domain for b in registry.all_backends()}
    assert by["nsight"] == "gpu_kernel"
    assert by["linux_perf"] == "cpu_function"
    assert by["metal"] == "gpu_pass"


def test_each_backend_has_a_default_core_set():
    for b in registry.all_backends():
        assert b.default_core_set, f"{b.name} has empty default core set"


def test_readable_backends_are_universal():
    # Pure-Python readers are always readable regardless of OS/GPU; this is the
    # tier-1 universality guarantee (a Mac can digest an AMD/NVIDIA report).
    readable = {b.name for b in registry.readable_backends()}
    assert {"nsight_csv", "rocm", "metal", "linux_perf"} <= readable
