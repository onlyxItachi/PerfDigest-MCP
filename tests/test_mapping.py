"""Mapping + converter unit tests — pure, no fixture or GPU.

Guards the most-changed file: the vendor->standard dict must stay consistent,
the default core set must be fully resolvable (or get_metrics(None) would emit
spurious 'not_available' for a default term), and the unit converters must
normalize each unit ncu can report.
"""

import pytest

from perfdigest.adapters.nsight.mapping import (
    CONVERTERS,
    DEFAULT_CORE_SET,
    METRIC_MAP,
    STANDARD_TO_VENDOR,
    to_gbps,
    to_microseconds,
)


def test_metric_map_has_no_duplicate_standard_terms():
    # If two vendor names mapped to one standard term, STANDARD_TO_VENDOR would
    # silently drop one and the reverse lookup would be lossy.
    terms = list(METRIC_MAP.values())
    assert len(terms) == len(set(terms))


def test_standard_to_vendor_is_exact_inverse():
    assert STANDARD_TO_VENDOR == {v: k for k, v in METRIC_MAP.items()}
    for vendor, term in METRIC_MAP.items():
        assert STANDARD_TO_VENDOR[term] == vendor


def test_every_default_core_term_is_resolvable():
    # Each default term must be a known standard term, or the special
    # 'duration_us'. Otherwise get_metrics(metrics=None) returns absence for a
    # metric we promised is in the default package.
    for term in DEFAULT_CORE_SET:
        assert term == "duration_us" or term in STANDARD_TO_VENDOR, term


def test_default_core_set_has_no_duplicates():
    assert len(DEFAULT_CORE_SET) == len(set(DEFAULT_CORE_SET))


def test_converters_only_cover_unit_bearing_terms():
    # duration and throughput need unit normalization; the rest are already
    # percentages / counts and must NOT be silently scaled.
    assert set(CONVERTERS) == {"duration_us", "mem_throughput_gbps"}


@pytest.mark.parametrize(
    "value,unit,expected",
    [
        (1000.0, "nsecond", 1.0),
        (1000.0, "ns", 1.0),
        (5.0, "usecond", 5.0),
        (5.0, "us", 5.0),
        (2.0, "msecond", 2000.0),
        (1.0, "second", 1_000_000.0),
        (1.0, "s", 1_000_000.0),
        (1000.0, None, 1.0),  # PRI's canonical duration unit is nanoseconds
    ],
)
def test_to_microseconds(value, unit, expected):
    assert to_microseconds(value, unit) == pytest.approx(expected)


@pytest.mark.parametrize(
    "value,unit,expected",
    [
        (1.0, "Gbyte/second", 1.0),
        (1000.0, "Mbyte/second", 1.0),
        (1_000_000.0, "Kbyte/second", 1.0),
        (1_000_000_000.0, "byte/second", 1.0),
        (251_000_000_000.0, None, 251.0),
    ],
)
def test_to_gbps(value, unit, expected):
    assert to_gbps(value, unit) == pytest.approx(expected)
