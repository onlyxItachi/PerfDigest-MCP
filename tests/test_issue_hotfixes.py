"""v1.1.1 hotfix contracts — issues #2 (edge-case semantics) and #4 (diagnostics).

Each test pins one fix:
  * registry collision detection uses the NORMALIZED format key
  * metrics=[] is an empty request, not "give me the defaults"
  * exact unit-name match precedes any numeric-index interpretation; '#3' is
    the explicit index form
  * an unreadable backend carries a diagnostic instead of a silent False
"""

import pytest

from perfdigest.adapters import registry
from perfdigest.core.backend import Backend, CapabilityReport
from perfdigest.core.metrics import NormalizedUnit
from perfdigest.platform import capabilities
from perfdigest.server import tools
from perfdigest.server.tools import _find_unit


def _fake_backend(name, formats, reader_available):
    return Backend(
        name=name,
        formats=frozenset(formats),
        suffixes=(".fake",),
        domain="gpu_kernel",
        platforms=frozenset({"linux"}),
        standard_to_vendor={"m": "m"},
        default_core_set=("m",),
        usage_prompt="fake",
        load_units=lambda path: [],
        raw_metrics=lambda path, i, f: {},
        probe=lambda: CapabilityReport(False, "fake backend", None),
        reader_available=reader_available,
    )


def _unregister(name, formats):
    registry._BY_NAME.pop(name, None)
    for fmt in formats:
        owner = registry._BY_FORMAT.get(fmt)
        if owner is not None and owner.name == name:  # never evict a real backend
            registry._BY_FORMAT.pop(fmt)


# ---------------------------------------------------- issue #2: registry


def test_case_variant_format_cannot_hijack_a_backend():
    # v1.1.0 checked the RAW string but stored the NORMALIZED key, so a backend
    # claiming 'NCU-REP' silently overwrote nsight's 'ncu-rep' registration.
    hijacker = _fake_backend("hijacker", {"NCU-REP"}, lambda: True)
    try:
        with pytest.raises(ValueError, match="already claimed"):
            registry.register(hijacker)
        assert registry.get_backend("ncu-rep").name == "nsight"  # unharmed
        # Registration is atomic: the rejected backend leaves NO trace — no
        # ghost in all_backends(), no partial format claims.
        assert "hijacker" not in {b.name for b in registry.all_backends()}
    finally:
        _unregister("hijacker", {"ncu-rep"})


def test_rejected_multiformat_backend_leaves_no_partial_claims():
    # One good format + one colliding format: validation must reject the WHOLE
    # registration, not claim 'fresh-fmt' before dying on the collision.
    mixed = _fake_backend("mixed", {"fresh-fmt", "ncu-rep"}, lambda: True)
    try:
        with pytest.raises(ValueError, match="already claimed"):
            registry.register(mixed)
        assert "mixed" not in {b.name for b in registry.all_backends()}
        with pytest.raises(ValueError, match="unsupported format"):
            registry.get_backend("fresh-fmt")  # never claimed
    finally:
        _unregister("mixed", {"fresh-fmt", "ncu-rep"})


# ------------------------------------------- issue #2: unit selection


def _units():
    return [
        NormalizedUnit(name="3", index=0, duration_us=1.0, raw_ref="r"),
        NormalizedUnit(name="alpha", index=1, duration_us=2.0, raw_ref="r"),
        NormalizedUnit(name="beta", index=3, duration_us=3.0, raw_ref="r"),
    ]


def test_exact_name_wins_over_index_interpretation():
    # A unit literally named "3" (arbitrary trace emitters produce these) must
    # be selectable by name even though index 3 also exists.
    assert _find_unit(_units(), "3").name == "3"


def test_hash_prefix_is_the_explicit_index_form():
    assert _find_unit(_units(), "#3").name == "beta"
    with pytest.raises(ValueError, match="not present"):
        _find_unit(_units(), "#99")


def test_plain_numeric_still_falls_back_to_index():
    # No unit is named "1", so the numeric fallback resolves index 1.
    assert _find_unit(_units(), "1").name == "alpha"


# ------------------------------------------- issue #2: metrics=[] vs None


def test_empty_metrics_list_is_an_empty_request(fixtures_dir):
    path = str(fixtures_dir / "nvidia_sample.ncu-csv")
    d = tools.get_metrics(path, "ncu-csv", "add_kernel", metrics=[])
    assert d["metrics"] == {}

    s = tools.summarize_report(path, "ncu-csv", top_n=1, metrics=[])
    assert s["units"][0]["metrics"] == {}

    c = tools.compare_metrics(path, path, "ncu-csv", "add_kernel", metrics=[])
    assert c["metrics"] == {}


def test_none_still_means_the_default_core_set(fixtures_dir):
    path = str(fixtures_dir / "nvidia_sample.ncu-csv")
    d = tools.get_metrics(path, "ncu-csv", "add_kernel", metrics=None)
    assert len(d["metrics"]) > 0


# ------------------------------------------- issue #4: reader diagnostics


def test_unreadable_backend_carries_a_diagnostic():
    broken = _fake_backend(
        "broken_wheel", {"broken-fmt"},
        lambda: (_ for _ in ()).throw(RuntimeError("libnvperf.so: cannot open")),
    )
    missing = _fake_backend("missing_dep", {"missing-fmt"}, lambda: False)
    try:
        registry.register(broken)
        registry.register(missing)

        readable = {b.name for b in registry.readable_backends()}
        assert "broken_wheel" not in readable and "missing_dep" not in readable

        issues = {i["backend"]: i["issue"] for i in registry.reader_issues()}
        assert "RuntimeError" in issues["broken_wheel"]  # crash is distinguishable
        assert "libnvperf" in issues["broken_wheel"]
        assert "returned False" in issues["missing_dep"]  # from a mere absence

        summ = capabilities.capability_summary()
        flagged = {i["backend"] for i in summ.get("can_digest_issues", [])}
        assert {"broken_wheel", "missing_dep"} <= flagged
    finally:
        _unregister("broken_wheel", {"broken-fmt"})
        _unregister("missing_dep", {"missing-fmt"})


def test_no_issues_key_when_all_readers_import():
    # Additive contract: the key is absent on a healthy host, so existing
    # consumers of platform_capabilities see an unchanged shape.
    summ = capabilities.capability_summary()
    for issue in summ.get("can_digest_issues", []):
        assert issue["backend"] not in {b.name for b in registry.readable_backends()}
