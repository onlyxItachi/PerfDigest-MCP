"""cargo build-diagnostics (BuildDigest) backend — against REAL cargo 1.94 captures.

The committed fixtures are verbatim `cargo build --message-format=json`
stdout, captured from the tiny 2-crate workspace committed for provenance at
tests/fixtures/cargo_diag_ws/ (mathlib lib crate with deliberate
unused_variables + dead_code warnings; mathapp bin crate depending on it):

  * cargo_build_warnings_sample.cargo-diag.jsonl — fresh build of the clean
    workspace (mathapp/src/main.rs): mathlib emits 2 warnings, mathapp
    compiles clean, build-finished success=true.
  * cargo_build_error_sample.cargo-diag.jsonl — fresh (post-`cargo clean`)
    build after introducing a real E0308 type error in mathapp
    (committed as mathapp/src/main.rs.error-variant): mathlib's 2 warnings,
    then mathapp's error + rustc's failure-note, build-finished success=false.

Both were captured with cargo 1.94.0 from a neutral workspace path
(/tmp/perfdigest-ws) so the absolute paths cargo unavoidably embeds in
package_id / manifest_path / artifact filenames carry no host-identifying
information; nothing was hand-edited. Span file_names are workspace-relative
(cargo run from the workspace root — verified), which is what the per-file
concentration keys in expand() are built from.

Real-artifact lesson (verified against these fixtures): cargo 1.94 package_id
is the Package ID Spec URL (`path+file:///ws/mathlib#0.1.0` — bare-version
fragment when the crate is named like its directory), NOT the legacy
`name version (source)` string older cargo wrote; the parser handles both,
and the synthetic-shape test below pins the registry/git PURL variants no
path-dependency fixture can produce.
"""

import json

import pytest

from perfdigest.adapters import registry
from perfdigest.core.digest import NOT_AVAILABLE
from perfdigest.server import tools

FMT = "cargo-diag"
ALIAS = "cargo-json"


def _warnings_path(fixtures_dir):
    p = fixtures_dir / "cargo_build_warnings_sample.cargo-diag.jsonl"
    assert p.exists(), f"committed fixture missing: {p}"
    return str(p)


def _error_path(fixtures_dir):
    p = fixtures_dir / "cargo_build_error_sample.cargo-diag.jsonl"
    assert p.exists(), f"committed fixture missing: {p}"
    return str(p)


def test_registered_and_formats():
    assert registry.get_backend(FMT).name == "cargo_diag"
    assert registry.get_backend(ALIAS).name == "cargo_diag"


def test_units_are_crates_named_from_package_id(fixtures_dir):
    units = tools.list_kernels(_warnings_path(fixtures_dir), FMT)
    # 2 crates, first-appearance order, short name@version parsed from the
    # modern PURL-style package_id (path+file:///ws/mathlib#0.1.0).
    assert [u["name"] for u in units] == ["mathlib@0.1.0", "mathapp@0.1.0"]
    assert all(u["domain"] == "build_diag" for u in units)


def test_duration_is_honestly_absent_on_every_unit(fixtures_dir):
    # THE honesty story of this backend: cargo's stream carries no timing.
    p = _warnings_path(fixtures_dir)
    assert all(u["duration_us"] is None for u in tools.list_kernels(p, FMT))
    # ...end-to-end through get_metrics: the digest says not-available, and
    # the sibling count in the same call stays a real number.
    m = tools.get_metrics(p, FMT, "mathlib", ["duration_us", "warnings"])["metrics"]
    assert m["duration_us"] == NOT_AVAILABLE
    assert m["warnings"] == 2.0


def test_clean_crate_counts_are_genuine_zero(fixtures_dir):
    # mathapp compiled clean in the warnings fixture: 0.0 is MEASURED (the
    # stream is a complete enumeration of this build's diagnostics — the
    # ptxas 'Used ...' rule), not absence, so no NOT_AVAILABLE here.
    m = tools.get_metrics(_warnings_path(fixtures_dir), FMT, "mathapp")["metrics"]
    assert m == {"errors": 0.0, "warnings": 0.0, "notes_helps": 0.0}


def test_warning_and_error_counts_from_real_captures(fixtures_dir):
    mw = tools.get_metrics(_warnings_path(fixtures_dir), FMT, "mathlib")["metrics"]
    assert mw == {"errors": 0.0, "warnings": 2.0, "notes_helps": 0.0}

    # Error fixture: mathapp FAILED (no compiler-artifact record at all) but
    # must still be a unit — its messages are the whole point.
    me = tools.get_metrics(_error_path(fixtures_dir), FMT, "mathapp")["metrics"]
    assert me["errors"] == 1.0  # the real E0308
    assert me["warnings"] == 0.0
    assert me["notes_helps"] == 1.0  # rustc's 'aborting due to ...' failure-note


def test_summarize_falls_back_to_file_order_with_all_none_durations(fixtures_dir):
    s = tools.summarize_report(_error_path(fixtures_dir), FMT, top_n=5)
    assert s["sorted_by"] == "file_order"  # no duration, no self_pct -> honest fallback
    assert [u["name"] for u in s["units"]] == ["mathlib@0.1.0", "mathapp@0.1.0"]
    assert all(u["duration_us"] is None for u in s["units"])
    # Coverage over durations would be fabrication here — the keys must be absent.
    assert "total_duration_us" not in s
    assert "coverage_pct_of_total_duration" not in s


def test_expand_concentration_keys_say_where(fixtures_dir):
    # The digest's WHERE story: per-level-per-file and per-code counts.
    raw = tools.expand(_error_path(fixtures_dir), FMT, "mathapp", "all")["metrics"]
    assert raw["error@mathapp/src/main.rs"] == 1.0  # workspace-relative file
    assert raw["failure-note@(no-span)"] == 1.0  # span-less diag, explicit bucket
    assert raw["code:E0308"] == 1.0

    raw2 = tools.expand(_warnings_path(fixtures_dir), FMT, "mathlib", "all")["metrics"]
    assert raw2["warning@mathlib/src/lib.rs"] == 2.0
    assert raw2["code:unused_variables"] == 1.0
    assert raw2["code:dead_code"] == 1.0

    # ...and the substring section filter reaches them.
    only_codes = tools.expand(_warnings_path(fixtures_dir), FMT, "mathlib", "code:")["metrics"]
    assert set(only_codes) == {"code:unused_variables", "code:dead_code"}


def test_expand_carries_report_level_build_finished(fixtures_dir):
    # No report-level slot in the digest schema (core stays unbent), so the
    # build-finished outcome rides expand on every unit.
    ok = tools.expand(_warnings_path(fixtures_dir), FMT, "mathlib", "all")["metrics"]
    assert ok["build_finished_success"] is True
    bad = tools.expand(_error_path(fixtures_dir), FMT, "mathapp", "all")["metrics"]
    assert bad["build_finished_success"] is False


def test_failed_crate_has_no_artifact_but_lib_does(fixtures_dir):
    p = _error_path(fixtures_dir)
    lib = tools.expand(p, FMT, "mathlib", "artifacts")["metrics"]
    app = tools.expand(p, FMT, "mathapp", "artifacts")["metrics"]
    assert lib["artifacts"] == 1
    assert app["artifacts"] == 0  # E0308 stopped it — honest zero, crate still a unit


def test_expand_and_list_indices_agree(fixtures_dir):
    p = _error_path(fixtures_dir)
    for u in tools.list_kernels(p, FMT):
        raw = tools.expand(p, FMT, str(u["index"]), "all")
        assert raw["kernel"] == u["name"]


def test_registry_dispatch_on_both_format_names(fixtures_dir):
    p = _warnings_path(fixtures_dir)
    a = tools.list_kernels(p, FMT)
    b = tools.list_kernels(p, ALIAS)
    assert [u["name"] for u in a] == [u["name"] for u in b]


def test_package_id_shapes_across_cargo_versions(tmp_path):
    # SYNTHETIC lines (clearly marked): the real path-dependency fixture can
    # only ever produce the path+file shape, so the registry PURL
    # (name@version fragment), git PURL (repo-basename + ?rev= query), and
    # legacy 'name version (source)' shapes are pinned here.
    trace = tmp_path / "shapes.cargo-diag.jsonl"
    lines = [
        {"reason": "compiler-artifact",
         "package_id": "registry+https://github.com/rust-lang/crates.io-index#serde@1.0.219"},
        {"reason": "compiler-artifact",
         "package_id": "git+https://github.com/owner/mytool.git?rev=abc123#0.3.0"},
        {"reason": "compiler-artifact",
         "package_id": "legacycrate 0.2.0 (registry+https://github.com/rust-lang/crates.io-index)"},
        {"reason": "build-finished", "success": True},
    ]
    trace.write_text("\n".join(json.dumps(l) for l in lines) + "\n")
    units = tools.list_kernels(str(trace), FMT)
    assert [u["name"] for u in units] == [
        "serde@1.0.219",
        "mytool@0.3.0",
        "legacycrate@0.2.0",
    ]


def test_non_ndjson_input_raises_loud_valueerror(tmp_path):
    # A Chrome trace is ONE multi-line JSON document — its first line alone
    # is not valid JSON, and the error must name the real format.
    chrome = tmp_path / "trace.jsonl"
    chrome.write_text('{\n  "traceEvents": []\n}\n')
    with pytest.raises(Exception, match="NDJSON"):
        tools.list_kernels(str(chrome), FMT)

    # Valid JSON-lines that is NOT cargo's (a perf-stat export has no
    # 'reason' key) must be loud too, with the redirect hint.
    stat = tmp_path / "stat.jsonl"
    stat.write_text('{"counter-value": "1", "event": "cycles"}\n')
    with pytest.raises(Exception, match="reason"):
        tools.list_kernels(str(stat), FMT)

    # A scalar line is JSON but not an object.
    scalar = tmp_path / "scalar.jsonl"
    scalar.write_text("42\n")
    with pytest.raises(Exception, match="reason"):
        tools.list_kernels(str(scalar), FMT)


def test_vocabulary_hint_fires_for_foreign_terms(fixtures_dir):
    d = tools.get_metrics(
        _warnings_path(fixtures_dir), FMT, "mathlib", ["achieved_occupancy", "errors"]
    )
    assert d["metrics"]["achieved_occupancy"] == NOT_AVAILABLE
    assert d["metrics"]["errors"] == 0.0
    assert "unknown_metrics" in d
