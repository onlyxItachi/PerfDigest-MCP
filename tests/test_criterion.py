"""criterion benchmark backend — parsing a REAL ``target/criterion`` tree.

The committed fixture (tests/fixtures/criterion_sample/criterion/) is the
verbatim output of ``cargo bench`` (cargo 1.94.0, criterion 0.5.1) on the tiny
crate committed alongside it (tests/fixtures/criterion_sample/{Cargo.toml,
src/lib.rs,benches/fib_bench.rs}), built in a neutral temp directory and PRUNED
to just the */new/estimates.json files. Two benchmarks cover both real layouts:

  * ``fib_plain``  — plain bench_function (linear sampling, so ``slope`` is a
    real estimate)
  * ``fib/fib_20`` — inside a benchmark_group (nested one level) configured
    with SamplingMode::Flat, where criterion GENUINELY does not estimate a
    slope and serializes ``"slope": null`` — the honest-absence case with real
    data, nothing hand-trimmed or fabricated.

The report_ref for this backend is the criterion output DIRECTORY, not a file.
Expected values below are read straight from the committed estimates.json.
"""

import json
import shutil

import pytest

from perfdigest.adapters import registry
from perfdigest.core.digest import NOT_AVAILABLE
from perfdigest.server import tools

FMT = "criterion"

# point_estimates from the committed fixture (ns)
FIB20_MEAN = 11766.474870466322
FIB20_MEDIAN = 11440.422279792747
FIB20_STD_DEV = 1419.904227474325
FIB20_MAD = 519.8580609779276
PLAIN_MEAN = 86.6994796535408
PLAIN_SLOPE = 86.0904495604189


def _root(fixtures_dir):
    p = fixtures_dir / "criterion_sample" / "criterion"
    assert p.is_dir(), f"committed fixture tree missing: {p}"
    return str(p)


def test_registered_and_domain():
    b = registry.get_backend(FMT)
    assert b.name == "criterion"
    assert b.domain == "benchmark"
    assert b.report_is_directory is True
    assert registry.get_backend("criterion-json").name == "criterion"
    assert registry.get_backend("CRITERION").name == "criterion"  # case-insensitive


def test_walks_both_nesting_depths_sorted(fixtures_dir):
    units = tools.list_kernels(_root(fixtures_dir), FMT)
    # Grouped bench nests one level deeper; walk order is sorted + deterministic.
    assert [u["name"] for u in units] == ["fib/fib_20", "fib_plain"]
    assert [u["index"] for u in units] == [0, 1]
    assert all(u["domain"] == "benchmark" for u in units)
    # duration_us = mean point_estimate (ns) / 1000
    assert units[0]["duration_us"] == pytest.approx(FIB20_MEAN / 1000.0)
    assert units[1]["duration_us"] == pytest.approx(PLAIN_MEAN / 1000.0)


def test_estimator_point_estimates(fixtures_dir):
    m = tools.get_metrics(_root(fixtures_dir), FMT, "fib/fib_20")["metrics"]
    assert m["mean_ns"] == pytest.approx(FIB20_MEAN)
    assert m["median_ns"] == pytest.approx(FIB20_MEDIAN)
    assert m["std_dev_ns"] == pytest.approx(FIB20_STD_DEV)
    assert m["median_abs_dev_ns"] == pytest.approx(FIB20_MAD)


def test_slope_null_in_flat_sampling_is_honest_absence(fixtures_dir):
    # criterion serializes an unestimated slope as JSON null (it is a Rust
    # Option) — REAL flat-sampling output, not a trimmed file. null means NOT
    # MEASURED and must never surface as 0.0.
    flat = tools.get_metrics(_root(fixtures_dir), FMT, "fib/fib_20")["metrics"]
    assert flat["slope_ns"] == NOT_AVAILABLE
    # ...while the linear-sampling bench has a genuine slope estimate.
    linear = tools.get_metrics(_root(fixtures_dir), FMT, "fib_plain")["metrics"]
    assert linear["slope_ns"] == pytest.approx(PLAIN_SLOPE)


def test_ranking_by_mean_duration(fixtures_dir):
    summary = tools.summarize_report(_root(fixtures_dir), FMT, top_n=1)
    assert summary["sorted_by"] == "duration_us"
    assert [u["name"] for u in summary["units"]] == ["fib/fib_20"]  # ~11.8us > ~87ns
    assert summary["total_units"] == 2


def test_directory_ref_resolution(fixtures_dir):
    # The directory IS the report_ref — resolution must return it as-is.
    units = tools.list_kernels(_root(fixtures_dir), FMT)
    assert len(units) == 2

    # A nonexistent ref errors as a missing DIRECTORY (no suffix guessing).
    with pytest.raises(FileNotFoundError) as exc:
        tools.list_kernels(str(fixtures_dir / "no_such_criterion_dir"), FMT)
    assert "directory" in str(exc.value)

    # Passing an individual file inside the tree is redirected to the dir rule.
    est = fixtures_dir / "criterion_sample" / "criterion" / "fib_plain" / "new" / "estimates.json"
    with pytest.raises(FileNotFoundError) as exc:
        tools.list_kernels(str(est), FMT)
    assert "DIRECTORY" in str(exc.value)


def test_dir_without_estimates_raises_loudly_never_silent_empty(tmp_path):
    (tmp_path / "not_criterion").mkdir()
    with pytest.raises(ValueError) as exc:
        tools.list_kernels(str(tmp_path / "not_criterion"), FMT)
    assert "no */new/estimates.json" in str(exc.value)


def test_criterion_report_bookkeeping_dirs_are_skipped(tmp_path, fixtures_dir):
    # criterion writes its own HTML under report/ dirs. Real trees put no
    # estimates.json there, but the guard must hold even if one appeared
    # (synthetic tree: a real estimates.json copied under a report/ path).
    root = tmp_path / "criterion"
    real = fixtures_dir / "criterion_sample" / "criterion" / "fib_plain" / "new" / "estimates.json"
    for rel in ("mybench/new", "report/mybench/new"):
        d = root / rel
        d.mkdir(parents=True)
        shutil.copy(real, d / "estimates.json")
    units = tools.list_kernels(str(root), FMT)
    assert [u["name"] for u in units] == ["mybench"]  # report/ never a benchmark


def test_rerun_overwrite_is_seen_immediately(tmp_path, fixtures_dir):
    # cargo bench overwrites <bench>/new/estimates.json IN PLACE without
    # touching the root directory's mtime — a directory ref therefore bypasses
    # the parse-once cache, so a re-run must be visible on the very next call.
    root = tmp_path / "criterion"
    shutil.copytree(fixtures_dir / "criterion_sample" / "criterion", root)
    before = tools.get_metrics(str(root), FMT, "fib_plain")["metrics"]
    assert before["mean_ns"] == pytest.approx(PLAIN_MEAN)

    est = root / "fib_plain" / "new" / "estimates.json"
    data = json.loads(est.read_text())
    data["mean"]["point_estimate"] = 500.0
    est.write_text(json.dumps(data))

    after = tools.get_metrics(str(root), FMT, "fib_plain")["metrics"]
    assert after["mean_ns"] == pytest.approx(500.0)


def test_malformed_estimates_json_raises_loudly(tmp_path):
    bad = tmp_path / "criterion" / "broken" / "new"
    bad.mkdir(parents=True)
    (bad / "estimates.json").write_text("{not json")
    with pytest.raises(ValueError) as exc:
        tools.list_kernels(str(tmp_path / "criterion"), FMT)
    assert "malformed estimates.json" in str(exc.value)


def test_absence_honesty_for_a_metric_the_tree_lacks(fixtures_dir):
    # A GPU-vocabulary term has no meaning for a wall-time benchmark: honest
    # absence plus the vocabulary hint, never 0.0.
    d = tools.get_metrics(_root(fixtures_dir), FMT, "fib_plain", ["dram_pct_peak", "mean_ns"])
    assert d["metrics"]["dram_pct_peak"] == NOT_AVAILABLE
    assert d["metrics"]["mean_ns"] == pytest.approx(PLAIN_MEAN)
    assert "unknown_metrics" in d


def test_expand_exposes_confidence_intervals(fixtures_dir):
    raw = tools.expand(_root(fixtures_dir), FMT, "fib_plain", "all")["metrics"]
    assert raw["mean.point_estimate"] == pytest.approx(PLAIN_MEAN)
    assert raw["mean.confidence_interval.lower_bound"] < PLAIN_MEAN
    assert raw["mean.confidence_interval.upper_bound"] > PLAIN_MEAN
    assert "mean.standard_error" in raw
    # Substring filtering works over the dot-flattened raw names.
    only_median = tools.expand(_root(fixtures_dir), FMT, "fib_plain", "median")["metrics"]
    assert only_median and all("median" in k for k in only_median)
