"""ninja build-step backend — parsing a REAL ``.ninja_log`` (ninja 1.13.2).

The committed fixture (tests/fixtures/ninja_log_sample.ninja_log) is the
verbatim log ninja wrote for a tiny 4-target project (tests/fixtures/
ninja_log_sample/{build.ninja,a.c,b.c,c.c}: compile a.o/b.o/c.o, link app) built
in a neutral temp directory, followed by editing a.c and rebuilding — a genuine
incremental rebuild that appends a SECOND entry each for a.o and app (real
supersede data, not hand-fabricated). Regenerate with:

    cd $(mktemp -d) && cp <fixture dir>/{build.ninja,a.c,b.c,c.c} . \\
        && ninja && touch a.c && ninja

ninja 1.13.2 (the version installed here) writes header ``# ninja log v7`` —
one bump past the ``v6`` the task brief names as current for ninja>=1.12, same
tab-separated shape. The reader accepts v5/v6/v7.
"""

import pytest

from perfdigest.adapters import registry
from perfdigest.core.digest import NOT_AVAILABLE
from perfdigest.server import tools

FMT = "ninja-log"


def _path(fixtures_dir):
    p = fixtures_dir / "ninja_log_sample.ninja_log"
    assert p.exists(), f"committed fixture missing: {p}"
    return str(p)


def test_registered_and_domain():
    b = registry.get_backend(FMT)
    assert b.name == "ninja_log"
    assert b.domain == "build_step"
    assert registry.get_backend("ninja_log").name == "ninja_log"  # underscore alias


def test_registry_dispatch_on_both_formats():
    assert registry.get_backend("ninja-log").name == "ninja_log"
    assert registry.get_backend("ninja_log").name == "ninja_log"
    assert registry.get_backend("NINJA-LOG").name == "ninja_log"  # case-insensitive


def test_parses_real_fixture_with_supersede(fixtures_dir):
    # 4 distinct output paths even though a.o and app each appear TWICE in the
    # raw file (initial build + incremental rebuild after editing a.c).
    units = tools.list_kernels(_path(fixtures_dir), FMT)
    names = [u["name"] for u in units]
    assert sorted(names) == ["a.o", "app", "b.o", "c.o"]
    assert len(names) == len(set(names))  # no duplicate output_path units
    assert all(u["domain"] == "build_step" for u in units)


def test_duplicate_output_supersede_keeps_last_timing(fixtures_dir):
    # Raw file has TWO lines for a.o: '35\t58\t...' (first build, 23ms) and
    # '0\t29\t...' (rebuild, 29ms). The rebuild is what actually happened last
    # and must win — the stale 23ms reading must not surface anywhere.
    a_o = tools.get_metrics(_path(fixtures_dir), FMT, "a.o")["metrics"]
    assert a_o["start_ms"] == pytest.approx(0.0)
    assert a_o["end_ms"] == pytest.approx(29.0)
    assert a_o["duration_ms"] == pytest.approx(29.0)

    # Same story for 'app': first link ('58\t75', 17ms) superseded by the
    # relink after a.o changed ('29\t54', 25ms).
    app = tools.get_metrics(_path(fixtures_dir), FMT, "app")["metrics"]
    assert app["start_ms"] == pytest.approx(29.0)
    assert app["end_ms"] == pytest.approx(54.0)
    assert app["duration_ms"] == pytest.approx(25.0)


def test_duration_us_is_derived_from_ms(fixtures_dir):
    units = {u["name"]: u for u in tools.list_kernels(_path(fixtures_dir), FMT)}
    assert units["a.o"]["duration_us"] == pytest.approx(29_000.0)
    assert units["b.o"]["duration_us"] == pytest.approx(22_000.0)  # 36 -> 58
    assert units["c.o"]["duration_us"] == pytest.approx(22_000.0)  # 36 -> 58


def test_ranking_by_duration(fixtures_dir):
    # a.o (29ms, post-rebuild) is the single slowest edge; app (25ms) second.
    summary = tools.summarize_report(_path(fixtures_dir), FMT, top_n=2)
    assert summary["sorted_by"] == "duration_us"
    ranked_names = [u["name"] for u in summary["units"]]
    assert ranked_names == ["a.o", "app"]
    # Parallel-build honesty: summed edge time can exceed wall-clock, so
    # coverage is explicitly documented (usage_prompt) as coverage over SUMMED
    # edge time, not wall-clock — here it's simply the normal ratio math.
    assert summary["total_duration_us"] == pytest.approx(29_000.0 + 25_000.0 + 22_000.0 * 2)


def test_malformed_line_raises_loudly(tmp_path):
    bad = tmp_path / "bad.ninja_log"
    bad.write_text("# ninja log v5\n36\t58\tonly-three-fields\n")
    with pytest.raises(ValueError) as exc:
        tools.list_kernels(str(bad), FMT)
    msg = str(exc.value)
    assert "malformed" in msg and "line 2" in msg


def test_non_numeric_timestamp_raises_loudly(tmp_path):
    bad = tmp_path / "bad_ts.ninja_log"
    bad.write_text("# ninja log v5\nSTART\t58\t123\ta.o\tdeadbeef\n")
    with pytest.raises(ValueError) as exc:
        tools.list_kernels(str(bad), FMT)
    assert "numeric" in str(exc.value)


def test_bad_header_raises_loudly_never_silent_empty(tmp_path):
    bad = tmp_path / "bad_header.ninja_log"
    bad.write_text("not a ninja log at all\n36\t58\t123\ta.o\tdeadbeef\n")
    with pytest.raises(ValueError) as exc:
        tools.list_kernels(str(bad), FMT)
    assert "unrecognized ninja log header" in str(exc.value)
    assert "not a ninja log at all" in str(exc.value)


def test_unsupported_version_raises_loudly(tmp_path):
    bad = tmp_path / "future_version.ninja_log"
    bad.write_text("# ninja log v99\n36\t58\t123\ta.o\tdeadbeef\n")
    with pytest.raises(ValueError) as exc:
        tools.list_kernels(str(bad), FMT)
    assert "unsupported ninja log version" in str(exc.value) and "99" in str(exc.value)


def test_empty_file_raises_loudly_never_silent_empty(tmp_path):
    empty = tmp_path / "empty.ninja_log"
    empty.write_text("")
    with pytest.raises(ValueError) as exc:
        tools.list_kernels(str(empty), FMT)
    assert "empty ninja log" in str(exc.value)


def test_absence_honesty_for_a_metric_the_log_lacks(fixtures_dir):
    # A GPU/CPU-vocabulary term has no meaning for a build edge; the log never
    # measured it, so it must come back honestly absent, never 0.0.
    m = tools.get_metrics(
        _path(fixtures_dir), FMT, "b.o", ["dram_pct_peak", "duration_ms"]
    )
    assert m["metrics"]["dram_pct_peak"] == NOT_AVAILABLE
    assert m["metrics"]["duration_ms"] == pytest.approx(22.0)
    assert "unknown_metrics" in m  # vocabulary hint, since it's a typo/OOV term


def test_restat_mtime_and_cmd_hash_are_not_metrics_but_are_expandable(fixtures_dir):
    # The non-negotiable: cmd_hash/restat_mtime must never leak into the
    # standard metrics vocabulary...
    default = tools.get_metrics(_path(fixtures_dir), FMT, "b.o")["metrics"]
    assert set(default) == {"duration_ms", "start_ms", "end_ms"}
    # ...but raw access must still expose them (they are real fields in the log).
    raw = tools.expand(_path(fixtures_dir), FMT, "b.o", "all")["metrics"]
    assert raw["cmd_hash"] == "c2cf39cfcfb76050"
    assert raw["restat_mtime"] == "1784198739738277865"
