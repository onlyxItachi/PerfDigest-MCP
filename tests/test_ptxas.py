"""ptxas codegen backend — parsing a REAL `nvcc -Xptxas -v` capture (CUDA 13.3).

The committed fixture (tests/fixtures/gpu_codegen_sample.ptxas.txt) is the
verbatim stderr of compiling tests/fixtures/gpu_codegen_sample.cu with
``nvcc -O3 -arch=sm_89 -maxrregcount 24 -Xptxas -v`` — four kernels covering
genuine spills, static smem, barriers, and the preamble noise lines.
"""

import pytest

from perfdigest.adapters import registry
from perfdigest.core.digest import NOT_AVAILABLE
from perfdigest.server import tools

FMT = "ptxas-verbose"


def _path(fixtures_dir):
    p = fixtures_dir / "gpu_codegen_sample.ptxas.txt"
    assert p.exists(), f"committed fixture missing: {p}"
    return str(p)


def test_registered_and_domain():
    b = registry.get_backend(FMT)
    assert b.name == "ptxas"
    assert b.domain == "kernel_codegen"
    assert registry.get_backend("ptxas").name == "ptxas"  # short-format alias


def test_lists_all_entry_functions_in_file_order(fixtures_dir):
    units = tools.list_kernels(_path(fixtures_dir), FMT)
    names = [u["name"] for u in units]
    # Mangled name tagged with the -gencode target arch.
    assert names == [
        "_Z17true_spill_kernelPKfPfi [sm_89]",
        "_Z16smem_tile_kernelPKfPfi [sm_89]",
        "_Z13spilly_kernelPKfPfii [sm_89]",
        "_Z10lean_saxpyfPKfPfl [sm_89]",
    ]
    # A compile artifact has no runtime duration — honest null, not 0.0.
    assert all(u["duration_us"] is None for u in units)
    assert all(u["domain"] == "kernel_codegen" for u in units)


def test_spill_kernel_digest(fixtures_dir):
    # Mangled names contain the plain name: substring matching must work.
    m = tools.get_metrics(_path(fixtures_dir), FMT, "true_spill_kernel")["metrics"]
    assert m["registers_per_thread"] == 24.0
    assert m["spill_stores_bytes"] == 660.0
    assert m["spill_loads_bytes"] == 656.0
    assert m["stack_frame_bytes"] == 312.0
    assert m["const_mem_bytes"] == 372.0
    assert m["compile_time_ms"] == pytest.approx(21.988)


def test_smem_and_barriers(fixtures_dir):
    m = tools.get_metrics(_path(fixtures_dir), FMT, "smem_tile_kernel")["metrics"]
    assert m["static_smem_bytes"] == 4224.0
    assert m["barriers"] == 1.0
    assert m["registers_per_thread"] == 10.0


def test_omitted_component_is_a_genuine_zero(fixtures_dir):
    # ptxas omits zero components from the (complete) `Used ...` enumeration, so
    # lean_saxpy's absent smem is a TRUE 0.0 — unlike a missing export, this is
    # measured-and-zero, and must not surface as NOT_AVAILABLE.
    m = tools.get_metrics(_path(fixtures_dir), FMT, "lean_saxpy")["metrics"]
    assert m["static_smem_bytes"] == 0.0
    assert m["spill_stores_bytes"] == 0.0
    assert m["barriers"] == 0.0  # 13.x prints 'used 0 barriers' explicitly


def test_expand_exposes_raw_fields_including_arch(fixtures_dir):
    raw = tools.expand(_path(fixtures_dir), FMT, "lean_saxpy", "all")["metrics"]
    assert raw["arch"] == "sm_89"
    assert raw["registers"] == 10.0


def test_requested_metric_not_in_codegen_export(fixtures_dir):
    # A runtime metric requested from a compile artifact: honest absence path
    # (plus the vocabulary hint, since dram_pct_peak is not a codegen term).
    d = tools.get_metrics(_path(fixtures_dir), FMT, "lean_saxpy", ["dram_pct_peak"])
    assert d["metrics"]["dram_pct_peak"] == NOT_AVAILABLE
    assert "unknown_metrics" in d


def test_multiarch_compile_stays_name_addressable(fixtures_dir):
    # The normal CI invocation: one entry per -gencode target, same mangled
    # names. The [sm_XX] tag must keep every unit distinguishable and reachable
    # by NAME (not only by index). Fixture: verbatim sm_80 + sm_89 compile.
    p = str(fixtures_dir / "gpu_codegen_multiarch_sample.ptxas.txt")
    units = tools.list_kernels(p, FMT)
    assert len(units) == 8  # 4 kernels x 2 arches
    names = [u["name"] for u in units]
    assert len(set(names)) == 8  # all distinguishable

    # Bare kernel name is ambiguous across arches -> the error must be actionable.
    with pytest.raises(Exception) as exc:
        tools.get_metrics(p, FMT, "true_spill_kernel")
    assert "ambiguous" in str(exc.value)

    # ...and the arch-tagged name resolves uniquely, with per-arch values.
    sm80 = tools.get_metrics(p, FMT, "true_spill_kernelPKfPfi [sm_80]")["metrics"]
    sm89 = tools.get_metrics(p, FMT, "true_spill_kernelPKfPfi [sm_89]")["metrics"]
    assert sm80["registers_per_thread"] == 24.0
    assert sm89["registers_per_thread"] == 24.0
    assert sm80["spill_stores_bytes"] > 0.0 and sm89["spill_stores_bytes"] > 0.0


def test_missing_used_line_keeps_resources_absent(tmp_path):
    # The OTHER half of the honesty nuance: omission-means-zero applies ONLY
    # inside a present `Used ...` enumeration. A truncated log with no such line
    # must keep registers/smem/cmem/barriers absent — never zero-filled — while
    # the present frame line's explicit zeros stay genuine 0.0.
    log = tmp_path / "truncated.ptxas.txt"
    log.write_text(
        "ptxas info    : Compiling entry function '_Z1kPf' for 'sm_89'\n"
        "ptxas info    : Function properties for _Z1kPf\n"
        "    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
    )
    m = tools.get_metrics(str(log), FMT, "0")["metrics"]
    assert m["registers_per_thread"] == NOT_AVAILABLE
    assert m["static_smem_bytes"] == NOT_AVAILABLE
    assert m["const_mem_bytes"] == NOT_AVAILABLE
    assert m["barriers"] == NOT_AVAILABLE
    assert m["stack_frame_bytes"] == 0.0  # explicitly printed: measured zero
    assert m["spill_stores_bytes"] == 0.0
