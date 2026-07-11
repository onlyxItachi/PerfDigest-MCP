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
    assert names == [
        "_Z17true_spill_kernelPKfPfi",
        "_Z16smem_tile_kernelPKfPfi",
        "_Z13spilly_kernelPKfPfii",
        "_Z10lean_saxpyfPKfPfl",
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
