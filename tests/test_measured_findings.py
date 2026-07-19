"""Fixes for findings measured against released 1.2.0 (issues #24-#28).

Each test below pins behaviour that a real capture proved wrong or invisible,
so the study that found it does not have to be re-run to keep it fixed.

The compiled-training fixture (tests/fixtures/torch_compiled_train_sample.
torch-trace.json) is a REAL ``torch.profiler`` export of a torch.compile'd
training step (torch 2.12.1+cu132, RTX 4060: MLP + LayerNorm + GELU, AdamW).
It was made smaller the honest way — whole events were DROPPED, none were
edited or invented — by keeping every torch.compile dispatch span plus the
heaviest real ops/kernels; ``host_name`` was redacted as elsewhere.
"""

import math

import pytest

from perfdigest.core.digest import build_summary
from perfdigest.core.metrics import NormalizedUnit
from perfdigest.server import tools

FMT = "torch-trace"


def _compiled(fixtures_dir):
    p = fixtures_dir / "torch_compiled_train_sample.torch-trace.json"
    assert p.exists(), f"committed fixture missing: {p}"
    return str(p)


# --- #24: torch.compile envelope spans must carry the discount signal --------


def test_compile_dispatch_spans_are_tagged(fixtures_dir):
    """Dynamo/AOTDispatcher/Inductor envelopes come back tagged [dispatch].

    They arrive with cat 'cpu_op' — the same cat as a genuine aten op — so only
    a name rule can catch them; that is the whole reason this fix exists.
    """
    names = [u["name"] for u in tools.list_kernels(_compiled(fixtures_dir), FMT)]

    for expected in (
        "Torch-Compiled Region: 0/0 [dispatch]",
        "CompiledFunction [dispatch]",
        "CompiledFunctionBackward [dispatch]",
        "TorchDynamo Cache Lookup [dispatch]",
        "AOTDispatcher Runtime Wrapper Prologue [dispatch]",
    ):
        assert expected in names, f"{expected!r} not tagged; got {names}"

    # Inductor's graph-call marker carries a per-graph hash, so match the shape.
    assert any(
        n.startswith("## Call CompiledFxGraph") and n.endswith("[dispatch]")
        for n in names
    ), names

    # The autograd engine's envelope around the COMPILED backward is tagged too
    # (the compile-family token is matched anywhere in the name) — a prefix-only
    # rule would have left this one, a top-3 span, looking like real work.
    assert any(
        n.startswith("autograd::engine::evaluate_function: CompiledFunctionBackward")
        and n.endswith("[dispatch]")
        for n in names
    ), names


def test_real_work_is_not_tagged_as_dispatch(fixtures_dir):
    """The rule must not swallow genuine ops — false tagging misleads the other way."""
    units = tools.list_kernels(_compiled(fixtures_dir), FMT)
    for u in units:
        if u["name"].startswith(("aten::", "void ", "Optimizer.step")):
            assert "[dispatch]" not in u["name"], u["name"]

    # Eager autograd envelopes stay deliberately untagged (documented in the
    # usage prompt instead): they are the engine's normal structure, not
    # compile machinery. Tagging them would be a separate decision.
    eager_envelopes = [
        u["name"]
        for u in units
        if u["name"].startswith("autograd::engine::evaluate_function:")
        and "CompiledFunction" not in u["name"]
    ]
    for name in eager_envelopes:
        assert "[dispatch]" not in name, name


def test_untouched_backends_keep_their_tagging(fixtures_dir):
    """clang's [total] rule still wins, and its spans never gain a [dispatch]."""
    units = tools.list_kernels(
        str(fixtures_dir / "template_heavy.ftime-trace.json"), "clang-time-trace"
    )
    names = [u["name"] for u in units]
    assert any(n.endswith("[total]") for n in names)
    assert not any("[dispatch]" in n for n in names)


# --- #28: top-N can hide a whole unit name ----------------------------------


def _unit(name: str, index: int, duration_us: float) -> NormalizedUnit:
    return NormalizedUnit(
        name=name,
        index=index,
        duration_us=duration_us,
        raw_ref="/tmp/report",
        metrics={"duration_us": duration_us},
        domain="gpu_kernel",
    )


def test_summary_reports_names_hidden_by_top_n():
    """Three variants, three launches each: top-3 can drop one variant entirely.

    This is the measured CUDA case — a variant lost a 0.15% near-tie and
    vanished from the summary, which read as if the report held two kernels.
    """
    units = [
        _unit("v0_naive", 0, 6000.0),
        _unit("v0_naive", 1, 6001.0),
        _unit("v0_naive", 2, 6002.0),
        _unit("v2_vectorized", 3, 2280.6),
        _unit("v2_vectorized", 4, 2280.7),
        _unit("v1_coalesced", 5, 2277.1),
    ]
    summary = build_summary(units, "ncu-rep", ["duration_us"], 5)

    assert summary["total_units"] == 6
    assert summary["distinct_names_total"] == 3
    assert summary["distinct_names_omitted"] == 1  # v1_coalesced never appears
    assert "v1_coalesced" not in [u["name"] for u in summary["units"]]


def test_omitted_key_absent_when_every_name_is_shown():
    """Additive-when-meaningful, like can_digest_issues and the coverage keys."""
    units = [_unit("a", 0, 3.0), _unit("b", 1, 2.0)]
    summary = build_summary(units, "ncu-rep", ["duration_us"], 5)

    assert summary["distinct_names_total"] == 2
    assert "distinct_names_omitted" not in summary


# --- #26: coverage must be bit-identical on every interpreter ---------------


def test_coverage_uses_exact_summation():
    """Values chosen so naive left-to-right summation loses the small terms.

    ``sum()`` returns a different float here on CPython < 3.12 than on >= 3.12
    (Neumaier compensation landed in 3.12); ``math.fsum`` is exactly rounded, so
    the digest is the same bytes everywhere. Asserting against fsum pins that.
    """
    values = [1e16, 1.0, 1.0, 1.0, -1e16]
    units = [_unit(f"k{i}", i, v) for i, v in enumerate(values)]

    summary = build_summary(units, "ncu-rep", ["duration_us"], len(values))

    assert summary["total_duration_us"] == math.fsum(values)
    # The naive accumulation genuinely differs — otherwise this test proves nothing.
    assert math.fsum(values) != sum(values)
    assert summary["coverage_pct_of_total_duration"] == pytest.approx(100.0)


# --- #25 / #27: caveats must actually reach the agent -----------------------


def test_usage_prompts_carry_the_measured_caveats():
    from perfdigest.adapters.chrome_trace.backend import CHROME_TRACE_USAGE
    from perfdigest.adapters.ninja_log.backend import NINJA_LOG_USAGE
    from perfdigest.server.prompts import build_usage_convention

    # #25: fusion changes kernel cardinality (the wrong-sign case).
    assert "[dispatch]" in CHROME_TRACE_USAGE
    assert "kernel_b" in CHROME_TRACE_USAGE and "+101%" in CHROME_TRACE_USAGE

    # #27: cross-run start_ms/end_ms deltas subtract unrelated origins.
    assert "duration_ms only" in NINJA_LOG_USAGE
    assert "start_ms/end_ms" in NINJA_LOG_USAGE

    # The cross-backend statement of both lives in the shared convention.
    convention = build_usage_convention()
    assert "distinct_names_omitted" in convention
    assert "SAME THING on both sides" in convention
