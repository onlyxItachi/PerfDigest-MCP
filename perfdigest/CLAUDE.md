# CLAUDE.md — perfdigest

Guidance for Claude Code working in this repo. Read `docs/INIT_PROMPT.md` first — it is the
**canonical spec** with all locked decisions and rationale. This file is the short operational
contract; the init prompt is the source of truth.

## What this project is

A **local MCP server** that makes NVIDIA Nsight Compute (`ncu`) profiler output
**token-efficient** for LLM coding agents. It reads a report from disk and returns a small,
structured, numeric signal, keeping a `raw_ref` pointer for lazy expansion.

We are a **translator/router, NOT a judge.** Provide clean numeric metrics deterministically;
interpretation ("is this kernel memory-bound?") is the model's job. Do not build verdict/threshold
heuristics.

## Non-negotiable rules (violating any of these is a bug)

1. **`None` ≠ `0.0`.** A missing metric means "not measured in this export", never zero. Never
   fill a gap with `0.0`. `get_metrics` returns `"not_available_in_this_export"` for a requested
   metric the export lacks. This is the single most dangerous failure mode.
2. **`format` is mandatory** in v1 — the agent passes what it produced. No auto-detect /
   `detect_format()` in v1.
3. **Lazy-import `ncu_report`** (and other heavy/CUDA-only libs) inside the function that uses
   them, so CSV-only paths never load them.
4. **`mapping.py`: vendor jargon → standard term**, never → invented abstract name
   (`l2_hit_rate`, not `low_level_cache_usage`).
5. **`server/` is a thin shell** — no business logic. Each tool = call reader + filter + return
   JSON. Logic lives in `core/` + `adapters/` so a future CLI can reuse it.
6. **Compression is lazy, not lossy** — never delete the raw report; always expose `expand`.
7. **Profiler invocation suppresses stdout:** `ncu --set full -o report.ncu-rep ./app`. Document
   this; if `ncu` prints to stdout the raw table pollutes the agent's context before perfdigest
   runs.

## Build order — do NOT one-shot. Complete a phase, show it, then proceed.

1. **Contract** — `core/metrics.py` (`NormalizedKernel`) + `core/digest.py` (JSON schema +
   absence convention). Types only, no parsing.
2. **PRI reader in isolation** — `adapters/nsight/pri_reader.py`: `.ncu-rep` path →
   `NormalizedKernel`, tested with a plain script against a REAL report. ~80% of the work.
3. **`mapping.py`** — the metric-name dict (most-changed file, isolated).
4. **Server shell** — FastMCP + the 3 tools (`list_kernels`, `get_metrics`, `expand`).
5. **Convention** — `server/prompts.py` (usage convention/vocabulary) + `report_store/discovery.py`.
6. Then **`csv_reader.py`** as a second reader on the same contract (extension-ready proof).

Current state: **v1 implemented and validated.** All phases complete — contract, PRI
reader, mapping (14 terms; default core set confirmed with the user), the 3-tool FastMCP
server, prompts, and report discovery. 12 pytest tests pass (4 contract + 8 against a real
`.ncu-rep`). End-to-end verified through Claude Code headless with Sonnet 4.6, and an A/B
context-efficiency benchmark is in `eval/RESULTS.md`. The CSV fallback reader
(`csv_reader.py`) is the one remaining stub — next extension point.

Real-report lesson (baked into `mapping.py`): the init prompt's example
`dram__throughput.avg.pct_of_peak_sustained_elapsed` does **not** exist in ncu 2026.1 — the
DRAM %-of-peak is `gpu__dram_throughput...`. The absence convention caught it (surfaced
`not_available_in_this_export` instead of a fake 0.0). Always verify metric names against a
real report before trusting a mapping entry.

## The 3 tools (target signatures)

```python
list_kernels(report_ref: str, format: str) -> list[dict]   # [{name, index, duration_us}]
get_metrics(report_ref: str, format: str, kernel: str,
            metrics: list[str] | None = None) -> KernelDigest  # None => default core set
expand(report_ref: str, format: str, kernel: str, section: str) -> dict
```

## Open items — ASK the user, do not invent silently

- **Phase 3** (`mapping.py`): the exact full `ncu` metric → standard term list, and what belongs
  in the `metrics=None` default core set.
- **Phase 5** (`prompts.py`): the init prompt vocabulary + the "after profiling, don't cat the
  CSV, call perfdigest" usage convention.

## Environment & commands

- **Python 3.11+**, packaged with **uv** (src-layout). Entry point: `perfdigest =
  perfdigest.server.app:main`.

```bash
uv sync                 # base server (CSV path works without CUDA)
uv sync --extra cuda    # + ncu_report (NVIDIA PRI) for native .ncu-rep parsing
uv run perfdigest       # run the MCP server over stdio
uv run pytest           # tests (after fixtures land)
```

- **`ncu_report` is now on PyPI** as `ncu-report` (the init prompt §3 predates this and says
  "not on PyPI" — that assumption is outdated). It is declared as the `cuda` optional extra and
  imported lazily. It can read `.ncu-rep` files **without a GPU present**, so the PRI reader is
  fully developable/testable on any machine; only *producing* fixture reports needs a GPU
  (`ncu --set full -o report.ncu-rep ./app`).
- `uv.lock` is committed for reproducible installs. Generated on first `uv lock`/`uv sync`.

## Conventions

- Standard hardware metric names in output JSON (lowercase, e.g. `dram_pct_peak`,
  `achieved_occupancy`, `l2_hit_rate`).
- Keep `core/` free of any vendor metric names — it only knows `NormalizedKernel`.
- `eval/` is a separate top-level dir and must **not** ship inside the package.
- `tests/fixtures/` holds REAL `.ncu-rep` samples (gitignored binaries; curate a small set).
