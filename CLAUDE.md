# CLAUDE.md — perfdigest

Guidance for Claude Code working in this repo. Read `docs/INIT_PROMPT.md` first — it is the
**canonical spec** with all locked decisions and rationale. This file is the short operational
contract; the init prompt is the source of truth.

## What this project is

A **local, multi-backend MCP server** that makes performance-profiler output
**token-efficient** for LLM coding agents. It reads a report from disk and returns a small,
structured, numeric signal, keeping a `raw_ref` pointer for lazy expansion. v1.0.0 backends:
NVIDIA `ncu` (native + CSV), AMD `rocprof`, Linux `perf` (CPU C++/Rust), Apple Metal. Dispatch
is a `format → Backend` registry (`core/backend.py`, `adapters/registry.py`); the `server/`
shell never imports a concrete reader.

We are a **translator/router, NOT a judge.** Provide clean numeric metrics deterministically;
interpretation ("is this kernel memory-bound?") is the model's job. Do not build verdict/threshold
heuristics.

**Two operations (split in the prompt):** *digest* (read) is universal/ungated — a Mac digests a
CI-produced NVIDIA report; *capture* (run a profiler) is platform-verified in `platform/`. Gating
lives on the capture/advisor tier (`platform/capabilities.py`), never on readers. A new backend =
a folder under `adapters/` with a reader + `mapping.py` + a `backend.py` that `register()`s a
`Backend` (formats, suffixes, domain, platforms, default_core_set, probe, capture_command).

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

Current state: **v1.0.0 — multi-backend.** The v1 NVIDIA path is refactored onto the registry;
the `csv_reader.py` stub is now a real pure-Python reader (NVIDIA CSV digest with no GPU/PRI
wheel). Added: AMD `rocprof`, Linux `perf` (CPU `cpu_function` units + CPU vocabulary), Metal
(`gpu_pass`), the `platform/` capability layer, and the two capture-advisory tools. 58 pytest
tests pass (+16 hardware-gated skips), CI runs the Linux/macOS/Windows matrix, Apache-2.0,
PyPI + Claude Code/Codex configs ready. The original A/B benchmark is in `eval/RESULTS.md`.

Real-report lesson (baked into `mapping.py`): the init prompt's example
`dram__throughput.avg.pct_of_peak_sustained_elapsed` does **not** exist in ncu 2026.1 — the
DRAM %-of-peak is `gpu__dram_throughput...`. The absence convention caught it (surfaced
`not_available_in_this_export` instead of a fake 0.0). Always verify metric names against a
real report before trusting a mapping entry.

## The tools (5 — registry-dispatched, thin shell)

Tier 1 — digest (any backend, any host; `format` is mandatory):
```python
list_kernels(report_ref, format) -> list[dict]   # [{name, index, duration_us, domain}]
get_metrics(report_ref, format, kernel, metrics=None) -> dict   # None => backend default core set
expand(report_ref, format, kernel, section) -> dict             # raw vendor metrics
```
Tier 2 — capture advisory (platform-verified):
```python
platform_capabilities() -> dict                  # can_digest (universal) vs can_capture_here (gated)
suggest_profile_command(backend, target) -> dict # correct invocation, or a refusal that redirects
```

## Extending — adding a backend

A backend is a folder under `src/perfdigest/adapters/<name>/`: a reader
(`*_reader.py` → `list[NormalizedUnit]` + `raw_metrics`), a `mapping.py` (vendor →
standard terms + `DEFAULT_CORE_SET`), and a `backend.py` that builds + `register()`s
a `Backend`. Then add its import to `server/app.py:_register_backends()`. The server
shell and `core/` never change. Templates: `adapters/linux_perf/` (CPU vocabulary),
`adapters/rocm/` (GPU, wide CSV).

## Environment & commands

- **Python 3.11+**, packaged with **uv** (src-layout). Entry points: `perfdigest-mcp`
  / `perfdigest` = `perfdigest.server.app:main`. Repo root **is** the package now.

```bash
uv sync --extra dev      # base + pytest; all pure-Python readers work with no GPU
uv sync --extra nvidia   # + ncu_report (NVIDIA PRI) for native .ncu-rep (alias: --extra cuda)
uv run perfdigest        # run the multi-backend MCP server over stdio
uv run pytest            # 58 pass + 16 hardware-gated skips
```

- **`ncu_report` is now on PyPI** as `ncu-report` (the init prompt §3 predates this and says
  "not on PyPI" — that assumption is outdated). It is declared as the `nvidia` optional extra and
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
