# CLAUDE.md — perfdigest

Guidance for Claude Code working in this repo. **This file is the live contract** —
where it and any other document disagree about the current state, this one wins.

`docs/INIT_PROMPT.md` is the original kickoff spec, kept deliberately FROZEN as the
record of why the locked design decisions were made (the absence convention, the
digest/capture split, mandatory `format`, mapping-not-inventing). Read it for
rationale, never for current facts: it predates the release and its concrete
numbers are superseded — it says Python 3.11+ (the floor is now 3.10) and describes
three tier-1 tools (there are five, plus two capture-advisory ones), and its claim
that `ncu_report` is not on PyPI stopped being true. Those are history, not bugs.

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

Current state: **v1.2.0 — the Development Observatory release (RELEASED, on PyPI).**
The digest matrix now covers the four feedback channels of the dev loop — RepoState (what
changed) -> BuildDigest (does it build) -> CIDigest (does it pass) -> PerfDigest (how fast) —
through the SAME seven tools; a backend is a registry row, never a new API surface, and every
reader binds to an on-disk artifact, never a live API. On the v1.1 base (compare_metrics,
summarize_report, parse-once cache, `ptxas`, `chrome_trace`), v1.2 adds: `clang_time_trace`
(reuses chrome_trace's `_grouped` with a `tag_override` hook; `Total *` aggregates are tagged
`[total]` because below `-ftime-trace-granularity` they are sometimes the ONLY complete
measurement), `cmake_profile` (cmake emits bare B/E pairs with NO ph=='X' events — the
`fold_be_pairs` hook stack-folds them; unpaired B is dropped, never given a fabricated
duration), `ninja_log` (v5-v7 logs, last-occurrence-wins rebuild semantics, per-edge times
overlap under parallel builds), `cargo_diag` (crate units, durations honestly absent — the
stream carries no timing; counts are genuine 0.0, complete-enumeration rule like ptxas),
`criterion` (FIRST directory-shaped report_ref: `Backend.report_is_directory` +
`resolve_report(accept_dir=)` + a cache bypass because a dir's mtime doesn't change on nested
rewrites), `gha_log` (saved `gh run view --log` files; current gh exports often can't attribute
lines to steps -> units degrade honestly to per-job; failed runs typically finish FASTER than
green — they died early), and `git_numstat` (RepoState: session-retrospective `git diff
--numstat -M` snapshots; binary files are `-`/`-` -> honest absence, never 0.0; renames parsed
from git's real brace grammar). ptxas honesty nuance still holds: the `Used ...` line is a
complete enumeration, so an omitted component there is a GENUINE 0.0; `barriers` stays None on
old toolkits. 206 tests pass (+16 hardware-gated skips). Benchmark artifacts live in the
companion repo https://github.com/onlyxItachi/PerfDigest-MCP-Bench (token A/Bs, cross-backend
runs, demanding-workload studies; local `eval/README.md` is the slim pointer).

Real-report lesson (baked into `mapping.py`): the init prompt's example
`dram__throughput.avg.pct_of_peak_sustained_elapsed` does **not** exist in ncu 2026.1 — the
DRAM %-of-peak is `gpu__dram_throughput...`. The absence convention caught it (surfaced
`not_available_in_this_export` instead of a fake 0.0). Always verify metric names against a
real report before trusting a mapping entry.

## The tools (7 — registry-dispatched, thin shell)

Tier 1 — digest (any backend, any host; `format` is mandatory):
```python
summarize_report(report_ref, format, top_n=5, metrics=None) -> dict  # one-call top-N + coverage
list_kernels(report_ref, format) -> list[dict]   # [{name, index, duration_us, domain}]
get_metrics(report_ref, format, kernel, metrics=None) -> dict   # None => backend default core set
compare_metrics(report_a, report_b, format, kernel, kernel_b=None, metrics=None) -> dict
    # measure->edit->measure deltas: {a, b, delta, delta_pct}, delta = b - a
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
`adapters/rocm/` (GPU, wide CSV), `adapters/clang_time_trace/` (reusing the chrome-trace
machinery via hooks), `adapters/criterion/` (directory-shaped report refs).

## Environment & commands

- **Python 3.10+** (the floor `pyproject.toml` declares and CI tests), packaged with
  **uv** (src-layout). CI runs 3.10/3.11/3.12/3.13/3.14 and free-threaded 3.14t on
  Linux, macOS and Windows; the one known gap is Windows + 3.14t, where `mcp`'s
  `pywin32` dependency has no cp314t wheel yet (that cell is an allowed failure).
  Entry points: `perfdigest-mcp` / `perfdigest` = `perfdigest.server.app:main`.
  Repo root **is** the package now.

```bash
uv sync --extra dev      # base + pytest; all pure-Python readers work with no GPU
uv sync --extra nvidia   # + ncu_report (NVIDIA PRI) for native .ncu-rep (alias: --extra cuda)
uv run perfdigest        # run the multi-backend MCP server over stdio
uv run pytest            # 206 pass + 16 hardware-gated skips
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
- Heavy benchmark artifacts (workload sources, result docs, captured reports like
  `.ncu-rep`) belong in the companion repo
  https://github.com/onlyxItachi/PerfDigest-MCP-Bench — never in this repo. Local
  `eval/` keeps only the slim pointer README and must **not** ship inside the package.
- `tests/fixtures/` holds REAL `.ncu-rep` samples (gitignored binaries; curate a small set).
