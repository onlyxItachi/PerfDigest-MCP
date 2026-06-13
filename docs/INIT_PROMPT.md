# perfdigest — Project Initialization Prompt

> Kickoff context for the coding agent (e.g. Claude Code). It encodes **locked decisions**
> and their rationale so you do not re-litigate them, plus explicit **non-goals** so you do
> not expand scope. Build in the phase order given. **Do not one-shot the whole thing** —
> complete a phase, show it, then proceed.

## 1. Mission

Build a local MCP server that makes GPU profiler output (NVIDIA Nsight Compute / `ncu`)
token-efficient for LLM coding agents.

The problem: profiler reports are written for humans — huge tables (a single `ncu` kernel
report can be thousands of tokens of comma/equals/semicolon noise). When an LLM agent reads
that raw output into its context, the signal it actually needs (is the kernel memory-bound? is
occupancy low? is L2 hit suspiciously high?) drowns in noise, the context window fills, and
decision quality drops.

perfdigest sits between the profiler and the agent: it reads the report from disk and returns
a small, structured, numeric signal the agent can act on, while keeping a pointer back to the
raw report for lazy expansion.

## 2. What this is — and is NOT (read carefully)

- This is **NOT a verdict engine.** Do not build heuristics that classify a kernel as
  `memory_bound` from thresholds. Interpretation is the model's job; the model is already
  smart. Our job is efficient, deterministic access to clean numeric metrics. We are a
  **translator/router, not a judge.**
- This is **NOT a profiler runner** (in v1). The agent runs `ncu` itself with its own shell
  (build/run/flags/error-loops are the agent's job). perfdigest only reads, parses, and
  returns efficiently.
- The whole point is **token/context efficiency, not latency.** The tool is I/O-bound and
  called infrequently; the dominant cost around it is the profiler run (seconds) and the LLM
  turn. Do not optimize Python runtime speed, do not reach for C++/Cython/GIL tricks — there
  is no hot loop here.

### Core thesis (the win)

The gain is **cumulative, not per-call.** An agent optimizing a kernel runs the
measure→edit→measure loop many times. Raw CSV in context is permanent and stacks every turn
(10 turns × ~2500 tokens = 25k tokens all resident). Our return is ~150 dense tokens per turn.
We are buying **signal-to-noise ratio**, which also improves the model's decisions — not just
saving tokens.

Compression must be **lazy, not lossy**: never delete the raw report, reference it (`raw_ref`)
and expose an `expand` tool. Success criterion: the returned set must be **sufficient** enough
that the agent never feels the urge to read the raw file. Sufficiency, not token count, is the
hard design target.

## 3. Locked technical decisions

- **First vertical: CUDA / Nsight Compute (`ncu`).** Keep the core profiler-agnostic so
  `perf` / `cargo-bench` / `wasm` adapters can be added later without touching decision logic.
- **Language: Python only (3.11+).** One language does everything; "terminal interaction" is
  just `subprocess`. No JS/TS, no separate shell layer. The profiled code may be CUDA/Rust/WASM,
  but that only changes which adapter calls which profiler — never the toolkit's own language.
- **Packaging: uv + src-layout + FastMCP (`mcp[cli]`).**
- **Output: structured JSON**, using standard hardware terms (`l1_hit_rate`,
  `achieved_occupancy`, `dram_pct_peak`) — NOT invented abstract names like
  `low_level_cache_usage`. The model already knows the standard terms; inventing a dialect adds
  ambiguity and re-teaching cost.
- **Tool granularity: few tools + a `metrics=[...]` parameter** to select what the agent wants.
  NOT one tool per metric (that means extra round-trips and a bloated schema).
- **Parsing strategy:** primary reader = NVIDIA's **Python Report Interface** (`ncu_report`, a
  C-backed module). It reads the native `.ncu-rep` binary structurally — we never guess byte
  layout; NVIDIA wrote both the writer and the reader. **CSV reader is a fallback**, added after
  PRI works. Import `ncu_report` **lazily** (only when actually parsing a CUDA report) so
  CSV-only paths never load it and cold-start stays minimal.

## 4. v1 scope boundaries (explicit non-goals — do not implement)

- **Only the agent-produced flow.** The agent ran the profiler, so it KNOWS the format. The
  tool's `format` parameter is **mandatory** in v1; the agent passes what it produced. The
  format is not a guess — it is the record of the agent's own action.
- **No auto-detect / `detect_format()` in v1.** The "orphan file" case (user hand-points at a
  pre-existing file the agent did not produce) is deferred to v2. A path tells you *where* a
  file is, not *what* format it is — different questions; we deliberately do not solve the
  "what" question in v1.
- **Profiler is not run by the toolkit in v1.** Reading only.
- **CSV reader exists but the v1 skeleton stands up on PRI first.**

## 5. Architecture & folder structure

```
perfdigest/
├── pyproject.toml          # uv; entry point + deps (see install gotcha below)
├── README.md
├── src/perfdigest/
│   ├── core/               # pure, profiler-agnostic
│   │   ├── metrics.py      # NormalizedKernel — the neutral contract
│   │   └── digest.py       # JSON response schema + absence marking
│   ├── adapters/nsight/    # one adapter per profiler; readers live inside it
│   │   ├── pri_reader.py   # PRIMARY: ncu_report (lazy import) -> NormalizedKernel
│   │   ├── csv_reader.py   # FALLBACK: --csv -> NormalizedKernel (after PRI)
│   │   └── mapping.py      # ncu metric name -> standard term
│   ├── report_store/
│   │   └── discovery.py    # report path resolution / "newest report" handling
│   └── server/
│       ├── app.py          # FastMCP instance
│       ├── tools.py        # tool signatures (thin shell, no business logic)
│       └── prompts.py      # init convention + vocabulary (MCP prompt primitive)
├── tests/fixtures/         # REAL .ncu-rep samples
└── eval/                   # SEPARATE — A/B harness, must NOT ship in the package
```

**Design rationale:** `core/` never sees a vendor metric name — it only knows
`NormalizedKernel`. Adding `perf`/`cargo` later = writing a new adapter, not touching core.
`mapping.py` is the most-changed file (metric translation), so it is isolated. `server/` is a
thin shell that calls core+adapter+store; keep business logic out of it so a future CLI can wrap
the same core. `eval/` is a separate top-level dir because it tests the toolkit and must not be
in the shipped package.

## 6. The three load-bearing contracts

Everything else is detail. Get these three right and the project stands.

### 6.1 Tool signatures (3 tools)

```python
list_kernels(report_ref: str, format: str) -> list[dict]
    # -> [{ "name": str, "index": int, "duration_us": float }]

get_metrics(report_ref: str, format: str, kernel: str,
            metrics: list[str] | None = None) -> KernelDigest
    # metrics=None  =>  return the default core set (a meaningful package)
    # metrics=[...]  =>  return only those fields

expand(report_ref: str, format: str, kernel: str, section: str) -> dict
    # lazy raw access — the safety valve when the digest is not enough,
    # so the agent never resorts to cat-ing the raw file
```

`format` is mandatory in v1. `metrics=None` returns a sensible default package so a vague call
still yields signal. `expand` is what kills the agent's urge to read raw output — do not omit it.

### 6.2 NormalizedKernel + absence honesty (the most dangerous failure mode)

`NormalizedKernel` is the neutral contract every reader fills in. The critical rule:

> A field that is `None` means **"not measured in this export format"** — it **NEVER** means
> zero. Readers must NEVER fill a missing metric with `0.0`.

And `get_metrics` must surface this transparently: for a requested metric the given export does
not contain, return `"<metric>": "not_available_in_this_export"` rather than a fake value.
Different export methods carry different information (a `.ncu-rep` from `--set full` has
everything; a `--page source` dump may lack high-level occupancy/DRAM/cache metrics entirely).
Silently returning zero for missing data = lying to the model = **the single worst bug this tool
can have.** Reading a format ≠ extracting the same rich signal from it; be honest about the gap.

### 6.3 Mapping direction rule

`mapping.py`'s one rule: **vendor-jargon → known standard term, never → invented abstract name.**

```
sm__throughput.avg.pct_of_peak_sustained_elapsed    -> compute_pct_peak
dram__throughput.avg.pct_of_peak_sustained_elapsed  -> dram_pct_peak
lts__t_sector_hit_rate.pct                           -> l2_hit_rate
sm__warps_active.avg.pct_of_peak_sustained_active    -> achieved_occupancy
```

It is a plain Python dict; extending it is trivial. (Exact full list is an open item — see §10.)

## 7. Build order (build inside-out; wrong order paints you into a corner)

1. **Contract first.** Write `NormalizedKernel` (`core/metrics.py`) and the JSON schema
   (`core/digest.py`). No parsing logic — just types and the absence convention. Both reader and
   server lean on this; fix the shared language before stitching ends together.
2. **Reader in isolation.** Write `pri_reader.py` completely independent of MCP: input = a
   `.ncu-rep` path, output = a `NormalizedKernel`. Test with a plain Python script against a REAL
   report file. ~80% of the work finishes here with zero MCP involvement.
3. **Mapping dict** (`mapping.py`), separate file (most-changed part).
4. **Server shell.** FastMCP, the 3 tool signatures, each tool = "call reader + filter + return
   JSON". Thin shell, no logic.
5. **Convention / prompt.** Init system-prompt (`prompts.py`) + report_store disambiguation.
   Behavioral polish, last.

Then add `csv_reader.py` as a second reader bound to the same contract — a cheap proof the
architecture is extension-ready.

## 8. Invariants & gotchas (put the first two in the README)

1. **Profiler command must suppress stdout and write to a file**, e.g.
   `ncu --set full -o report.ncu-rep ./app`. If stdout is not suppressed, `ncu`'s summary enters
   the agent's context before perfdigest runs — defeating the entire purpose. Document the
   correct invocation prominently.
2. **uvx install must include dependencies in the launch command.** A known pitfall: even with
   deps in `pyproject.toml`/lockfile, the default `uv run` invocation may omit them and the
   server fails with `ModuleNotFoundError` at launch. Define the entry point and deps correctly
   so install does not break.
3. **Collect wide once.** The agent should profile with a broad default collection set
   (`--set full`: roofline + occupancy + cache + stalls) so the file contains everything; any
   later `get_metrics` question is answerable from the one file with no re-profiling round-trip.
   Slow-but-single-run beats fast-but-maybe-rerun.
4. **Report disambiguation.** A debug dir accumulates many reports over time.
   `report_store/discovery.py` resolves the right one (kernel + timestamp in the export name, or
   "newest"). Avoid reading stale data.
5. **Lazy imports for heavy libs** (`ncu_report`, etc.) to keep cold-start near zero. The stdio
   server is long-lived (resident), so this is a one-time cost anyway — but lazy-load the
   CUDA-only deps so CSV-only paths skip them.
6. **Convention, not sandbox.** "The agent won't cat the raw file" is enforced by the init
   prompt, not by the OS. A well-behaved agent that follows the prompt is where the value comes
   from. Be honest about this in docs.

## 9. Evaluation (separate effort, `eval/` dir)

- The toolkit is built with the agent; it is evaluated with a small A/B harness that calls the
  API in tool-use mode (toolkit-on vs toolkit-off, where "off" = the realistic baseline of the
  agent `cat`-ing the raw output — an honest baseline is the test's integrity).
- **First metric: context efficiency** (tokens spent, same task, on vs off). It is the central
  thesis and is measured exactly.
- **Subject: a weak model / low reasoning effort** (e.g. Haiku, or Sonnet at high not max
  effort). A strong model can parse raw CSV anyway and masks the toolkit's contribution;
  weakening the model amplifies the visible value — proper ablation.
- **Batch only single-turn diagnosis tests** (independent, fixed prompts). The agentic
  multi-turn loop CANNOT be batched: turn N+1's input depends on turn N's model output. If you
  script the steps to batch them, you are testing your script, not the model's decision-making —
  which erases what the toolkit is for.

## 10. Open items to resolve with the user before/while coding

- **Full `mapping.py` contents:** exact `ncu` metric → standard term list, and what belongs in
  the `metrics=None` default core set.
- **Init prompt text (`prompts.py`):** the vocabulary + the "after profiling, don't cat the CSV,
  call perfdigest" usage convention.

Surface these as questions when you reach Phase 3 and Phase 5; do not invent the full list
silently.
