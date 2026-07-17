# perfdigest

Local **MCP server** that makes development-loop reports **token-efficient** for
LLM coding agents — profiler output first, and since v1.2 the rest of the loop's
feedback artifacts too (build traces, build diagnostics, benchmark results, CI
run logs, repo change state). It sits between the tool and the agent: reads the
report from disk and returns a small, structured, numeric digest the agent can
act on, while keeping a pointer back to the raw report for lazy expansion.

It is a **translator/router, not a judge** — interpretation ("is this kernel
memory-bound?") is the model's job. perfdigest only provides efficient,
deterministic access to clean numeric metrics across vendors and languages.

## Two operations — keep them separate

perfdigest deliberately splits what an agent does with a profiler into two tiers:

1. **Digest (read a report) — universal, available on every machine.** A report's
   origin is irrelevant to digesting it. An NVIDIA `.ncu-rep` (or its CSV export)
   captured on a CI/remote GPU runner can be pulled to a **Mac** and digested
   there — that is how you do CUDA work on a Mac via CI. Tier-1 is never gated by
   local hardware, only by whether a reader's dependency imports.
2. **Capture (produce a report) — platform-verified.** You cannot run `ncu` on a
   Mac, Metal on Linux, or hardware PMU counters under WSL2. `platform_capabilities`
   and `suggest_profile_command` gate this so the agent never spends context on a
   capture that can't run here, and redirect it to "capture elsewhere, digest here."

## The four feedback channels (v1.2 — the Development Observatory)

Every change a session makes propagates through the same four stages, each with
its own verbose artifact: *what changed* (RepoState) → *does it build, at what
cost* (BuildDigest) → *does it pass* (CIDigest) → *how fast is it* (PerfDigest).
perfdigest digests all four through the **same seven tools** — a backend is a
registry row, not a new API surface. One principle binds the whole matrix: we
**bind to artifacts, never to live APIs** — every reader parses a file on disk
whose format is frozen by its ecosystem's backward-compat pressure. (CI
*artifacts* need no dedicated backend at all: a downloaded `.ncu-rep` from a CI
runner is digested by `nsight` — capture anywhere, digest anywhere, by
composition.)

| Channel | Backend | `format` | Domain | Capture tool | Digest anywhere |
|---|---|---|---|---|---|
| Perf | `nsight` | `ncu-rep` | gpu_kernel | NVIDIA `ncu` (Linux/Windows) | needs `ncu-report` wheel |
| Perf | `nsight_csv` | `ncu-csv` | gpu_kernel | (export of `ncu`) | ✅ pure Python |
| Perf | `rocm` | `rocprof-csv` | gpu_kernel | AMD `rocprof` (Linux/Windows) | ✅ pure Python |
| Perf | `linux_perf` | `perf-stat-json`, `perf-report` | cpu_function | Linux `perf` | ✅ pure Python |
| Perf | `metal` | `metal-trace` | gpu_pass | Apple `xctrace` (macOS) | ✅ pure Python |
| Perf | `chrome_trace` | `torch-trace`, `chrome-trace` | framework_op + gpu_kernel | `torch.profiler` → `export_chrome_trace()` | ✅ pure Python |
| Build | `ptxas` | `ptxas-verbose` | kernel_codegen | `nvcc -Xptxas -v` (**no GPU needed**) | ✅ pure Python |
| Build | `clang_time_trace` | `clang-time-trace`, `ftime-trace` | build_phase | `clang -ftime-trace` | ✅ pure Python |
| Build | `cmake_profile` | `cmake-profile`, `cmake-trace` | build_phase | `cmake --profiling-format=google-trace` (≥3.18) | ✅ pure Python |
| Build | `ninja_log` | `ninja-log` | build_step | `.ninja_log` (byproduct of any `ninja` build) | ✅ pure Python |
| Build | `cargo_diag` | `cargo-diag`, `cargo-json` | build_diag | `cargo build --message-format=json` | ✅ pure Python |
| Build | `criterion` | `criterion`, `criterion-json` | benchmark | `cargo bench` (`target/criterion` dir) | ✅ pure Python |
| CI | `gha_log` | `gha-log`, `gh-run-log` | ci_step | saved `gh run view <id> --log` | ✅ pure Python |
| RepoState | `git_numstat` | `git-numstat`, `numstat` | repo_change | saved `git diff --numstat -M` | ✅ pure Python |

GPU backends share one vocabulary (`compute_pct_peak`, `dram_pct_peak`,
`l2_hit_rate`, `achieved_occupancy`, …); the CPU backend introduces a CPU
vocabulary (`ipc`, `cache_miss_rate`, `llc_miss_rate`, `branch_mispredict_rate`,
`self_pct`, …). The `ptxas` backend adds the **codegen layer**: runtime counters
say *what* is slow, `registers_per_thread` / `spill_stores_bytes` /
`static_smem_bytes` say *why the code became what it is* — captured at compile
time, so it works in any CI job with no GPU attached.

RepoState is deliberately **session-retrospective, not GitHub-shaped**: a saved
`numstat` snapshot is the token-cheap answer to "what has this session changed
so far", and it doubles as the provenance anchor that pins perf/build/CI reports
to a code state. It emits facts (files, line counts, honest `None` for binary
files) — never verdicts like "release ready"; concluding is the agent's job.
Future: Go, Java, and other perf-critical runtimes.

## Two load-bearing invariants (read before running)

1. **Suppress profiler stdout — write to a file.** e.g. `ncu --set full -o report.ncu-rep ./app`.
   If the profiler prints its summary to stdout, that raw table enters the agent's
   context *before* perfdigest runs — defeating the entire purpose.
2. **`None` means "not measured in this export", NEVER zero.** A metric the export
   does not contain is returned as `not_available_in_this_export`, not a fake `0.0`.
   A genuine `0.0` (e.g. zero branch divergence) is preserved. Silently returning
   zero = lying to the model = the worst bug this tool can have.

## Tools

Tier 1 — digest (any backend, any host):

- `summarize_report(report_ref, format, top_n=5)` → the N hottest units + core
  metrics in one call (ranked by `duration_us`, falling back to `self_pct`, then
  file order; reports duration coverage of the returned units)
- `list_kernels(report_ref, format)` → `[{name, index, duration_us, domain}]`
- `get_metrics(report_ref, format, kernel, metrics=None)` → compact digest
  (`metrics=None` → the backend's default core set)
- `compare_metrics(report_a, report_b, format, kernel, kernel_b=None)` → the
  measure→edit→measure tool: `{a, b, delta, delta_pct}` per metric with
  `delta = b − a`; a metric missing on either side yields an honest
  `not_available_in_this_export` delta, never a fake `0.0`
- `expand(report_ref, format, kernel, section)` → raw vendor metrics (the safety valve)

Reports are parsed once per file version (an mtime/size-keyed cache), so
repeated digests of the same report cost no re-parse.

Tier 2 — capture advisory (platform-verified):

- `platform_capabilities()` → machine identity + `can_digest` (universal) vs
  `can_capture_here` (gated)
- `suggest_profile_command(backend, target)` → the correct, platform-aware
  invocation, or a refusal that redirects to the tier-1 path

`format` is **mandatory** — the agent passes what it produced; a path says *where*
a file is, not *what* format it is.

## Install & connect

```bash
uvx perfdigest-mcp                      # run from PyPI (downloadable)
uv tool install "perfdigest-mcp[nvidia]"  # + NVIDIA native binary reader (Linux/Windows)
```

> PyPI/install name is **`perfdigest-mcp`**; the command and import package are
> `perfdigest` (e.g. `uvx perfdigest-mcp`, `import perfdigest`).

Claude Code and OpenAI Codex setup (both stdio MCP): see [`docs/clients.md`](docs/clients.md).

## Status

**v1.2.0 (pre-release, in review)** — the Development Observatory release:
seven new backends extend the digest matrix from profilers to the whole
edit→build→verify→measure loop. BuildDigest: `clang_time_trace` (compiler-phase
traces, with `[total]` aggregate tagging), `cmake_profile` (configure-step
traces — cmake emits bare `B`/`E` event pairs, stack-folded honestly),
`ninja_log` (per-edge build timings, v5–v7 logs), `cargo_diag` (rustc
diagnostics — a backend whose durations are *honestly absent*), `criterion`
(Rust benchmark estimates; first directory-shaped report ref). CIDigest:
`gha_log` (saved `gh run view --log` files, error/warning annotations,
previous-green comparison proven on real two-run fixtures). RepoState:
`git_numstat` (session-retrospective change digest; binary files surface as
honest absence, never `0.0`). Zero new tools — every backend rides the same
seven; 206 tests. Every fixture in the release was produced by the real tool it
mimics — several briefs were corrected by what the artifacts actually contained.

**v1.1.1** — contract-hardening hotfix (issues #2/#4/#5): registry format
collisions are detected on the normalized key (a case-variant format can no
longer silently hijack a backend); `metrics=[]` now means an empty request
(only `None` selects the default core set); exact unit names win over numeric
index interpretation, with `#3` as the explicit index form; unreadable backends
surface a diagnostic in `platform_capabilities` (`can_digest_issues`) instead
of a silent `False`; and publishing is gated — a tag only reaches PyPI after
the full suite, a clean-venv wheel install, and a tool-surface check pass.

**v1.1.0** — the agent-loop release: `compare_metrics` (before/after deltas),
`summarize_report` (one-call top-N), a parse-once report cache, the `ptxas`
codegen backend (compile-time registers/spills/smem; capture needs `nvcc`, not a
GPU), and the `chrome_trace` backend (torch/Kineto and any Chrome-trace emitter:
framework ops + the CUDA kernels they launched, one report). Also normalizes
`perf` scope-modifier event names (`cycles:u` on `perf_event_paranoid>=2`
hosts).

**v1.0.0** — multi-backend registry; NVIDIA (native + CSV), AMD HIP, Linux perf
(C++/Rust), and Apple Metal adapters; platform capability gating; cross-client
config. Validated on the Linux/macOS/Windows CI matrix (pure-Python readers run
hardware-free against committed fixtures). Real binary-capture tests are
fixture-gated and skip without the device.

**Benchmarks & evaluation:** token-efficiency A/B studies and real-hardware
cross-backend runs live in
[PerfDigest-MCP-Bench](https://github.com/onlyxItachi/PerfDigest-MCP-Bench)
(digest ≈14–130x fewer tokens per turn than raw `ncu` output; see
[`eval/README.md`](eval/README.md)).

## Related / similar projects

perfdigest was authored independently; these adjacent MCP servers occupy a nearby
space and likely work well for their narrower scope. The differences are the
reason perfdigest exists:

- **nsys-mcp** (NVIDIA Nsight *Systems*) — profiles binaries and aggregates trace
  timeline stats. perfdigest targets Nsight *Compute* per-kernel counters, is
  read-only (does not run the profiler), and spans multiple vendors.
- **pprof-analyzer-mcp** / **Profiler-MCP** — Go (and Go/Python/Java) CPU/memory
  profiles, often rendering flamegraphs. perfdigest is a numeric digester focused
  on token/context efficiency rather than visualization.

What is distinct here: the **multi-backend matrix** (NVIDIA + AMD HIP + CPU perf +
Metal under one neutral contract), the **token-efficiency thesis** with the
`None`≠`0.0` honesty rule, and the **read/capture split with platform capability
gating** (digest anywhere, capture only where supported).

## License

Apache-2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
