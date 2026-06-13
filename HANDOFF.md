# Handoff — continue perfdigest on Linux

Written 2026-06-13. Picking up on a Linux box (no Windows Smart App Control to
fight). This file is the full context to resume; read it top to bottom.

## TL;DR — what's left

perfdigest **v1 is implemented, committed, and validated** on Windows. The only
unfinished item is **regenerating the profiler fixture from the richer 8-launch
benchmark harness and adding the family-specific tests** — the Windows step that
got blocked. On Linux this is ~10 minutes. Jump to "Resume on Linux".

## Why we moved

Windows **Smart App Control** (enforce mode) blocks every freshly-compiled
*unsigned* binary — `gafime_bench.exe` was refused by Code Integrity ("did not
meet Enterprise signing level", event IDs 3033/3077), so `ncu` couldn't launch
it to profile. Admin rights, self-signing, and local certs do **not** bypass SAC;
only disabling it (a one-way door on Windows) or building on Linux does. Linux has
no SAC — locally built binaries just run. Nothing in the repo is Windows-specific.

## Project state (committed, branch `main`)

- **perfdigest v1**: contract (`core/metrics.py`, `core/digest.py`), PRI reader
  (`adapters/nsight/pri_reader.py`), mapping (`adapters/nsight/mapping.py`, 14
  standard terms + the confirmed default core set), 3-tool FastMCP server
  (`server/tools.py`: `list_kernels`/`get_metrics`/`expand`), usage prompt,
  report discovery. The CSV fallback reader (`adapters/nsight/csv_reader.py`) is
  still a stub — the one remaining feature.
- **Tests**: 49 total. 33 are pure/filesystem/contract and always pass; 16 are
  fixture-gated and **skip** when no `.ncu-rep` is present (graceful — see
  `tests/conftest.py`). Run `uv run pytest`.
- **Benchmark**: already done and committed in `perfdigest/eval/RESULTS.md`
  (Sonnet 4.6, toolkit ON vs OFF, same correct diagnosis, ~36x fewer profiler
  tokens per turn). Harness: `perfdigest/eval/run_ab.ps1` (PowerShell — port to
  `.sh` on Linux if you want to re-run; logic is just two `claude -p` calls).
- **Benchmark workload** (`test_script/`): `kernels.cu` is the real GAFIME CUDA
  source (the user's). `interfaces.h` + `bench_main.cu` are the standalone
  harness I reconstructed so it compiles without the rest of GAFIME.
  `bench_main.cu` launches **two distinct kernels, 8 launches total**:
  - `gafime_global_continuous_kernel<ARITY>` (matrix path) at arity 1,2,3,4,5
  - `gafime_batched_kernel<ARITY>` (bucket path) in 3 configs: continuous
    `log*identity` (arity 2), continuous `square+abs` (arity 3), and a
    **time-series rolling-mean** transform (arity 2) that hits the windowed
    code path and should profile more compute-heavy than the memory-bound
    continuous kernels.

## The fixture is gitignored

`*.ncu-rep` is intentionally NOT committed (they're large + regenerable; the
54 MB one was deleted by a failed `ncu --force-overwrite`). The fixture dir is
kept via `tests/fixtures/.gitkeep`. Regenerate locally — see below.

## Resume on Linux

Prereqs on the Linux box: NVIDIA driver + CUDA toolkit (`nvcc`) + Nsight Compute
(`ncu`), `uv`, and (for the optional benchmark re-run) the `claude` CLI. Same GPU
class assumed (RTX 4060 Laptop, **sm_89** — change `-arch` if different).

```bash
git clone git@github.com:onlyxItachi/PerfDigest-MCP.git
cd PerfDigest-MCP/perfdigest
uv sync --extra cuda --extra dev          # ncu-report + pytest

# 1. Build the workload (Linux: no .exe suffix)
cd ../test_script
nvcc -O3 -arch=sm_89 -o gafime_bench kernels.cu bench_main.cu

# 2. GPU perf-counter access (Linux equiv of Windows' admin-only counters).
#    Either run ncu as root, OR (persistent) allow non-root profiling:
#      echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' \
#        | sudo tee /etc/modprobe.d/nvidia-profiling.conf && sudo reboot
#    See: https://developer.nvidia.com/ERR_NVGPUCTRPERM

# 3. Profile the 8 launches into the fixture (full set; a few minutes)
ncu --set full -o ../perfdigest/tests/fixtures/gafime ./gafime_bench

# 4. Run the suite — the 16 fixture-gated tests now execute
cd ../perfdigest
uv run pytest -v
```

### Then finish the tests (the actual remaining work)

The richer harness produces **8 kernels** instead of 4, so:

1. `tests/test_pri_reader.py`: bump `EXPECTED_KERNELS = 4` to the real count
   (expected **8** — verify with `ncu`'s output / `list_kernels`). The existing
   `test_durations_descend_when_sorted` still expects the arity-5 continuous
   kernel as hottest; confirm that still holds.
2. Add **family-specific tests** (the goal that was blocked):
   - both kernel families present: some kernel name contains
     `gafime_global_continuous_kernel` AND some contains `gafime_batched_kernel`.
   - **distinct profiles**: the rolling-mean batched kernel should NOT look like
     the memory-bound continuous ones — e.g. assert the set of kernels is not
     uniformly DRAM-bound (compare `dram_pct_peak` / `compute_pct_peak` across
     families; the continuous arity-1 kernel is ~96% DRAM, the rolling-mean one
     should show a different balance). This proves the digest captures real
     per-kernel differences, not a constant.
3. `uv run pytest` green, then commit + push.

## Invariants — do not regress (from `perfdigest/CLAUDE.md`)

- **`None` ≠ `0.0`**: a missing metric is `not_available_in_this_export`, never
  zero. This is the project's core honesty rule.
- **Real-report metric-name lesson**: the init prompt's example
  `dram__throughput.avg.pct_of_peak_sustained_elapsed` does **not** exist in ncu
  2026.1; DRAM %-of-peak is `gpu__dram_throughput...` (already fixed in
  `mapping.py`). Always verify vendor metric names against a real report — the
  absence rule will surface a wrong name as `not_available` rather than lying.
- `format` is mandatory (no auto-detect); `server/` stays a thin shell; heavy
  CUDA imports stay lazy.

## Toolchain that was in use (Windows; match or newer on Linux)

uv 0.11.20 · Python 3.11 (uv-managed) · CUDA/nvcc 13.2 · Nsight Compute 2026.1 ·
claude CLI 2.1.177 · GPU: RTX 4060 Laptop (sm_89). MCP server entry point:
`uv run perfdigest` (stdio). Claude Code MCP config: `perfdigest/.mcp.json`
(uses `uv --directory <path> run perfdigest` — path is Windows-absolute there;
update it to the Linux clone path).
