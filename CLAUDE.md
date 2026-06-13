# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A development kit for building **MCP (Model Context Protocol) servers**. The main (and currently only real) project is **`perfdigest/`** — everything else at the root is supporting scratch space.

- **`perfdigest/`** — a local MCP server that makes NVIDIA Nsight Compute (`ncu`) GPU profiler output token-efficient for LLM coding agents. It reads profiler reports from disk and returns small, structured, numeric digests instead of letting raw multi-thousand-token tables flood the agent's context. This is the project all real work happens in.
- **`MCP_Developing_Workshop/`** — a Node scratchpad (npm package `mcp_developing_workshop`) with `@anthropic-ai/claude-agent-sdk`, `@anthropic-ai/sdk`, `tsx`, and `typescript` as dependencies. No source files yet.
- **`workshop-env/`** — a POSIX Python 3.10 venv (gitignored, never commit; not used by perfdigest, which targets Python 3.11+ via uv).

## Working in perfdigest — read these first, in this order

1. **`perfdigest/docs/INIT_PROMPT.md`** — the canonical spec. All architectural decisions are locked there with rationale; do not re-litigate them.
2. **`perfdigest/CLAUDE.md`** — the short operational contract: non-negotiable invariants, build order, tool signatures, and open items that must be asked of the user rather than invented.

Current state: **v1 implemented and validated** (all phases done except the `csv_reader.py` fallback stub). 12 pytest tests pass, end-to-end verified through Claude Code headless with Sonnet 4.6, and an A/B context-efficiency benchmark lives in `perfdigest/eval/RESULTS.md` (~36x fewer profiler tokens per turn vs reading the raw `ncu` details, same diagnosis). The `test_script/` folder holds the CUDA workload (`kernels.cu`) and benchmark harness used to generate the real `.ncu-rep` fixture.

## Commands (run from `perfdigest/`)

```bash
uv sync                 # base install (CSV path works without CUDA)
uv sync --extra cuda    # adds ncu-report for native .ncu-rep parsing (works without a GPU)
uv run perfdigest       # run the MCP server over stdio
uv run pytest           # tests (once fixtures land)
```

## The invariants that must never break (full detail in perfdigest/CLAUDE.md)

- **`None` ≠ `0.0`** — a missing metric means "not measured in this export", never zero. `get_metrics` returns `"not_available_in_this_export"` for absent metrics. Silently substituting zero is the single worst bug this tool can have.
- **`format` is mandatory** in v1 — no auto-detect.
- **Translator, not judge** — no verdict/threshold heuristics; interpretation is the model's job.
- **`server/` is a thin shell** — business logic lives in `core/` and `adapters/`; `core/` never sees vendor metric names.
- **Heavy/CUDA-only imports (`ncu_report`) are lazy** so CSV-only paths never load them.
