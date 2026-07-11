# eval — moved to PerfDigest-MCP-Bench

Benchmark workloads and evaluation results live in a dedicated companion repo:
**<https://github.com/onlyxItachi/PerfDigest-MCP-Bench>** — keeping large report
artifacts and result documents out of this code repo. Unit tests stay here
(they gate CI); only benchmark/eval artifacts moved.

Headline results measured so far:

- **Token efficiency:** the digest costs ~14–130x fewer tokens per turn than raw
  `ncu` output for the same correct diagnosis —
  [results/RESULTS.md](https://github.com/onlyxItachi/PerfDigest-MCP-Bench/blob/main/results/RESULTS.md).
- **Cross-backend, real hardware:** NVIDIA (`nsight`) and CPU (`linux_perf`)
  digests validated on a real RTX 4060 + Ryzen host through one MCP call shape —
  [results/CROSS_BACKEND_2026-06-15.md](https://github.com/onlyxItachi/PerfDigest-MCP-Bench/blob/main/results/CROSS_BACKEND_2026-06-15.md).

To reproduce, see
[workloads/README.md](https://github.com/onlyxItachi/PerfDigest-MCP-Bench/blob/main/workloads/README.md)
in the bench repo (capture commands for both workloads).
