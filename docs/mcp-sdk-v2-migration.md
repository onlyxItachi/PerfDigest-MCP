# MCP SDK 2 migration and protocol compatibility

This is an **unreleased migration**, based on `390ad80a952f7af2c86c3e18baddb8d2423a2192`.
The package version remains 1.2.0; no release, tag or publication is part of this
change. Evidence was collected on 12 October 2026. `CLAUDE.md` remains the live
behavior contract; `docs/INIT_PROMPT.md` is unchanged.

## SDK change versus protocol change

The dependency is now `mcp[cli]>=2.3,<3`, with 2.3.0 in `uv.lock`. The tested floor
is deliberately 2.3, not an untested claim about every earlier 2.x release.
The server imports `MCPServer` instead of the removed `FastMCP` module. Decorators
and seven tool signatures stay unchanged. Server identity remains `perfdigest`;
its version now comes from **perfdigest-mcp's** distribution metadata rather
than the SDK's version or an empty default.

Two additional SDK changes matter here: synchronous handlers run on worker
threads, and ordinary exceptions no longer expose their details to the client.
The existing report-cache lock already protects the shared LRU. A small
`MCPServer.call_tool` override translates the readers' deliberate value, index,
file and import failures to `ToolError`, preserving their corrective messages.
Direct Python calls retain their exception types. Unexpected crashes retain the
SDK's generic error handling; a regression test checks this explicitly.
Sources: [migration guide](https://github.com/modelcontextprotocol/python-sdk/blob/main/docs/migration.md),
[SDK 2 overview](https://github.com/modelcontextprotocol/python-sdk/blob/main/docs/whats-new.md).

SDK 2 also changes Python protocol-field access to snake_case and moves several
client/transport APIs. Our server previously used none of those low-level APIs.
The compatibility probe serializes with `by_alias=True`, so it compares actual
camelCase wire keys. `mcp-types` follows the SDK's exact dependency; it is not
pinned separately. The resolved dependency graph adds httpx2, mcp-types and
opentelemetry-api while removing the old SDK's httpx/httpx-sse dependencies.

The **wire protocol** is selected independently:

| Client path | Connection behavior | Wire revision |
| --- | --- | --- |
| Genuine SDK 1 client | `initialize`, then ordinary requests | 2025-11-25 |
| SDK 2 `Client(..., mode="legacy")` | Legacy initialization | 2025-11-25 |
| SDK 2 `Client(..., mode="auto")` | `server/discover` against this server | 2026-07-28 |
| SDK 2 pinned modern mode | Locally adopts the version, no discovery round trip | 2026-07-28 |
| Raw JSON-RPC client | May make a tool call as its first message | 2026-07-28 |

Modern requests carry `io.modelcontextprotocol/protocolVersion` and
`io.modelcontextprotocol/clientCapabilities` inside `params._meta` on **every**
request. Client identity is optional. Results include `resultType: complete`;
the SDK places server identity in result `_meta`, including on discovery.
The raw stdio test verifies these fields, no-handshake tool execution,
unsupported-version rejection and rejection of missing capabilities after a
previous valid request. Report state is addressed explicitly by report_ref and
format; the parse cache does not infer protocol/client session state.
Sources: [2026 specification](https://modelcontextprotocol.io/specification/2026-07-28),
[base protocol](https://modelcontextprotocol.io/specification/2026-07-28/basic),
[version compatibility](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning).

## Transport and feature boundary

Both console commands still launch **stdio**. No HTTP listener, auth service or
new transport is introduced. SDK 2's dual-era support handles both connection
styles on that endpoint. `stateless_http=True` is an HTTP setting for legacy
connections; it is not the switch enabling the modern protocol. HTTP/SSE are
not exercised by this repository's stdio evidence. The modern protocol no
longer has server-initiated requests; perfdigest has no sampling, elicitation,
roots or subscription-dependent tool handlers to port.
Sources: [SDK 2 overview](https://github.com/modelcontextprotocol/python-sdk/blob/main/docs/whats-new.md),
[transport bindings](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports).

## Contract and executable evidence

`tests/fixtures/mcp_v1_contract.json` was captured from a separately installed
SDK 1.27.2 client and the base server over a real stdio pipe. It contains every
tool's name, description, input schema and output schema, plus deterministic
success/error results. Fixture paths, JSON text whitespace and the top-level
description's common docstring indentation are normalized. Names and all input/
output schema fields remain exact. Tests discard the two new protocol-envelope fields when comparing
payloads; content-block layout, structuredContent and isError remain checked.

The seven tools remain `list_kernels`, `get_metrics`, `expand`,
`summarize_report`, `compare_metrics`, `platform_capabilities` and
`suggest_profile_command`. The probes exercise all seven, mandatory format,
absent metrics, explicit empty metric lists, raw expansion, comparison identity,
unknown formats/backends, missing reports and missing units. Digest remains
host-independent, capture remains advisory and gated, raw_ref provenance is
preserved, and no interpretation/threshold heuristic is introduced.

Local Linux evidence at the first implementation checkpoint:

| Client | Server | Transport / result |
| --- | --- | --- |
| Python SDK 1.27.2 | Base source + SDK 1.27.2 | Real stdio; legacy schema/payload baseline captured |
| Python SDK 1.27.2 | Migrated SDK 2.3.0 | Real stdio; seven-tool contract preserved |
| Python SDK 1.28.1 | Base source + SDK 1.28.1 | Real stdio; all seven tools exercised |
| Python SDK 1.28.1 | Migrated SDK 2.3.0 | Real stdio; schema and successful payload parity |
| Python SDK 2.3.0 | Migrated SDK 2.3.0 | Legacy, auto and pinned-modern modes passed |
| Raw JSON-RPC | Migrated SDK 2.3.0 | Modern first-call/no-discovery, metadata and version checks passed |
| Plugin JS SDK 1.32.1 | Migrated SDK 2.3.0 | Independent production gatherEvidence smoke: seven tools, two CPU units, capture SHA retained |

The locked Python 3.14.3 suite passes **229 tests**, with **16 native report
fixture skips**. The 32 concurrent stdio reads test exercises the SDK's worker
thread dispatch and verifies per-unit identity, values and missingness. These
are fixture/transport checks, not new hardware captures or performance claims.

`tests/run_legacy_client.py` creates isolated SDK 1.27.2 and 1.30.0 environments
and checks the installed migrated server. `tests/verify_wheel.py` builds the
wheel from an sdist, checks packaged source bytes and excluded directories,
installs into a clean environment, checks dependencies and both entrypoints,
then exercises real modern stdio calls. The CI matrix runs the genuine legacy
clients in every installable OS/Python cell; separate clean-wheel jobs cover
Python 3.10 and 3.14. Neither helper changes the user's global installation.

## Python / OS policy and hosted comparison

The declared Python 3.10–3.14 and free-threaded 3.14t matrix stays present on
Linux, macOS and Windows. The SDK 2.3.0 distribution declares Python >=3.10.
Hardware tests remain fixture-gated. The legacy comparison is
[PR 35](https://github.com/onlyxItachi/PerfDigest-MCP/pull/35), commit
`996a4b4c67c12e852ef61773e151ca75b2ed20d8`,
[CI run 38227999286](https://github.com/onlyxItachi/PerfDigest-MCP/actions/runs/38227999286):
17 test jobs passed; [Windows 3.14t failed before tests](https://github.com/onlyxItachi/PerfDigest-MCP/actions/runs/38227999286/job/114740166173)
because pywin32 312 has no applicable distribution.

That pre-existing gap is reevaluated, not broadened. This branch removes the
whole-job allowed-failure rule. `tests/ci_sync.py` recognizes only the exact
pywin32 312/no-distribution error on Windows 3.14t, writes **NOT EXECUTED**, and
skips that cell's test steps. Every different install error and every executed
test failure fails CI. A future successful installation automatically runs the
suite. The classifier has positive and negative regression tests. Its green
job conclusion is **not** a Windows 3.14t support/execution pass.

Hosted migration results and clean-wheel evidence will be appended after the
final draft PR run. The [first run](https://github.com/onlyxItachi/PerfDigest-MCP/actions/runs/38229043242)
exposed a test-fixture portability error: all nine Python 3.10–3.12 cells and the
3.10 wheel check failed on description indentation. A separately captured SDK
1.30.0/Python 3.10 legacy listing confirms this difference already existed in
v1. [Python 3.13's compiler change](https://docs.python.org/3/whatsnew/3.13.html#other-language-changes)
strips common docstring indentation. Comparing only descriptions with
`inspect.cleandoc` fixes the test without changing server behavior or relaxing
any schema/payload assertions. The first failed run remains part of the record.

Locally, downloads of some frozen-lock/floor wheels encountered a
files.pythonhosted.org TLS hostname mismatch; those failed attempts are not
counted as passing installation checks and TLS verification was not disabled.

## Release boundary and reproduction

CI runs on pull requests and pushes to main; publishing runs only on `v*` tags.
The publish workflow's test, floor and clean-wheel gates are unchanged. This
work creates a draft PR only, with no main write, tag, merge or publication.

```sh
uv sync --locked --extra dev --python 3.14
uv run --locked --extra dev --python 3.14 pytest -q
uv run --locked --extra dev --python 3.14 python tests/run_legacy_client.py
uv run --no-project --python 3.10 python tests/verify_wheel.py
```
