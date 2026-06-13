# Connecting perfdigest to Claude Code and OpenAI Codex

perfdigest is a local **stdio** MCP server. Once published it runs straight from
PyPI with `uvx perfdigest-mcp` (no clone, no manual venv). Both clients below speak
stdio MCP; the only difference is the config file format.

## Install (downloadable)

```bash
uvx perfdigest-mcp            # run on demand from PyPI (recommended)
# or pin backends' optional deps:
uv tool install "perfdigest-mcp[nvidia]"   # adds NVIDIA's native binary reader (Linux/Windows)
```

Backends whose readers are pure Python (AMD rocprof CSV, Linux perf, Metal JSON,
and the NVIDIA **CSV** path) need no extra — they work everywhere, so a Mac can
digest an NVIDIA report captured on a CI/remote GPU runner.

## Claude Code

Project scope — commit `.mcp.json` at the repo root (already provided here):

```json
{
  "mcpServers": {
    "perfdigest": { "type": "stdio", "command": "uvx", "args": ["perfdigest-mcp"] }
  }
}
```

Or add it from the CLI (pick a scope: `local` | `project` | `user`):

```bash
claude mcp add perfdigest --scope user -- uvx perfdigest-mcp
```

Then inject the workflow convention once per session via the MCP prompt
`perfdigest_usage`, and call `/mcp` to confirm the three digest tools +
`platform_capabilities` + `suggest_profile_command` are listed.

## OpenAI Codex

Codex stores MCP servers in `~/.codex/config.toml` (or a trusted project's
`.codex/config.toml`):

```toml
[mcp_servers.perfdigest]
command = "uvx"
args = ["perfdigest-mcp"]
```

Or from the CLI:

```bash
codex mcp add perfdigest -- uvx perfdigest-mcp
```

Confirm with `/mcp` inside Codex.

## Local development (from source, before publishing)

Point either client at the working tree instead of PyPI:

```bash
# Claude Code
claude mcp add perfdigest --scope local -- uv --directory /abs/path/to/PerfDigest-MCP run perfdigest-mcp
```
```toml
# Codex ~/.codex/config.toml
[mcp_servers.perfdigest]
command = "uv"
args = ["--directory", "/abs/path/to/PerfDigest-MCP", "run", "perfdigest-mcp"]
```
