"""Real stdio probe, runnable in either an independently installed v1 or v2 SDK.

This file deliberately imports no perfdigest code: the server runs in a separate
interpreter. Use --output to retain full tool schemas and results as evidence.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

from mcp import StdioServerParameters


def wire(value):
    return value.model_dump(mode="json", by_alias=True, exclude_none=True)


async def exercise(client, fixtures):
    report = str((fixtures / "nvidia_sample.ncu-csv").resolve())
    common = {"report_ref": report, "format": "ncu-csv"}
    cases = {
        "list": ("list_kernels", common),
        "defaults": ("get_metrics", {**common, "kernel": "0"}),
        "missing_metric": ("get_metrics", {**common, "kernel": "0", "metrics": ["duration_us", "never_measured"]}),
        "empty_metrics": ("get_metrics", {**common, "kernel": "0", "metrics": []}),
        "expand": ("expand", {**common, "kernel": "0", "section": "all"}),
        "summary": ("summarize_report", {**common, "top_n": 2}),
        "compare": ("compare_metrics", {"report_a": report, "report_b": report, "format": "ncu-csv", "kernel": "0", "kernel_b": "1", "metrics": ["duration_us", "never_measured"]}),
        "capabilities": ("platform_capabilities", {}),
        "suggest": ("suggest_profile_command", {"backend": "metal", "target": "./example with spaces"}),
        "missing_format": ("list_kernels", {"report_ref": report}),
        "unknown_format": ("list_kernels", {**common, "format": "unknown-format"}),
        "missing_report": ("list_kernels", {**common, "report_ref": str(fixtures / "absent-report")}),
        "missing_unit": ("get_metrics", {**common, "kernel": "no-such-unit"}),
        "unknown_backend": ("suggest_profile_command", {"backend": "unknown-backend", "target": "./example"}),
    }
    tools = wire(await client.list_tools())["tools"]
    calls = {}
    for label, (name, arguments) in cases.items():
        calls[label] = wire(await client.call_tool(name, arguments))
    return {
        "tools": sorted(tools, key=lambda t: t["name"]),
        "calls": calls,
        "prompts": wire(await client.list_prompts()),
        "usage": wire(await client.get_prompt("perfdigest_usage")),
    }


async def probe(args):
    # Query the exact server interpreter, not the client environment. Keep the
    # distribution identity separate from the legacy SDK's serverInfo version.
    metadata = subprocess.run([
        str(args.server_python.absolute()), "-c",
        "import json, sys; from importlib.metadata import version; "
        "print(json.dumps({'python': sys.version, 'perfdigest': version('perfdigest-mcp'), "
        "'mcp': version('mcp')}))",
    ], check=True, capture_output=True, text=True, timeout=20)
    server_metadata = json.loads(metadata.stdout)
    server = StdioServerParameters(
        # Do not resolve symlinks: a venv's Python may link to the base runtime.
        command=str(args.server_python.absolute()),
        args=["-c", "from perfdigest.server.app import main; main()"],
    )
    if args.mode == "v1":
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        async with stdio_client(server) as (read, write):
            async with ClientSession(read, write) as client:
                handshake = wire(await client.initialize())
                data = await exercise(client, args.fixtures)
        connection = {"initialize": handshake, "protocol": handshake["protocolVersion"]}
    else:
        from mcp import Client

        async with Client(server, mode=args.mode, read_timeout_seconds=30) as client:
            data = await exercise(client, args.fixtures)
            connection = {"protocol": client.protocol_version}
            if client.session.initialize_result is not None:
                connection["initialize"] = wire(client.session.initialize_result)
            if client.session.discover_result is not None:
                # A pinned Client adopts a synthetic DiscoverResult locally;
                # only auto mode sends server/discover over the transport.
                key = "discover" if args.mode == "auto" else "adopted"
                connection[key] = wire(client.session.discover_result)
    return {
        "client_sdk": version("mcp"), "client_python": sys.version,
        "server": server_metadata, "mode": args.mode, "connection": connection, **data,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-python", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=Path(__file__).parent / "fixtures")
    parser.add_argument("--mode", choices=["v1", "legacy", "auto", "2026-07-28"], required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(asyncio.wait_for(probe(args), timeout=90))
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)
    # Keep the received wire evidence even if verification fails.
    if args.verify:
        from mcp_contract import verify_contract

        verify_contract(result, args.fixtures)
