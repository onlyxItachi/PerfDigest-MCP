"""SDK/protocol regressions use real stdio subprocesses, not mocked transport."""
import asyncio
import json
import subprocess
import sys
from contextlib import asynccontextmanager
from importlib.metadata import version
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters

from mcp_contract import verify_contract

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures"
SERVER_CODE = "from perfdigest.server.app import main; main()"
META = {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {},
}


@pytest.mark.parametrize("mode", ["legacy", "auto", "2026-07-28"])
def test_all_tools_over_stdio_preserve_v1_contract(mode, tmp_path):
    output = tmp_path / "probe.json"
    subprocess.run([
        sys.executable, str(ROOT / "tests/mcp_stdio_probe.py"),
        "--server-python", sys.executable, "--mode", mode,
        "--output", str(output),
    ], check=True, timeout=100)
    data = json.loads(output.read_text(encoding="utf-8"))
    verify_contract(data, FIXTURES)
    connection = data["connection"]
    if mode == "legacy":
        assert connection["protocol"] == "2025-11-25"
        assert "initialize" in connection and "discover" not in connection
        identity = connection["initialize"]["serverInfo"]
    elif mode == "auto":
        assert connection["protocol"] == "2026-07-28"
        assert "discover" in connection and "initialize" not in connection
        identity = connection["discover"]["_meta"]["io.modelcontextprotocol/serverInfo"]
    else:
        assert connection["protocol"] == "2026-07-28"
        assert "adopted" in connection and "initialize" not in connection and "discover" not in connection
        identity = data["calls"]["defaults"]["_meta"]["io.modelcontextprotocol/serverInfo"]
    assert identity == {"name": "perfdigest", "version": version("perfdigest-mcp")}


@asynccontextmanager
async def raw_server():
    """Bounded, SDK-independent JSON-RPC exchange over the installed entrypoint."""
    scripts = Path(sys.executable).parent
    executable = scripts / ("perfdigest-mcp.exe" if sys.platform == "win32" else "perfdigest-mcp")
    process = await asyncio.create_subprocess_exec(
        str(executable), stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    async def request(identifier, method, params):
        message = {"jsonrpc": "2.0", "id": identifier, "method": method, "params": params}
        process.stdin.write((json.dumps(message) + "\n").encode())
        await process.stdin.drain()
        response = json.loads(await asyncio.wait_for(process.stdout.readline(), 20))
        assert response["id"] == identifier
        return response
    try:
        yield request
    finally:
        process.stdin.close()
        try:
            await asyncio.wait_for(process.wait(), 5)
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()


def test_modern_stdio_is_stateless_and_discovery_is_optional():
    async def run():
        async with raw_server() as request:
            # A tool call is the FIRST message: no initialize and no discover.
            result = await request(1, "tools/call", {
                "name": "get_metrics", "arguments": {
                    "report_ref": str(FIXTURES / "nvidia_sample.ncu-csv"),
                    "format": "ncu-csv", "kernel": "0", "metrics": ["never_measured"],
                }, "_meta": META,
            })
            result = result["result"]
            assert result["resultType"] == "complete"
            assert result["isError"] is False
            assert result["_meta"]["io.modelcontextprotocol/serverInfo"]["name"] == "perfdigest"
            digest = json.loads(result["content"][0]["text"])
            assert digest["metrics"]["never_measured"] == "not_available_in_this_export"
            assert digest["raw_ref"] == str(FIXTURES / "nvidia_sample.ncu-csv")
            discovery = (await request(2, "server/discover", {"_meta": META}))["result"]
            assert "2026-07-28" in discovery["supportedVersions"]
            assert "tools" in discovery["capabilities"]
            assert "serverInfo" not in discovery  # identity is result metadata in 2026
            unsupported = await request(3, "tools/list", {"_meta": {
                **META, "io.modelcontextprotocol/protocolVersion": "1900-01-01",
            }})
            assert unsupported["error"]["code"] == -32022
            # The previous request's capabilities must not leak into this one.
            incomplete = await request(4, "tools/list", {"_meta": {
                "io.modelcontextprotocol/protocolVersion": "2026-07-28",
            }})
            assert "error" in incomplete
            valid = await request(5, "tools/list", {"_meta": META})
            assert len(valid["result"]["tools"]) == 7
    asyncio.run(run())


def test_concurrent_stdio_reads_preserve_identity_and_absence():
    async def run():
        server = StdioServerParameters(command=sys.executable, args=["-c", SERVER_CODE])
        async with Client(server) as client:
            async def read(index):
                response = await client.call_tool("get_metrics", {
                    "report_ref": str(FIXTURES / "nvidia_sample.ncu-csv"),
                    "format": "ncu-csv", "kernel": str(index),
                    "metrics": ["duration_us", "never_measured"],
                })
                assert not response.is_error
                value = json.loads(response.content[0].text)
                assert value["index"] == index
                assert value["metrics"] == {
                    "duration_us": 5.0 if index == 0 else 20.0,
                    "never_measured": "not_available_in_this_export",
                }
            await asyncio.gather(*(read(i % 2) for i in range(32)))
    asyncio.run(run())


def test_unexpected_crashes_keep_sdk_error_redaction():
    from perfdigest.server.app import PerfDigestServer
    server = PerfDigestServer("error-contract")

    @server.tool()
    def crash() -> str:
        raise RuntimeError("internal crash detail")

    async def run():
        async with Client(server) as client:
            result = await client.call_tool("crash", {})
            assert result.is_error
            assert result.content[0].text == "Error executing tool crash"
    asyncio.run(run())
