"""app — MCPServer instance + entry point.

The ``mcp`` instance is created here; ``tools`` and ``prompts`` import it and
register against it via decorators. ``main`` imports those modules for their
registration side effects before running, so a bare ``uv run perfdigest``
serves the full tool + prompt surface.
"""
from importlib.metadata import version
from typing import Any

from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError
from mcp.types import CallToolResult, InputRequiredResult


class PerfDigestServer(MCPServer):
    """Keep actionable report/input errors visible across SDK versions.

    SDK 2 masks ordinary exceptions. Our readers deliberately use ValueError,
    IndexError and file/import errors to explain how to correct a report request.
    Translate those at the MCP boundary; direct Python calls keep their original
    exception types, and unexpected crashes retain the SDK's safe error handling.
    """

    async def call_tool(
        self, name: str, arguments: dict[str, Any], context: Context | None = None,
    ) -> CallToolResult | InputRequiredResult:
        try:
            return await super().call_tool(name, arguments, context)
        except UnexpectedToolError as exc:
            cause = exc.__cause__
            if isinstance(cause, (ValueError, IndexError, OSError, ImportError)):
                raise ToolError(f"Error executing tool {name}: {cause}") from cause
            raise


mcp = PerfDigestServer("perfdigest", version=version("perfdigest-mcp"))


def _register_backends() -> None:
    """Import every adapter for its registry side effect (registry.register)."""
    from perfdigest.adapters.cargo_diag import backend as _cargo_diag  # noqa: F401
    from perfdigest.adapters.chrome_trace import backend as _chrome  # noqa: F401
    from perfdigest.adapters.clang_time_trace import backend as _clang_tt  # noqa: F401
    from perfdigest.adapters.cmake_profile import backend as _cmake_profile  # noqa: F401
    from perfdigest.adapters.criterion import backend as _criterion  # noqa: F401
    from perfdigest.adapters.gha_log import backend as _gha_log  # noqa: F401
    from perfdigest.adapters.git_numstat import backend as _git_numstat  # noqa: F401
    from perfdigest.adapters.linux_perf import backend as _perf  # noqa: F401
    from perfdigest.adapters.metal import backend as _metal  # noqa: F401
    from perfdigest.adapters.ninja_log import backend as _ninja_log  # noqa: F401
    from perfdigest.adapters.nsight import backend as _nsight  # noqa: F401
    from perfdigest.adapters.ptxas import backend as _ptxas  # noqa: F401
    from perfdigest.adapters.rocm import backend as _rocm  # noqa: F401


def main() -> None:
    """Run the perfdigest MCP server over stdio."""
    # Side-effect imports: executing these modules runs the registry.register()
    # calls (backends) and the @mcp.tool() / @mcp.prompt() decorators (surface).
    _register_backends()
    from perfdigest.server import prompts, tools  # noqa: F401

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
