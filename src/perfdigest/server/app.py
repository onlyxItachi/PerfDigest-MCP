"""app — FastMCP instance + entry point.

The ``mcp`` instance is created here; ``tools`` and ``prompts`` import it and
register against it via decorators. ``main`` imports those modules for their
registration side effects before running, so a bare ``uv run perfdigest``
serves the full tool + prompt surface.
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("perfdigest")


def _register_backends() -> None:
    """Import every adapter for its registry side effect (registry.register)."""
    from perfdigest.adapters.linux_perf import backend as _perf  # noqa: F401
    from perfdigest.adapters.metal import backend as _metal  # noqa: F401
    from perfdigest.adapters.nsight import backend as _nsight  # noqa: F401
    from perfdigest.adapters.rocm import backend as _rocm  # noqa: F401


def main() -> None:
    """Run the perfdigest MCP server over stdio."""
    # Side-effect imports: executing these modules runs the registry.register()
    # calls (backends) and the @mcp.tool() / @mcp.prompt() decorators (surface).
    _register_backends()
    from perfdigest.server import prompts, tools  # noqa: F401

    mcp.run()


if __name__ == "__main__":
    main()
