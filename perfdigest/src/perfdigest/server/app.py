"""app — FastMCP instance + entry point.

The ``mcp`` instance is created here; ``tools`` and ``prompts`` import it and
register against it via decorators. ``main`` imports those modules for their
registration side effects before running, so a bare ``uv run perfdigest``
serves the full tool + prompt surface.
"""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("perfdigest")


def main() -> None:
    """Run the perfdigest MCP server over stdio."""
    # Side-effect imports: executing these modules runs the @mcp.tool() /
    # @mcp.prompt() decorators that register the surface on `mcp`.
    from perfdigest.server import prompts, tools  # noqa: F401

    mcp.run()


if __name__ == "__main__":
    main()
