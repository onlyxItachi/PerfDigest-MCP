"""Platform-correct command construction for the capture advisor.

This module only *builds strings*; perfdigest never executes a profiler (it is a
digester, not a runner). The strings differ by host: POSIX shells vs Windows
**PowerShell** quote differently, and **WSL** needs path translation so a Windows
path handed to a Linux profiler (or vice versa) actually resolves.
"""

from __future__ import annotations

import shlex


def quote(arg: str, shell: str) -> str:
    """Quote one argument for the given shell ('posix' | 'powershell')."""
    if shell == "powershell":
        return _quote_powershell(arg)
    return shlex.quote(arg)


def _quote_powershell(arg: str) -> str:
    if arg and all(c.isalnum() or c in "-_./:\\" for c in arg):
        return arg
    # PowerShell single-quote literal: escape ' by doubling it.
    return "'" + arg.replace("'", "''") + "'"


def join(parts: list[str], shell: str) -> str:
    """Join an argv into a single command line for ``shell``."""
    return " ".join(quote(p, shell) for p in parts)


def wsl_to_windows_path(path: str) -> str:
    """``/mnt/c/x`` -> ``C:\\x`` (heuristic; prefer real ``wslpath`` at runtime)."""
    if path.startswith("/mnt/") and len(path) > 6 and path[6] == "/":
        drive = path[5].upper()
        rest = path[7:].replace("/", "\\")
        return f"{drive}:\\{rest}"
    return path


def windows_to_wsl_path(path: str) -> str:
    """``C:\\x`` -> ``/mnt/c/x`` (heuristic; prefer real ``wslpath`` at runtime)."""
    if len(path) >= 2 and path[1] == ":":
        drive = path[0].lower()
        rest = path[2:].replace("\\", "/").lstrip("/")
        return f"/mnt/{drive}/{rest}"
    return path
