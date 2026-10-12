"""CI check with a genuine, isolated v1 SDK client and this installed server.

Run with the migrated server's Python. uv creates separate client environments;
it never replaces the server interpreter's SDK. No profiler or model is run.
"""
import json
import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    output = Path("work/mcp-compat")
    output.mkdir(parents=True, exist_ok=True)
    probe = Path(__file__).with_name("mcp_stdio_probe.py")
    # The repository's pre-migration lock, plus the current v1 maintenance line.
    for sdk in ("1.27.2", "1.30.0"):
        destination = output / f"legacy-{sdk}-to-migrated.json"
        environment = output / f"client-{sdk}"
        legacy_python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        subprocess.run(["uv", "venv", "--python", sys.executable, str(environment)], check=True, timeout=60)
        subprocess.run([
            "uv", "pip", "install", "--python", str(legacy_python), f"mcp=={sdk}",
        ], check=True, timeout=180)
        subprocess.run([
            str(legacy_python), str(probe),
            "--server-python", sys.executable, "--mode", "v1", "--verify",
            "--output", str(destination),
        ], check=True, timeout=180)
        evidence = json.loads(destination.read_text(encoding="utf-8"))
        assert evidence["client_sdk"] == sdk
        assert evidence["connection"]["protocol"] == "2025-11-25"
        print(f"SDK {sdk} client -> migrated stdio server: seven-tool contract passed")
