"""Real v1 clients compare the released 1.2.1 and installed migrated servers.

Run with the migrated server's Python. Each old client and the actual PyPI
baseline live in their own environment; the server SDK is never replaced.
No profiler or model is run.
"""
import json
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path


if __name__ == "__main__":
    output = Path("work/mcp-compat")
    output.mkdir(parents=True, exist_ok=True)
    probe = Path(__file__).with_name("mcp_stdio_probe.py")
    # The pre-migration lock and current v1 maintenance line both remain real
    # clients. Verify the actual published baseline, not old version metadata
    # from a source checkout that already contained post-1.2.0 fixes.
    for sdk in ("1.27.2", "1.30.0"):
        environment = output / f"client-{sdk}"
        legacy_python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        subprocess.run(["uv", "venv", "--python", sys.executable, str(environment)], check=True, timeout=60)
        subprocess.run([
            "uv", "pip", "install", "--python", str(legacy_python),
            f"mcp=={sdk}", "perfdigest-mcp==1.2.1",
        ], check=True, timeout=180)
        targets = (
            (legacy_python, "released-1.2.1", "1.2.1", sdk),
            (Path(sys.executable), "migrated", version("perfdigest-mcp"), version("mcp")),
        )
        for server_python, label, package_version, server_sdk in targets:
            destination = output / f"legacy-{sdk}-to-{label}.json"
            subprocess.run([
                str(legacy_python), str(probe),
                "--server-python", str(server_python), "--mode", "v1", "--verify",
                "--output", str(destination),
            ], check=True, timeout=180)
            evidence = json.loads(destination.read_text(encoding="utf-8"))
            assert evidence["client_sdk"] == sdk
            assert evidence["server"]["perfdigest"] == package_version
            assert evidence["server"]["mcp"] == server_sdk
            assert evidence["connection"]["protocol"] == "2025-11-25"
            print(f"SDK {sdk} client -> {label} server: seven-tool contract passed")
