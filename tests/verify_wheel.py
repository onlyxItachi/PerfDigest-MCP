"""Build and exercise the actual wheel in a clean, non-editable environment."""
import json
import subprocess
import sys
import zipfile
from pathlib import Path


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    output = root / "work/wheel-check"
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run(["uv", "build", "--out-dir", str(output / "dist")], check=True, timeout=120)
    wheel, = (output / "dist").glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        assert not any(n.startswith(("tests/", "eval/", "work/")) for n in archive.namelist())
        for source in (root / "src/perfdigest").rglob("*.py"):
            assert archive.read(source.relative_to(root / "src").as_posix()) == source.read_bytes()
    environment = output / "venv"
    subprocess.run(["uv", "venv", "--python", sys.executable, str(environment)], check=True, timeout=60)
    python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    subprocess.run(["uv", "pip", "install", "--python", str(python), str(wheel)], check=True, timeout=180)
    subprocess.run(["uv", "pip", "check", "--python", str(python)], check=True, timeout=60)
    for command in ("perfdigest", "perfdigest-mcp"):
        assert python.with_name(command + (".exe" if sys.platform == "win32" else "")).is_file()
    subprocess.run([
        str(python), str(root / "tests/mcp_stdio_probe.py"),
        "--server-python", str(python), "--mode", "auto", "--verify",
        "--output", str(output / "modern-stdio.json"),
    ], check=True, timeout=100)
    evidence = json.loads((output / "modern-stdio.json").read_text(encoding="utf-8"))
    print(f"Clean wheel: SDK {evidence['client_sdk']}, protocol {evidence['connection']['protocol']}, seven-tool contract passed")
