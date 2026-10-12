"""Install the CI cell, recognizing only the already documented wheel gap.

A whole-job allowed failure would also swallow new SDK or test regressions.
This narrowly skips tests only after the exact pywin32 312 install failure on
Windows 3.14t. A future available wheel runs the full suite automatically.
"""
import os
import subprocess
import sys
from pathlib import Path


def known_windows_gap(python_tag, platform, output):
    # Require the sole uv error line to match; unrelated install errors must fail.
    errors = [line for line in output.splitlines() if line.startswith("error:")]
    return (
        python_tag == "3.14t" and platform == "win32" and len(errors) == 1
        and errors[0] == "error: Distribution `pywin32==312 @ registry+https://pypi.org/simple` can't be installed because it doesn't have a source distribution or wheel for the current platform"
    )


if __name__ == "__main__":
    python_tag = sys.argv[1]
    result = subprocess.run(
        ["uv", "sync", "--locked", "--extra", "dev", "--python", python_tag],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
    )
    print(result.stdout, end="")
    gap = result.returncode != 0 and known_windows_gap(python_tag, sys.platform, result.stdout)
    if result.returncode and not gap:
        raise SystemExit(result.returncode)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
            output.write(f"run_tests={'false' if gap else 'true'}\n")
    if gap:
        message = "Windows 3.14t: NOT EXECUTED — known pywin32 312 distribution gap; no test/support pass is claimed."
        print(f"::warning::{message}")
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as summary:
                summary.write(message + "\n")
        evidence = Path("work/mcp-compat")
        evidence.mkdir(parents=True, exist_ok=True)
        (evidence / "windows-314t-NOT-EXECUTED.txt").write_text(message + "\n" + result.stdout, encoding="utf-8")
