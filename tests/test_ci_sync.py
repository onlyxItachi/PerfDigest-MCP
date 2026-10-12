"""The historical Windows gap must never excuse a different failure."""
import pytest

from ci_sync import known_windows_gap

GAP = "error: Distribution `pywin32==312 @ registry+https://pypi.org/simple` can't be installed because it doesn't have a source distribution or wheel for the current platform"


def test_only_exact_known_windows_install_failure_is_exempt():
    assert known_windows_gap("3.14t", "win32", "Resolved packages\n" + GAP + "\n")


@pytest.mark.parametrize("python_tag,platform,output", [
    ("3.14", "win32", GAP),
    ("3.14t", "linux", GAP),
    ("3.14t", "win32", GAP.replace("pywin32==312", "mcp==2.3.0")),
    ("3.14t", "win32", GAP.replace("pywin32==312", "pywin32==313")),
    ("3.14t", "win32", GAP + "\nerror: another package failed"),
    ("3.14t", "win32", "error: certificate verification failed"),
])
def test_new_failures_are_not_exempt(python_tag, platform, output):
    assert not known_windows_gap(python_tag, platform, output)
