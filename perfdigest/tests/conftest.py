import pytest

from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "gafime.ncu-rep"


@pytest.fixture(scope="session")
def report_path() -> str:
    if not FIXTURE.exists():
        pytest.skip(
            "real .ncu-rep fixture absent (gitignored). Regenerate with: "
            "ncu --set full -o tests/fixtures/gafime test_script/gafime_bench.exe"
        )
    return str(FIXTURE)
