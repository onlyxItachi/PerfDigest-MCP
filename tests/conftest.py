import pytest

from pathlib import Path

# Register every backend once for the whole test session (the same side-effect
# imports the server does at startup) so registry/tools tests see all formats.
from perfdigest.server.app import _register_backends

_register_backends()

FIXTURES = Path(__file__).parent / "fixtures"
FIXTURE = FIXTURES / "gafime.ncu-rep"


@pytest.fixture(scope="session")
def report_path() -> str:
    if not FIXTURE.exists():
        pytest.skip(
            "real .ncu-rep fixture absent (gitignored binary). Regenerate from any "
            "CUDA app with: ncu --set full -o tests/fixtures/gafime <app>"
        )
    return str(FIXTURE)


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES
