import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Must be set before app.main is imported: it reads settings at import time.
_TMP = Path(tempfile.mkdtemp(prefix="cashdesk-test-"))
os.environ.setdefault("API_KEY", "test-key")
os.environ.setdefault("DATA_DIR", str(_TMP / "data"))

from make_synthetic import make  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_video() -> Path:
    path = _TMP / "synthetic.mov"
    make(str(path), seconds=12.0, fps=30)
    return path


@pytest.fixture(scope="session")
def long_synthetic_video() -> Path:
    path = _TMP / "synthetic_long.mov"
    make(str(path), seconds=31.0, fps=30)
    return path


@pytest.fixture
def corrupt_file(tmp_path) -> Path:
    path = tmp_path / "broken.mov"
    path.write_bytes(b"\x00\x00\x00\x18ftypqt  " + os.urandom(4096))
    return path
