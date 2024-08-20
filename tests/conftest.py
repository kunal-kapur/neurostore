import pytest
from pathlib import Path
import os


@pytest.fixture(autouse=True)
def change_dir(monkeypatch, request=None) -> Path:
    monkeypatch.chdir(os.path.join(Path(__file__).parent))
