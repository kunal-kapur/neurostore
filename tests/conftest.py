import pytest
from pathlib import Path
import os


@pytest.fixture
def change_dir(monkeypatch, request) -> Path:
    monkeypatch.chdir(os.path.join(Path(__file__).parent, request.param))
