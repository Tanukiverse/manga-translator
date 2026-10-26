import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))

import synth  # noqa: E402


@pytest.fixture(scope="session")
def ja_page():
    return synth.japanese_page()


@pytest.fixture(scope="session")
def en_page():
    return synth.english_page()


@pytest.fixture(scope="session")
def detector():
    from manga_translator.core.detect import TextDetector

    return TextDetector()


def pytest_configure(config):
    config.addinivalue_line("markers", "ollama: needs a running Ollama server")
    config.addinivalue_line("markers", "gpu: loads a model on the GPU")
