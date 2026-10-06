import pytest

from app.core.config import clear_config_cache
from app.core.settings import get_settings, reset_settings


@pytest.fixture(autouse=True)
def _reset():
    reset_settings()
    yield
    reset_settings()


def test_loads_real_config_files():
    settings = get_settings()
    assert settings.engines.active == "vllm"
    assert any(e.name == "vllm" for e in settings.engines.engines)
    assert settings.rag.chunking.chunk_size > settings.rag.chunking.chunk_overlap


def test_active_engine_resolves():
    settings = get_settings()
    engine = settings.engines.active_engine()
    assert engine.name == settings.engines.active


def test_get_settings_is_cached():
    clear_config_cache()
    first = get_settings()
    second = get_settings()
    assert first is second
