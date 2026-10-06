import pytest

from app.core.settings import EngineConfig, EnginesConfig
from app.services.engines.errors import EngineNotFoundError
from app.services.engines.fake import FakeEngine
from app.services.engines.registry import EngineRegistry


def test_resolve_by_name_returns_matching_engine():
    fake = FakeEngine(name="fake", model="fake-model")
    registry = EngineRegistry(engines={"fake": fake}, active_name="fake")
    assert registry.resolve("fake") is fake


def test_resolve_falls_back_to_model_name():
    fake = FakeEngine(name="fake", model="some-model-id")
    registry = EngineRegistry(engines={"fake": fake}, active_name="fake")
    assert registry.resolve("some-model-id") is fake


def test_resolve_none_uses_active():
    fake = FakeEngine(name="fake", model="fake-model")
    registry = EngineRegistry(engines={"fake": fake}, active_name="fake")
    assert registry.resolve(None) is fake


def test_resolve_unknown_raises():
    registry = EngineRegistry(engines={}, active_name="fake")
    with pytest.raises(EngineNotFoundError):
        registry.resolve("unknown")


def test_from_config_only_includes_enabled_engines():
    config = EnginesConfig(
        active="vllm",
        engines=[
            EngineConfig(
                name="vllm", base_url="http://localhost:8000/v1", model="m", enabled=True
            ),
            EngineConfig(
                name="sglang", base_url="http://localhost:30000/v1", model="m", enabled=False
            ),
        ],
    )
    registry = EngineRegistry.from_config(config)
    names = {engine.name for engine in registry.list_engines()}
    assert names == {"vllm"}
