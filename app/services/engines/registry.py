from app.core.settings import EnginesConfig
from app.services.engines.base import Engine
from app.services.engines.errors import EngineNotFoundError
from app.services.engines.openai_compatible import OpenAICompatibleEngine


class EngineRegistry:
    def __init__(self, engines: dict[str, Engine], active_name: str) -> None:
        self._engines = engines
        self._active_name = active_name

    @classmethod
    def from_config(cls, config: EnginesConfig) -> "EngineRegistry":
        engines = {
            engine.name: OpenAICompatibleEngine(
                name=engine.name, base_url=engine.base_url, model=engine.model
            )
            for engine in config.engines
            if engine.enabled
        }
        return cls(engines=engines, active_name=config.active)

    def list_engines(self) -> list[Engine]:
        return list(self._engines.values())

    def active(self) -> Engine:
        return self.resolve(self._active_name)

    def resolve(self, name: str | None) -> Engine:
        if name is None:
            name = self._active_name
        try:
            return self._engines[name]
        except KeyError:
            # Also allow resolving by the underlying model name, e.g. clients that pass
            # the model id straight through rather than our engine name.
            for engine in self._engines.values():
                if engine.model == name:
                    return engine
            raise EngineNotFoundError(f"No enabled engine or model named '{name}'") from None
