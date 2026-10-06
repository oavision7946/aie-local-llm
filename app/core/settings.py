from functools import lru_cache

from pydantic import BaseModel, ConfigDict

from app.core.config import clear_config_cache, load_yaml


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EngineConfig(_Strict):
    name: str
    base_url: str
    model: str
    enabled: bool = True


class EnginesConfig(_Strict):
    active: str
    engines: list[EngineConfig]

    def get(self, name: str) -> EngineConfig:
        for engine in self.engines:
            if engine.name == name:
                return engine
        raise KeyError(f"No engine named '{name}' in engines.yaml")

    def active_engine(self) -> EngineConfig:
        return self.get(self.active)


class EmbeddingsConfig(_Strict):
    model: str
    device: str = "cpu"


class ChunkingConfig(_Strict):
    chunk_size: int
    chunk_overlap: int


class RetrievalConfig(_Strict):
    top_k: int
    default_corpus: str


class RagConfig(_Strict):
    embeddings: EmbeddingsConfig
    chunking: ChunkingConfig
    retrieval: RetrievalConfig


class Settings(_Strict):
    engines: EnginesConfig
    rag: RagConfig


@lru_cache
def get_settings() -> Settings:
    return Settings(
        engines=EnginesConfig(**load_yaml("engines")),
        rag=RagConfig(**load_yaml("rag")),
    )


def reset_settings() -> None:
    get_settings.cache_clear()
    clear_config_cache()
