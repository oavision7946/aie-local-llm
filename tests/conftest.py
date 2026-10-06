import os

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_chat_service, get_engine_registry, get_rag_service, get_vector_store
from app.main import app
from app.services.chat.service import ChatService
from app.services.engines.fake import FakeEngine
from app.services.engines.registry import EngineRegistry
from app.services.rag.embeddings.fake import FakeEmbedder
from app.services.rag.service import RagService
from app.services.rag.vector_store.memory import MemoryVectorStore

for _key in ("OPENAI_API_KEY", "VLLM_API_KEY", "AIE_LOCAL_LLM_API_KEY"):
    os.environ.pop(_key, None)


@pytest.fixture
def fake_engine() -> FakeEngine:
    return FakeEngine(name="fake", model="fake-model")


@pytest.fixture
def fake_registry(fake_engine: FakeEngine) -> EngineRegistry:
    return EngineRegistry(engines={"fake": fake_engine}, active_name="fake")


@pytest.fixture
def memory_vector_store() -> MemoryVectorStore:
    return MemoryVectorStore()


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def rag_service(fake_embedder: FakeEmbedder, memory_vector_store: MemoryVectorStore) -> RagService:
    return RagService(
        embedder=fake_embedder, vector_store=memory_vector_store, chunk_size=500, chunk_overlap=50
    )


@pytest.fixture
def chat_service(fake_registry: EngineRegistry, rag_service: RagService) -> ChatService:
    return ChatService(
        registry=fake_registry, rag_service=rag_service, default_top_k=5, default_corpus="default"
    )


@pytest.fixture
def client(
    fake_registry: EngineRegistry, rag_service: RagService, memory_vector_store: MemoryVectorStore,
    chat_service: ChatService,
) -> TestClient:
    app.dependency_overrides[get_engine_registry] = lambda: fake_registry
    app.dependency_overrides[get_rag_service] = lambda: rag_service
    app.dependency_overrides[get_vector_store] = lambda: memory_vector_store
    app.dependency_overrides[get_chat_service] = lambda: chat_service
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
