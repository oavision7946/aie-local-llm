from functools import lru_cache

from app.core.settings import get_settings
from app.services.chat.service import ChatService
from app.services.engines.registry import EngineRegistry
from app.services.rag.embeddings.base import Embedder
from app.services.rag.service import RagService
from app.services.rag.vector_store.base import VectorStore
from app.services.rag.vector_store.memory import MemoryVectorStore


@lru_cache
def get_engine_registry() -> EngineRegistry:
    return EngineRegistry.from_config(get_settings().engines)


@lru_cache
def get_embedder() -> Embedder:
    from app.services.rag.embeddings.sentence_transformers_embedder import (
        SentenceTransformersEmbedder,
    )

    embeddings_config = get_settings().rag.embeddings
    return SentenceTransformersEmbedder(embeddings_config.model, device=embeddings_config.device)


@lru_cache
def get_vector_store() -> VectorStore:
    return MemoryVectorStore()


@lru_cache
def get_rag_service() -> RagService:
    rag_config = get_settings().rag
    return RagService(
        embedder=get_embedder(),
        vector_store=get_vector_store(),
        chunk_size=rag_config.chunking.chunk_size,
        chunk_overlap=rag_config.chunking.chunk_overlap,
    )


@lru_cache
def get_chat_service() -> ChatService:
    rag_config = get_settings().rag
    return ChatService(
        registry=get_engine_registry(),
        rag_service=get_rag_service(),
        default_top_k=rag_config.retrieval.top_k,
        default_corpus=rag_config.retrieval.default_corpus,
    )
