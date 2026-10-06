from uuid import uuid4

from app.services.rag.chunking import chunk_text
from app.services.rag.embeddings.base import Embedder
from app.services.rag.vector_store.base import Chunk, ScoredChunk, VectorStore


class RagService:
    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        *,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def ingest(self, content: str, *, corpus: str, metadata: dict | None = None) -> tuple[str, int]:
        doc_id = str(uuid4())
        texts = chunk_text(content, chunk_size=self._chunk_size, chunk_overlap=self._chunk_overlap)
        if not texts:
            return doc_id, 0

        vectors = self._embedder.embed(texts)
        chunks = [
            Chunk(text=text, corpus=corpus, doc_id=doc_id, metadata=metadata or {}, vector=vector)
            for text, vector in zip(texts, vectors, strict=True)
        ]
        self._vector_store.add(chunks)
        return doc_id, len(chunks)

    def retrieve(self, query: str, *, corpus: str, top_k: int) -> list[ScoredChunk]:
        (query_vector,) = self._embedder.embed([query])
        return self._vector_store.query(query_vector, corpus=corpus, top_k=top_k)

    def delete_document(self, doc_id: str) -> int:
        return self._vector_store.delete(doc_id)

    def list_documents(self, corpus: str | None = None):
        return self._vector_store.list_documents(corpus)

    @staticmethod
    def format_context(chunks: list[ScoredChunk]) -> str:
        if not chunks:
            return ""
        parts = [f"[{i + 1}] {sc.chunk.text}" for i, sc in enumerate(chunks)]
        return (
            "Use the following retrieved context to answer the user's question. "
            "Cite sources by their bracketed number when relevant.\n\n" + "\n\n".join(parts)
        )
