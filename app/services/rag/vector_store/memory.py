import numpy as np

from app.services.rag.vector_store.base import Chunk, ScoredChunk, VectorStore


class MemoryVectorStore(VectorStore):
    """In-memory, numpy cosine-similarity vector store. Corpora are isolated namespaces.

    Not persisted across restarts. Implements the same VectorStore ABC a future
    Postgres/pgvector-backed store would, so callers never need to change.
    """

    def __init__(self) -> None:
        self._chunks: dict[str, list[Chunk]] = {}

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            if chunk.vector is None:
                raise ValueError("Chunk must have a vector before being added to the store")
            self._chunks.setdefault(chunk.corpus, []).append(chunk)

    def query(self, vector: list[float], *, corpus: str, top_k: int) -> list[ScoredChunk]:
        candidates = self._chunks.get(corpus, [])
        if not candidates:
            return []

        query_vec = np.array(vector)
        query_norm = np.linalg.norm(query_vec)
        scored: list[ScoredChunk] = []
        for chunk in candidates:
            vec = np.array(chunk.vector)
            denom = query_norm * np.linalg.norm(vec)
            score = float(np.dot(query_vec, vec) / denom) if denom else 0.0
            scored.append(ScoredChunk(chunk=chunk, score=score))

        scored.sort(key=lambda sc: sc.score, reverse=True)
        return scored[:top_k]

    def delete(self, doc_id: str) -> int:
        removed = 0
        for corpus, chunks in list(self._chunks.items()):
            kept = [c for c in chunks if c.doc_id != doc_id]
            removed += len(chunks) - len(kept)
            self._chunks[corpus] = kept
        return removed

    def list_documents(self, corpus: str | None = None) -> list[Chunk]:
        if corpus is not None:
            return list(self._chunks.get(corpus, []))
        return [chunk for chunks in self._chunks.values() for chunk in chunks]
