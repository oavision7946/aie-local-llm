import hashlib

from app.services.rag.embeddings.base import Embedder

_DIM = 16


class FakeEmbedder(Embedder):
    """Deterministic hash-based embedder. No model download, no network — for tests only."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i] / 255.0 for i in range(_DIM)]
