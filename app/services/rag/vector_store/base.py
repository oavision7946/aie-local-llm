from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class Chunk:
    text: str
    corpus: str
    doc_id: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = field(default_factory=lambda: str(uuid4()))
    vector: list[float] | None = None


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class VectorStore(ABC):
    @abstractmethod
    def add(self, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    def query(self, vector: list[float], *, corpus: str, top_k: int) -> list[ScoredChunk]: ...

    @abstractmethod
    def delete(self, doc_id: str) -> int:
        """Delete all chunks belonging to doc_id. Returns the number of chunks removed."""

    @abstractmethod
    def list_documents(self, corpus: str | None = None) -> list[Chunk]: ...
