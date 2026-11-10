from app.services.rag.embeddings.base import Embedder
from app.services.rag.embeddings.fake import FakeEmbedder
from app.services.rag.service import RagService
from app.services.rag.vector_store.memory import MemoryVectorStore


def make_service(chunk_size: int = 500, chunk_overlap: int = 50) -> RagService:
    return RagService(
        embedder=FakeEmbedder(),
        vector_store=MemoryVectorStore(),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def test_ingest_creates_chunks_and_returns_doc_id() -> None:
    service = make_service()
    doc_id, chunk_count = service.ingest("The sky is blue. " * 50, corpus="facts")
    assert chunk_count > 0
    assert len(doc_id) > 0


def test_ingest_empty_content_creates_no_chunks() -> None:
    service = make_service()
    doc_id, chunk_count = service.ingest("   ", corpus="facts")
    assert chunk_count == 0


def test_retrieve_finds_exact_text_match_first() -> None:
    service = make_service(chunk_size=5000, chunk_overlap=0)
    service.ingest("The capital of France is Paris.", corpus="facts")
    service.ingest("The capital of Japan is Tokyo.", corpus="facts")

    results = service.retrieve("The capital of France is Paris.", corpus="facts", top_k=1)

    assert len(results) == 1
    assert "Paris" in results[0].chunk.text


def test_retrieve_respects_corpus() -> None:
    service = make_service(chunk_size=5000, chunk_overlap=0)
    service.ingest("secret content", corpus="private")
    results = service.retrieve("secret content", corpus="public", top_k=5)
    assert results == []


def test_delete_document_removes_its_chunks() -> None:
    service = make_service(chunk_size=5000, chunk_overlap=0)
    doc_id, _ = service.ingest("some content to delete", corpus="facts")
    removed = service.delete_document(doc_id)
    assert removed >= 1
    assert service.list_documents("facts") == []


def test_format_context_includes_bracketed_citations() -> None:
    service = make_service(chunk_size=5000, chunk_overlap=0)
    service.ingest("fact one", corpus="facts")
    results = service.retrieve("fact one", corpus="facts", top_k=1)
    context = RagService.format_context(results)
    assert "[1]" in context
    assert "fact one" in context


def test_format_context_empty_when_no_results() -> None:
    assert RagService.format_context([]) == ""


class _RecordingEmbedder(Embedder):
    """Tracks which Embedder method each call used, like a model needing an instruction
    prefix only on queries — this would need embed_query, not embed, to be the one called."""

    def __init__(self) -> None:
        self.embed_calls: list[list[str]] = []
        self.embed_query_calls: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(texts)
        return FakeEmbedder().embed(texts)

    def embed_query(self, text: str) -> list[float]:
        self.embed_query_calls.append(text)
        return FakeEmbedder().embed([text])[0]


def test_retrieve_calls_embed_query_not_embed() -> None:
    embedder = _RecordingEmbedder()
    service = RagService(
        embedder=embedder, vector_store=MemoryVectorStore(), chunk_size=5000, chunk_overlap=0
    )
    service.ingest("some content", corpus="facts")
    embedder.embed_calls.clear()  # drop the ingest call's record; only retrieve's call matters here

    service.retrieve("a query", corpus="facts", top_k=1)

    assert embedder.embed_query_calls == ["a query"]
    assert embedder.embed_calls == []
