from app.services.rag.embeddings.fake import FakeEmbedder
from app.services.rag.service import RagService
from app.services.rag.vector_store.memory import MemoryVectorStore


def make_service(chunk_size=500, chunk_overlap=50):
    return RagService(
        embedder=FakeEmbedder(),
        vector_store=MemoryVectorStore(),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def test_ingest_creates_chunks_and_returns_doc_id():
    service = make_service()
    doc_id, chunk_count = service.ingest("The sky is blue. " * 50, corpus="facts")
    assert chunk_count > 0
    assert len(doc_id) > 0


def test_ingest_empty_content_creates_no_chunks():
    service = make_service()
    doc_id, chunk_count = service.ingest("   ", corpus="facts")
    assert chunk_count == 0


def test_retrieve_finds_exact_text_match_first():
    service = make_service(chunk_size=5000, chunk_overlap=0)
    service.ingest("The capital of France is Paris.", corpus="facts")
    service.ingest("The capital of Japan is Tokyo.", corpus="facts")

    results = service.retrieve("The capital of France is Paris.", corpus="facts", top_k=1)

    assert len(results) == 1
    assert "Paris" in results[0].chunk.text


def test_retrieve_respects_corpus():
    service = make_service(chunk_size=5000, chunk_overlap=0)
    service.ingest("secret content", corpus="private")
    results = service.retrieve("secret content", corpus="public", top_k=5)
    assert results == []


def test_delete_document_removes_its_chunks():
    service = make_service(chunk_size=5000, chunk_overlap=0)
    doc_id, _ = service.ingest("some content to delete", corpus="facts")
    removed = service.delete_document(doc_id)
    assert removed >= 1
    assert service.list_documents("facts") == []


def test_format_context_includes_bracketed_citations():
    service = make_service(chunk_size=5000, chunk_overlap=0)
    service.ingest("fact one", corpus="facts")
    results = service.retrieve("fact one", corpus="facts", top_k=1)
    context = RagService.format_context(results)
    assert "[1]" in context
    assert "fact one" in context


def test_format_context_empty_when_no_results():
    assert RagService.format_context([]) == ""
