import pytest

from app.services.rag.vector_store.base import Chunk
from app.services.rag.vector_store.memory import MemoryVectorStore


def make_chunk(text, corpus="default", doc_id="doc1", vector=None):
    return Chunk(text=text, corpus=corpus, doc_id=doc_id, vector=vector or [1.0, 0.0])


def test_add_requires_vector():
    store = MemoryVectorStore()
    with pytest.raises(ValueError):
        store.add([Chunk(text="x", corpus="c", doc_id="d")])


def test_query_returns_closest_first():
    store = MemoryVectorStore()
    store.add(
        [
            make_chunk("exact match", vector=[1.0, 0.0]),
            make_chunk("orthogonal", vector=[0.0, 1.0]),
            make_chunk("close match", vector=[0.9, 0.1]),
        ]
    )
    results = store.query([1.0, 0.0], corpus="default", top_k=2)
    assert [r.chunk.text for r in results] == ["exact match", "close match"]
    assert results[0].score > results[1].score


def test_query_respects_corpus_isolation():
    store = MemoryVectorStore()
    store.add([make_chunk("in corpus a", corpus="a", vector=[1.0, 0.0])])
    store.add([make_chunk("in corpus b", corpus="b", vector=[1.0, 0.0])])
    results = store.query([1.0, 0.0], corpus="a", top_k=5)
    assert [r.chunk.text for r in results] == ["in corpus a"]


def test_query_empty_corpus_returns_empty():
    store = MemoryVectorStore()
    assert store.query([1.0, 0.0], corpus="missing", top_k=5) == []


def test_delete_removes_all_chunks_for_doc():
    store = MemoryVectorStore()
    store.add(
        [
            make_chunk("chunk1", doc_id="doc-a", vector=[1.0, 0.0]),
            make_chunk("chunk2", doc_id="doc-a", vector=[0.0, 1.0]),
            make_chunk("chunk3", doc_id="doc-b", vector=[1.0, 1.0]),
        ]
    )
    removed = store.delete("doc-a")
    assert removed == 2
    remaining = store.list_documents()
    assert [c.doc_id for c in remaining] == ["doc-b"]


def test_list_documents_filters_by_corpus():
    store = MemoryVectorStore()
    store.add([make_chunk("a", corpus="x", vector=[1.0, 0.0])])
    store.add([make_chunk("b", corpus="y", vector=[1.0, 0.0])])
    assert len(store.list_documents(corpus="x")) == 1
    assert len(store.list_documents()) == 2
