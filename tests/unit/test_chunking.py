import pytest

from app.services.rag.chunking import chunk_text


def test_short_text_is_single_chunk():
    chunks = chunk_text("hello world", chunk_size=100, chunk_overlap=10)
    assert chunks == ["hello world"]


def test_empty_text_yields_no_chunks():
    assert chunk_text("   ", chunk_size=100, chunk_overlap=10) == []


def test_long_text_is_split_with_overlap():
    text = "a" * 10 + " " + "b" * 10 + " " + "c" * 10
    chunks = chunk_text(text, chunk_size=15, chunk_overlap=5)
    assert len(chunks) > 1
    # consecutive chunks overlap by `chunk_overlap` characters
    assert chunks[0][-5:] == chunks[1][:5]


def test_covers_entire_text():
    text = " ".join(f"word{i}" for i in range(100))
    normalized = " ".join(text.split())
    chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)
    assert chunks[-1][-1] == normalized[-1]


@pytest.mark.parametrize("chunk_size,chunk_overlap", [(0, 0), (-1, 0), (10, 10), (10, 20)])
def test_invalid_sizes_raise(chunk_size, chunk_overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
