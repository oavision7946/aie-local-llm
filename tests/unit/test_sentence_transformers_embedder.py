from typing import Any

import pytest

from app.services.rag.embeddings.sentence_transformers_embedder import (
    SentenceTransformersEmbedder,
)


class FakeSentenceTransformer:
    """Stands in for sentence_transformers.SentenceTransformer: no model download."""

    last_init_kwargs: dict[str, Any] = {}

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        self.model_name = model_name
        FakeSentenceTransformer.last_init_kwargs = kwargs

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> "FakeVectors":
        del normalize_embeddings
        return FakeVectors([[float(len(text))] for text in texts])


class FakeVectors:
    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def tolist(self) -> list[list[float]]:
        return self._rows


@pytest.fixture(autouse=True)
def _patch_sentence_transformer(monkeypatch: pytest.MonkeyPatch) -> None:
    import sentence_transformers

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", FakeSentenceTransformer)


def test_forwards_load_kwargs_to_sentence_transformer() -> None:
    SentenceTransformersEmbedder(
        "some/model",
        device="cuda",
        trust_remote_code=True,
        truncate_dim=256,
        dtype="bfloat16",
    )
    kwargs = FakeSentenceTransformer.last_init_kwargs
    assert kwargs["device"] == "cuda"
    assert kwargs["trust_remote_code"] is True
    assert kwargs["truncate_dim"] == 256
    assert kwargs["model_kwargs"] == {"dtype": "bfloat16"}


def test_omits_model_kwargs_when_dtype_not_set() -> None:
    SentenceTransformersEmbedder("some/model")
    assert FakeSentenceTransformer.last_init_kwargs["model_kwargs"] is None


def test_embed_applies_document_prompt_to_every_text() -> None:
    embedder = SentenceTransformersEmbedder("some/model", document_prompt="passage: ")
    assert embedder.embed(["a", "bb"]) == [[len("passage: a")], [len("passage: bb")]]


def test_embed_without_document_prompt_is_unprefixed() -> None:
    embedder = SentenceTransformersEmbedder("some/model")
    assert embedder.embed(["a"]) == [[1.0]]


def test_embed_empty_list_returns_empty_without_calling_model() -> None:
    embedder = SentenceTransformersEmbedder("some/model", document_prompt="passage: ")
    assert embedder.embed([]) == []


def test_embed_query_applies_query_prompt() -> None:
    embedder = SentenceTransformersEmbedder("some/model", query_prompt="query: ")
    assert embedder.embed_query("x") == [len("query: x")]


def test_embed_query_without_prompt_is_unprefixed() -> None:
    embedder = SentenceTransformersEmbedder("some/model")
    assert embedder.embed_query("abc") == [3.0]
