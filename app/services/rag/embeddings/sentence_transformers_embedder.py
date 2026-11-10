from typing import cast

from app.services.rag.embeddings.base import Embedder


class SentenceTransformersEmbedder(Embedder):
    """Loads a sentence-transformers model once and embeds in-process.

    Defaults to CPU: the local engine (vLLM/SGLang) typically reserves most of the
    GPU's memory, and this box's unified memory architecture means an embedding
    model competing for GPU memory can OOM the whole process. Pass device="cuda"
    only if you've sized --gpu-memory-utilization on the engine to leave headroom.

    Several newer embedding models expect a literal instruction prefix on the query and/or
    document text for good retrieval quality (sometimes required, not just recommended — see
    each model's card); `query_prompt`/`document_prompt` prepend one verbatim. `trust_remote_code`
    is needed for models that ship custom modeling code (e.g. MoE architectures). `dtype` is
    forwarded as `model_kwargs={"dtype": ...}` so a large model doesn't default to float32.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        *,
        trust_remote_code: bool = False,
        query_prompt: str | None = None,
        document_prompt: str | None = None,
        truncate_dim: int | None = None,
        dtype: str | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            model_name,
            device=device,
            trust_remote_code=trust_remote_code,
            truncate_dim=truncate_dim,
            model_kwargs={"dtype": dtype} if dtype else None,
        )
        self._query_prompt = query_prompt
        self._document_prompt = document_prompt

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._document_prompt:
            texts = [f"{self._document_prompt}{text}" for text in texts]
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return cast(list[list[float]], vectors.tolist())

    def embed_query(self, text: str) -> list[float]:
        if self._query_prompt:
            text = f"{self._query_prompt}{text}"
        vectors = self._model.encode([text], normalize_embeddings=True)
        (vector,) = cast(list[list[float]], vectors.tolist())
        return vector
