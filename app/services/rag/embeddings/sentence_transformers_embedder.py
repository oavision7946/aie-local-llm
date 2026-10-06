from app.services.rag.embeddings.base import Embedder


class SentenceTransformersEmbedder(Embedder):
    """Loads a sentence-transformers model once and embeds in-process.

    Defaults to CPU: the local engine (vLLM/SGLang) typically reserves most of the
    GPU's memory, and this box's unified memory architecture means an embedding
    model competing for GPU memory can OOM the whole process. Pass device="cuda"
    only if you've sized --gpu-memory-utilization on the engine to leave headroom.
    """

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return vectors.tolist()
