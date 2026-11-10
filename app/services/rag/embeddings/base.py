from abc import ABC, abstractmethod


class Embedder(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]:
        """Embed a single search query.

        Some embedding models expect a different instruction prefix for queries than for the
        documents they're matched against; overriding this lets an implementation apply one.
        The default embeds the query the same way as a document.
        """
        (vector,) = self.embed([text])
        return vector
