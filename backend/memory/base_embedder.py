"""
The contract every embedder must satisfy.

`Chroma` is handed an embedder and calls it for both sides of the index: `embed_documents` when
ingesting chunks and `embed_query` when searching. Both sides must come from the same model, or the
vectors stop being comparable, so the two methods belong to one object rather than being wired up
independently.
"""

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Interface for turning text into vectors."""

    model_name: str
    """Which model produced these vectors. Part of the contract because an index is only valid
    for the model that built it, and the index needs a way to record what that was -- vectors
    themselves carry no such record, so a swapped model degrades results with no error."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of documents.

        Args:
            texts: The texts to embed

        Returns:
            list[list[float]]: One embedding per input text, in the same order. Implementations must
                preserve both order and length, since callers zip the result against ids and metadata.
        """

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single search query.

        Args:
            text: The query text

        Returns:
            list[float]: The query embedding
        """
