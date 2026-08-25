"""
Embedder backed by OpenAI's embeddings API.

Note that this talks to OpenAI directly, not through OpenRouter: OpenRouter proxies chat
completions, not embeddings. Running the LLM on OpenRouter and the embedder on OpenAI therefore
needs two separate keys.
"""

from typing import Any

from helpers.log import get_logger
from openai import OpenAI

from memory.base_embedder import BaseEmbedder

logger = get_logger(__name__)

# The API accepts up to 2048 inputs per request, but the real constraint is the token total. Chunks
# are ~1000 characters by default, so a smaller batch keeps requests comfortably inside the limit
# while still cutting the number of round trips by two orders of magnitude.
DEFAULT_BATCH_SIZE = 256

# The API rejects an empty input. A chunk can end up blank after cleaning, and dropping it would
# desynchronise the embeddings from the ids and metadata the caller zips them against, so blanks are
# substituted rather than removed.
EMPTY_TEXT_PLACEHOLDER = " "


class OpenAIEmbedder(BaseEmbedder):
    """Computes embeddings using an OpenAI embedding model."""

    def __init__(
        self,
        api_key: str,
        model_name: str = "text-embedding-3-small",
        base_url: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        timeout: int = 120,
        **kwargs: Any,
    ):
        """
        Initialize the OpenAI embedder.

        Args:
            api_key: OpenAI API key. Required.
            model_name: Embedding model to use. `text-embedding-3-small` returns 1536 dimensions.
            base_url: Optional override, for an OpenAI-compatible embeddings endpoint
            batch_size: How many texts to send per request
            timeout: Request timeout in seconds
            **kwargs: Additional keyword arguments passed to the OpenAI client

        Raises:
            ValueError: If no API key was supplied
        """
        if not api_key:
            raise ValueError("OPENAI_API_KEY is empty. Set it in .env before selecting EMBEDDING_PROVIDER=openai.")

        self.model_name = model_name
        self.batch_size = max(1, batch_size)
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, **kwargs)

    @staticmethod
    def _clean_texts(texts: list[str]) -> list[str]:
        """
        Normalise texts before embedding.

        Newlines are collapsed to spaces, matching the sentence-transformers embedder so that
        switching providers does not silently change what gets embedded. Blank results are replaced,
        because the API rejects empty input.

        Args:
            texts: The texts to clean

        Returns:
            list[str]: The cleaned texts, same order and length as the input
        """
        cleaned = []
        for text in texts:
            normalised = text.replace("\n", " ").strip()
            cleaned.append(normalised or EMPTY_TEXT_PLACEHOLDER)
        return cleaned

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Compute embeddings for a batch of documents.

        Args:
            texts: The texts to embed

        Returns:
            list[list[float]]: One embedding per input text, in input order
        """
        if not texts:
            return []

        cleaned = self._clean_texts(texts)
        embeddings: list[list[float]] = []

        for start in range(0, len(cleaned), self.batch_size):
            batch = cleaned[start : start + self.batch_size]
            logger.info(
                f"Embedding documents {start + 1}-{start + len(batch)} of {len(cleaned)} with {self.model_name}"
            )
            try:
                response = self.client.embeddings.create(model=self.model_name, input=batch)
            except Exception as e:
                logger.error(f"Error embedding documents with {self.model_name}: {e}")
                raise

            # The API documents that results come back in input order, but it also returns an
            # explicit index per item. Sorting on it costs nothing and removes the assumption.
            ordered = sorted(response.data, key=lambda item: item.index)
            embeddings.extend(item.embedding for item in ordered)

        if len(embeddings) != len(texts):
            raise RuntimeError(f"Embedding count mismatch: asked for {len(texts)}, received {len(embeddings)}.")

        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        Compute the embedding for a single search query.

        Args:
            text: The query text

        Returns:
            list[float]: The query embedding
        """
        cleaned = self._clean_texts([text])[0]

        try:
            response = self.client.embeddings.create(model=self.model_name, input=cleaned)
        except Exception as e:
            logger.error(f"Error embedding query with {self.model_name}: {e}")
            raise

        return response.data[0].embedding
