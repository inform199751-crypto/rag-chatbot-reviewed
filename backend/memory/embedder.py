from typing import Any

import sentence_transformers
import torch

from memory.base_embedder import BaseEmbedder

EMBEDDER_ARGS = {
    "jinaai/jina-embeddings-v5-text-nano-retrieval": {
        "trust_remote_code": True,
        "model_kwargs": {"dtype": torch.bfloat16},  # Recommended for GPUs
        "config_kwargs": {},
    },
    "jinaai/jina-embeddings-v5-text-small-retrieval": {
        "trust_remote_code": True,
        "model_kwargs": {"dtype": torch.bfloat16},  # Recommended for GPUs
        "config_kwargs": {},
    },
}


class Embedder(BaseEmbedder):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_folder: str | None = None, **kwargs: Any):
        """
        Initialize the Embedder class with the specified parameters.

        Args:
            model_name (str): The name of the SentenceTransformer model to use for embedding.
            cache_folder (str | None): The directory where the model will be cached.
            **kwargs (Any): Additional keyword arguments to pass to the SentenceTransformer model.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        args = EMBEDDER_ARGS.get(model_name, None)
        if args is not None:
            # Copy before touching `model_kwargs`: EMBEDDER_ARGS is module-level state shared by
            # every instantiation, so mutating it in place would leak into later ones.
            args = {**args, "model_kwargs": dict(args.get("model_kwargs", {}))}
            if device == "cpu":
                # bfloat16 is a GPU recommendation. On CPU it is only competitive where
                # AVX512-BF16 or AMX is available and is markedly slower than float32 elsewhere,
                # so drop the override and let the model load at its default precision.
                args["model_kwargs"].pop("dtype", None)
            kwargs.update(args)

        self.client = sentence_transformers.SentenceTransformer(
            model_name_or_path=model_name, device=device, cache_folder=cache_folder, **kwargs
        )

    @staticmethod
    def _clean_texts(texts: list[str]) -> list[str]:
        """
        Clean the input texts by replacing newline characters with spaces.

        Args:
            texts (list[str]): The list of texts to clean.

        Returns:
            list[str]: The cleaned list of texts.
        """
        return [x.replace("\n", " ") for x in texts]

    def embed_documents(self, texts: list[str], multi_process: bool = False, **encode_kwargs: Any) -> list[list[float]]:
        """
        Compute document embeddings using a transformer model.

        Notes:
            The more general `SentenceTransformer.encode` method differs in two ways from
            `SentenceTransformer.encode_query` and `SentenceTransformer.encode_document`:
            - If no prompt_name or prompt is provided, it uses a predefined “query” or “document” prompt, if specified
              in the model’s prompts dictionary.
            - It sets the task to “document”. If the model has a Router module, it will use the “query” or “document”
              task type to route the input through the appropriate submodules.

        Args:
            texts (list[str]): The list of texts to embed.
            multi_process (bool): If True, use multiple processes to compute embeddings.
            **encode_kwargs (Any): Additional keyword arguments to pass when calling the `encode` method of the model.
                `show_progress_bar` defaults to True here because indexing is a batch job someone is watching;
                pass `show_progress_bar=False` to silence it.

        Returns:
            list[list[float]]: A list of embeddings, one for each text.
        """

        texts = self._clean_texts(texts)
        if multi_process:
            pool = self.client.start_multi_process_pool()
            embeddings = self.client.encode(sentences=texts, pool=pool)
            sentence_transformers.SentenceTransformer.stop_multi_process_pool(pool)
        else:
            encode_kwargs.setdefault("show_progress_bar", True)
            embeddings = self.client.encode(sentences=texts, normalize_embeddings=False, **encode_kwargs)

        return embeddings.tolist()

    def embed_query(self, text: str, **encode_kwargs) -> list[float]:
        """
        Compute query embeddings using a transformer model.

        Notes:
            The more general `SentenceTransformer.encode` method differs in two ways from
            `SentenceTransformer.encode_query` and `SentenceTransformer.encode_document`:
            - If no prompt_name or prompt is provided, it uses a predefined “query” or “document” prompt, if specified
              in the model’s prompts dictionary.
            - It sets the task to “document”. If the model has a Router module, it will use the “query” or “document”
              task type to route the input through the appropriate submodules.

        Args:
            text (str): The text to embed.
            **encode_kwargs: Additional keyword arguments to pass when calling the `encode` method of the model.

        Returns:
            list[float]: Embeddings for the text.
        """
        text = self._clean_texts([text])[0]

        # No progress bar on the query path. This runs once per user message, always on a single
        # string, so the bar can never show anything useful -- but it does write a tqdm frame to
        # stderr on every request. In a served deployment that buries the real log lines, and the
        # bar redraws with carriage returns, so a log aggregator sees one unreadable line per query.
        encode_kwargs.setdefault("show_progress_bar", False)
        embeddings = self.client.encode(sentences=text, normalize_embeddings=False, **encode_kwargs)

        return embeddings.tolist()
