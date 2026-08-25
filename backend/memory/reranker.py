"""Second-stage reranking with a cross-encoder.

Why a second stage at all
    Dense retrieval scores a query against a document by comparing two vectors that were
    computed independently. That is what makes it cheap enough to run over the whole corpus,
    and it is also its ceiling: the model never sees the query and the passage together, so it
    cannot tell that a passage merely mentions the query's topic rather than answering it.

    A cross-encoder does see both at once. That makes it far more accurate and far too slow to
    run over every chunk -- hence two stages: dense retrieval casts a wide net (top-N), the
    cross-encoder reorders that net and keeps the best few.

Why the scores are squashed to [0, 1]
    Cross-encoders emit raw logits, unbounded and centred wherever training put them. The
    retrieval path already has a threshold expressed on the cosine relevance scale of [0, 1],
    and mixing the two scales silently would mean the configured threshold no longer refers to
    anything. Applying a sigmoid keeps one comparable scale -- but note the two are still not
    interchangeable: a reranker's 0.5 is not a cosine 0.5, which is why the threshold has to be
    re-measured against the eval set after turning reranking on.
"""

from __future__ import annotations

import logging

import torch
from sentence_transformers import CrossEncoder
from services.ingest_documents_service.document import Document

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reorders candidate documents by scoring each one jointly with the query."""

    def __init__(self, model_name: str, max_length: int = 512, **kwargs) -> None:
        """
        Args:
            model_name: A cross-encoder model. Must cover the corpus languages -- an
                English-only reranker will happily reorder Chinese passages into nonsense,
                and nothing about the output looks wrong.
            max_length: Query and passage are concatenated into one sequence, so this budget is
                shared. Passages longer than the remainder are truncated, which means a chunk
                size close to `max_length` gets its tail ignored during reranking.
            **kwargs: Passed through to `CrossEncoder`.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.client = CrossEncoder(model_name, max_length=max_length, device=device, **kwargs)
        logger.info(f"Reranker loaded: {model_name} on {device}")

    def rerank(
        self,
        query: str,
        documents: list[Document],
        top_k: int | None = None,
        normalise: bool = True,
    ) -> list[tuple[Document, float]]:
        """
        Score every candidate against the query and return them best-first.

        Args:
            query: The user's question, as it will be answered -- pass the same text the dense
                stage used, or the two stages are ranking against different questions.
            documents: Candidates from the first stage.
            top_k: How many to keep. None keeps all, reordered.
            normalise: Squash logits through a sigmoid into [0, 1]. See the module docstring.

        Returns:
            (document, score) pairs sorted by descending score. Empty input returns empty
            output rather than raising: "nothing was retrieved" is a normal outcome upstream,
            and turning it into an exception here would only move the handling somewhere worse.
        """
        if not documents:
            return []

        pairs = [(query, doc.page_content) for doc in documents]
        scores = self.client.predict(pairs, show_progress_bar=False)

        if normalise:
            scores = torch.sigmoid(torch.as_tensor(scores, dtype=torch.float32)).tolist()
        else:
            scores = [float(s) for s in scores]

        ranked = sorted(zip(documents, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_k] if top_k else ranked


def create_reranker(model_name: str | None = None, enabled: bool | None = None) -> CrossEncoderReranker | None:
    """
    Build the reranker, or None when it is switched off.

    Returning None rather than a pass-through object is deliberate: the caller has to decide
    what a missing second stage means for its scoring, and a no-op reranker would let that
    decision be skipped by accident.
    """
    from config import settings

    if enabled is None:
        enabled = settings.RERANK_ENABLED
    if not enabled:
        return None

    return CrossEncoderReranker(model_name or settings.RERANK_MODEL)
