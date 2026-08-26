"""Retrieval, with an optional cross-encoder second stage.

This lives in the service layer rather than on `Chroma` on purpose. Reranking is not a vector
store concern -- the store's job is nearest neighbours, and pushing a cross-encoder into it would
mean every caller pays for the import and every alternative store has to reimplement the same
orchestration.

The two paths deliberately return the same `(documents, sources)` shape, so the caller does not
branch on whether reranking is on. That matters because the alternative -- two response formats
depending on a setting -- is the sort of difference that only shows up once the setting is
flipped in production.
"""

from __future__ import annotations

import logging
from typing import Any

from services.ingest_documents_service.document import Document

logger = logging.getLogger(__name__)


def _to_sources(docs_and_scores: list[tuple[Document, float]]) -> list[dict[str, Any]]:
    """Mirror the shape `Chroma.similarity_search_with_threshold` produces, so
    `prettify_source` keeps working regardless of which stage assigned the score."""
    return [
        {
            "score": round(score, 3),
            "document": doc.metadata.get("source"),
            "content_preview": f"{doc.page_content[0:256]}...",
        }
        for doc, score in docs_and_scores
    ]


def retrieve(
    index,
    query: str,
    num_retrievals: int,
    threshold: float,
    reranker=None,
    candidates: int = 20,
    rerank_threshold: float = 0.3,
) -> tuple[list[Document], list[dict[str, Any]]]:
    """
    Fetch the chunks that will go into the prompt.

    Args:
        index: The vector store.
        query: The question to retrieve for. Pass the same text to both stages -- the refined
            question, if the caller refined one -- or the two stages rank against different
            questions and the second undoes the first.
        num_retrievals: How many chunks survive to the prompt.
        threshold: Minimum relevance on the embedding model's scale. Used when not reranking.
        reranker: A `CrossEncoderReranker`, or None for single-stage retrieval.
        candidates: First-stage width when reranking. Measured, not maximised: on the golden set,
            widening this from 20 to the whole corpus lowered English recall@3 by 10 points,
            because the extra candidates are mostly noise and the reranker promotes some of it.
        rerank_threshold: Minimum reranked score. Separate from `threshold` because a
            cross-encoder sigmoid and a cosine relevance score are different scales.

    Returns:
        (documents, sources). Empty lists when nothing clears the threshold -- which the caller
        must treat as "no context", not as "context I can ignore".
    """
    if reranker is None:
        return index.similarity_search_with_threshold(query=query, k=num_retrievals, threshold=threshold)

    # First stage casts the wide net. No threshold here: it is expressed on the embedding
    # model's scale, and anything it filters out is something the reranker never gets to judge.
    pool = index.similarity_search_with_relevance_scores(query=query, k=candidates)
    if not pool:
        return [], []

    reranked = reranker.rerank(query, [doc for doc, _ in pool], top_k=num_retrievals)
    kept = [(doc, score) for doc, score in reranked if score > rerank_threshold]

    if not kept:
        logger.warning(
            "Reranker kept nothing: %d candidates, best score %.3f, threshold %.3f",
            len(pool),
            reranked[0][1] if reranked else float("nan"),
            rerank_threshold,
        )

    return [doc for doc, _ in kept], _to_sources(kept)
