"""Retrieval, with an optional cross-encoder second stage.

This lives in the service layer rather than on `Chroma` on purpose. Reranking is not a vector
store concern -- the store's job is nearest neighbours, and pushing a cross-encoder into it would
mean every caller pays for the import and every alternative store has to reimplement the same
orchestration.

Both paths return the same `RetrievalOutcome`, so the caller does not branch on whether
reranking is on. That matters because the alternative -- two response shapes depending on a
setting -- is the sort of difference that only shows up once the setting is flipped in
production.

Why the outcome carries a reason
    "Nothing came back" has at least three causes that need different responses: the index is
    empty, the corpus has nothing on the topic, or the threshold is set too high for the
    embedding model in use. Collapsing them into an empty list makes them indistinguishable to
    the caller, and an operator debugging "the bot says it found nothing" has no way in.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, NamedTuple

from services.ingest_documents_service.document import Document

logger = logging.getLogger(__name__)


class NoContextReason(str, Enum):
    """Why retrieval produced nothing. Only set when `documents` is empty."""

    EMPTY_INDEX = "empty_index"
    """The store returned no neighbours at all -- nothing is indexed, or the index is not the
    one the app is pointed at."""

    BELOW_THRESHOLD = "below_threshold"
    """Neighbours came back but none cleared RETRIEVAL_THRESHOLD. Either the corpus does not
    cover the question, or the threshold does not suit the embedding model."""

    BELOW_RERANK_THRESHOLD = "below_rerank_threshold"
    """The dense stage found candidates and the cross-encoder rejected all of them."""


class RetrievalOutcome(NamedTuple):
    documents: list[Document]
    sources: list[dict[str, Any]]
    reason: NoContextReason | None = None
    """None when documents were found."""

    best_rejected_score: float | None = None
    """Highest score among the chunks a threshold discarded. This is the number that says
    whether the threshold is slightly too high or the corpus simply has nothing -- 0.19 against
    a 0.2 threshold is a tuning problem, 0.02 is not."""


def explain_no_context(outcome: RetrievalOutcome) -> str:
    """
    Turn an empty outcome into something the reader can act on.

    The original message -- "I did not detect any pertinent chunk of text from the documents" --
    is the same sentence for an empty index, an off-topic question and a mistuned threshold.
    A user cannot tell whether to rephrase, upload something, or tell an operator, and an
    operator reading a bug report cannot tell which of the three happened.
    """
    if outcome.reason is NoContextReason.EMPTY_INDEX:
        return (
            "No documents are indexed yet, so there was nothing to search. "
            "Upload a Markdown file, or run `scripts/memory_builder.py` over `docs/`. \n\n"
        )

    near_miss = outcome.best_rejected_score is not None and outcome.best_rejected_score > 0.0
    detail = f" The closest match scored {outcome.best_rejected_score:.3f}, below the cutoff." if near_miss else ""

    if outcome.reason is NoContextReason.BELOW_RERANK_THRESHOLD:
        return (
            "The documents were searched and candidates were found, but the reranker judged "
            f"none of them to actually answer this question.{detail} \n\n"
        )

    return (
        "Nothing in the indexed documents is close enough to this question. "
        f"Try rephrasing it using wording from your documents.{detail} \n\n"
    )


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
) -> RetrievalOutcome:
    """
    Fetch the chunks that will go into the prompt.

    Args:
        index: The vector store.
        query: The question to retrieve for. Pass the same text to both stages -- the refined
            question, if the caller refined one -- or the two stages rank against different
            questions and the second undoes the first.
        num_retrievals: How many chunks survive to the prompt.
        threshold: Minimum relevance on the embedding model's scale. Ignored when reranking,
            where `rerank_threshold` applies instead.
        reranker: A `CrossEncoderReranker`, or None for single-stage retrieval.
        candidates: First-stage width when reranking. Measured, not maximised: on the golden
            set, widening this from 20 to the whole corpus lowered English recall@3 by 10
            points, because the extra candidates are mostly noise and the reranker promotes
            some of it.
        rerank_threshold: Minimum reranked score. Separate from `threshold` because a
            cross-encoder sigmoid and a cosine relevance score are different scales.

    Returns:
        A `RetrievalOutcome`. When `documents` is empty, `reason` says which of the three
        failure modes occurred.
    """
    # One code path for both stages so the diagnostics are the same either way. Filtering here
    # rather than calling `similarity_search_with_threshold` is what makes EMPTY_INDEX
    # distinguishable from BELOW_THRESHOLD -- that method returns an empty list for both.
    pool_size = candidates if reranker is not None else num_retrievals
    pool = index.similarity_search_with_relevance_scores(query=query, k=pool_size)

    if not pool:
        logger.warning("Retrieval returned no neighbours at all for query: %r", query)
        return RetrievalOutcome([], [], NoContextReason.EMPTY_INDEX)

    if reranker is None:
        scored = sorted(pool, key=lambda pair: pair[1], reverse=True)
        cutoff, why = threshold, NoContextReason.BELOW_THRESHOLD
    else:
        # No first-stage threshold when reranking: it is expressed on the embedding model's
        # scale, and anything it removes is something the cross-encoder never gets to judge.
        scored = reranker.rerank(query, [doc for doc, _ in pool], top_k=num_retrievals)
        cutoff, why = rerank_threshold, NoContextReason.BELOW_RERANK_THRESHOLD

    kept = [(doc, score) for doc, score in scored if score > cutoff][:num_retrievals]

    if not kept:
        best = max((score for _, score in scored), default=None)
        logger.warning(
            "All %d candidates fell below the %s cutoff of %.3f (best was %.3f) for query: %r",
            len(scored),
            why.value,
            cutoff,
            best if best is not None else float("nan"),
            query,
        )
        return RetrievalOutcome([], [], why, best)

    return RetrievalOutcome([doc for doc, _ in kept], _to_sources(kept))
