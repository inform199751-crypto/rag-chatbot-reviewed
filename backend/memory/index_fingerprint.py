"""What an index was built with, recorded alongside it.

The problem this exists for
    Vectors carry no record of how they were produced. Change the embedding model and the stored
    vectors keep answering queries -- they are just answering them from a different semantic
    space than the query was embedded into, so the results are noise. Nothing raises.

    A dimension change is the lucky case: Chroma rejects the insert outright with
    "Collection expecting embedding with dimension of 384, got 1024". Two models of the *same*
    width -- all-MiniLM-L6-v2 and paraphrase-multilingual-MiniLM-L12-v2 are both 384 -- swap
    silently, and retrieval quality collapses with no error anywhere.

    Chunk size has a quieter version of the same problem: the CLI indexer takes it as an
    argument while the upload endpoint reads it from settings, so a corpus can end up half
    indexed at 1000 and half at 400 without anything to compare against.

The fix is to write down what was used and check it on load. Cheap, and it converts a silent
wrong-answer failure into a startup message that names the mismatch.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

EMBEDDING_MODEL_KEY = "embedding_model"
EMBEDDING_PROVIDER_KEY = "embedding_provider"
CHUNK_SIZE_KEY = "chunk_size"
CHUNK_OVERLAP_KEY = "chunk_overlap"


def build_fingerprint(
    embedding_model: str,
    embedding_provider: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> dict[str, str | int]:
    """
    Describe how an index was built, in a shape Chroma will accept as collection metadata.

    Chroma only stores scalars in metadata, so this stays flat deliberately -- no nesting, no
    lists.
    """
    fingerprint: dict[str, str | int] = {
        EMBEDDING_MODEL_KEY: embedding_model,
        EMBEDDING_PROVIDER_KEY: embedding_provider,
    }
    if chunk_size is not None:
        fingerprint[CHUNK_SIZE_KEY] = chunk_size
    if chunk_overlap is not None:
        fingerprint[CHUNK_OVERLAP_KEY] = chunk_overlap
    return fingerprint


def compare(stored: dict | None, expected: dict[str, str | int]) -> list[str]:
    """
    Return one message per mismatch, empty when everything agrees.

    An index with no fingerprint at all -- anything built before this existed -- is not a
    mismatch. Refusing to serve those would break working deployments to enforce bookkeeping,
    and the check has no evidence of a problem, only an absence of evidence.
    """
    if not stored:
        return []

    problems = []
    for key, want in expected.items():
        if key not in stored:
            continue
        have = stored[key]
        if str(have) != str(want):
            problems.append(f"{key}: index was built with {have!r}, config says {want!r}")
    return problems


def verify(stored: dict | None, expected: dict[str, str | int], strict: bool = False) -> None:
    """
    Compare and complain.

    Args:
        stored: Metadata read back off the collection.
        expected: What the current configuration would produce.
        strict: Raise instead of warning. The embedding model belongs in strict mode -- results
            are meaningless, not merely degraded -- while a chunk size difference is worth
            knowing about but still returns useful answers.

    Raises:
        ValueError: When `strict` and the fingerprints disagree.
    """
    problems = compare(stored, expected)
    if not problems:
        if not stored:
            logger.info("Index has no fingerprint; skipping the compatibility check.")
        return

    detail = "; ".join(problems)
    if strict:
        raise ValueError(
            f"The vector index does not match the current configuration -- {detail}. "
            f"Queries would be embedded into a different space than the stored vectors, so "
            f"retrieval would return noise without failing. Rebuild the index "
            f"(delete vector_store/docs_index and re-run scripts/memory_builder.py), or set "
            f"the configuration back to what built it."
        )
    logger.warning("Vector index fingerprint mismatch -- %s", detail)
