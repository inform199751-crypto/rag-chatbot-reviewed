from pathlib import Path

from config import settings
from memory.factory import create_embedder
from memory.index_fingerprint import build_fingerprint, verify
from memory.vector_database.chroma import Chroma


def init_vector_database(vector_store_path: Path) -> Chroma:
    """
    Loads a Vector Database index based on the specified vector store path.

    Fails fast when the index was built with a different embedding model. Serving in that state
    means every answer is drawn from vectors in a different semantic space than the query --
    wrong, and wrong without any error to notice.

    Args:
        vector_store_path (Path): The path to the vector store.

    Returns:
        Chroma: An instance of the Vector Database.

    Raises:
        ValueError: When the index fingerprint disagrees with the configured embedding model.
    """
    embedding = create_embedder()
    expected = build_fingerprint(
        embedding_model=embedding.model_name,
        embedding_provider=settings.EMBEDDING_PROVIDER,
    )
    index = Chroma(
        is_persistent=True,
        persist_directory=str(vector_store_path),
        embedding=embedding,
        collection_metadata=expected,
    )
    # Read back rather than trusting what we passed: `get_or_create_collection` keeps the
    # metadata of an existing collection and ignores the argument, which is exactly the case
    # this check is for.
    verify(index.collection.metadata, expected, strict=True)

    return index
