from pathlib import Path

from memory.factory import create_embedder
from memory.vector_database.chroma import Chroma


def init_vector_database(vector_store_path: Path) -> Chroma:
    """
    Loads a Vector Database index based on the specified vector store path.

    Args:
        vector_store_path (Path): The path to the vector store.

    Returns:
        Chroma: An instance of the Vector Database.
    """
    embedding = create_embedder()
    index = Chroma(is_persistent=True, persist_directory=str(vector_store_path), embedding=embedding)

    return index
