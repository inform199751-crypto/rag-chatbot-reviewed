"""
Global application state.
Holds singleton instances that are initialized during app startup.
"""

from llm_providers.llamacpp_client import LlamaCppClient
from memory.reranker import CrossEncoderReranker
from memory.vector_database.chroma import Chroma
from sqlalchemy import Engine

# Global singleton instances
db_engine: Engine | None = None
llm_client: LlamaCppClient | None = None
vector_database: Chroma | None = None
# None both when reranking is disabled and before startup has run. Loading a cross-encoder takes
# seconds and hundreds of MB, so it has to be built once here rather than per request -- doing it
# in the request path would add that cost to every message and look like a slow model.
reranker: CrossEncoderReranker | None = None
