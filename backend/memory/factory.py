"""
Builds the embedder named by configuration.

Both the ingestion script and the API must end up with the same embedder, or the vectors they write
and the vectors they search with will not be comparable. Routing both through one factory is what
keeps them in step.
"""

from config import settings
from helpers.log import get_logger

from memory.base_embedder import BaseEmbedder
from memory.embedder import Embedder
from memory.openai_embedder import OpenAIEmbedder

logger = get_logger(__name__)

SENTENCE_TRANSFORMERS = "sentence-transformers"
OPENAI = "openai"

SUPPORTED_PROVIDERS = (SENTENCE_TRANSFORMERS, OPENAI)


def create_embedder(provider: str | None = None, model_name: str | None = None) -> BaseEmbedder:
    """
    Create the configured embedder.

    Args:
        provider: Which backend to use. Defaults to `settings.EMBEDDING_PROVIDER`.
        model_name: Override for the model. Defaults to the setting matching the chosen provider,
            so that leaving it unset cannot pair one provider with another provider's model name.

    Returns:
        BaseEmbedder: The embedder instance.

    Raises:
        ValueError: If the provider is not recognised.
    """
    provider = (provider or settings.EMBEDDING_PROVIDER).strip().lower()

    if provider == OPENAI:
        model = model_name or settings.OPENAI_EMBEDDING_MODEL
        logger.info(f"Using OpenAI embeddings: {model}")
        return OpenAIEmbedder(api_key=settings.OPENAI_API_KEY, model_name=model)

    if provider in (SENTENCE_TRANSFORMERS, "sentence_transformers", "local"):
        model = model_name or settings.EMBEDDING_MODEL
        logger.info(f"Using local sentence-transformers embeddings: {model}")
        return Embedder(model_name=model)

    raise ValueError(f"Unsupported EMBEDDING_PROVIDER '{provider}'. Supported: {', '.join(SUPPORTED_PROVIDERS)}.")
