from pathlib import Path
from urllib.parse import urlparse

from config import settings
from helpers.log import get_logger
from llm_providers.llamacpp_client import LlamaCppClient
from llm_providers.openrouter_client import OpenRouterClient
from schemas.model import ModelSettings

logger = get_logger(__name__)

LLMClient = LlamaCppClient | OpenRouterClient

LLAMACPP = "llamacpp"
OPENROUTER = "openrouter"
SUPPORTED_PROVIDERS = (LLAMACPP, OPENROUTER)


def init_llm_client(model_folder: Path) -> LLMClient:
    """
    Build the LLM client named by `settings.LLM_PROVIDER`.

    Args:
        model_folder: Where local GGUF weights live. Ignored by hosted providers.

    Returns:
        LLMClient: The provider client.

    Raises:
        ValueError: If the provider is not recognised.
    """
    provider = settings.LLM_PROVIDER.strip().lower()

    if provider == OPENROUTER:
        logger.info(f"Using OpenRouter model: {settings.OPENROUTER_MODEL}")
        return OpenRouterClient(
            api_key=settings.OPENROUTER_API_KEY,
            model_settings=ModelSettings(
                name=settings.OPENROUTER_MODEL,
                reasoning_start_tag=settings.MODEL_REASONING_START_TAG,
                reasoning_stop_tag=settings.MODEL_REASONING_STOP_TAG,
                system_template=settings.MODEL_SYSTEM_TEMPLATE,
                reasoning=settings.MODEL_REASONING,
            ),
            base_url=settings.OPENROUTER_BASE_URL,
            timeout=settings.OPENROUTER_TIMEOUT,
            app_url=settings.OPENROUTER_APP_URL,
            app_title=settings.OPENROUTER_APP_TITLE,
        )

    if provider != LLAMACPP:
        raise ValueError(f"Unsupported LLM_PROVIDER '{provider}'. Supported: {', '.join(SUPPORTED_PROVIDERS)}.")

    logger.info(f"Using local llama.cpp server at {settings.LLAMA_SERVER_BASE_URL}")
    settings.MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

    model_url_path = urlparse(settings.MODEL_URL).path
    model_file_name = Path(model_url_path).name or f"{settings.MODEL}.gguf"

    model_settings = ModelSettings(
        url=settings.MODEL_URL,
        name=settings.MODEL,
        file_name=model_file_name,
        reasoning_start_tag=settings.MODEL_REASONING_START_TAG,
        reasoning_stop_tag=settings.MODEL_REASONING_STOP_TAG,
        system_template=settings.MODEL_SYSTEM_TEMPLATE,
        reasoning=settings.MODEL_REASONING,
    )

    return LlamaCppClient(
        base_url=settings.LLAMA_SERVER_BASE_URL,
        model_folder=model_folder,
        model_settings=model_settings,
        timeout=settings.LLAMA_SERVER_TIMEOUT,
    )
