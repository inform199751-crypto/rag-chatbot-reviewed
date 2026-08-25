from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_PATH = Path(__file__).parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT_PATH / ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    PROJECT_NAME: str = "Chatbot API"
    VERSION: str = "0.1.0"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Logging Configuration
    LOG_LEVEL: str = "INFO"

    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ]

    MODEL_FOLDER: Path = ROOT_PATH / "models"
    VECTOR_STORE_PATH: Path = ROOT_PATH / "vector_store" / "docs_index"
    DOCS_PATH: Path = ROOT_PATH / Path("docs")

    DATABASE_URL: str = f"sqlite:///{ROOT_PATH / 'vector_store' / 'registry.db'}"

    # LLM Provider Selection
    # "llamacpp" talks to a local llama.cpp server; "openrouter" calls a hosted model.
    LLM_PROVIDER: str = "llamacpp"

    # OpenRouter Configuration (used when LLM_PROVIDER="openrouter")
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_MODEL: str = "nvidia/nemotron-3.5-lightning"
    OPENROUTER_TIMEOUT: int = 300
    # Optional attribution headers shown on the OpenRouter dashboard.
    OPENROUTER_APP_URL: str = ""
    OPENROUTER_APP_TITLE: str = ""

    # llama.cpp Server Configuration
    LLAMA_SERVER_BASE_URL: str = "http://localhost:8080"
    LLAMA_SERVER_TIMEOUT: int = 300

    # LLM Model Configuration
    MODEL: str = "Llama-3.2-1B-Instruct-Q5_K_M"
    MODEL_URL: str = (
        "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q5_K_M.gguf"
    )
    MODEL_SYSTEM_TEMPLATE: str = ""
    MODEL_REASONING: bool = False
    MODEL_REASONING_START_TAG: str = "<think>"
    MODEL_REASONING_STOP_TAG: str = "</think>"
    MAX_NEW_TOKENS: int = 512

    # Retrieval Configuration
    # "sentence-transformers" embeds locally; "openai" calls OpenAI's embeddings API.
    # Note: OpenRouter proxies chat completions only, so embeddings need an OpenAI key of
    # their own even when the LLM is served by OpenRouter.
    EMBEDDING_PROVIDER: str = "sentence-transformers"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    OPENAI_API_KEY: str = ""
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    SYNTHESIS_STRATEGY: str = "tree-summarization"
    NUM_RETRIEVALS: int = 2
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 50

    # Chat History Configuration
    CHAT_HISTORY_LENGTH: int = 2

    # WebSocket Configuration
    WEBSOCKET_MAX_SIZE: int = 10 * 1024 * 1024  # 10 MB

    # File Upload Configuration
    ALLOWED_UPLOAD_EXTENSIONS: list[str] = [".md"]


settings = Settings()
