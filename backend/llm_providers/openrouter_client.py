"""
Client for OpenRouter, exposing the same surface as `LlamaCppClient`.

OpenRouter speaks the OpenAI chat-completions API, so the request shapes are identical to the
llama.cpp server's. What differs is that the model is remote: there is no GGUF file to download,
nothing to load into or unload from memory, and no local server to poll. Those methods are absent
here rather than stubbed, so calling one is an obvious error instead of a silent no-op.
"""

from typing import AsyncIterator, Iterator

from helpers.log import get_logger
from openai import AsyncOpenAI, OpenAI
from openai.types.chat import ChatCompletionChunk
from schemas.model import ModelSettings

from llm_providers.prompt_builder import PromptBuilder

logger = get_logger(__name__)


class OpenRouterClient(PromptBuilder):
    """
    Client for communicating with OpenRouter via the OpenAI-compatible REST API.
    """

    def __init__(
        self,
        api_key: str,
        model_settings: ModelSettings,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: int = 300,
        app_url: str = "",
        app_title: str = "",
    ):
        """
        Initialize the OpenRouter client.

        Args:
            api_key: OpenRouter API key. Required -- OpenRouter rejects unauthenticated requests.
            model_settings: Model configuration settings. `name` must be an OpenRouter model slug,
                for example "nvidia/nemotron-3.5-lightning".
            base_url: OpenRouter API base URL
            timeout: Request timeout in seconds (default: 300)
            app_url: Optional value for the `HTTP-Referer` header, used by OpenRouter to attribute
                traffic to an application on its dashboard
            app_title: Optional value for the `X-Title` header, same purpose as `app_url`

        Raises:
            ValueError: If no API key was supplied
        """
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is empty. Set it in .env before selecting LLM_PROVIDER=openrouter.")

        self.base_url = base_url if base_url.rstrip("/").endswith("/v1") else f"{base_url.rstrip('/')}/v1"
        self.timeout = timeout

        self.model_settings = model_settings
        self.model_name = self.model_settings.name

        # OpenRouter uses these two headers purely for attribution; neither is required.
        default_headers = {}
        if app_url:
            default_headers["HTTP-Referer"] = app_url
        if app_title:
            default_headers["X-Title"] = app_title

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=self.timeout,
            default_headers=default_headers or None,
        )

        self.async_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=self.timeout,
            default_headers=default_headers or None,
        )

        self._validate_connection()

    def _validate_connection(self) -> None:
        """
        Confirm the key is accepted and the configured model is actually served.

        Failing here rather than on the first user question turns an authentication or typo problem
        into a startup error, where it is far cheaper to diagnose.

        Raises:
            RuntimeError: If OpenRouter cannot be reached or rejects the credentials
        """
        try:
            available = {model.id for model in self.client.models.list().data}
        except Exception as e:
            logger.error(f"Failed to reach OpenRouter at {self.base_url}: {e}")
            raise RuntimeError(f"Failed to reach OpenRouter at {self.base_url}.") from e

        logger.info(f"Connected to OpenRouter, {len(available)} models available")

        if available and self.model_name not in available:
            # Not fatal: OpenRouter can serve variants that the catalogue endpoint omits, and a wrong
            # slug fails loudly on the first request anyway. A warning is the proportionate response.
            logger.warning(
                f"Model '{self.model_name}' was not found in the OpenRouter catalogue. "
                f"Check the slug at https://openrouter.ai/models if requests start failing."
            )

    def close(self):
        """
        Closes the underlying HTTP clients.
        """
        self.client.close()

    def generate_answer(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        Generates an answer based on the given prompt using the language model.

        Args:
            prompt: The input prompt for generating the answer
            max_new_tokens: The maximum number of new tokens to generate (default is 512)

        Returns:
            str: The generated answer
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.model_settings.system_template},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_new_tokens,
                stream=False,
            )

            return response.choices[0].message.content or ""

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            raise

    async def async_generate_answer(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        Generates an answer based on the given prompt using the language model asynchronously.

        Args:
            prompt: The input prompt for generating the answer
            max_new_tokens: The maximum number of new tokens to generate (default is 512)

        Returns:
            str: The generated answer
        """
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.model_settings.system_template},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_new_tokens,
                stream=False,
            )

            return response.choices[0].message.content or ""

        except Exception as e:
            logger.error(f"Error generating answer asynchronously: {e}")
            raise

    def stream_answer(self, prompt: str, max_new_tokens: int = 512) -> str:
        """
        Generates an answer by streaming tokens.

        Args:
            prompt: The input prompt for generating the answer
            max_new_tokens: The maximum number of new tokens to generate (default is 512)

        Returns:
            str: The generated answer
        """
        answer = ""
        stream = self.start_answer_iterator_streamer(prompt, max_new_tokens=max_new_tokens)

        for output in stream:
            token = self.parse_token(output)
            if token:
                answer += token
                print(token, end="", flush=True)

        return answer

    def start_answer_iterator_streamer(self, prompt: str, max_new_tokens: int = 512) -> Iterator[ChatCompletionChunk]:
        """
        Start an answer iterator streamer for a given prompt.

        Args:
            prompt: The input prompt for generating the answer
            max_new_tokens: The maximum number of new tokens to generate (default is 512)

        Returns:
            Iterator yielding streaming response chunks
        """
        try:
            return self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.model_settings.system_template},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_new_tokens,
                stream=True,
            )

        except Exception as e:
            logger.error(f"Error starting stream: {e}")
            raise

    async def async_start_answer_iterator_streamer(
        self, prompt: str, max_new_tokens: int = 512
    ) -> AsyncIterator[ChatCompletionChunk]:
        """
        Asynchronously start an answer iterator streamer for streaming response generation.

        Args:
            prompt: The input prompt for generating the answer
            max_new_tokens: The maximum number of new tokens to generate (default is 512)

        Returns:
            AsyncIterator yielding streaming response chunks
        """
        try:
            return await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.model_settings.system_template},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_new_tokens,
                stream=True,
            )

        except Exception as e:
            logger.error(f"Error starting async stream: {e}")
            raise

    @staticmethod
    def parse_token(token: ChatCompletionChunk) -> str:
        """
        Parse a streaming token to extract content.

        Args:
            token: The streaming response chunk

        Returns:
            str: The content from the token, or empty string if no content
        """
        if token.choices and len(token.choices) > 0:
            delta = token.choices[0].delta
            if delta and delta.content:
                return delta.content
        return ""
