"""PydanticAI powered LLM providers and schema validation."""

from typing import Any, Dict, Optional, Union

import httpx
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider as PydanticOpenAIProvider

from .config import get_gemini_api_key, get_openai_api_key, get_openai_base_url
from .constants import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
    DEFAULT_RETRY_DELAY,
    DEFAULT_TIMEOUT,
)
from .llm_base import LLMProvider
from .scoring import ComplexityResult

_GEMINI_PREFIXES = ("gemini:", "google-gla:", "google-vertex:", "google:")
_OPENAI_PREFIXES = ("openai:", "openai-chat:")


def _strip_prefix(name: str, prefixes: tuple[str, ...]) -> str:
    """Strip provider prefix from model name if present."""
    for prefix in prefixes:
        if name.startswith(prefix):
            return name.removeprefix(prefix)
    return name


class LLMError(Exception):
    """LLM provider error."""


class PydanticAIProvider(LLMProvider):
    """PydanticAI LLM provider supporting Gemini and OpenAI."""

    def __init__(
        self,
        provider: str = "openai",
        api_key: Optional[str] = None,
        model: Optional[Union[str, Model]] = None,
        timeout: float = DEFAULT_TIMEOUT,
        base_url: Optional[str] = None,
        retries: int = DEFAULT_MAX_RETRIES,
    ):
        """
        Initialize PydanticAI provider.

        Args:
            provider: Provider name ('openai', 'gemini', etc.)
            api_key: API key for the provider (or None to infer from environment)
            model: Model name string or Model instance
            timeout: Request timeout in seconds
            base_url: Optional base URL for API endpoints
            retries: Number of schema validation retries handled natively by PydanticAI
        """
        self._provider_name = provider.lower().strip()
        self.timeout = timeout

        if self._provider_name not in ("openai", "openai-chat", "gemini", "google"):
            raise LLMError(f"Unsupported provider: '{provider}'")

        if self._provider_name == "google":
            self._provider_name = "gemini"

        if isinstance(model, Model):
            self._model_instance = model
            self._model_name = model.model_name
        elif self._provider_name == "gemini":
            resolved_key = api_key or get_gemini_api_key()
            if not resolved_key:
                raise LLMError(
                    "Missing API key for Gemini provider. Set GEMINI_API_KEY or GOOGLE_API_KEY."
                )
            model_str = model or DEFAULT_GEMINI_MODEL
            self._model_name = _strip_prefix(model_str, _GEMINI_PREFIXES)
            http_client = httpx.AsyncClient(timeout=timeout)
            google_prov = GoogleProvider(
                api_key=resolved_key,
                http_client=http_client,
                base_url=base_url,
            )
            self._model_instance = GoogleModel(self._model_name, provider=google_prov)
        else:  # openai or openai-chat
            resolved_key = api_key or get_openai_api_key()
            resolved_base_url = base_url or get_openai_base_url()
            if not resolved_key and not resolved_base_url:
                raise LLMError("Missing API key for OpenAI provider. Set OPENAI_API_KEY.")
            model_str = model or DEFAULT_MODEL
            self._model_name = _strip_prefix(model_str, _OPENAI_PREFIXES)
            http_client = httpx.AsyncClient(timeout=timeout)
            openai_prov = PydanticOpenAIProvider(
                api_key=resolved_key,
                base_url=resolved_base_url,
                http_client=http_client,
            )
            self._model_instance = OpenAIChatModel(self._model_name, provider=openai_prov)

        self._agent = Agent(self._model_instance, output_type=ComplexityResult, retries=retries)

    @property
    def provider_name(self) -> str:
        """Return the provider name."""
        return self._provider_name

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model_name

    # Backward compatibility
    @property
    def provider(self) -> str:
        """Return the provider name (backward compatible)."""
        return self._provider_name

    @property
    def model(self) -> str:
        """Return the model name (backward compatible)."""
        return self._model_name

    def analyze_complexity(
        self,
        prompt: str,
        diff_excerpt: str,
        stats_json: str,
        title: str,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
    ) -> Dict[str, Any]:
        """
        Analyze PR complexity using PydanticAI and return score with explanation.

        Args:
            prompt: System prompt/instructions
            diff_excerpt: Formatted diff excerpt
            stats_json: JSON string with stats
            title: PR title
            max_retries: Accepted for backward compatibility with the original
                OpenAIProvider signature; retries are configured via the
                ``retries`` constructor argument and handled natively by PydanticAI.
            retry_delay: Accepted for backward compatibility; unused (see max_retries).

        Returns:
            Dict with 'complexity' (int 1..10), 'explanation' (str),
            'provider' (str), 'model' (str), and 'tokens' (int or None)

        Raises:
            LLMError: If analysis fails
        """
        user_prompt = (
            f"diff_excerpt:\n{diff_excerpt}\n\nstats_json:\n{stats_json}\n\ntitle:\n{title}"
        )

        try:
            result = self._agent.run_sync(user_prompt, instructions=prompt)
            output = result.output
            usage = result.usage()
            tokens = usage.total_tokens
            if not tokens and usage.details:
                tokens = sum(v for v in usage.details.values() if isinstance(v, (int, float)))

            return {
                "complexity": output.complexity,
                "explanation": output.explanation,
                "provider": self.provider_name,
                "model": self.model_name,
                "tokens": tokens,
            }

        except UnexpectedModelBehavior as e:
            raise LLMError(
                f"Failed to parse or validate LLM response from {self.provider_name}: {e}"
            ) from e
        except Exception as e:
            raise LLMError(f"{self.provider_name} API error: {e}") from e


class OpenAIProvider(PydanticAIProvider):
    """OpenAI API provider implementation using PydanticAI."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        base_url: Optional[str] = None,
        retries: int = DEFAULT_MAX_RETRIES,
    ):
        """Initialize OpenAI provider."""
        super().__init__(
            provider="openai",
            api_key=api_key,
            model=model,
            timeout=timeout,
            base_url=base_url,
            retries=retries,
        )


class GeminiProvider(PydanticAIProvider):
    """Gemini API provider implementation using PydanticAI."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_GEMINI_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        base_url: Optional[str] = None,
        retries: int = DEFAULT_MAX_RETRIES,
    ):
        """Initialize Gemini provider."""
        super().__init__(
            provider="gemini",
            api_key=api_key,
            model=model,
            timeout=timeout,
            base_url=base_url,
            retries=retries,
        )


_PROVIDERS: Dict[str, tuple[type[PydanticAIProvider], str]] = {
    "openai": (OpenAIProvider, DEFAULT_MODEL),
    "openai-chat": (OpenAIProvider, DEFAULT_MODEL),
    "gemini": (GeminiProvider, DEFAULT_GEMINI_MODEL),
    "google": (GeminiProvider, DEFAULT_GEMINI_MODEL),
}


def get_provider(
    provider: str = "auto",
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_MAX_RETRIES,
) -> LLMProvider:
    """
    Factory function to get an LLM provider instance.

    Args:
        provider: Provider name ("openai", "gemini", "auto", etc.)
        api_key: API key for the provider
        model: Model name to use
        base_url: Optional base URL for API endpoints
        timeout: Request timeout in seconds
        retries: Number of schema validation retries

    Returns:
        LLMProvider instance
    """
    normalized_provider = provider.lower().strip()

    if normalized_provider in _PROVIDERS:
        provider_cls, default_model = _PROVIDERS[normalized_provider]
        return provider_cls(
            api_key=api_key,
            model=model or default_model,
            timeout=timeout,
            base_url=base_url,
            retries=retries,
        )

    if normalized_provider == "auto":
        if api_key:
            if api_key.startswith("AIza"):
                return GeminiProvider(
                    api_key=api_key,
                    model=model or DEFAULT_GEMINI_MODEL,
                    timeout=timeout,
                    base_url=base_url,
                    retries=retries,
                )
            if api_key.startswith("sk-"):
                return OpenAIProvider(
                    api_key=api_key,
                    model=model or DEFAULT_MODEL,
                    timeout=timeout,
                    base_url=base_url,
                    retries=retries,
                )
            raise LLMError(
                "Explicit api_key passed with provider='auto'. "
                "Please specify provider='gemini' or provider='openai'."
            )
        gemini_key = get_gemini_api_key()
        openai_key = get_openai_api_key()
        if gemini_key and not openai_key:
            return GeminiProvider(
                api_key=gemini_key,
                model=model or DEFAULT_GEMINI_MODEL,
                timeout=timeout,
                base_url=base_url,
                retries=retries,
            )
        if openai_key:
            return OpenAIProvider(
                api_key=openai_key,
                model=model or DEFAULT_MODEL,
                timeout=timeout,
                base_url=base_url,
                retries=retries,
            )
        raise LLMError(
            "No API key found in environment. "
            "Set GEMINI_API_KEY / GOOGLE_API_KEY or OPENAI_API_KEY."
        )

    raise LLMError(f"Unsupported provider: '{provider}'")
