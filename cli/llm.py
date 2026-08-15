"""LLM provider layer with PydanticAI integration and schema validation."""

import time
from typing import Any, Dict, Optional, Union

from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider as PydanticOpenAIProvider

from .config import get_gemini_api_key, get_openai_api_key, get_openai_base_url
from .constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
    DEFAULT_RETRY_DELAY,
    DEFAULT_TIMEOUT,
)
from .llm_base import LLMProvider
from .scoring import ComplexityResult


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
    ):
        """
        Initialize PydanticAI provider.

        Args:
            provider: Provider name ('openai', 'gemini', etc.)
            api_key: API key for the provider (or None to infer from environment)
            model: Model name string or Model instance
            timeout: Request timeout in seconds
            base_url: Optional base URL for OpenAI-compatible API endpoints
        """
        self._provider_name = provider.lower().strip() if isinstance(provider, str) else "openai"
        self._timeout = timeout
        self._base_url = base_url

        if isinstance(model, Model):
            self._model_instance = model
            self._model_name = getattr(model, "model_name", "test")
            self._api_key = api_key
            return

        if self._provider_name in ("gemini", "google"):
            self._api_key = api_key or get_gemini_api_key()
            if not self._api_key:
                raise LLMError(
                    "Missing API key for Gemini provider. Set GEMINI_API_KEY or GOOGLE_API_KEY."
                )
            model_str = model or "gemini-2.5-flash"
            self._model_name = model_str
            clean_name = model_str
            for prefix in ("gemini:", "google-gla:", "google-vertex:", "google:"):
                if clean_name.startswith(prefix):
                    clean_name = clean_name[len(prefix) :]
                    break
            google_prov = GoogleProvider(api_key=self._api_key)
            self._model_instance = GoogleModel(clean_name, provider=google_prov)
        else:  # openai or other
            self._api_key = api_key or get_openai_api_key()
            self._base_url = base_url or get_openai_base_url()
            if not self._api_key and not self._base_url:
                raise LLMError("Missing API key for OpenAI provider. Set OPENAI_API_KEY.")
            model_str = model or DEFAULT_MODEL
            self._model_name = model_str
            clean_name = model_str
            for prefix in ("openai:", "openai-chat:"):
                if clean_name.startswith(prefix):
                    clean_name = clean_name[len(prefix) :]
                    break
            openai_prov = PydanticOpenAIProvider(api_key=self._api_key, base_url=self._base_url)
            self._model_instance = OpenAIChatModel(clean_name, provider=openai_prov)

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
            max_retries: Maximum retry attempts
            retry_delay: Initial delay between retries (exponential backoff)

        Returns:
            Dict with 'complexity' (int 1..10), 'explanation' (str),
            'provider' (str), 'model' (str), and 'tokens' (int or None)

        Raises:
            LLMError: If analysis fails after retries
        """
        user_prompt = (
            f"diff_excerpt:\n{diff_excerpt}\n\nstats_json:\n{stats_json}\n\ntitle:\n{title}"
        )
        last_error: Optional[Exception] = None

        for attempt in range(max_retries):
            try:
                try:
                    agent = Agent(
                        self._model_instance,
                        system_prompt=prompt,
                        output_type=ComplexityResult,
                        retries=max_retries,
                    )
                except TypeError:
                    agent = Agent(
                        self._model_instance,
                        system_prompt=prompt,
                        result_type=ComplexityResult,
                        retries=max_retries,
                    )

                result = agent.run_sync(user_prompt)

                output = getattr(result, "output", None)
                if output is None:
                    output = getattr(result, "data", None)
                if output is None:
                    raise LLMError(f"Empty response from {self.provider_name}")

                tokens = None
                if hasattr(result, "usage"):
                    usage = result.usage() if callable(result.usage) else result.usage
                    if usage is not None:
                        tokens = getattr(usage, "total_tokens", None)

                return {
                    "complexity": output.complexity,
                    "explanation": output.explanation,
                    "provider": self.provider_name,
                    "model": self.model_name,
                    "tokens": tokens,
                }

            except LLMError:
                raise
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = retry_delay * (2**attempt)
                    delay += (time.time() % 1) * 0.1
                    time.sleep(delay)
                    continue
                raise LLMError(
                    f"{self.provider_name} API error after {max_retries} attempts: {e}"
                ) from e

        raise LLMError(f"Failed after {max_retries} attempts: {last_error}")


class OpenAIProvider(PydanticAIProvider):
    """OpenAI API provider implementation using PydanticAI."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        base_url: Optional[str] = None,
    ):
        """Initialize OpenAI provider."""
        super().__init__(
            provider="openai",
            api_key=api_key,
            model=model,
            timeout=timeout,
            base_url=base_url,
        )


class GeminiProvider(PydanticAIProvider):
    """Gemini API provider implementation using PydanticAI."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        timeout: float = DEFAULT_TIMEOUT,
    ):
        """Initialize Gemini provider."""
        super().__init__(
            provider="gemini",
            api_key=api_key,
            model=model,
            timeout=timeout,
        )


def get_provider(
    provider: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> LLMProvider:
    """
    Factory function to get an LLM provider instance.

    Args:
        provider: Provider name ("openai", "gemini", "auto", etc.)
        api_key: API key for the provider
        model: Model name to use
        base_url: Optional base URL for OpenAI-compatible API endpoints
        timeout: Request timeout in seconds

    Returns:
        LLMProvider instance
    """
    provider_norm = provider.lower().strip() if isinstance(provider, str) else "auto"

    if provider_norm == "openai":
        return OpenAIProvider(
            api_key=api_key,
            model=model or DEFAULT_MODEL,
            timeout=timeout,
            base_url=base_url,
        )
    elif provider_norm in ("gemini", "google"):
        return GeminiProvider(
            api_key=api_key,
            model=model or "gemini-2.5-flash",
            timeout=timeout,
        )
    elif provider_norm == "auto":
        gemini_key = api_key or get_gemini_api_key()
        openai_key = api_key or get_openai_api_key()
        if gemini_key and not openai_key:
            return GeminiProvider(
                api_key=gemini_key,
                model=model or "gemini-2.5-flash",
                timeout=timeout,
            )
        return OpenAIProvider(
            api_key=openai_key or api_key,
            model=model or DEFAULT_MODEL,
            timeout=timeout,
            base_url=base_url,
        )
    else:
        return PydanticAIProvider(
            provider=provider_norm,
            api_key=api_key,
            model=model,
            timeout=timeout,
            base_url=base_url,
        )
