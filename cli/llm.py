"""PydanticAI powered LLM providers and schema validation."""

from typing import Any, Dict, Optional, Union

import httpx
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider as PydanticOpenAIProvider

from .config import get_gemini_api_key, get_openai_api_key, get_openai_base_url
from .constants import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_MODEL,
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
        retries: int = DEFAULT_MAX_RETRIES,
    ):
        """
        Initialize PydanticAI provider.

        Args:
            provider: Provider name ('openai', 'gemini', etc.)
            api_key: API key for the provider (or None to infer from environment)
            model: Model name string or Model instance
            timeout: Request timeout in seconds
            base_url: Optional base URL for OpenAI-compatible API endpoints
            retries: Number of schema validation retries handled natively by PydanticAI
        """
        self._provider_name = provider.lower().strip() if isinstance(provider, str) else "openai"
        self._timeout = timeout
        self._base_url = base_url

        if isinstance(model, Model):
            self._model_instance = model
            self._model_name = getattr(model, "model_name", "test-model")
            self._agent = Agent(self._model_instance, output_type=ComplexityResult, retries=retries)
            return

        if self._provider_name in ("gemini", "google"):
            self._provider_name = "gemini"
            resolved_key = api_key or get_gemini_api_key()
            if not resolved_key:
                raise LLMError(
                    "Missing API key for Gemini provider. Set GEMINI_API_KEY or GOOGLE_API_KEY."
                )
            model_str = model or "gemini-2.5-flash"
            clean_name = model_str
            for prefix in ("gemini:", "google-gla:", "google-vertex:", "google:"):
                if clean_name.startswith(prefix):
                    clean_name = clean_name[len(prefix) :]
                    break
            self._model_name = clean_name
            http_client = httpx.AsyncClient(timeout=timeout)
            google_prov = GoogleProvider(
                api_key=resolved_key,
                http_client=http_client,
                base_url=base_url,
            )
            self._model_instance = GoogleModel(clean_name, provider=google_prov)

        elif self._provider_name in ("openai", "openai-chat"):
            self._provider_name = "openai"
            resolved_key = api_key or get_openai_api_key()
            resolved_base_url = base_url or get_openai_base_url()
            if not resolved_key and not resolved_base_url:
                raise LLMError("Missing API key for OpenAI provider. Set OPENAI_API_KEY.")
            model_str = model or DEFAULT_MODEL
            clean_name = model_str
            for prefix in ("openai:", "openai-chat:"):
                if clean_name.startswith(prefix):
                    clean_name = clean_name[len(prefix) :]
                    break
            self._model_name = clean_name
            client = AsyncOpenAI(
                api_key=resolved_key or "api-key-not-set",
                base_url=resolved_base_url,
                timeout=timeout,
            )
            openai_prov = PydanticOpenAIProvider(openai_client=client)
            self._model_instance = OpenAIChatModel(clean_name, provider=openai_prov)

        else:
            raise LLMError(f"Unsupported provider: '{provider}'")

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
    def model(self) -> str:
        """Return the model name (backward compatible)."""
        return self._model_name

    def analyze_complexity(
        self,
        prompt: str,
        diff_excerpt: str,
        stats_json: str,
        title: str,
        max_retries: Optional[int] = None,
        retry_delay: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Analyze PR complexity using PydanticAI and return score with explanation.

        Args:
            prompt: System prompt/instructions
            diff_excerpt: Formatted diff excerpt
            stats_json: JSON string with stats
            title: PR title
            max_retries: Optional retry attempts override (maintained for compatibility)
            retry_delay: Optional retry delay (maintained for compatibility)

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
            tokens = None
            if hasattr(result, "usage") and result.usage is not None:
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

        except UnexpectedModelBehavior as e:
            raise LLMError(
                f"Failed to parse or validate LLM response from {self.provider_name}: {e}"
            ) from e
        except LLMError:
            raise
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
        model: str = "gemini-2.5-flash",
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


def get_provider(
    provider: str,
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
    normalized_provider = provider.lower().strip() if isinstance(provider, str) else "auto"

    if normalized_provider in ("openai", "openai-chat"):
        return OpenAIProvider(
            api_key=api_key,
            model=model or DEFAULT_MODEL,
            timeout=timeout,
            base_url=base_url,
            retries=retries,
        )
    elif normalized_provider in ("gemini", "google"):
        return GeminiProvider(
            api_key=api_key,
            model=model or "gemini-2.5-flash",
            timeout=timeout,
            base_url=base_url,
            retries=retries,
        )
    elif normalized_provider == "auto":
        if api_key:
            if api_key.startswith("AIza"):
                return GeminiProvider(
                    api_key=api_key,
                    model=model or "gemini-2.5-flash",
                    timeout=timeout,
                    base_url=base_url,
                    retries=retries,
                )
            elif api_key.startswith("sk-"):
                return OpenAIProvider(
                    api_key=api_key,
                    model=model or DEFAULT_MODEL,
                    timeout=timeout,
                    base_url=base_url,
                    retries=retries,
                )
            else:
                raise LLMError(
                    "Explicit api_key passed with provider='auto'. Please specify provider='gemini' or provider='openai'."
                )
        gemini_key = get_gemini_api_key()
        openai_key = get_openai_api_key()
        if gemini_key and not openai_key:
            return GeminiProvider(
                api_key=gemini_key,
                model=model or "gemini-2.5-flash",
                timeout=timeout,
                base_url=base_url,
                retries=retries,
            )
        elif openai_key:
            return OpenAIProvider(
                api_key=openai_key,
                model=model or DEFAULT_MODEL,
                timeout=timeout,
                base_url=base_url,
                retries=retries,
            )
        else:
            raise LLMError(
                "No API key found in environment. Set GEMINI_API_KEY / GOOGLE_API_KEY or OPENAI_API_KEY."
            )
    else:
        raise LLMError(f"Unsupported provider: '{provider}'")
