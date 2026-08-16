"""PydanticAI powered LLM providers and schema validation."""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import httpx
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel
from pydantic_ai.providers.anthropic import AnthropicProvider as PydanticAnthropicProvider
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider as PydanticOpenAIProvider
from pydantic_ai.usage import RequestUsage

from .config import (
    get_anthropic_api_key,
    get_anthropic_base_url,
    get_gemini_api_key,
    get_openai_api_key,
    get_openai_base_url,
)
from .constants import (
    DEFAULT_ANTHROPIC_MODEL,
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
_ANTHROPIC_PREFIXES = ("anthropic:", "claude:")

# PydanticAI's `RequestUsage.extract` delegates to genai-prices, which for OpenAI
# chat completions can return an `output_reasoning_tokens` field that isn't a
# valid `RequestUsage` constructor argument (RequestUsage has no such field).
# That raises a TypeError that `extract` swallows via a blanket `except Exception`,
# silently collapsing token usage to 0. Wrap `extract` to recover the real
# prompt/completion token counts straight from the raw response dict when this
# happens, so OpenAI token usage isn't reported as 0.
_orig_request_usage_extract = RequestUsage.extract


@classmethod
def _safe_request_usage_extract(cls, data: Any, *args, **kwargs) -> RequestUsage:
    res = _orig_request_usage_extract(data, *args, **kwargs)
    if res.total_tokens == 0 and isinstance(data, dict):
        usage_dict = data.get("usage")
        if isinstance(usage_dict, dict):
            prompt_tokens = usage_dict.get("prompt_tokens") or usage_dict.get("input_tokens")
            completion_tokens = usage_dict.get("completion_tokens") or usage_dict.get(
                "output_tokens"
            )
            if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                details = kwargs.get("details") or {}
                return cls(
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    details=details,
                )
    return res


RequestUsage.extract = _safe_request_usage_extract


def _strip_prefix(name: str, prefixes: tuple[str, ...]) -> str:
    """Strip provider prefix from model name if present."""
    for prefix in prefixes:
        if name.startswith(prefix):
            return name.removeprefix(prefix)
    return name


def _uses_openai_responses_api(model_name: str) -> bool:
    """Return whether an OpenAI model requires the Responses API."""
    return model_name.endswith("-luna")


class LLMError(Exception):
    """LLM provider error."""


class PydanticAIProvider(LLMProvider):
    """PydanticAI LLM provider supporting Gemini, OpenAI, and Anthropic."""

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
            provider: Provider name ('openai', 'gemini', 'anthropic', etc.)
            api_key: API key for the provider (or None to infer from environment)
            model: Model name string or Model instance
            timeout: Request timeout in seconds
            base_url: Optional base URL for API endpoints
            retries: Number of schema validation retries handled natively by PydanticAI
        """
        self._provider_name = provider.lower().strip()
        self.timeout = timeout

        if self._provider_name not in (
            "openai",
            "openai-chat",
            "gemini",
            "google",
            "anthropic",
            "claude",
        ):
            raise LLMError(f"Unsupported provider: '{provider}'")

        if self._provider_name == "google":
            self._provider_name = "gemini"
        elif self._provider_name == "claude":
            self._provider_name = "anthropic"

        if isinstance(model, Model):
            self._model_instance = model
            self._model_name = model.model_name
        elif self._provider_name == "anthropic":
            resolved_key = api_key or get_anthropic_api_key()
            resolved_base_url = base_url or get_anthropic_base_url()
            if not resolved_key and not resolved_base_url:
                raise LLMError("Missing API key for Anthropic provider. Set ANTHROPIC_API_KEY.")
            model_str = model or DEFAULT_ANTHROPIC_MODEL
            self._model_name = _strip_prefix(model_str, _ANTHROPIC_PREFIXES)
            http_client = httpx.AsyncClient(timeout=timeout)
            anthropic_prov = PydanticAnthropicProvider(
                api_key=resolved_key,
                base_url=resolved_base_url,
                http_client=http_client,
            )
            self._model_instance = AnthropicModel(self._model_name, provider=anthropic_prov)
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
            model_cls = (
                OpenAIResponsesModel
                if _uses_openai_responses_api(self._model_name)
                else OpenAIChatModel
            )
            self._model_instance = model_cls(self._model_name, provider=openai_prov)

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
            tokens = getattr(usage, "total_tokens", None)
            if not isinstance(tokens, (int, float)) or tokens <= 0:
                in_tok = getattr(usage, "input_tokens", None)
                out_tok = getattr(usage, "output_tokens", None)
                if (
                    isinstance(in_tok, (int, float))
                    and isinstance(out_tok, (int, float))
                    and (in_tok + out_tok > 0)
                ):
                    tokens = in_tok + out_tok
                elif isinstance(in_tok, (int, float)) and in_tok > 0:
                    tokens = in_tok
                elif isinstance(out_tok, (int, float)) and out_tok > 0:
                    tokens = out_tok
                else:
                    tokens = None

            if not tokens and getattr(usage, "details", None) and isinstance(usage.details, dict):
                tokens = sum(v for v in usage.details.values() if isinstance(v, (int, float)))

            if not tokens:
                msg_tokens = 0
                for msg in result.all_messages() if hasattr(result, "all_messages") else []:
                    if hasattr(msg, "usage") and msg.usage:
                        u = msg.usage
                        t = getattr(u, "total_tokens", None)
                        if not isinstance(t, (int, float)) or t <= 0:
                            i_t = getattr(u, "input_tokens", 0)
                            o_t = getattr(u, "output_tokens", 0)
                            if isinstance(i_t, (int, float)) and isinstance(o_t, (int, float)):
                                t = i_t + o_t
                            else:
                                t = 0
                        if isinstance(t, (int, float)) and t > 0:
                            msg_tokens += int(t)
                if msg_tokens:
                    tokens = msg_tokens

            if not isinstance(tokens, (int, float)) or tokens <= 0:
                tokens = None

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


class AnthropicProvider(PydanticAIProvider):
    """Anthropic Claude API provider implementation using PydanticAI."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_ANTHROPIC_MODEL,
        timeout: float = DEFAULT_TIMEOUT,
        base_url: Optional[str] = None,
        retries: int = DEFAULT_MAX_RETRIES,
    ):
        """Initialize Anthropic provider."""
        super().__init__(
            provider="anthropic",
            api_key=api_key,
            model=model,
            timeout=timeout,
            base_url=base_url,
            retries=retries,
        )


@dataclass(frozen=True)
class ProviderSpec:
    """Specification for an LLM provider."""

    name: str
    provider_cls: type[PydanticAIProvider]
    default_model: str
    env_var_keys: tuple[str, ...]
    aliases: tuple[str, ...] = ()


PROVIDER_SPECS: Dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        name="openai",
        provider_cls=OpenAIProvider,
        default_model=DEFAULT_MODEL,
        env_var_keys=("OPENAI_API_KEY",),
        aliases=("openai-chat",),
    ),
    "gemini": ProviderSpec(
        name="gemini",
        provider_cls=GeminiProvider,
        default_model=DEFAULT_GEMINI_MODEL,
        env_var_keys=("GEMINI_API_KEY", "GOOGLE_API_KEY"),
        aliases=("google",),
    ),
    "anthropic": ProviderSpec(
        name="anthropic",
        provider_cls=AnthropicProvider,
        default_model=DEFAULT_ANTHROPIC_MODEL,
        env_var_keys=("ANTHROPIC_API_KEY",),
        aliases=("claude",),
    ),
}

DEFAULT_PROVIDER_PRECEDENCE: tuple[str, ...] = ("openai", "gemini", "anthropic")


def get_provider_spec(name_or_alias: str) -> Optional[ProviderSpec]:
    """Find ProviderSpec by canonical name or alias (case-insensitive)."""
    norm = name_or_alias.lower().strip()
    if norm in PROVIDER_SPECS:
        return PROVIDER_SPECS[norm]
    for spec in PROVIDER_SPECS.values():
        if norm in spec.aliases:
            return spec
    return None


def resolve_provider_credentials(
    provider: str = "auto",
    openai_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    env_openai_api_key: Optional[str] = None,
    env_gemini_api_key: Optional[str] = None,
    env_anthropic_api_key: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    """
    Resolve provider credentials and perform fast pre-flight auth validation.

    Returns:
        (effective_provider, effective_api_key)

    Raises:
        ValueError: If required API key is missing.
        LLMError: If provider is unsupported.
    """
    prov_norm = (provider or "auto").lower().strip()
    openai_env = env_openai_api_key or get_openai_api_key()
    gemini_env = env_gemini_api_key or get_gemini_api_key()
    anthropic_env = env_anthropic_api_key or get_anthropic_api_key()

    if prov_norm == "auto":
        # Check explicit CLI argument overrides first
        if openai_api_key:
            effective_prov = "openai"
            effective_key = openai_api_key
        elif gemini_api_key:
            effective_prov = "gemini"
            effective_key = gemini_api_key
        elif anthropic_api_key:
            effective_prov = "anthropic"
            effective_key = anthropic_api_key
        else:
            # Auto-detect using DEFAULT_PROVIDER_PRECEDENCE ("openai", "gemini", "anthropic")
            effective_prov = None
            effective_key = None
            env_keys = {
                "openai": openai_env,
                "gemini": gemini_env,
                "anthropic": anthropic_env,
            }
            for p_name in DEFAULT_PROVIDER_PRECEDENCE:
                if env_keys.get(p_name):
                    effective_prov = p_name
                    effective_key = env_keys[p_name]
                    break
            if not effective_prov:
                raise ValueError(
                    "Either GEMINI_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY is required"
                )
    else:
        spec = get_provider_spec(prov_norm)
        if not spec:
            raise LLMError(f"Unsupported provider: '{provider}'")
        effective_prov = spec.name
        if effective_prov == "openai":
            effective_key = openai_api_key or openai_env
            if not effective_key:
                raise ValueError("OPENAI_API_KEY environment variable or argument is required")
        elif effective_prov == "gemini":
            effective_key = gemini_api_key or gemini_env
            if not effective_key:
                raise ValueError(
                    "GEMINI_API_KEY or GOOGLE_API_KEY environment variable or argument is required"
                )
        elif effective_prov == "anthropic":
            effective_key = anthropic_api_key or anthropic_env
            if not effective_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable or argument is required")
        else:
            effective_key = (
                openai_api_key
                or gemini_api_key
                or anthropic_api_key
                or openai_env
                or gemini_env
                or anthropic_env
            )

    return effective_prov, effective_key


def resolve_provider(
    provider: str = "auto",
    openai_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    model: Optional[str] = None,
    env_openai_api_key: Optional[str] = None,
    env_gemini_api_key: Optional[str] = None,
    env_anthropic_api_key: Optional[str] = None,
) -> tuple[str, Optional[str], str]:
    """
    Resolve effective provider, API key, and model.

    Returns:
        (effective_provider, effective_api_key, effective_model)
    """
    effective_prov, effective_key = resolve_provider_credentials(
        provider=provider,
        openai_api_key=openai_api_key,
        gemini_api_key=gemini_api_key,
        anthropic_api_key=anthropic_api_key,
        env_openai_api_key=env_openai_api_key,
        env_gemini_api_key=env_gemini_api_key,
        env_anthropic_api_key=env_anthropic_api_key,
    )

    spec = get_provider_spec(effective_prov)
    default_model = spec.default_model if spec else DEFAULT_MODEL

    if not model or model == "" or (model == DEFAULT_MODEL and effective_prov != "openai"):
        effective_model = default_model
    else:
        effective_model = model

    return effective_prov, effective_key, effective_model


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
        provider: Provider name ("openai", "gemini", "anthropic", "auto", etc.)
        api_key: API key for the provider
        model: Model name to use
        base_url: Optional base URL for API endpoints
        timeout: Request timeout in seconds
        retries: Number of schema validation retries

    Returns:
        LLMProvider instance
    """
    normalized_provider = provider.lower().strip()

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
            if api_key.startswith("sk-ant-"):
                return AnthropicProvider(
                    api_key=api_key,
                    model=model or DEFAULT_ANTHROPIC_MODEL,
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
                "Please specify provider='gemini', provider='openai', or provider='anthropic'."
            )
        try:
            eff_prov, eff_key, eff_model = resolve_provider(provider="auto", model=model)
        except ValueError as e:
            raise LLMError(
                "No API key found in environment. "
                "Set GEMINI_API_KEY / GOOGLE_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY."
            ) from e

        spec = get_provider_spec(eff_prov)
        if not spec:
            raise LLMError(f"Unsupported provider: '{eff_prov}'")
        return spec.provider_cls(
            api_key=eff_key,
            model=eff_model,
            timeout=timeout,
            base_url=base_url,
            retries=retries,
        )

    spec = get_provider_spec(normalized_provider)
    if spec:
        eff_model = model or spec.default_model
        return spec.provider_cls(
            api_key=api_key,
            model=eff_model,
            timeout=timeout,
            base_url=base_url,
            retries=retries,
        )

    raise LLMError(f"Unsupported provider: '{provider}'")
