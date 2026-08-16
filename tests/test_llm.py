"""Tests for LLM module and PydanticAI provider integration."""

from unittest.mock import MagicMock, patch
import pytest
from pydantic import ValidationError
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.test import TestModel

from cli.config import get_anthropic_api_key, get_anthropic_base_url
from cli.constants import DEFAULT_ANTHROPIC_MODEL, DEFAULT_GEMINI_MODEL, DEFAULT_MODEL
from cli.llm import (
    AnthropicProvider,
    GeminiProvider,
    LLMError,
    OpenAIProvider,
    PydanticAIProvider,
    get_provider,
)

from cli.llm_base import LLMProvider
from cli.scoring import ComplexityResult


class TestComplexityResultSchema:
    """Tests for ComplexityResult Pydantic schema."""

    def test_valid_complexity_result(self):
        """Test valid ComplexityResult construction."""
        res = ComplexityResult(complexity=5, explanation="Valid explanation")
        assert res.complexity == 5
        assert res.explanation == "Valid explanation"

    def test_boundary_values(self):
        """Test complexity boundary values (1 and 10)."""
        res_min = ComplexityResult(complexity=1, explanation="Min")
        assert res_min.complexity == 1

        res_max = ComplexityResult(complexity=10, explanation="Max")
        assert res_max.complexity == 10

    def test_complexity_below_bounds_raises(self):
        """Test complexity < 1 raises ValidationError."""
        with pytest.raises(ValidationError):
            ComplexityResult(complexity=0, explanation="Too low")

    def test_complexity_above_bounds_raises(self):
        """Test complexity > 10 raises ValidationError."""
        with pytest.raises(ValidationError):
            ComplexityResult(complexity=11, explanation="Too high")

    def test_missing_fields_raises(self):
        """Test missing fields raise ValidationError."""
        with pytest.raises(ValidationError):
            ComplexityResult(complexity=5)  # missing explanation
        with pytest.raises(ValidationError):
            ComplexityResult(explanation="Missing score")  # missing complexity


class TestOpenAIProviderBase:
    """Tests for OpenAIProvider base functionality."""

    def test_inherits_from_llm_provider(self):
        """Test that OpenAIProvider inherits from LLMProvider and PydanticAIProvider."""
        assert issubclass(OpenAIProvider, LLMProvider)
        assert issubclass(OpenAIProvider, PydanticAIProvider)

    def test_provider_name(self):
        """Test provider_name property."""
        provider = OpenAIProvider("test-key")
        assert provider.provider_name == "openai"

    def test_model_name(self):
        """Test model_name property."""
        provider = OpenAIProvider("test-key", model="gpt-4")
        assert provider.model_name == "gpt-4"

    def test_model_name_strips_prefix(self):
        """Test model_name strips 'openai:' prefix."""
        provider = OpenAIProvider("test-key", model="openai:gpt-4o")
        assert provider.model_name == "gpt-4o"
        assert provider.model == "gpt-4o"

    def test_model_backward_compat(self):
        """Test model property for backward compatibility."""
        provider = OpenAIProvider("test-key", model="gpt-5.2")
        assert provider.model == "gpt-5.2"

    def test_default_model(self):
        """Test default model is set correctly."""
        provider = OpenAIProvider("test-key")
        assert provider.model_name == DEFAULT_MODEL

    def test_timeout_passed_to_client(self):
        """Test timeout is correctly passed to the underlying AsyncOpenAI client."""
        provider = OpenAIProvider("test-key", timeout=15.0)
        assert provider._model_instance._provider.client.timeout.read == 15.0

    def test_base_url_passed_to_client(self):
        """Test base_url is passed through to the AsyncOpenAI client."""
        provider = OpenAIProvider("test-key", base_url="https://my-proxy.example.com/v1")
        assert (
            str(provider._model_instance._provider.client.base_url)
            == "https://my-proxy.example.com/v1/"
        )

    def test_missing_api_key_raises_llm_error(self):
        """Test missing API key raises LLMError when env vars are clear."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("cli.llm.get_openai_api_key", return_value=None):
                with patch("cli.llm.get_openai_base_url", return_value=None):
                    with pytest.raises(LLMError, match="Missing API key"):
                        OpenAIProvider()

    def test_builds_openai_chat_model(self):
        """Test OpenAI initialization builds a PydanticAI OpenAIChatModel."""
        provider = OpenAIProvider("test-key", model="gpt-4o")
        assert isinstance(provider._model_instance, OpenAIChatModel)
        assert provider._model_instance.model_name == "gpt-4o"

    def test_api_key_forwarded_to_client(self):
        """Test the API key reaches the underlying AsyncOpenAI client."""
        provider = OpenAIProvider("sk-test-key")
        assert provider._model_instance._provider.client.api_key == "sk-test-key"

    def test_api_key_read_from_environment(self):
        """Test the OpenAI key is resolved from the environment when not passed."""
        with patch("cli.llm.get_openai_api_key", return_value="sk-env-key"):
            provider = OpenAIProvider()
            assert provider._model_instance._provider.client.api_key == "sk-env-key"

    def test_base_url_only_is_allowed_without_api_key(self):
        """Test a local OpenAI-compatible endpoint works without an API key."""
        with (
            patch("cli.llm.get_openai_api_key", return_value=None),
            patch("cli.llm.get_openai_base_url", return_value=None),
        ):
            provider = OpenAIProvider(base_url="http://localhost:1234/v1")
            assert provider.provider_name == "openai"
            assert (
                str(provider._model_instance._provider.client.base_url)
                == "http://localhost:1234/v1/"
            )

    def test_base_url_from_environment(self):
        """Test base_url falls back to OPENAI_BASE_URL from the environment."""
        with patch("cli.llm.get_openai_base_url", return_value="https://env-proxy.example.com/v1"):
            provider = OpenAIProvider("test-key")
            assert (
                str(provider._model_instance._provider.client.base_url)
                == "https://env-proxy.example.com/v1/"
            )


class TestGeminiProviderBase:
    """Tests for GeminiProvider base functionality."""

    def test_inherits_from_llm_provider(self):
        """Test that GeminiProvider inherits from LLMProvider and PydanticAIProvider."""
        assert issubclass(GeminiProvider, LLMProvider)
        assert issubclass(GeminiProvider, PydanticAIProvider)

    def test_provider_name(self):
        """Test provider_name property."""
        provider = GeminiProvider("test-key")
        assert provider.provider_name == "gemini"

    def test_model_name(self):
        """Test model_name property."""
        provider = GeminiProvider("test-key", model="gemini-2.5-pro")
        assert provider.model_name == "gemini-2.5-pro"

    def test_model_name_strips_prefix(self):
        """Test model_name strips 'gemini:' and 'google-gla:' prefixes."""
        p1 = GeminiProvider("test-key", model="gemini:gemini-2.5-pro")
        assert p1.model_name == "gemini-2.5-pro"

        p2 = GeminiProvider("test-key", model="google-gla:gemini-2.5-flash")
        assert p2.model_name == "gemini-2.5-flash"

    def test_model_backward_compat(self):
        """Test model property for backward compatibility."""
        provider = GeminiProvider("test-key", model="gemini-2.5-flash")
        assert provider.model == "gemini-2.5-flash"

    def test_default_model(self):
        """Test default model is set correctly."""
        provider = GeminiProvider("test-key")
        assert provider.model_name == DEFAULT_GEMINI_MODEL

    def test_timeout_passed_to_client(self):
        """Test timeout is correctly passed to the Google client httpx client."""
        provider = GeminiProvider("test-key", timeout=15.0)
        client = provider._model_instance._provider.client
        http_options = client._api_client._http_options
        assert http_options.httpx_async_client.timeout.read == 15.0

    def test_base_url_passed_to_client(self):
        """Test base_url is passed through to the Google provider."""
        provider = GeminiProvider("test-key", base_url="https://custom.gemini.endpoint/v1")
        client = provider._model_instance._provider.client
        http_options = client._api_client._http_options
        assert http_options.base_url == "https://custom.gemini.endpoint/v1"

    def test_missing_api_key_raises_llm_error(self):
        """Test missing API key raises LLMError when env vars are clear."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("cli.llm.get_gemini_api_key", return_value=None):
                with pytest.raises(LLMError, match="Missing API key"):
                    GeminiProvider()

    def test_builds_google_model(self):
        """Test Gemini initialization builds a PydanticAI GoogleModel."""
        provider = GeminiProvider("test-key", model="gemini-2.5-flash")
        assert isinstance(provider._model_instance, GoogleModel)
        assert provider._model_instance.model_name == "gemini-2.5-flash"

    def test_api_key_read_from_environment(self):
        """Test the Gemini key is resolved from the environment when not passed."""
        with patch("cli.llm.get_gemini_api_key", return_value="AIza-env-key") as mock_key:
            provider = GeminiProvider()
            mock_key.assert_called_once()
            assert provider.provider_name == "gemini"

    def test_openai_base_url_is_not_used_for_gemini(self):
        """Test OPENAI_BASE_URL never leaks into the Google provider."""
        with patch("cli.llm.get_openai_base_url", return_value="https://openai-proxy.test/v1"):
            provider = GeminiProvider("test-key")
            client = provider._model_instance._provider.client
            http_options = client._api_client._http_options
            assert "openai-proxy" not in (http_options.base_url or "")


class TestAnthropicProviderBase:
    """Tests for AnthropicProvider base functionality."""

    def test_inherits_from_llm_provider(self):
        """Test that AnthropicProvider inherits from LLMProvider and PydanticAIProvider."""
        assert issubclass(AnthropicProvider, LLMProvider)
        assert issubclass(AnthropicProvider, PydanticAIProvider)

    def test_provider_name(self):
        """Test provider_name property."""
        provider = AnthropicProvider("test-key")
        assert provider.provider_name == "anthropic"

    def test_model_name(self):
        """Test model_name property."""
        provider = AnthropicProvider("test-key", model="claude-3-7-sonnet-latest")
        assert provider.model_name == "claude-3-7-sonnet-latest"

    def test_model_name_strips_prefix(self):
        """Test model_name strips 'anthropic:' and 'claude:' prefixes."""
        p1 = AnthropicProvider("test-key", model="anthropic:claude-3-7-sonnet-latest")
        assert p1.model_name == "claude-3-7-sonnet-latest"

        p2 = AnthropicProvider("test-key", model="claude:claude-3-5-haiku-latest")
        assert p2.model_name == "claude-3-5-haiku-latest"

    def test_model_backward_compat(self):
        """Test model property for backward compatibility."""
        provider = AnthropicProvider("test-key", model="claude-sonnet-latest")
        assert provider.model == "claude-sonnet-latest"

    def test_default_model(self):
        """Test default model is set correctly."""
        provider = AnthropicProvider("test-key")
        assert provider.model_name == DEFAULT_ANTHROPIC_MODEL

    def test_missing_api_key_raises_llm_error(self):
        """Test missing API key raises LLMError when env vars are clear."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("cli.llm.get_anthropic_api_key", return_value=None):
                with patch("cli.llm.get_anthropic_base_url", return_value=None):
                    with pytest.raises(LLMError, match="Missing API key"):
                        AnthropicProvider()

    def test_builds_anthropic_model(self):
        """Test Anthropic initialization builds a PydanticAI AnthropicModel."""
        provider = AnthropicProvider("test-key", model="claude-sonnet-latest")
        assert isinstance(provider._model_instance, AnthropicModel)
        assert provider._model_instance.model_name == "claude-sonnet-latest"

    def test_api_key_read_from_environment(self):
        """Test the Anthropic key is resolved from the environment when not passed."""
        with patch("cli.llm.get_anthropic_api_key", return_value="sk-ant-env-key") as mock_key:
            provider = AnthropicProvider()
            mock_key.assert_called_once()
            assert provider.provider_name == "anthropic"

    def test_get_anthropic_api_key_and_base_url_helpers(self):
        """Test get_anthropic_api_key and get_anthropic_base_url helper functions."""
        with patch.dict(
            "os.environ",
            {
                "ANTHROPIC_API_KEY": "sk-ant-test",
                "ANTHROPIC_BASE_URL": "https://api.anthropic.test",
            },
        ):
            assert get_anthropic_api_key() == "sk-ant-test"
            assert get_anthropic_base_url() == "https://api.anthropic.test"


class TestProviderAliases:
    """Tests for provider-name normalization on PydanticAIProvider."""

    def test_google_alias_normalizes_to_gemini(self):
        """Test 'google' is normalized to 'gemini'."""
        provider = PydanticAIProvider(provider="google", api_key="test-key")
        assert provider.provider_name == "gemini"
        assert isinstance(provider._model_instance, GoogleModel)

    def test_claude_alias_normalizes_to_anthropic(self):
        """Test 'claude' is normalized to 'anthropic'."""
        provider = PydanticAIProvider(provider="claude", api_key="test-key")
        assert provider.provider_name == "anthropic"
        assert isinstance(provider._model_instance, AnthropicModel)

    def test_openai_chat_alias_builds_openai_model(self):
        """Test 'openai-chat' is accepted and builds an OpenAI chat model."""
        provider = PydanticAIProvider(provider="openai-chat", api_key="test-key")
        assert provider.provider_name == "openai-chat"
        assert isinstance(provider._model_instance, OpenAIChatModel)

    def test_provider_name_is_case_and_whitespace_insensitive(self):
        """Test provider names are lowercased and stripped."""
        assert PydanticAIProvider(provider="  GEMINI ", api_key="k").provider_name == "gemini"
        assert PydanticAIProvider(provider="OpenAI", api_key="k").provider_name == "openai"
        assert (
            PydanticAIProvider(provider="  ANTHROPIC  ", api_key="k").provider_name == "anthropic"
        )

    def test_default_model_per_provider(self):
        """Test each provider falls back to its own default model."""
        assert PydanticAIProvider(provider="gemini", api_key="k").model_name == (
            DEFAULT_GEMINI_MODEL
        )
        assert PydanticAIProvider(provider="openai", api_key="k").model_name == DEFAULT_MODEL
        assert PydanticAIProvider(provider="anthropic", api_key="k").model_name == (
            DEFAULT_ANTHROPIC_MODEL
        )


class TestStructuredOutput:
    """Tests that the agent is wired to the ComplexityResult schema."""

    def test_agent_output_type_is_complexity_result(self):
        """Test the PydanticAI agent validates against ComplexityResult."""
        provider = PydanticAIProvider(provider="openai", model=TestModel())
        assert provider._agent.output_type is ComplexityResult

    def test_agent_run_returns_complexity_result_instance(self):
        """Test the agent output is a validated ComplexityResult, not a raw dict."""

        def respond(messages, info):
            return ModelResponse(
                parts=[ToolCallPart("final_result", {"complexity": 3, "explanation": "ok"})]
            )

        provider = PydanticAIProvider(provider="openai", model=FunctionModel(respond))
        output = provider._agent.run_sync("prompt").output
        assert isinstance(output, ComplexityResult)
        assert output.complexity == 3

    def test_test_model_generates_schema_valid_output(self):
        """Test TestModel's schema-derived output satisfies ComplexityResult bounds."""
        provider = PydanticAIProvider(provider="openai", model=TestModel())
        result = provider.analyze_complexity(
            prompt="Analyze",
            diff_excerpt="diff",
            stats_json="{}",
            title="Title",
        )
        assert 1 <= result["complexity"] <= 10
        assert isinstance(result["explanation"], str)


class TestGetProvider:
    """Tests for get_provider factory function."""

    def test_get_provider_openai(self):
        """Test get_provider with openai."""
        provider = get_provider("openai", api_key="test-key", model="openai:gpt-4o")
        assert isinstance(provider, OpenAIProvider)
        assert provider.provider_name == "openai"
        assert provider.model_name == "gpt-4o"

    def test_get_provider_gemini(self):
        """Test get_provider with gemini."""
        provider = get_provider("gemini", api_key="test-key", model="gemini-2.5-pro")
        assert isinstance(provider, GeminiProvider)
        assert provider.provider_name == "gemini"
        assert provider.model_name == "gemini-2.5-pro"

    def test_get_provider_google_alias(self):
        """Test get_provider with google alias."""
        provider = get_provider("google", api_key="test-key")
        assert isinstance(provider, GeminiProvider)
        assert provider.provider_name == "gemini"

    def test_get_provider_anthropic(self):
        """Test get_provider with anthropic."""
        provider = get_provider(
            "anthropic", api_key="test-key", model="anthropic:claude-3-7-sonnet-latest"
        )
        assert isinstance(provider, AnthropicProvider)
        assert provider.provider_name == "anthropic"
        assert provider.model_name == "claude-3-7-sonnet-latest"

    def test_get_provider_claude_alias(self):
        """Test get_provider with claude alias."""
        provider = get_provider("claude", api_key="test-key")
        assert isinstance(provider, AnthropicProvider)
        assert provider.provider_name == "anthropic"
        assert provider.model_name == DEFAULT_ANTHROPIC_MODEL

    def test_get_provider_auto_with_gemini_env(self):
        """Test get_provider auto-detection with Gemini key in environment."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("cli.llm.get_gemini_api_key", return_value="gemini-key"):
                with patch("cli.llm.get_openai_api_key", return_value=None):
                    with patch("cli.llm.get_anthropic_api_key", return_value=None):
                        provider = get_provider("auto")
                        assert isinstance(provider, GeminiProvider)
                        assert provider.provider_name == "gemini"

    def test_get_provider_auto_with_openai_env(self):
        """Test get_provider auto-detection with OpenAI key in environment."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("cli.llm.get_gemini_api_key", return_value=None):
                with patch("cli.llm.get_openai_api_key", return_value="openai-key"):
                    with patch("cli.llm.get_anthropic_api_key", return_value=None):
                        provider = get_provider("auto")
                        assert isinstance(provider, OpenAIProvider)
                        assert provider.provider_name == "openai"

    def test_get_provider_auto_with_anthropic_env(self):
        """Test get_provider auto-detection with Anthropic key in environment."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("cli.llm.get_anthropic_api_key", return_value="sk-ant-key"):
                with patch("cli.llm.get_gemini_api_key", return_value=None):
                    with patch("cli.llm.get_openai_api_key", return_value=None):
                        provider = get_provider("auto")
                        assert isinstance(provider, AnthropicProvider)
                        assert provider.provider_name == "anthropic"

    def test_get_provider_openai_chat_alias(self):
        """Test get_provider with the openai-chat alias."""
        provider = get_provider("openai-chat", api_key="test-key")
        assert isinstance(provider, OpenAIProvider)
        assert provider.provider_name == "openai"

    def test_get_provider_uses_provider_default_model(self):
        """Test get_provider substitutes each provider's default model when none is given."""
        assert get_provider("gemini", api_key="test-key").model_name == DEFAULT_GEMINI_MODEL
        assert get_provider("openai", api_key="test-key").model_name == DEFAULT_MODEL
        assert get_provider("anthropic", api_key="test-key").model_name == DEFAULT_ANTHROPIC_MODEL

    def test_get_provider_forwards_base_url_and_timeout(self):
        """Test get_provider plumbs base_url and timeout to the built client."""
        provider = get_provider(
            "openai",
            api_key="test-key",
            base_url="https://proxy.example.com/v1",
            timeout=7.0,
        )
        client = provider._model_instance._provider.client
        assert str(client.base_url) == "https://proxy.example.com/v1/"
        assert client.timeout.read == 7.0

    def test_get_provider_auto_prefers_openai_when_both_env_keys_set(self):
        """Test auto-detection picks OpenAI when both env keys are present."""
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("cli.llm.get_gemini_api_key", return_value="AIza-key"),
            patch("cli.llm.get_openai_api_key", return_value="sk-key"),
            patch("cli.llm.get_anthropic_api_key", return_value="sk-ant-key"),
        ):
            provider = get_provider("auto")
            assert isinstance(provider, OpenAIProvider)
            assert provider.provider_name == "openai"

    def test_get_provider_name_is_case_insensitive(self):
        """Test provider names are normalized before lookup."""
        assert get_provider("GEMINI", api_key="test-key").provider_name == "gemini"
        assert get_provider(" OpenAI ", api_key="test-key").provider_name == "openai"
        assert get_provider(" Anthropic ", api_key="test-key").provider_name == "anthropic"

    def test_get_provider_auto_with_explicit_gemini_key(self):
        """Test get_provider auto with explicit Gemini key (starts with AIza)."""
        with patch.dict("os.environ", {}, clear=True):
            provider = get_provider("auto", api_key="AIzaSyTestKey123")
            assert isinstance(provider, GeminiProvider)
            assert provider.provider_name == "gemini"

    def test_get_provider_auto_with_explicit_openai_key(self):
        """Test get_provider auto with explicit OpenAI key (starts with sk-)."""
        with patch.dict("os.environ", {}, clear=True):
            provider = get_provider("auto", api_key="sk-proj-testkey123")
            assert isinstance(provider, OpenAIProvider)
            assert provider.provider_name == "openai"

    def test_get_provider_auto_with_explicit_anthropic_key(self):
        """Test get_provider auto with explicit Anthropic key (starts with sk-ant-)."""
        with patch.dict("os.environ", {}, clear=True):
            provider = get_provider("auto", api_key="sk-ant-api03-testkey123")
            assert isinstance(provider, AnthropicProvider)
            assert provider.provider_name == "anthropic"

    def test_get_provider_auto_unrecognized_key_raises(self):
        """Test get_provider auto with unrecognized key format raises LLMError."""
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(LLMError, match="Explicit api_key passed with provider='auto'"):
                get_provider("auto", api_key="unrecognized_format_key")

    def test_get_provider_auto_no_keys_raises(self):
        """Test get_provider auto with no keys in env raises LLMError."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("cli.llm.get_gemini_api_key", return_value=None):
                with patch("cli.llm.get_openai_api_key", return_value=None):
                    with patch("cli.llm.get_anthropic_api_key", return_value=None):
                        with pytest.raises(LLMError, match="No API key found in environment"):
                            get_provider("auto")

    def test_get_provider_unsupported_provider_raises(self):
        """Test get_provider with unknown provider string raises LLMError."""
        with pytest.raises(LLMError, match="Unsupported provider: 'unknown_prov'"):
            get_provider("unknown_prov", api_key="test-key")

    def test_pydantic_ai_provider_unsupported_raises(self):
        """Test PydanticAIProvider constructor with unsupported provider raises LLMError."""
        with pytest.raises(LLMError, match="Unsupported provider: 'cohere'"):
            PydanticAIProvider(provider="cohere", api_key="test-key")

    def test_pydantic_ai_provider_model_instance_validates_provider(self):
        """Test Model-instance path also validates provider name."""
        with pytest.raises(LLMError, match="Unsupported provider: 'cohere'"):
            PydanticAIProvider(provider="cohere", model=TestModel())


class TestAnalyzeComplexity:
    """Tests for analyze_complexity execution using TestModel and FunctionModel."""

    def test_system_prompt_reaches_the_model(self):
        """Test that system prompt instructions actually reach the model request."""
        seen = {}

        def capture(messages, info):
            seen["req"] = messages[0]
            return ModelResponse(
                parts=[ToolCallPart("final_result", {"complexity": 4, "explanation": "x"})]
            )

        provider = PydanticAIProvider(provider="openai", model=FunctionModel(capture))
        result = provider.analyze_complexity(
            prompt="RUBRIC",
            diff_excerpt="diff content",
            stats_json="{}",
            title="Title",
        )

        assert result["complexity"] == 4
        assert seen["req"].instructions == "RUBRIC"

    def test_analyze_complexity_success_openai(self):
        """Test successful complexity analysis with OpenAI provider using TestModel."""
        test_model = TestModel(
            custom_output_args={"complexity": 5, "explanation": "Medium complexity"}
        )
        provider = PydanticAIProvider(provider="openai", model=test_model)

        result = provider.analyze_complexity(
            prompt="Analyze this PR",
            diff_excerpt="diff content",
            stats_json='{"additions": 10}',
            title="Fix bug",
        )

        assert result["complexity"] == 5
        assert result["explanation"] == "Medium complexity"
        assert result["provider"] == "openai"
        assert result["model"] == "test"
        assert isinstance(result["tokens"], int)

    def test_analyze_complexity_success_gemini(self):
        """Test successful complexity analysis with Gemini provider using TestModel."""
        test_model = TestModel(
            custom_output_args={
                "complexity": 8,
                "explanation": "High complexity architectural change",
            }
        )
        provider = PydanticAIProvider(provider="gemini", model=test_model)

        result = provider.analyze_complexity(
            prompt="Analyze this PR",
            diff_excerpt="diff content",
            stats_json='{"additions": 500}',
            title="Refactor core architecture",
        )

        assert result["complexity"] == 8
        assert result["explanation"] == "High complexity architectural change"
        assert result["provider"] == "gemini"
        assert result["model"] == "test"
        assert isinstance(result["tokens"], int)

    def test_analyze_complexity_out_of_bounds_fails_fast_with_schema_error(self):
        """Test out-of-bounds complexity fails with descriptive schema validation error."""
        test_model = TestModel(
            custom_output_args={"complexity": 15, "explanation": "Out of bounds"}
        )
        provider = PydanticAIProvider(provider="openai", model=test_model)

        with pytest.raises(LLMError, match="Failed to parse or validate LLM response"):
            provider.analyze_complexity(
                prompt="Analyze",
                diff_excerpt="diff",
                stats_json="{}",
                title="Title",
            )

    def test_analyze_complexity_wrong_field_types_raise_llm_error(self):
        """Test structurally invalid tool arguments surface as LLMError."""

        def wrong_types(messages, info):
            return ModelResponse(
                parts=[
                    ToolCallPart("final_result", {"complexity": "very high", "explanation": None})
                ]
            )

        provider = PydanticAIProvider(
            provider="openai", model=FunctionModel(wrong_types), retries=1
        )

        with pytest.raises(LLMError, match="Failed to parse or validate LLM response"):
            provider.analyze_complexity(
                prompt="Analyze",
                diff_excerpt="diff",
                stats_json="{}",
                title="Title",
            )

    def test_analyze_complexity_plain_text_response_raises_llm_error(self):
        """Test a prose reply with no structured output surfaces as LLMError."""

        def text_only(messages, info):
            return ModelResponse(parts=[TextPart("This PR looks like a 5 to me.")])

        provider = PydanticAIProvider(provider="gemini", model=FunctionModel(text_only), retries=1)

        with pytest.raises(LLMError, match="Failed to parse or validate LLM response from gemini"):
            provider.analyze_complexity(
                prompt="Analyze",
                diff_excerpt="diff",
                stats_json="{}",
                title="Title",
            )

    def test_analyze_complexity_recovers_after_invalid_response(self):
        """Test PydanticAI retries a schema violation and returns the corrected result."""
        calls = {"n": 0}

        def flaky(messages, info):
            calls["n"] += 1
            if calls["n"] == 1:
                return ModelResponse(
                    parts=[ToolCallPart("final_result", {"complexity": 99, "explanation": "bad"})]
                )
            return ModelResponse(
                parts=[ToolCallPart("final_result", {"complexity": 7, "explanation": "good"})]
            )

        provider = PydanticAIProvider(provider="gemini", model=FunctionModel(flaky), retries=2)
        result = provider.analyze_complexity(
            prompt="Analyze",
            diff_excerpt="diff",
            stats_json="{}",
            title="Title",
        )

        assert calls["n"] == 2
        assert result["complexity"] == 7
        assert result["explanation"] == "good"
        assert result["provider"] == "gemini"

    def test_bounded_request_count_on_schema_error(self):
        """Test that schema validation retries are bounded to retries + 1 (e.g. 4 total)."""
        request_count = 0
        original_request = TestModel.request

        async def counting_request(self, *args, **kwargs):
            nonlocal request_count
            request_count += 1
            return await original_request(self, *args, **kwargs)

        test_model = TestModel(
            custom_output_args={"complexity": 15, "explanation": "Out of bounds"}
        )
        provider = PydanticAIProvider(provider="openai", model=test_model, retries=3)

        with patch.object(TestModel, "request", counting_request):
            with pytest.raises(LLMError, match="Failed to parse or validate LLM response"):
                provider.analyze_complexity(
                    prompt="Analyze",
                    diff_excerpt="diff",
                    stats_json="{}",
                    title="Title",
                )

        assert request_count == 4  # 1 initial + 3 retries

    def test_analyze_complexity_api_error_handling(self):
        """Test API errors from provider are caught and wrapped as LLMError."""
        test_model = TestModel()
        provider = PydanticAIProvider(provider="openai", model=test_model)

        with patch.object(
            provider._agent, "run_sync", side_effect=Exception("API connection refused")
        ):
            with pytest.raises(LLMError, match="openai API error: API connection refused"):
                provider.analyze_complexity(
                    prompt="Analyze",
                    diff_excerpt="diff",
                    stats_json="{}",
                    title="Title",
                )

    def test_pydantic_ai_provider_properties(self):
        """Test provider_name, provider, model_name, and model properties."""
        test_model = TestModel()
        provider = PydanticAIProvider(provider="gemini", model=test_model)
        assert provider.provider_name == "gemini"
        assert provider.provider == "gemini"
        assert provider.model_name == test_model.model_name
        assert provider.model == test_model.model_name

    def test_analyze_complexity_token_fallback_details(self):
        """Test fallback to usage.details when total_tokens is zero or falsy."""
        test_model = TestModel()
        provider = PydanticAIProvider(provider="gemini", model=test_model)

        mock_result = MagicMock()
        mock_result.output = ComplexityResult(complexity=5, explanation="Solid PR")
        mock_usage = MagicMock()
        mock_usage.total_tokens = 0
        mock_usage.details = {
            "text_prompt_tokens": 150,
            "thoughts_tokens": 50,
            "candidates_tokens": 20,
        }
        mock_result.usage.return_value = mock_usage

        with patch.object(provider._agent, "run_sync", return_value=mock_result):
            res = provider.analyze_complexity(
                prompt="Analyze",
                diff_excerpt="diff",
                stats_json="{}",
                title="Title",
            )
            assert res["tokens"] == 220


class TestLLMError:
    """Tests for LLMError exception."""

    def test_llm_error_message(self):
        """Test LLMError stores message correctly."""
        error = LLMError("Something went wrong")
        assert str(error) == "Something went wrong"

    def test_llm_error_is_exception(self):
        """Test LLMError is an Exception."""
        assert issubclass(LLMError, Exception)
