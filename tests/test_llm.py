"""Tests for LLM module and PydanticAI provider integration."""

from unittest.mock import patch
import pytest
from pydantic import ValidationError
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel

from cli.constants import DEFAULT_GEMINI_MODEL, DEFAULT_MODEL
from cli.llm import (
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

    def test_get_provider_auto_with_gemini_env(self):
        """Test get_provider auto-detection with Gemini key in environment."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("cli.llm.get_gemini_api_key", return_value="gemini-key"):
                with patch("cli.llm.get_openai_api_key", return_value=None):
                    provider = get_provider("auto")
                    assert isinstance(provider, GeminiProvider)
                    assert provider.provider_name == "gemini"

    def test_get_provider_auto_with_openai_env(self):
        """Test get_provider auto-detection with OpenAI key in environment."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("cli.llm.get_gemini_api_key", return_value=None):
                with patch("cli.llm.get_openai_api_key", return_value="openai-key"):
                    provider = get_provider("auto")
                    assert isinstance(provider, OpenAIProvider)
                    assert provider.provider_name == "openai"

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
                    with pytest.raises(LLMError, match="No API key found in environment"):
                        get_provider("auto")

    def test_get_provider_unsupported_provider_raises(self):
        """Test get_provider with unknown provider string raises LLMError."""
        with pytest.raises(LLMError, match="Unsupported provider: 'anthropic'"):
            get_provider("anthropic", api_key="test-key")

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


class TestLLMError:
    """Tests for LLMError exception."""

    def test_llm_error_message(self):
        """Test LLMError stores message correctly."""
        error = LLMError("Something went wrong")
        assert str(error) == "Something went wrong"

    def test_llm_error_is_exception(self):
        """Test LLMError is an Exception."""
        assert issubclass(LLMError, Exception)
