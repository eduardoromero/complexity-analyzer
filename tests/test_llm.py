"""Tests for LLM module and PydanticAI provider integration."""

from unittest.mock import MagicMock, patch
import pytest
from pydantic import ValidationError
from pydantic_ai.models.test import TestModel

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

    def test_model_backward_compat(self):
        """Test model property for backward compatibility."""
        provider = OpenAIProvider("test-key", model="gpt-5.2")
        assert provider.model == "gpt-5.2"

    def test_default_model(self):
        """Test default model is set correctly."""
        provider = OpenAIProvider("test-key")
        assert provider.model_name == "gpt-5.2"

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

    def test_model_backward_compat(self):
        """Test model property for backward compatibility."""
        provider = GeminiProvider("test-key", model="gemini-2.5-flash")
        assert provider.model == "gemini-2.5-flash"

    def test_default_model(self):
        """Test default model is set correctly."""
        provider = GeminiProvider("test-key")
        assert provider.model_name == "gemini-2.5-flash"

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
        provider = get_provider("openai", api_key="test-key", model="gpt-4o")
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

    def test_get_provider_auto_with_gemini(self):
        """Test get_provider auto-detection with Gemini key."""
        with patch("cli.llm.get_gemini_api_key", return_value="gemini-key"):
            with patch("cli.llm.get_openai_api_key", return_value=None):
                provider = get_provider("auto")
                assert isinstance(provider, GeminiProvider)
                assert provider.provider_name == "gemini"

    def test_get_provider_auto_with_openai(self):
        """Test get_provider auto-detection with OpenAI key."""
        with patch("cli.llm.get_gemini_api_key", return_value=None):
            with patch("cli.llm.get_openai_api_key", return_value="openai-key"):
                provider = get_provider("auto")
                assert isinstance(provider, OpenAIProvider)
                assert provider.provider_name == "openai"


class TestAnalyzeComplexity:
    """Tests for analyze_complexity execution using TestModel."""

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

    def test_analyze_complexity_out_of_bounds_raises_llm_error(self):
        """Test that invalid/out-of-bounds complexity results in LLMError."""
        test_model = TestModel(
            custom_output_args={"complexity": 15, "explanation": "Out of bounds"}
        )
        provider = PydanticAIProvider(provider="openai", model=test_model)

        with pytest.raises(LLMError, match="API error after"):
            provider.analyze_complexity(
                prompt="Analyze",
                diff_excerpt="diff",
                stats_json="{}",
                title="Title",
                max_retries=1,
            )

    @patch("cli.llm.Agent")
    def test_analyze_complexity_empty_response(self, mock_agent_class):
        """Test handling of empty LLM response."""
        mock_result = MagicMock()
        mock_result.output = None
        mock_result.data = None

        mock_agent = MagicMock()
        mock_agent.run_sync.return_value = mock_result
        mock_agent_class.return_value = mock_agent

        test_model = TestModel()
        provider = PydanticAIProvider(provider="openai", model=test_model)

        with pytest.raises(LLMError, match="Empty response"):
            provider.analyze_complexity(
                prompt="Analyze",
                diff_excerpt="diff",
                stats_json="{}",
                title="Title",
                max_retries=1,
            )

    @patch("cli.llm.time.sleep")
    def test_analyze_complexity_retry_on_error(self, mock_sleep):
        """Test retry logic on transient errors."""
        test_model = TestModel(
            custom_output_args={"complexity": 3, "explanation": "Low complexity"}
        )
        provider = PydanticAIProvider(provider="openai", model=test_model)

        # Mock run_sync to fail on first attempt, succeed on second
        mock_success_result = MagicMock()
        mock_success_result.output = ComplexityResult(complexity=3, explanation="Low complexity")
        mock_success_result.usage.return_value = MagicMock(total_tokens=500)

        with patch("cli.llm.Agent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent.run_sync.side_effect = [
                Exception("Temporary network glitch"),
                mock_success_result,
            ]
            mock_agent_class.return_value = mock_agent

            result = provider.analyze_complexity(
                prompt="Analyze",
                diff_excerpt="diff",
                stats_json="{}",
                title="Title",
                max_retries=3,
            )

            assert result["complexity"] == 3
            assert result["explanation"] == "Low complexity"
            assert result["tokens"] == 500
            assert mock_agent.run_sync.call_count == 2
            mock_sleep.assert_called_once()

    def test_analyze_complexity_all_retries_fail(self):
        """Test behavior when all retries fail."""
        test_model = TestModel()
        provider = PydanticAIProvider(provider="openai", model=test_model)

        with patch("cli.llm.Agent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent.run_sync.side_effect = Exception("Persistent API error")
            mock_agent_class.return_value = mock_agent

            with pytest.raises(LLMError, match="after 2 attempts"):
                provider.analyze_complexity(
                    prompt="Analyze",
                    diff_excerpt="diff",
                    stats_json="{}",
                    title="Title",
                    max_retries=2,
                )

    def test_analyze_complexity_no_usage(self):
        """Test handling when usage info is missing."""
        mock_result = MagicMock()
        mock_result.output = ComplexityResult(complexity=7, explanation="High")
        del mock_result.usage  # simulate missing usage attribute

        with patch("cli.llm.Agent") as mock_agent_class:
            mock_agent = MagicMock()
            mock_agent.run_sync.return_value = mock_result
            mock_agent_class.return_value = mock_agent

            test_model = TestModel()
            provider = PydanticAIProvider(provider="openai", model=test_model)
            result = provider.analyze_complexity(
                prompt="Analyze",
                diff_excerpt="diff",
                stats_json="{}",
                title="Title",
            )

            assert result["complexity"] == 7
            assert result["tokens"] is None


class TestLLMError:
    """Tests for LLMError exception."""

    def test_llm_error_message(self):
        """Test LLMError stores message correctly."""
        error = LLMError("Something went wrong")
        assert str(error) == "Something went wrong"

    def test_llm_error_is_exception(self):
        """Test LLMError is an Exception."""
        assert issubclass(LLMError, Exception)
