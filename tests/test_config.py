"""Tests for config module."""

import os
import pytest
from unittest.mock import patch
from cli.config import (
    validate_owner_repo,
    validate_pr_number,
    get_github_tokens,
    get_gemini_api_key,
    get_openai_api_key,
    get_openai_base_url,
)
from cli.config_types import AnalysisConfig, BatchConfig, OutputConfig
from cli.constants import DEFAULT_MODEL


def test_validate_owner_repo_valid():
    """Test valid owner/repo names."""
    validate_owner_repo("owner", "repo")
    validate_owner_repo("owner-name", "repo_name")
    validate_owner_repo("owner.name", "repo-123")


def test_validate_owner_repo_invalid():
    """Test invalid owner/repo names."""
    with pytest.raises(ValueError):
        validate_owner_repo("owner/repo", "repo")
    with pytest.raises(ValueError):
        validate_owner_repo("owner", "repo@name")


def test_validate_pr_number():
    """Test PR number validation."""
    validate_pr_number(1)
    validate_pr_number(123)
    with pytest.raises(ValueError):
        validate_pr_number(0)
    with pytest.raises(ValueError):
        validate_pr_number(-1)


# get_github_tokens tests


class TestGetGitHubTokens:
    """Tests for the get_github_tokens function."""

    @patch.dict(os.environ, {}, clear=True)
    def test_no_tokens_returns_empty(self):
        """Test that empty list is returned when no tokens are set."""
        tokens = get_github_tokens()
        assert tokens == []

    @patch.dict(os.environ, {"GH_TOKEN": "single_token"}, clear=True)
    def test_single_token_from_gh_token(self):
        """Test getting single token from GH_TOKEN."""
        tokens = get_github_tokens()
        assert tokens == ["single_token"]

    @patch.dict(os.environ, {"GITHUB_TOKEN": "single_token"}, clear=True)
    def test_single_token_from_github_token(self):
        """Test getting single token from GITHUB_TOKEN."""
        tokens = get_github_tokens()
        assert tokens == ["single_token"]

    @patch.dict(os.environ, {"GH_TOKENS": "token1,token2,token3"}, clear=True)
    def test_multiple_tokens_comma_separated(self):
        """Test getting multiple tokens from GH_TOKENS (comma-separated)."""
        tokens = get_github_tokens()
        assert tokens == ["token1", "token2", "token3"]

    @patch.dict(os.environ, {"GH_TOKENS": "token1\ntoken2\ntoken3"}, clear=True)
    def test_multiple_tokens_newline_separated(self):
        """Test getting multiple tokens from GH_TOKENS (newline-separated)."""
        tokens = get_github_tokens()
        assert tokens == ["token1", "token2", "token3"]

    @patch.dict(os.environ, {"GH_TOKENS": "token1, token2 , token3"}, clear=True)
    def test_tokens_are_stripped(self):
        """Test that tokens are stripped of whitespace."""
        tokens = get_github_tokens()
        assert tokens == ["token1", "token2", "token3"]

    @patch.dict(os.environ, {"GH_TOKENS": "token1,,token2,,,token3"}, clear=True)
    def test_empty_tokens_filtered(self):
        """Test that empty tokens are filtered out."""
        tokens = get_github_tokens()
        assert tokens == ["token1", "token2", "token3"]

    @patch.dict(os.environ, {"GITHUB_TOKENS": "token1,token2"}, clear=True)
    def test_multiple_tokens_from_github_tokens(self):
        """Test getting multiple tokens from GITHUB_TOKENS."""
        tokens = get_github_tokens()
        assert tokens == ["token1", "token2"]

    @patch.dict(os.environ, {"GH_TOKENS": "multi1,multi2", "GH_TOKEN": "single"}, clear=True)
    def test_gh_tokens_takes_precedence_over_gh_token(self):
        """Test that GH_TOKENS takes precedence over GH_TOKEN."""
        tokens = get_github_tokens()
        assert tokens == ["multi1", "multi2"]

    @patch.dict(os.environ, {"GH_TOKENS": "", "GH_TOKEN": "fallback"}, clear=True)
    def test_falls_back_to_single_token_if_multi_empty(self):
        """Test fallback to single token if multi-token env var is empty."""
        tokens = get_github_tokens()
        assert tokens == ["fallback"]


# get_gemini_api_key tests


class TestGetGeminiApiKey:
    """Tests for the get_gemini_api_key function."""

    @patch.dict(os.environ, {}, clear=True)
    def test_no_key_returns_none(self):
        """Test that None is returned when no key is set."""
        assert get_gemini_api_key() is None

    @patch.dict(os.environ, {"GEMINI_API_KEY": "gemini_key"}, clear=True)
    def test_key_from_gemini_api_key(self):
        """Test getting key from GEMINI_API_KEY."""
        assert get_gemini_api_key() == "gemini_key"

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "google_key"}, clear=True)
    def test_key_from_google_api_key(self):
        """Test getting key from GOOGLE_API_KEY fallback."""
        assert get_gemini_api_key() == "google_key"

    @patch.dict(
        os.environ,
        {"GEMINI_API_KEY": "gemini_key", "GOOGLE_API_KEY": "google_key"},
        clear=True,
    )
    def test_gemini_api_key_takes_precedence(self):
        """Test that GEMINI_API_KEY takes precedence over GOOGLE_API_KEY."""
        assert get_gemini_api_key() == "gemini_key"

    @patch.dict(
        os.environ,
        {"GEMINI_API_KEY": "", "GOOGLE_API_KEY": "google_key"},
        clear=True,
    )
    def test_empty_gemini_key_falls_back_to_google_key(self):
        """Test an empty GEMINI_API_KEY falls back to GOOGLE_API_KEY."""
        assert get_gemini_api_key() == "google_key"

    @patch.dict(os.environ, {"OPENAI_API_KEY": "openai_key"}, clear=True)
    def test_gemini_key_ignores_openai_key(self):
        """Test that an OpenAI key never satisfies the Gemini lookup."""
        assert get_gemini_api_key() is None


# get_openai_api_key / get_openai_base_url tests


class TestGetOpenAIConfig:
    """Tests for the OpenAI credential helpers."""

    @patch.dict(os.environ, {}, clear=True)
    def test_no_key_returns_none(self):
        """Test that None is returned when OPENAI_API_KEY is unset."""
        assert get_openai_api_key() is None

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True)
    def test_key_from_openai_api_key(self):
        """Test getting the key from OPENAI_API_KEY."""
        assert get_openai_api_key() == "sk-test"

    @patch.dict(os.environ, {"GEMINI_API_KEY": "gemini_key"}, clear=True)
    def test_openai_key_ignores_gemini_key(self):
        """Test that a Gemini key never satisfies the OpenAI lookup."""
        assert get_openai_api_key() is None

    @patch.dict(os.environ, {}, clear=True)
    def test_no_base_url_returns_none(self):
        """Test that None is returned when OPENAI_BASE_URL is unset."""
        assert get_openai_base_url() is None

    @patch.dict(os.environ, {"OPENAI_BASE_URL": ""}, clear=True)
    def test_empty_base_url_returns_none(self):
        """Test that an empty OPENAI_BASE_URL is normalized to None."""
        assert get_openai_base_url() is None

    @patch.dict(os.environ, {"OPENAI_BASE_URL": "https://proxy.test/v1"}, clear=True)
    def test_base_url_from_env(self):
        """Test getting the base URL from OPENAI_BASE_URL."""
        assert get_openai_base_url() == "https://proxy.test/v1"


# AnalysisConfig validation tests


class TestAnalysisConfigValidation:
    """Tests for AnalysisConfig validation."""

    def test_valid_config(self):
        """Test that valid config is accepted."""
        config = AnalysisConfig(
            model="gpt-4",
            timeout=30.0,
            max_tokens=1000,
            hunks_per_file=5,
            sleep_seconds=0.5,
        )
        assert config.model == "gpt-4"
        assert config.timeout == 30.0

    def test_timeout_must_be_positive(self):
        """Test that timeout must be positive."""
        with pytest.raises(ValueError, match="timeout must be positive"):
            AnalysisConfig(timeout=-10.0)
        with pytest.raises(ValueError, match="timeout must be positive"):
            AnalysisConfig(timeout=0.0)

    def test_max_tokens_must_be_positive(self):
        """Test that max_tokens must be positive."""
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            AnalysisConfig(max_tokens=0)
        with pytest.raises(ValueError, match="max_tokens must be positive"):
            AnalysisConfig(max_tokens=-100)

    def test_hunks_per_file_must_be_positive(self):
        """Test that hunks_per_file must be positive."""
        with pytest.raises(ValueError, match="hunks_per_file must be positive"):
            AnalysisConfig(hunks_per_file=0)
        with pytest.raises(ValueError, match="hunks_per_file must be positive"):
            AnalysisConfig(hunks_per_file=-5)

    def test_sleep_seconds_cannot_be_negative(self):
        """Test that sleep_seconds cannot be negative."""
        with pytest.raises(ValueError, match="sleep_seconds cannot be negative"):
            AnalysisConfig(sleep_seconds=-1.0)
        # Zero is allowed
        config = AnalysisConfig(sleep_seconds=0.0)
        assert config.sleep_seconds == 0.0

    def test_model_cannot_be_empty(self):
        """Test that model cannot be empty."""
        with pytest.raises(ValueError, match="model cannot be empty"):
            AnalysisConfig(model="")
        with pytest.raises(ValueError, match="model cannot be empty"):
            AnalysisConfig(model="   ")

    def test_provider_defaults_to_auto(self):
        """Test that provider defaults to 'auto'."""
        config = AnalysisConfig()
        assert config.provider == "auto"

    def test_provider_accepts_valid_values(self):
        """Test that provider accepts 'auto', 'gemini', 'openai', and 'anthropic'."""
        assert AnalysisConfig(provider="auto").provider == "auto"
        assert AnalysisConfig(provider="gemini").provider == "gemini"
        assert AnalysisConfig(provider="openai").provider == "openai"
        assert AnalysisConfig(provider="anthropic").provider == "anthropic"

    def test_provider_rejects_invalid_values(self):
        """Test that an invalid provider is rejected."""
        with pytest.raises(ValueError, match="provider must be"):
            AnalysisConfig(provider="cohere")

    def test_provider_rejects_aliases_and_casing(self):
        """Test that only the canonical lowercase provider names are accepted."""
        for value in (
            "google",
            "openai-chat",
            "claude",
            "Gemini",
            "OPENAI",
            "ANTHROPIC",
            " gemini ",
            "",
        ):
            with pytest.raises(ValueError, match="provider must be"):
                AnalysisConfig(provider=value)

    def test_gemini_key_defaults_to_none(self):
        """Test that gemini_key defaults to None."""
        config = AnalysisConfig()
        assert config.gemini_key is None

    def test_anthropic_key_defaults_to_none(self):
        """Test that anthropic_key defaults to None."""
        config = AnalysisConfig()
        assert config.anthropic_key is None

    def test_gemini_provider_config(self):
        """Test a Gemini-flavoured config carries provider, model and key."""
        config = AnalysisConfig(
            provider="gemini",
            model="gemini-flash-latest",
            gemini_key="AIza-test-key",
        )
        assert config.provider == "gemini"
        assert config.model == "gemini-flash-latest"
        assert config.gemini_key == "AIza-test-key"
        assert config.openai_key is None
        assert config.anthropic_key is None

    def test_openai_provider_config(self):
        """Test an OpenAI-flavoured config carries provider, model and key."""
        config = AnalysisConfig(
            provider="openai",
            model="gpt-5.2",
            openai_key="sk-test-key",
        )
        assert config.provider == "openai"
        assert config.model == "gpt-5.2"
        assert config.openai_key == "sk-test-key"
        assert config.gemini_key is None
        assert config.anthropic_key is None

    def test_anthropic_provider_config(self):
        """Test an Anthropic-flavoured config carries provider, model and key."""
        config = AnalysisConfig(
            provider="anthropic",
            model="claude-sonnet-latest",
            anthropic_key="sk-ant-test-key",
        )
        assert config.provider == "anthropic"
        assert config.model == "claude-sonnet-latest"
        assert config.anthropic_key == "sk-ant-test-key"
        assert config.openai_key is None
        assert config.gemini_key is None

    def test_from_env_with_anthropic_key(self):
        """Test AnalysisConfig.from_env picks up ANTHROPIC_API_KEY from env."""
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-env-123"}, clear=True):
            config = AnalysisConfig.from_env()
            assert config.anthropic_key == "sk-ant-env-123"

    def test_all_provider_keys_can_coexist(self):
        """Test all keys may be supplied so 'auto' can choose between them."""
        config = AnalysisConfig(
            provider="auto",
            gemini_key="AIza-test-key",
            openai_key="sk-test-key",
            anthropic_key="sk-ant-test-key",
        )
        assert config.gemini_key == "AIza-test-key"
        assert config.openai_key == "sk-test-key"
        assert config.anthropic_key == "sk-ant-test-key"

    def test_defaults_to_openai_model(self):
        """Test the default model is the OpenAI default, matching provider='auto'."""
        assert AnalysisConfig().model == DEFAULT_MODEL


# BatchConfig validation tests


class TestBatchConfigValidation:
    """Tests for BatchConfig validation."""

    def test_valid_config(self):
        """Test that valid config is accepted."""
        config = BatchConfig(workers=4, label_prs=True, label_prefix="complexity:")
        assert config.workers == 4

    def test_workers_must_be_at_least_one(self):
        """Test that workers must be >= 1."""
        with pytest.raises(ValueError, match="workers must be >= 1"):
            BatchConfig(workers=0)
        with pytest.raises(ValueError, match="workers must be >= 1"):
            BatchConfig(workers=-1)

    def test_label_prefix_required_when_label_prs_true(self):
        """Test that label_prefix cannot be empty when label_prs is True."""
        with pytest.raises(ValueError, match="label_prefix cannot be empty when label_prs is True"):
            BatchConfig(label_prs=True, label_prefix="")
        # Empty prefix is allowed when label_prs is False
        config = BatchConfig(label_prs=False, label_prefix="")
        assert config.label_prefix == ""


# OutputConfig validation tests


class TestOutputConfigValidation:
    """Tests for OutputConfig validation."""

    def test_valid_json_format(self):
        """Test that json format is accepted."""
        config = OutputConfig(format="json")
        assert config.format == "json"

    def test_valid_markdown_format(self):
        """Test that markdown format is accepted."""
        config = OutputConfig(format="markdown")
        assert config.format == "markdown"

    def test_invalid_format_rejected(self):
        """Test that invalid formats are rejected."""
        with pytest.raises(ValueError, match="format must be 'json' or 'markdown'"):
            OutputConfig(format="xml")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="format must be 'json' or 'markdown'"):
            OutputConfig(format="MARKDOWN")  # type: ignore[arg-type]
