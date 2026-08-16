from unittest.mock import patch

import pytest

from cli.analyze import analyze_single_pr, is_automated_sync_pr
from cli.config_types import AnalysisConfig
from cli.constants import DEFAULT_GEMINI_MODEL, DEFAULT_MODEL

PR_URL = "https://github.com/owner/repo/pull/123"

SYNC_TITLE = "chore(cursor): [skip-ci] synced file(s) with lemonade-hq/cursor-rules"
SYNC_BODY = (
    "synced local file(s) with [lemonade-hq/cursor-rules](https://github.com/...).\n"
    "This PR was created automatically by the [repo-file-sync-action] workflow."
)


def test_sync_pr_detected_by_title_and_bot_author():
    assert is_automated_sync_pr(SYNC_TITLE, "github-actions[bot]") is True
    assert is_automated_sync_pr(SYNC_TITLE, "repo-file-sync-action") is True


def test_sync_pr_detected_by_title_and_body_signature():
    # Sync workflows often commit under a human PAT, so author_login is a
    # real username. The body signature is the fallback signal.
    assert is_automated_sync_pr(SYNC_TITLE, "dor-tzur-lmnd", body=SYNC_BODY) is True


def test_sync_title_alone_is_not_enough():
    # Human-authored PR — no bot login, no sync-action body marker.
    assert is_automated_sync_pr(SYNC_TITLE, "erez.dickman", body="rebased branch") is False
    assert is_automated_sync_pr(SYNC_TITLE, "erez.dickman", body=None) is False


def test_bot_author_alone_is_not_enough():
    # A bot PR with a substantive title should NOT short-circuit.
    title = "feat: bump dependency from 1.0 to 2.0"
    assert is_automated_sync_pr(title, "dependabot[bot]") is False


def test_skip_ci_alone_does_not_trigger():
    # "[skip-ci]" is widely used by humans for CI bypass on docs/typo fixes.
    # Without "synced file(s)", we must not short-circuit even with a bot author.
    assert is_automated_sync_pr("[skip-ci] fix typo in README", "github-actions[bot]") is False


def test_missing_signals():
    assert is_automated_sync_pr("", "github-actions[bot]") is False
    assert is_automated_sync_pr("synced file(s)", None) is False
    assert is_automated_sync_pr("synced file(s)", "") is False


def test_synced_local_file_phrasing():
    title = "chore: synced local file(s) with org/upstream"
    assert is_automated_sync_pr(title, "github-actions[bot]") is True


@patch("cli.analyze.fetch_pr")
@patch("cli.analyze.get_provider")
def test_analyze_single_pr_auto_detects_gemini(mock_get_provider, mock_fetch):
    mock_fetch.return_value = (
        "diff text",
        {"title": "Test PR", "additions": 1, "deletions": 1},
    )
    mock_provider = mock_get_provider.return_value
    mock_provider.analyze_complexity.return_value = {
        "complexity": 4,
        "explanation": "Simple",
        "provider": "gemini",
        "model": "gemini-flash-latest",
    }

    config = AnalysisConfig(gemini_key="gemini-key", provider="auto")
    result = analyze_single_pr("https://github.com/owner/repo/pull/123", config)

    assert result["score"] == 4
    assert result["provider"] == "gemini"
    mock_get_provider.assert_called_once_with(
        provider="gemini",
        api_key="gemini-key",
        model="gemini-flash-latest",
        timeout=config.timeout,
    )


def test_analyze_single_pr_missing_api_keys_raises_value_error():
    config = AnalysisConfig(provider="auto")
    with patch.dict("os.environ", {}, clear=True):
        with patch("cli.analyze.get_gemini_api_key", return_value=None):
            with patch("cli.analyze.get_openai_api_key", return_value=None):
                with pytest.raises(ValueError, match="GEMINI_API_KEY or OPENAI_API_KEY"):
                    analyze_single_pr(PR_URL, config)


@pytest.fixture
def analyze_mocks():
    """Patch the GitHub fetch and provider factory used by analyze_single_pr."""
    with (
        patch("cli.analyze.fetch_pr") as mock_fetch,
        patch("cli.analyze.get_provider") as mock_get_provider,
    ):
        mock_fetch.return_value = (
            "diff --git a/file.py b/file.py\n+line1",
            {"title": "Test PR", "additions": 10, "deletions": 5},
        )
        mock_get_provider.return_value.analyze_complexity.return_value = {
            "complexity": 4,
            "explanation": "Simple",
            "provider": "stub",
            "model": "stub-model",
            "tokens": 42,
        }
        yield mock_get_provider


class TestProviderAutoDetection:
    """Auto-detection of the LLM provider from the environment."""

    def test_auto_detects_gemini_from_env(self, analyze_mocks, monkeypatch):
        """GEMINI_API_KEY alone selects the Gemini provider and its default model."""
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-env-key")

        analyze_single_pr(PR_URL, AnalysisConfig(provider="auto"))

        analyze_mocks.assert_called_once()
        kwargs = analyze_mocks.call_args.kwargs
        assert kwargs["provider"] == "gemini"
        assert kwargs["api_key"] == "AIza-env-key"
        assert kwargs["model"] == DEFAULT_GEMINI_MODEL

    def test_auto_detects_gemini_from_google_api_key(self, analyze_mocks, monkeypatch):
        """GOOGLE_API_KEY is an accepted fallback for Gemini auto-detection."""
        monkeypatch.setenv("GOOGLE_API_KEY", "AIza-google-env-key")

        analyze_single_pr(PR_URL, AnalysisConfig(provider="auto"))

        kwargs = analyze_mocks.call_args.kwargs
        assert kwargs["provider"] == "gemini"
        assert kwargs["api_key"] == "AIza-google-env-key"

    def test_auto_detects_openai_from_env(self, analyze_mocks, monkeypatch):
        """OPENAI_API_KEY alone selects the OpenAI provider and its default model."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")

        analyze_single_pr(PR_URL, AnalysisConfig(provider="auto"))

        kwargs = analyze_mocks.call_args.kwargs
        assert kwargs["provider"] == "openai"
        assert kwargs["api_key"] == "sk-env-key"
        assert kwargs["model"] == DEFAULT_MODEL

    def test_auto_prefers_openai_when_both_env_keys_set(self, analyze_mocks, monkeypatch):
        """With both env keys present, OpenAI wins to preserve prior behaviour."""
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-env-key")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")

        analyze_single_pr(PR_URL, AnalysisConfig(provider="auto"))

        kwargs = analyze_mocks.call_args.kwargs
        assert kwargs["provider"] == "openai"
        assert kwargs["api_key"] == "sk-env-key"

    def test_config_key_overrides_ambient_env_key(self, analyze_mocks, monkeypatch):
        """An explicit gemini_key on the config beats an ambient OPENAI_API_KEY."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")

        analyze_single_pr(PR_URL, AnalysisConfig(provider="auto", gemini_key="AIza-explicit"))

        kwargs = analyze_mocks.call_args.kwargs
        assert kwargs["provider"] == "gemini"
        assert kwargs["api_key"] == "AIza-explicit"

    def test_auto_detects_openai_from_config_key(self, analyze_mocks):
        """An explicit openai_key on the config selects OpenAI with no env keys set."""
        analyze_single_pr(PR_URL, AnalysisConfig(provider="auto", openai_key="sk-explicit"))

        kwargs = analyze_mocks.call_args.kwargs
        assert kwargs["provider"] == "openai"
        assert kwargs["api_key"] == "sk-explicit"


class TestExplicitProviderSelection:
    """Explicit provider selection through AnalysisConfig."""

    def test_explicit_gemini_uses_env_key(self, analyze_mocks, monkeypatch):
        """provider='gemini' reads GEMINI_API_KEY from the environment."""
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-env-key")

        analyze_single_pr(PR_URL, AnalysisConfig(provider="gemini"))

        kwargs = analyze_mocks.call_args.kwargs
        assert kwargs["provider"] == "gemini"
        assert kwargs["api_key"] == "AIza-env-key"

    def test_explicit_gemini_ignores_openai_env_key(self, analyze_mocks, monkeypatch):
        """provider='gemini' does not fall back to an OpenAI key."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")

        with pytest.raises(ValueError, match="GEMINI_API_KEY or GOOGLE_API_KEY"):
            analyze_single_pr(PR_URL, AnalysisConfig(provider="gemini"))

    def test_explicit_openai_without_key_raises(self, analyze_mocks, monkeypatch):
        """provider='openai' without an OpenAI key fails with a clear error."""
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-env-key")

        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            analyze_single_pr(PR_URL, AnalysisConfig(provider="openai"))

    def test_gemini_keeps_explicit_model_override(self, analyze_mocks, monkeypatch):
        """A caller-supplied Gemini model is not replaced by the default."""
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-env-key")

        analyze_single_pr(PR_URL, AnalysisConfig(provider="gemini", model="gemini-2.5-pro"))

        assert analyze_mocks.call_args.kwargs["model"] == "gemini-2.5-pro"

    def test_gemini_swaps_openai_default_model(self, analyze_mocks, monkeypatch):
        """The OpenAI default model is swapped for the Gemini default under Gemini."""
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-env-key")

        analyze_single_pr(PR_URL, AnalysisConfig(provider="gemini", model=DEFAULT_MODEL))

        assert analyze_mocks.call_args.kwargs["model"] == DEFAULT_GEMINI_MODEL

    def test_openai_keeps_explicit_model_override(self, analyze_mocks, monkeypatch):
        """A caller-supplied OpenAI model is passed through untouched."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")

        analyze_single_pr(PR_URL, AnalysisConfig(provider="openai", model="gpt-4o"))

        assert analyze_mocks.call_args.kwargs["model"] == "gpt-4o"

    def test_result_carries_provider_metadata(self, analyze_mocks, monkeypatch):
        """Provider/model/token metadata from the LLM call reaches the result."""
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-env-key")

        result = analyze_single_pr(PR_URL, AnalysisConfig(provider="gemini"))

        assert result["score"] == 4
        assert result["provider"] == "stub"
        assert result["model"] == "stub-model"
        assert result["tokens"] == 42
        assert result["repo"] == "owner/repo"
        assert result["pr"] == 123
