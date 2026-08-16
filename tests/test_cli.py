"""CLI integration tests."""

import json
import re
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from cli.constants import DEFAULT_GEMINI_MODEL, DEFAULT_MODEL
from cli.llm import LLMError
from cli.main import app, resolve_provider_credentials

runner = CliRunner()


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


class TestAnalyzePrCommand:
    """Tests for the analyze-pr command."""

    def test_missing_pr_url_shows_help_on_infer_failure(self):
        """Test that missing PR URL attempts to infer from context."""
        result = runner.invoke(app, ["analyze-pr"])
        # Should fail since we can't infer from context
        assert result.exit_code != 0
        assert "Could not infer PR URL" in result.output or "Error" in result.output

    def test_invalid_pr_url(self):
        """Test error for invalid PR URL."""
        result = runner.invoke(app, ["analyze-pr", "https://not-a-valid-url"])
        assert result.exit_code != 0
        assert "Invalid PR URL" in result.output or "Error" in result.output

    def test_invalid_pr_url_gitlab(self):
        """Test error for GitLab URL."""
        result = runner.invoke(app, ["analyze-pr", "https://gitlab.com/owner/repo/pull/123"])
        assert result.exit_code != 0
        assert "Invalid PR URL" in result.output

    def test_missing_openai_key(self):
        """Test error when OpenAI API key is missing."""
        with patch.dict("os.environ", {}, clear=True):
            with patch("cli.main.get_openai_api_key", return_value=None):
                with patch("cli.main.get_github_token", return_value=None):
                    result = runner.invoke(
                        app, ["analyze-pr", "https://github.com/owner/repo/pull/123"]
                    )
                    assert result.exit_code != 0
                    assert "OPENAI_API_KEY" in result.output

    @patch("cli.main.fetch_pr")
    @patch("cli.main.get_openai_api_key")
    @patch("cli.main.get_github_token")
    def test_dry_run_mode(self, mock_token, mock_api_key, mock_fetch):
        """Test dry-run mode."""
        mock_token.return_value = "test-token"
        mock_api_key.return_value = "test-key"
        mock_fetch.return_value = (
            "diff --git a/file.py b/file.py\n+line1",
            {"title": "Test PR", "additions": 10, "deletions": 5, "files": [], "changed_files": 1},
        )

        result = runner.invoke(
            app, ["analyze-pr", "https://github.com/owner/repo/pull/123", "--dry-run"]
        )
        # Dry run should exit with code 0 and contain dry run message
        assert result.exit_code == 0
        assert "dry run" in result.output.lower() or "skipping llm" in result.output.lower()

    def test_help_shows_options(self):
        """Test that help shows available options."""
        result = runner.invoke(app, ["analyze-pr", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "--model" in output
        assert "--timeout" in output
        assert "--dry-run" in output
        assert "--verbose" in output
        assert "--provider" in output or "-p" in output
        assert "--gemini-api-key" in output
        assert "--openai-api-key" in output

    @patch("cli.main.fetch_pr")
    @patch("cli.main.get_provider")
    def test_auto_detect_gemini_provider(self, mock_get_provider, mock_fetch):
        """Test auto-detection selects gemini when GEMINI_API_KEY is present and no OPENAI_API_KEY."""
        mock_fetch.return_value = (
            "diff --git a/file.py b/file.py\n+line1",
            {"title": "Test PR", "additions": 10, "deletions": 5, "files": [], "changed_files": 1},
        )
        mock_prov_instance = mock_get_provider.return_value
        mock_prov_instance.analyze_complexity.return_value = {
            "complexity": 5,
            "explanation": "Medium",
            "provider": "gemini",
            "model": "gemini-flash-latest",
        }

        with patch.dict("os.environ", {"GEMINI_API_KEY": "gemini-secret"}, clear=True):
            result = runner.invoke(app, ["analyze-pr", "https://github.com/owner/repo/pull/123"])
            assert result.exit_code == 0
            mock_get_provider.assert_called_once()
            call_kwargs = mock_get_provider.call_args.kwargs
            assert call_kwargs["provider"] == "gemini"
            assert call_kwargs["api_key"] == "gemini-secret"
            assert call_kwargs["model"] == DEFAULT_GEMINI_MODEL

    @patch("cli.main.fetch_pr")
    @patch("cli.main.get_provider")
    def test_explicit_provider_option(self, mock_get_provider, mock_fetch):
        """Test explicit --provider option is respected."""
        mock_fetch.return_value = (
            "diff --git a/file.py b/file.py\n+line1",
            {"title": "Test PR", "additions": 10, "deletions": 5, "files": [], "changed_files": 1},
        )
        mock_prov_instance = mock_get_provider.return_value
        mock_prov_instance.analyze_complexity.return_value = {
            "complexity": 3,
            "explanation": "Low",
            "provider": "gemini",
            "model": "gemini-flash-latest",
        }

        result = runner.invoke(
            app,
            [
                "analyze-pr",
                "https://github.com/owner/repo/pull/123",
                "--provider",
                "gemini",
                "--gemini-api-key",
                "test-gemini-key",
            ],
        )
        assert result.exit_code == 0
        mock_get_provider.assert_called_once()
        call_kwargs = mock_get_provider.call_args.kwargs
        assert call_kwargs["provider"] == "gemini"
        assert call_kwargs["api_key"] == "test-gemini-key"

    def test_invalid_provider_option(self):
        """Test invalid --provider option raises clear error."""
        result = runner.invoke(
            app,
            [
                "analyze-pr",
                "https://github.com/owner/repo/pull/123",
                "--provider",
                "cohere",
            ],
        )
        assert result.exit_code != 0
        assert "Unsupported provider" in result.output or "Invalid provider" in result.output

    @patch("cli.main.fetch_pr")
    @patch("cli.main.get_provider")
    def test_explicit_anthropic_provider_option(self, mock_get_provider, mock_fetch):
        """Test --provider anthropic with --anthropic-api-key routes to the Anthropic provider."""
        mock_fetch.return_value = (
            "diff --git a/file.py b/file.py\n+line1",
            {"title": "Test PR", "additions": 10, "deletions": 5, "files": [], "changed_files": 1},
        )
        mock_get_provider.return_value.analyze_complexity.return_value = {
            "complexity": 7,
            "explanation": "High",
            "provider": "anthropic",
            "model": "claude-3-7-sonnet-latest",
        }

        result = runner.invoke(
            app,
            [
                "analyze-pr",
                "https://github.com/owner/repo/pull/123",
                "--provider",
                "anthropic",
                "--anthropic-api-key",
                "sk-ant-test-key",
            ],
        )
        assert result.exit_code == 0
        call_kwargs = mock_get_provider.call_args.kwargs
        assert call_kwargs["provider"] == "anthropic"
        assert call_kwargs["api_key"] == "sk-ant-test-key"

    @patch("cli.main.fetch_pr")
    @patch("cli.main.get_provider")
    def test_explicit_gemini_key_overrides_openai_env(self, mock_get_provider, mock_fetch):
        """Test explicit --gemini-api-key on CLI takes precedence over ambient OPENAI_API_KEY env var."""
        mock_fetch.return_value = (
            "diff --git a/file.py b/file.py\n+line1",
            {"title": "Test PR", "additions": 10, "deletions": 5, "files": [], "changed_files": 1},
        )
        mock_prov_instance = mock_get_provider.return_value
        mock_prov_instance.analyze_complexity.return_value = {
            "complexity": 4,
            "explanation": "Medium",
            "provider": "gemini",
            "model": "gemini-flash-latest",
        }

        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-ambient-secret"}, clear=True):
            result = runner.invoke(
                app,
                [
                    "analyze-pr",
                    "https://github.com/owner/repo/pull/123",
                    "--gemini-api-key",
                    "AIza-explicit-key",
                ],
            )
            assert result.exit_code == 0
            mock_get_provider.assert_called_once()
            call_kwargs = mock_get_provider.call_args.kwargs
            assert call_kwargs["provider"] == "gemini"
            assert call_kwargs["api_key"] == "AIza-explicit-key"

    @patch("cli.main.fetch_pr")
    @patch("cli.main.get_provider")
    def test_explicit_openai_provider_option(self, mock_get_provider, mock_fetch):
        """Test --provider openai with --openai-api-key routes to the OpenAI provider."""
        mock_fetch.return_value = (
            "diff --git a me.py b/me.py\n+line1",
            {"title": "Test PR", "additions": 10, "deletions": 5, "files": [], "changed_files": 1},
        )
        mock_get_provider.return_value.analyze_complexity.return_value = {
            "complexity": 6,
            "explanation": "Medium",
            "provider": "openai",
            "model": DEFAULT_MODEL,
        }

        result = runner.invoke(
            app,
            [
                "analyze-pr",
                "https://github.com/owner/repo/pull/123",
                "--provider",
                "openai",
                "--openai-api-key",
                "sk-test-key",
            ],
        )
        assert result.exit_code == 0
        call_kwargs = mock_get_provider.call_args.kwargs
        assert call_kwargs["provider"] == "openai"
        assert call_kwargs["api_key"] == "sk-test-key"
        assert call_kwargs["model"] == DEFAULT_MODEL

    @patch("cli.main.fetch_pr")
    @patch("cli.main.get_provider")
    def test_gemini_provider_keeps_model_override(self, mock_get_provider, mock_fetch):
        """Test --model is honoured alongside --provider gemini."""
        mock_fetch.return_value = (
            "diff --git a/file.py b/file.py\n+line1",
            {"title": "Test PR", "additions": 10, "deletions": 5, "files": [], "changed_files": 1},
        )
        mock_get_provider.return_value.analyze_complexity.return_value = {
            "complexity": 2,
            "explanation": "Low",
            "provider": "gemini",
            "model": "gemini-2.5-pro",
        }

        result = runner.invoke(
            app,
            [
                "analyze-pr",
                "https://github.com/owner/repo/pull/123",
                "--provider",
                "gemini",
                "--gemini-api-key",
                "AIza-test-key",
                "--model",
                "gemini-2.5-pro",
            ],
        )
        assert result.exit_code == 0
        assert mock_get_provider.call_args.kwargs["model"] == "gemini-2.5-pro"

    def test_explicit_gemini_provider_without_gemini_key_fails_preflight(self):
        """Test --provider gemini without a Gemini key fails pre-flight with a clean error."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-ambient"}, clear=True):
            result = runner.invoke(
                app,
                [
                    "analyze-pr",
                    "https://github.com/owner/repo/pull/123",
                    "--provider",
                    "gemini",
                ],
            )
            assert result.exit_code != 0
            assert "GEMINI_API_KEY" in result.output

    def test_explicit_openai_provider_without_openai_key_fails_preflight(self):
        """Test explicit --provider openai without OpenAI key fails pre-flight with clean error."""
        with patch.dict("os.environ", {"GEMINI_API_KEY": "AIza-secret"}, clear=True):
            result = runner.invoke(
                app,
                [
                    "analyze-pr",
                    "https://github.com/owner/repo/pull/123",
                    "--provider",
                    "openai",
                ],
            )
            assert result.exit_code != 0
            assert "OPENAI_API_KEY" in result.output

    def test_explicit_anthropic_provider_without_anthropic_key_fails_preflight(self):
        """Test explicit --provider anthropic without Anthropic key fails pre-flight with clean error."""
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-ambient"}, clear=True):
            result = runner.invoke(
                app,
                [
                    "analyze-pr",
                    "https://github.com/owner/repo/pull/123",
                    "--provider",
                    "anthropic",
                ],
            )
            assert result.exit_code != 0
            assert "ANTHROPIC_API_KEY" in result.output


class TestRateLimitCommand:
    """Tests for the rate-limit command."""

    @patch("cli.main.check_rate_limit")
    def test_json_output(self, mock_check):
        """Test rate-limit command JSON output."""
        mock_check.return_value = {
            "core": {"limit": 5000, "remaining": 4999, "reset": 1234567890, "used": 1},
            "search": {"limit": 30, "remaining": 30, "reset": 1234567890, "used": 0},
        }

        result = runner.invoke(app, ["rate-limit", "--format", "json"])
        assert result.exit_code == 0

        # Parse JSON output
        output = json.loads(result.output)
        assert "core" in output
        assert "search" in output
        assert output["core"]["limit"] == 5000

    @patch("cli.main.check_rate_limit")
    def test_human_output(self, mock_check):
        """Test rate-limit command human-readable output."""
        mock_check.return_value = {
            "core": {"limit": 5000, "remaining": 4999, "reset": 1234567890, "used": 1},
            "search": {"limit": 30, "remaining": 30, "reset": 1234567890, "used": 0},
        }

        result = runner.invoke(app, ["rate-limit", "--format", "human"])
        assert result.exit_code == 0
        assert "Core API" in result.output
        assert "Search API" in result.output
        assert "5000" in result.output

    def test_help_shows_format_option(self):
        """Test that help shows format option."""
        result = runner.invoke(app, ["rate-limit", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "--format" in output


class TestBatchAnalyzeCommand:
    """Tests for the batch-analyze command."""

    def test_missing_required_options(self):
        """Test error when required options are missing."""
        result = runner.invoke(app, ["batch-analyze"])
        assert result.exit_code != 0
        assert "Must specify" in result.output or "Error" in result.output

    def test_conflicting_options(self):
        """Test error when conflicting options are provided."""
        result = runner.invoke(
            app,
            [
                "batch-analyze",
                "--input-file",
                "prs.txt",
                "--org",
                "testorg",
                "--since",
                "2024-01-01",
                "--until",
                "2024-01-31",
            ],
        )
        assert result.exit_code != 0
        assert "Cannot specify both" in result.output

    def test_help_shows_options(self):
        """Test that help shows available options."""
        result = runner.invoke(app, ["batch-analyze", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "--input-file" in output
        assert "--org" in output
        assert "--since" in output
        assert "--until" in output
        assert "--workers" in output
        assert "--label" in output

    @patch("cli.main.resolve_provider_credentials")
    @patch("cli.main.load_pr_urls_from_file")
    @patch("cli.main.run_batch_analysis_with_labels")
    def test_batch_analyze_gemini_provider(self, mock_batch, mock_urls, mock_resolve):
        """Test batch-analyze with gemini provider and key."""
        mock_urls.return_value = ["https://github.com/owner/repo/pull/1"]
        result = runner.invoke(
            app,
            [
                "batch-analyze",
                "--input-file",
                "prs.txt",
                "--output",
                "out.csv",
                "--provider",
                "gemini",
                "--gemini-api-key",
                "test-gemini-key",
            ],
        )
        assert result.exit_code == 0
        mock_resolve.assert_called_once()
        call_kwargs = mock_resolve.call_args.kwargs
        assert call_kwargs["provider"] == "gemini"
        assert call_kwargs["gemini_api_key"] == "test-gemini-key"

    @patch("cli.main.resolve_provider_credentials")
    @patch("cli.main.load_pr_urls_from_file")
    @patch("cli.main.run_batch_analysis_with_labels")
    def test_batch_analyze_openai_provider(self, mock_batch, mock_urls, mock_resolve):
        """Test batch-analyze with openai provider and key."""
        mock_urls.return_value = ["https://github.com/owner/repo/pull/1"]
        result = runner.invoke(
            app,
            [
                "batch-analyze",
                "--input-file",
                "prs.txt",
                "--output",
                "out.csv",
                "--provider",
                "openai",
                "--openai-api-key",
                "sk-test-key",
            ],
        )
        assert result.exit_code == 0
        call_kwargs = mock_resolve.call_args.kwargs
        assert call_kwargs["provider"] == "openai"
        assert call_kwargs["openai_api_key"] == "sk-test-key"

    def test_help_shows_provider_options(self):
        """Test that batch-analyze help advertises the multi-provider options."""
        result = runner.invoke(app, ["batch-analyze", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "--provider" in output
        assert "--gemini-api-key" in output
        assert "--openai-api-key" in output


class TestLabelPrCommand:
    """Tests for the label-pr command."""

    def test_missing_github_token(self):
        """Test error when GitHub token is missing."""
        with patch("cli.main.get_github_token", return_value=None):
            with patch("cli.main.get_openai_api_key", return_value="test-key"):
                result = runner.invoke(app, ["label-pr", "https://github.com/owner/repo/pull/123"])
                assert result.exit_code != 0
                assert "token" in result.output.lower()

    def test_help_shows_options(self):
        """Test that help shows available options."""
        result = runner.invoke(app, ["label-pr", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        assert "--label-prefix" in output
        assert "--dry-run" in output
        assert "--provider" in output
        assert "--gemini-api-key" in output
        assert "--openai-api-key" in output


class TestResolveProviderCredentials:
    """Tests for the shared provider/credential resolution used by every command."""

    def test_auto_selects_gemini_from_env(self, monkeypatch):
        """GEMINI_API_KEY alone resolves to the Gemini provider."""
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-env")
        assert resolve_provider_credentials("auto") == ("gemini", "AIza-env")

    def test_auto_selects_openai_from_env(self, monkeypatch):
        """OPENAI_API_KEY alone resolves to the OpenAI provider."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        assert resolve_provider_credentials("auto") == ("openai", "sk-env")

    def test_auto_prefers_openai_when_both_env_keys_present(self, monkeypatch):
        """With both env keys set, OpenAI wins."""
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-env")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        assert resolve_provider_credentials("auto") == ("openai", "sk-env")

    def test_auto_prefers_explicit_argument_over_env(self, monkeypatch):
        """An explicit gemini_api_key argument beats an ambient OpenAI env key."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        assert resolve_provider_credentials("auto", gemini_api_key="AIza-arg") == (
            "gemini",
            "AIza-arg",
        )

    def test_auto_selects_anthropic_from_env(self, monkeypatch):
        """ANTHROPIC_API_KEY alone resolves to the Anthropic provider."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
        assert resolve_provider_credentials("auto") == ("anthropic", "sk-ant-env")

    def test_auto_selects_openai_when_all_three_keys_present(self, monkeypatch):
        """When all three API keys are set in environment, OpenAI takes precedence."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-env")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
        assert resolve_provider_credentials("auto") == ("openai", "sk-env")

    def test_auto_selects_openai_when_openai_and_anthropic_keys_present(self, monkeypatch):
        """When OPENAI_API_KEY and ANTHROPIC_API_KEY are set, OpenAI takes precedence."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
        assert resolve_provider_credentials("auto") == ("openai", "sk-env")

    def test_auto_selects_gemini_when_gemini_and_anthropic_keys_present(self, monkeypatch):
        """When GEMINI_API_KEY and ANTHROPIC_API_KEY are set, Gemini takes precedence."""
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-env")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
        assert resolve_provider_credentials("auto") == ("gemini", "AIza-env")

    def test_auto_without_any_key_raises(self):
        """No credentials at all is a ValueError naming env vars."""
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            resolve_provider_credentials("auto")

    def test_google_alias_resolves_to_gemini(self, monkeypatch):
        """The 'google' alias maps onto the Gemini provider."""
        monkeypatch.setenv("GOOGLE_API_KEY", "AIza-env")
        assert resolve_provider_credentials("google") == ("gemini", "AIza-env")

    def test_openai_chat_alias_resolves_to_openai(self, monkeypatch):
        """The 'openai-chat' alias maps onto the OpenAI provider."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        assert resolve_provider_credentials("openai-chat") == ("openai", "sk-env")

    def test_claude_alias_resolves_to_anthropic(self, monkeypatch):
        """The 'claude' alias maps onto the Anthropic provider."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env")
        assert resolve_provider_credentials("claude") == ("anthropic", "sk-ant-env")

    def test_provider_name_is_normalized(self, monkeypatch):
        """Provider names are lowercased and stripped before resolution."""
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-env")
        assert resolve_provider_credentials("  Gemini ") == ("gemini", "AIza-env")

    def test_explicit_gemini_without_key_raises(self, monkeypatch):
        """provider='gemini' never falls back to an OpenAI or Anthropic key."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        with pytest.raises(ValueError, match="GEMINI_API_KEY or GOOGLE_API_KEY"):
            resolve_provider_credentials("gemini")

    def test_explicit_openai_without_key_raises(self, monkeypatch):
        """provider='openai' never falls back to a Gemini or Anthropic key."""
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-env")
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            resolve_provider_credentials("openai")

    def test_explicit_anthropic_without_key_raises(self, monkeypatch):
        """provider='anthropic' never falls back to another key."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            resolve_provider_credentials("anthropic")

    def test_unsupported_provider_raises_llm_error(self):
        """An unknown provider name is rejected up front."""
        with pytest.raises(LLMError, match="Unsupported provider: 'unknown_prov'"):
            resolve_provider_credentials("unknown_prov")

    def test_none_provider_defaults_to_auto(self, monkeypatch):
        """A None provider is treated as 'auto'."""
        monkeypatch.setenv("GEMINI_API_KEY", "AIza-env")
        assert resolve_provider_credentials(None) == ("gemini", "AIza-env")


class TestMainCallback:
    """Tests for the main callback (direct URL invocation)."""

    def test_direct_url_invalid(self):
        """Test direct URL invocation with invalid URL."""
        result = runner.invoke(app, ["not-a-url"])
        assert result.exit_code != 0
        # Typer may interpret this as a command, so check for either error message
        assert "Invalid PR URL" in result.output or "No such command" in result.output

    def test_no_args_shows_usage(self):
        """Test no arguments shows usage message."""
        result = runner.invoke(app, [])
        assert result.exit_code != 0
        assert "PR URL is required" in result.output or "Usage" in result.output
