"""CLI integration tests."""

import json
import re
from unittest.mock import patch

from typer.testing import CliRunner

from cli.main import app

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
            assert call_kwargs["model"] == "gemini-flash-latest"

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
                "anthropic",
            ],
        )
        assert result.exit_code != 0
        assert "Unsupported provider" in result.output or "Invalid provider" in result.output

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
