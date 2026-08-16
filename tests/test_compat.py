"""Backward-compatibility and interface-preservation tests.

Guards the original (pre-multi-provider) CLI, GitHub Action, and Python API
surfaces so the PydanticAI/Gemini modernization stays strictly additive:
a user with only OPENAI_API_KEY set, legacy flags, or legacy workflow inputs
must see identical behavior to the original tool.
"""

import csv
import inspect
import json
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError
from pydantic_ai.models.test import TestModel
from typer.testing import CliRunner

from cli.config_types import AnalysisConfig, BatchConfig, OutputConfig
from cli.constants import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_HUNKS_PER_FILE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_SLEEP_SECONDS,
    DEFAULT_TIMEOUT,
)
from cli.csv_handler import CSVBatchWriter
from cli.llm import GeminiProvider, LLMError, OpenAIProvider, PydanticAIProvider, get_provider
from cli.main import analyze_pr_to_dict, app
from cli.scoring import ComplexityResult

runner = CliRunner()

REPO_ROOT = Path(__file__).parent.parent

TEST_PR_URL = "https://github.com/owner/repo/pull/123"

FAKE_DIFF = "diff --git a/file.py b/file.py\n+line1"
FAKE_META = {
    "title": "Test PR",
    "additions": 10,
    "deletions": 5,
    "files": [],
    "changed_files": 1,
}
FAKE_LLM_RESULT = {
    "complexity": 5,
    "explanation": "Moderate change across two modules",
    "provider": "openai",
    "model": DEFAULT_MODEL,
    "tokens": 1234,
}


def strip_ansi(text: str) -> str:
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
    return ansi_escape.sub("", text)


class TestLegacyCliInvocation:
    """Pure legacy invocation: only OPENAI_API_KEY set, original flags."""

    @patch("cli.main.fetch_pr")
    @patch("cli.main.get_provider")
    def test_openai_env_only_defaults_to_openai_gpt52(self, mock_get_provider, mock_fetch):
        """With only OPENAI_API_KEY set, provider must be openai and model gpt-5.2."""
        mock_fetch.return_value = (FAKE_DIFF, FAKE_META)
        mock_get_provider.return_value.analyze_complexity.return_value = FAKE_LLM_RESULT

        with patch.dict("os.environ", {"OPENAI_API_KEY": "legacy-env-key"}, clear=True):
            result = runner.invoke(app, ["analyze-pr", TEST_PR_URL])

        assert result.exit_code == 0
        call_kwargs = mock_get_provider.call_args.kwargs
        assert call_kwargs["provider"] == "openai"
        assert call_kwargs["api_key"] == "legacy-env-key"
        assert call_kwargs["model"] == "gpt-5.2"
        assert DEFAULT_MODEL == "gpt-5.2"

    @patch("cli.main.fetch_pr")
    @patch("cli.main.get_provider")
    def test_custom_model_passes_through_unaltered(self, mock_get_provider, mock_fetch):
        """--model <custom> with OpenAI must be forwarded without alteration."""
        mock_fetch.return_value = (FAKE_DIFF, FAKE_META)
        mock_get_provider.return_value.analyze_complexity.return_value = FAKE_LLM_RESULT

        with patch.dict("os.environ", {"OPENAI_API_KEY": "legacy-env-key"}, clear=True):
            result = runner.invoke(app, ["analyze-pr", TEST_PR_URL, "--model", "my-custom-model-x"])

        assert result.exit_code == 0
        assert mock_get_provider.call_args.kwargs["model"] == "my-custom-model-x"

    @patch("cli.main.fetch_pr")
    @patch("cli.main.get_provider")
    def test_direct_url_invocation_without_subcommand(self, mock_get_provider, mock_fetch):
        """`complexity-cli <PR_URL>` (no subcommand) must work seamlessly."""
        mock_fetch.return_value = (FAKE_DIFF, FAKE_META)
        mock_get_provider.return_value.analyze_complexity.return_value = FAKE_LLM_RESULT

        with patch.dict("os.environ", {"OPENAI_API_KEY": "legacy-env-key"}, clear=True):
            result = runner.invoke(app, [TEST_PR_URL])

        assert result.exit_code == 0
        output = json.loads(result.stdout)
        assert output["score"] == 5
        assert output["provider"] == "openai"

    @patch("cli.main.fetch_pr")
    @patch("cli.main.get_provider")
    def test_legacy_flags_still_accepted(self, mock_get_provider, mock_fetch, tmp_path):
        """Original flags (--github-token, --openai-api-key, --format, --output-file,
        --prompt-file) must all still be accepted together."""
        mock_fetch.return_value = (FAKE_DIFF, FAKE_META)
        mock_get_provider.return_value.analyze_complexity.return_value = FAKE_LLM_RESULT

        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Custom prompt")
        out_file = tmp_path / "result.json"

        with patch.dict("os.environ", {}, clear=True):
            result = runner.invoke(
                app,
                [
                    "analyze-pr",
                    TEST_PR_URL,
                    "--openai-api-key",
                    "sk-flag-key",
                    "--github-token",
                    "gh-token",
                    "--format",
                    "json",
                    "--output-file",
                    str(out_file),
                    "--prompt-file",
                    str(prompt_file),
                ],
            )

        assert result.exit_code == 0
        assert mock_get_provider.call_args.kwargs["api_key"] == "sk-flag-key"
        # Output file written with the full result payload
        written = json.loads(out_file.read_text())
        assert written["score"] == 5
        assert written["repo"] == "owner/repo"

    @patch("cli.main.fetch_pr")
    @patch("cli.main.get_provider")
    def test_stdout_stderr_stream_separation(self, mock_get_provider, mock_fetch):
        """stdout must contain only the machine-readable output; progress goes to stderr."""
        mock_fetch.return_value = (FAKE_DIFF, FAKE_META)
        mock_get_provider.return_value.analyze_complexity.return_value = FAKE_LLM_RESULT

        with patch.dict("os.environ", {"OPENAI_API_KEY": "legacy-env-key"}, clear=True):
            result = runner.invoke(app, ["analyze-pr", TEST_PR_URL])

        assert result.exit_code == 0
        # stdout parses cleanly as JSON on its own
        json.loads(result.stdout)
        # progress logs live on stderr, not stdout
        assert "Fetching PR" in result.stderr
        assert "Fetching PR" not in result.stdout

    @patch("cli.main.fetch_pr")
    @patch("cli.main.get_provider")
    def test_openai_env_wins_over_gemini_when_both_set(self, mock_get_provider, mock_fetch):
        """Auto-detection with both env keys present must keep the original OpenAI default."""
        mock_fetch.return_value = (FAKE_DIFF, FAKE_META)
        mock_get_provider.return_value.analyze_complexity.return_value = FAKE_LLM_RESULT

        env = {"OPENAI_API_KEY": "openai-key", "GEMINI_API_KEY": "gemini-key"}
        with patch.dict("os.environ", env, clear=True):
            result = runner.invoke(app, ["analyze-pr", TEST_PR_URL])

        assert result.exit_code == 0
        call_kwargs = mock_get_provider.call_args.kwargs
        assert call_kwargs["provider"] == "openai"
        assert call_kwargs["model"] == DEFAULT_MODEL

    @patch("cli.main.fetch_pr")
    @patch("cli.main.get_provider")
    def test_all_three_env_keys_present_resolves_to_openai(self, mock_get_provider, mock_fetch):
        """When OPENAI_API_KEY, GEMINI_API_KEY, and ANTHROPIC_API_KEY are set, CLI resolves to OpenAI."""
        mock_fetch.return_value = (FAKE_DIFF, FAKE_META)
        mock_get_provider.return_value.analyze_complexity.return_value = FAKE_LLM_RESULT

        env = {
            "OPENAI_API_KEY": "openai-key",
            "GEMINI_API_KEY": "gemini-key",
            "ANTHROPIC_API_KEY": "anthropic-key",
        }
        with patch.dict("os.environ", env, clear=True):
            result = runner.invoke(app, ["analyze-pr", TEST_PR_URL])

        assert result.exit_code == 0
        call_kwargs = mock_get_provider.call_args.kwargs
        assert call_kwargs["provider"] == "openai"
        assert call_kwargs["model"] == DEFAULT_MODEL

    @patch("cli.main.fetch_pr")
    @patch("cli.main.get_provider")
    def test_openai_and_anthropic_keys_present_resolves_to_openai(
        self, mock_get_provider, mock_fetch
    ):
        """When OPENAI_API_KEY and ANTHROPIC_API_KEY are set, CLI resolves to OpenAI."""
        mock_fetch.return_value = (FAKE_DIFF, FAKE_META)
        mock_get_provider.return_value.analyze_complexity.return_value = FAKE_LLM_RESULT

        env = {"OPENAI_API_KEY": "openai-key", "ANTHROPIC_API_KEY": "anthropic-key"}
        with patch.dict("os.environ", env, clear=True):
            result = runner.invoke(app, ["analyze-pr", TEST_PR_URL])

        assert result.exit_code == 0
        call_kwargs = mock_get_provider.call_args.kwargs
        assert call_kwargs["provider"] == "openai"
        assert call_kwargs["model"] == DEFAULT_MODEL

    @patch("cli.main.fetch_pr")
    @patch("cli.main.get_provider")
    def test_gemini_and_anthropic_keys_present_resolves_to_gemini(
        self, mock_get_provider, mock_fetch
    ):
        """When GEMINI_API_KEY and ANTHROPIC_API_KEY are set, CLI resolves to Gemini."""
        mock_fetch.return_value = (FAKE_DIFF, FAKE_META)
        mock_get_provider.return_value.analyze_complexity.return_value = {
            **FAKE_LLM_RESULT,
            "provider": "gemini",
            "model": DEFAULT_GEMINI_MODEL,
        }

        env = {"GEMINI_API_KEY": "gemini-key", "ANTHROPIC_API_KEY": "anthropic-key"}
        with patch.dict("os.environ", env, clear=True):
            result = runner.invoke(app, ["analyze-pr", TEST_PR_URL])

        assert result.exit_code == 0
        call_kwargs = mock_get_provider.call_args.kwargs
        assert call_kwargs["provider"] == "gemini"
        assert call_kwargs["model"] == DEFAULT_GEMINI_MODEL

    @patch("cli.main.fetch_pr")
    @patch("cli.main.get_provider")
    def test_anthropic_key_only_resolves_to_anthropic(self, mock_get_provider, mock_fetch):
        """When only ANTHROPIC_API_KEY is set, the CLI uses the Anthropic default."""
        from cli.constants import DEFAULT_ANTHROPIC_MODEL

        mock_fetch.return_value = (FAKE_DIFF, FAKE_META)
        mock_get_provider.return_value.analyze_complexity.return_value = {
            **FAKE_LLM_RESULT,
            "provider": "anthropic",
            "model": DEFAULT_ANTHROPIC_MODEL,
        }

        env = {"ANTHROPIC_API_KEY": "anthropic-key"}
        with patch.dict("os.environ", env, clear=True):
            result = runner.invoke(app, ["analyze-pr", TEST_PR_URL])

        assert result.exit_code == 0
        call_kwargs = mock_get_provider.call_args.kwargs
        assert call_kwargs["provider"] == "anthropic"
        assert call_kwargs["model"] == DEFAULT_ANTHROPIC_MODEL


class TestActionCompat:
    """GitHub Action interface: legacy input keys and default-value resolution."""

    @pytest.fixture(scope="class")
    def action_text(self):
        return (REPO_ROOT / "action.yml").read_text()

    def test_legacy_input_keys_preserved(self, action_text):
        """Workflows using the original input keys must keep working unmodified."""
        for input_key in (
            "pr-url:",
            "openai-api-key:",
            "github-token:",
            "model:",
            "format:",
            "output-file:",
            "timeout:",
            "max-tokens:",
            "hunks-per-file:",
            "sleep-seconds:",
        ):
            assert input_key in action_text, f"action.yml lost legacy input {input_key!r}"

    def test_legacy_inputs_not_required(self, action_text):
        """No newly-required inputs: legacy minimal workflows must not break."""
        assert "required: true" not in action_text

    def test_action_args_map_inputs_to_cli_flags(self, action_text):
        """The docker args must forward the legacy inputs to the same CLI flags."""
        for flag in (
            "'analyze-pr'",
            "'--openai-api-key'",
            "'--github-token'",
            "'--model'",
            "'--format'",
            "'--output-file'",
            "'--timeout'",
            "'--max-tokens'",
            "'--hunks-per-file'",
            "'--sleep-seconds'",
        ):
            assert flag in action_text, f"action.yml args lost {flag}"

    def test_action_outputs_preserved(self, action_text):
        """The action must keep exposing the original outputs."""
        outputs_section = action_text.split("outputs:", 1)[1]
        for output_key in ("score:", "explanation:", "output:"):
            assert output_key in outputs_section

    @patch("cli.main.fetch_pr")
    @patch("cli.main.get_provider")
    def test_action_default_inputs_simulation_openai(self, mock_get_provider, mock_fetch):
        """Simulate the exact arg vector action.yml passes with default inputs.

        Optional inputs render as empty strings; with only the OpenAI key
        supplied the run must resolve to the openai provider and gpt-5.2.
        """
        mock_fetch.return_value = (FAKE_DIFF, FAKE_META)
        mock_get_provider.return_value.analyze_complexity.return_value = FAKE_LLM_RESULT

        args = [
            "analyze-pr",
            "--provider",
            "auto",
            "--gemini-api-key",
            "",
            "--openai-api-key",
            "workflow-secret-key",
            "--github-token",
            "gh-token",
            "--model",
            "",
            "--format",
            "json",
            "--timeout",
            "120",
            "--max-tokens",
            "50000",
            "--hunks-per-file",
            "2",
            "--sleep-seconds",
            "0.7",
            "--api-base-url",
            "",
            TEST_PR_URL,
        ]
        with patch.dict("os.environ", {}, clear=True):
            result = runner.invoke(app, args)

        assert result.exit_code == 0
        call_kwargs = mock_get_provider.call_args.kwargs
        assert call_kwargs["provider"] == "openai"
        assert call_kwargs["api_key"] == "workflow-secret-key"
        # Empty model input must resolve to the original default model
        assert call_kwargs["model"] == DEFAULT_MODEL

    @patch("cli.main.fetch_pr")
    @patch("cli.main.get_provider")
    def test_action_empty_model_resolves_to_gemini_default(self, mock_get_provider, mock_fetch):
        """With only a Gemini key, the empty model input resolves to the Gemini default."""
        mock_fetch.return_value = (FAKE_DIFF, FAKE_META)
        mock_get_provider.return_value.analyze_complexity.return_value = {
            **FAKE_LLM_RESULT,
            "provider": "gemini",
            "model": DEFAULT_GEMINI_MODEL,
        }

        args = [
            "analyze-pr",
            "--provider",
            "auto",
            "--gemini-api-key",
            "workflow-gemini-key",
            "--openai-api-key",
            "",
            "--model",
            "",
            TEST_PR_URL,
        ]
        with patch.dict("os.environ", {}, clear=True):
            result = runner.invoke(app, args)

        assert result.exit_code == 0
        call_kwargs = mock_get_provider.call_args.kwargs
        assert call_kwargs["provider"] == "gemini"
        assert call_kwargs["model"] == DEFAULT_GEMINI_MODEL

    @patch("cli.main.fetch_pr")
    @patch("cli.main.get_provider")
    def test_github_output_keys_preserved(self, mock_get_provider, mock_fetch, tmp_path):
        """GITHUB_OUTPUT must keep receiving score, explanation, output, and model."""
        mock_fetch.return_value = (FAKE_DIFF, FAKE_META)
        mock_get_provider.return_value.analyze_complexity.return_value = FAKE_LLM_RESULT

        gh_output = tmp_path / "github_output"
        gh_output.write_text("")
        env = {"OPENAI_API_KEY": "legacy-env-key", "GITHUB_OUTPUT": str(gh_output)}
        with patch.dict("os.environ", env, clear=True):
            result = runner.invoke(app, ["analyze-pr", TEST_PR_URL])

        assert result.exit_code == 0
        content = gh_output.read_text()
        assert "score=5\n" in content
        assert "explanation=" in content
        assert "output=" in content
        assert f"model={DEFAULT_MODEL}\n" in content


class TestBatchCsvCompat:
    """batch-analyze flags and CSV output format parity."""

    def test_csv_header_structure(self, tmp_path):
        """The batch CSV header must remain pr_url,complexity,explanation."""
        out_file = tmp_path / "results.csv"
        writer = CSVBatchWriter(out_file)
        writer.add_row(TEST_PR_URL, 5, "Moderate change")
        writer.close()

        lines = out_file.read_text().splitlines()
        assert lines[0] == "pr_url,complexity,explanation"

    @patch("cli.main.analyze_pr_to_dict")
    def test_batch_analyze_short_flags_and_csv_parity(self, mock_analyze, tmp_path):
        """-i/-o short flags work and rows keep the original column mapping."""
        mock_analyze.side_effect = lambda pr_url, **kwargs: {
            "score": 7,
            "explanation": "Complex change",
            "provider": "openai",
            "model": DEFAULT_MODEL,
            "tokens": 100,
            "timestamp": "2024-01-01T00:00:00Z",
            "repo": "owner/repo",
            "pr": 123,
            "url": pr_url,
            "title": "Test PR",
        }

        input_file = tmp_path / "prs.txt"
        input_file.write_text(f"{TEST_PR_URL}\n")
        out_file = tmp_path / "results.csv"

        with patch.dict("os.environ", {"OPENAI_API_KEY": "legacy-env-key"}, clear=True):
            result = runner.invoke(
                app, ["batch-analyze", "-i", str(input_file), "-o", str(out_file)]
            )

        assert result.exit_code == 0
        with out_file.open() as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["pr_url"] == TEST_PR_URL
        assert rows[0]["complexity"] == "7"
        assert rows[0]["explanation"] == "Complex change"

    def test_batch_analyze_requires_output_unless_label(self, tmp_path):
        """Missing --output (without --label) must fail with the original error."""
        input_file = tmp_path / "prs.txt"
        input_file.write_text(f"{TEST_PR_URL}\n")

        with patch.dict("os.environ", {"OPENAI_API_KEY": "legacy-env-key"}, clear=True):
            result = runner.invoke(app, ["batch-analyze", "-i", str(input_file)])

        assert result.exit_code == 1
        assert "--output is required" in result.output


class TestOutputSchemaCompat:
    """JSON and Markdown output schema stability."""

    @patch("cli.main.fetch_pr")
    @patch("cli.main.get_provider")
    def _invoke(self, mock_get_provider, mock_fetch, extra_args=()):
        mock_fetch.return_value = (FAKE_DIFF, FAKE_META)
        mock_get_provider.return_value.analyze_complexity.return_value = FAKE_LLM_RESULT
        with patch.dict("os.environ", {"OPENAI_API_KEY": "legacy-env-key"}, clear=True):
            return runner.invoke(app, ["analyze-pr", TEST_PR_URL, *extra_args])

    def test_json_schema_keys_and_types(self):
        """JSON stdout must keep exactly the original keys with original types."""
        result = self._invoke()
        assert result.exit_code == 0
        output = json.loads(result.stdout)

        assert set(output.keys()) == {
            "score",
            "explanation",
            "provider",
            "model",
            "tokens",
            "timestamp",
        }
        assert isinstance(output["score"], int) and 1 <= output["score"] <= 10
        assert isinstance(output["explanation"], str)
        assert isinstance(output["provider"], str)
        assert isinstance(output["model"], str)
        assert isinstance(output["tokens"], int)
        assert isinstance(output["timestamp"], str) and output["timestamp"].endswith("Z")

    def test_markdown_output_structure(self):
        """Markdown format must keep the original headings and fields."""
        result = self._invoke(extra_args=("--format", "markdown"))
        assert result.exit_code == 0
        assert "# PR Complexity Analysis" in result.stdout
        assert "**Score:** 5/10" in result.stdout
        assert "**Explanation:**" in result.stdout
        assert "- Repository: owner/repo" in result.stdout


class TestErrorAndExitCodeCompat:
    """Error messages and exit codes for invalid inputs."""

    def test_invalid_pr_url_exits_nonzero(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "legacy-env-key"}, clear=True):
            result = runner.invoke(app, ["analyze-pr", "https://gitlab.com/x/y/pull/1"])
        assert result.exit_code == 1
        assert "Invalid PR URL" in result.output

    def test_missing_openai_key_message_preserved(self):
        """Explicit --provider openai without a key keeps the original error text."""
        with patch.dict("os.environ", {}, clear=True):
            result = runner.invoke(app, ["analyze-pr", TEST_PR_URL, "--provider", "openai"])
        assert result.exit_code == 1
        assert "OPENAI_API_KEY environment variable or argument is required" in result.output

    def test_missing_any_key_mentions_openai(self):
        """Auto mode without any key still points legacy users at OPENAI_API_KEY."""
        with patch.dict("os.environ", {}, clear=True):
            result = runner.invoke(app, ["analyze-pr", TEST_PR_URL])
        assert result.exit_code == 1
        assert "OPENAI_API_KEY" in result.output

    def test_nonexistent_prompt_file_exits_nonzero(self, tmp_path):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "legacy-env-key"}, clear=True):
            result = runner.invoke(
                app,
                [
                    "analyze-pr",
                    TEST_PR_URL,
                    "--prompt-file",
                    str(tmp_path / "missing.txt"),
                ],
            )
        assert result.exit_code == 1
        assert "Prompt file not found" in result.output

    def test_no_args_shows_original_usage_error(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "legacy-env-key"}, clear=True):
            result = runner.invoke(app, [])
        assert result.exit_code == 1
        assert "PR URL is required" in result.output


class TestPythonApiCompat:
    """Public Python API signatures and defaults."""

    def test_analyze_pr_to_dict_signature_is_superset_of_original(self):
        """Original parameters keep their names, order, and defaults."""
        params = inspect.signature(analyze_pr_to_dict).parameters
        names = list(params)
        # Original positional interface preserved
        assert names[:4] == ["pr_url", "prompt_text", "github_token", "openai_key"]
        # Original keyword parameters still present
        for legacy in (
            "model",
            "timeout",
            "max_tokens",
            "hunks_per_file",
            "sleep_seconds",
            "progress_callback",
            "token_rotator",
            "base_url",
        ):
            assert legacy in params, f"analyze_pr_to_dict lost parameter {legacy!r}"
        assert params["model"].default == DEFAULT_MODEL
        assert params["provider"].default == "auto"

    def test_analysis_config_defaults_backward_compatible(self):
        config = AnalysisConfig()
        assert config.model == DEFAULT_MODEL
        assert config.timeout == DEFAULT_TIMEOUT
        assert config.max_tokens == DEFAULT_MAX_TOKENS
        assert config.hunks_per_file == DEFAULT_HUNKS_PER_FILE
        assert config.sleep_seconds == DEFAULT_SLEEP_SECONDS
        assert config.provider == "auto"
        assert config.github_token is None
        assert config.openai_key is None
        assert config.gemini_key is None
        assert config.prompt_text is None

    def test_batch_and_output_config_defaults(self):
        batch = BatchConfig()
        assert batch.workers == 1
        assert batch.resume is True
        assert batch.label_prs is False
        assert batch.label_prefix == "complexity:"

        out = OutputConfig()
        assert out.format == "json"
        assert out.write_github_output is True

    def test_constants_defaults_preserved(self):
        assert DEFAULT_MODEL == "gpt-5.2"
        assert DEFAULT_TIMEOUT == 120.0
        assert DEFAULT_MAX_TOKENS == 50000
        assert DEFAULT_HUNKS_PER_FILE == 2
        assert DEFAULT_SLEEP_SECONDS == 0.7

    def test_provider_exposes_name_and_alias(self):
        provider = OpenAIProvider(api_key="sk-test")
        assert provider.provider_name == "openai"
        assert provider.provider == "openai"
        assert provider.model_name == DEFAULT_MODEL
        assert provider.model == DEFAULT_MODEL

        gemini = GeminiProvider(api_key="AIza-test")
        assert gemini.provider_name == "gemini"
        assert gemini.provider == "gemini"
        assert gemini.model_name == DEFAULT_GEMINI_MODEL

    def test_openai_provider_positional_construction(self):
        """The original positional call style OpenAIProvider(key, model) still works."""
        provider = OpenAIProvider("sk-test", "my-model", 30.0)
        assert provider.model == "my-model"
        assert provider.timeout == 30.0

    def test_analyze_complexity_accepts_legacy_retry_kwargs(self):
        """The original max_retries/retry_delay kwargs must still be accepted."""
        test_model = TestModel(custom_output_args={"complexity": 4, "explanation": "Simple change"})
        provider = PydanticAIProvider(provider="openai", model=test_model)
        result = provider.analyze_complexity(
            prompt="Analyze",
            diff_excerpt="diff",
            stats_json="{}",
            title="Title",
            max_retries=3,
            retry_delay=1.0,
        )
        assert result["complexity"] == 4
        assert result["explanation"] == "Simple change"
        assert result["provider"] == "openai"

    def test_analyze_complexity_result_keys(self):
        """The result dict keeps the original keys."""
        test_model = TestModel(custom_output_args={"complexity": 4, "explanation": "Simple change"})
        provider = PydanticAIProvider(provider="openai", model=test_model)
        result = provider.analyze_complexity(
            prompt="Analyze", diff_excerpt="diff", stats_json="{}", title="Title"
        )
        assert {"complexity", "explanation", "provider", "model", "tokens"} <= set(result)

    def test_legacy_llm_imports_still_work(self):
        """`from cli.llm import LLMError, OpenAIProvider` was the original import path."""
        assert issubclass(LLMError, Exception)
        assert issubclass(OpenAIProvider, PydanticAIProvider)
        assert callable(get_provider)

    def test_complexity_result_schema_bounds(self):
        """Structured output schema: score is an int clamped to 1-10."""
        assert ComplexityResult(complexity=1, explanation="min").complexity == 1
        assert ComplexityResult(complexity=10, explanation="max").complexity == 10
        with pytest.raises(ValidationError):
            ComplexityResult(complexity=0, explanation="too low")
        with pytest.raises(ValidationError):
            ComplexityResult(complexity=11, explanation="too high")

    def test_help_keeps_original_commands(self):
        """Top-level help must keep listing the original commands."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.output)
        for command in ("analyze-pr", "batch-analyze", "label-pr", "rate-limit"):
            assert command in output
