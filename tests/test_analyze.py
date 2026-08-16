from unittest.mock import patch

import pytest

from cli.analyze import analyze_single_pr, is_automated_sync_pr
from cli.config_types import AnalysisConfig

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
                    analyze_single_pr("https://github.com/owner/repo/pull/123", config)
