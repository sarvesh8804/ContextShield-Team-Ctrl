"""
Tests for inference.py baseline script (mocked LLM).
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock

from inference import run_single_task


def make_mock_client(content: str):
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def test_run_single_task_valid_json(capsys):
    """Baseline path completes and emits [START]/[END] logs."""
    payload = (
        '{"decision": "remove", "confidence": 0.9, '
        '"reasoning": "Spam violates marketplace policy in US region; prior_violations noted."}'
    )
    client = make_mock_client(payload)
    run_single_task(client, "test-model", "easy", seed=42)
    out = capsys.readouterr().out
    assert "[START]" in out
    assert "[END]" in out
    assert "task=ContextShield-easy" in out


def test_run_single_task_api_error_still_logs_end(capsys):
    """When the API raises, fallback JSON is used and [END] is still printed."""
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("API unavailable")
    run_single_task(client, "test-model", "medium", seed=0)
    out = capsys.readouterr().out
    assert "[END]" in out


def test_run_single_task_malformed_json_uses_fallback(capsys):
    client = make_mock_client("not json {{{")
    run_single_task(client, "test-model", "hard", seed=1)
    out = capsys.readouterr().out
    assert "[END]" in out
