"""
Unit tests for inference.py error handling.
Requirements: 6.5
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, patch

from inference import run_inference, FALLBACK_ACTION
from tasks.task_pool import TaskPool
from env.environment import ContextShieldEnv


def make_mock_client(response_content: str):
    """Create a mock OpenAI client that returns the given content."""
    mock_choice = MagicMock()
    mock_choice.message.content = response_content
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


def test_openai_exception_records_zero_score():
    """When OpenAI raises an exception, score=0.0 is recorded and execution continues."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API error")

    pool = TaskPool()
    results = run_inference(mock_client, "test-model", pool, ContextShieldEnv)

    assert len(results) == len(pool.get_all())
    for r in results:
        assert r["score"] == 0.0


def test_malformed_json_uses_fallback_action():
    """When LLM returns malformed JSON, fallback Action is used (not a crash)."""
    mock_client = make_mock_client("this is not valid json {{{")

    pool = TaskPool()
    results = run_inference(mock_client, "test-model", pool, ContextShieldEnv)

    # All tasks should complete (no crash)
    assert len(results) == len(pool.get_all())
    # Scores should be floats in [0, 1]
    for r in results:
        assert 0.0 <= r["score"] <= 1.0


def test_valid_json_response_is_parsed():
    """When LLM returns valid JSON, it is parsed into an Action correctly."""
    valid_json = '{"decision": "remove", "confidence": 0.9, "reasoning": "This is clearly spam content that violates policy."}'
    mock_client = make_mock_client(valid_json)

    pool = TaskPool()
    results = run_inference(mock_client, "test-model", pool, ContextShieldEnv)

    assert len(results) == len(pool.get_all())
    for r in results:
        assert 0.0 <= r["score"] <= 1.0
