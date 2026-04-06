"""
Tests for ContextShieldEnv.
- Property 7: Reset clears state (Validates: Requirements 1.4)
- Property 8: Step after done raises EpisodeTerminatedError (Validates: Requirements 1.5)
- Unit tests for step() return structure (Requirements 1.2)
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from models.action import Action
from models.observation import Observation
from models.reward import Reward
from env.environment import ContextShieldEnv
from env.exceptions import EpisodeTerminatedError

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

valid_action_strategy = st.builds(
    Action,
    decision=st.sampled_from(["allow", "remove", "escalate"]),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    reasoning=st.text(min_size=1, max_size=100),
)


def make_action(**kwargs) -> Action:
    defaults = dict(decision="allow", confidence=0.8, reasoning="This content looks fine.")
    defaults.update(kwargs)
    return Action(**defaults)


# ---------------------------------------------------------------------------
# Property 7: Reset clears state
# Validates: Requirements 1.4
# ---------------------------------------------------------------------------

@given(action=valid_action_strategy)
@settings(max_examples=50)
def test_reset_clears_state(action):
    """**Validates: Requirements 1.4**
    After running a step and calling reset(), step_number == 0 and history == [].
    """
    env = ContextShieldEnv(seed=42)
    env.reset()
    env.step(action)

    # Now reset and verify clean state
    env.reset()
    state = env.state()
    assert state.step_number == 0
    assert state.history == []


# ---------------------------------------------------------------------------
# Property 8: Step after done raises EpisodeTerminatedError
# Validates: Requirements 1.5
# ---------------------------------------------------------------------------

@given(action=valid_action_strategy)
@settings(max_examples=50)
def test_step_after_done_raises(action):
    """**Validates: Requirements 1.5**
    After 5 steps (episode termination), calling step() again raises EpisodeTerminatedError.
    """
    env = ContextShieldEnv(seed=42)
    env.reset()
    for _ in range(5):
        env.step(action)

    with pytest.raises(EpisodeTerminatedError):
        env.step(action)


# ---------------------------------------------------------------------------
# Task 8.2: Unit tests for step() return structure
# Requirements: 1.2
# ---------------------------------------------------------------------------

def test_step_returns_correct_types():
    """step() returns (Observation, Reward, bool, dict) with correct types."""
    env = ContextShieldEnv(seed=0)
    env.reset()
    action = make_action(decision="remove", confidence=0.9, reasoning="Clear spam content that violates policy.")
    result = env.step(action)

    obs, reward, done, info = result
    assert isinstance(obs, Observation)
    assert isinstance(reward, Reward)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_step_done_behavior():
    """done flag is True exactly after the 5th step."""
    env = ContextShieldEnv(seed=1)
    env.reset()
    for _ in range(4):
        _, _, done, _ = env.step(make_action())
        assert done is False
    
    _, _, done, _ = env.step(make_action())
    assert done is True


def test_step_info_has_required_keys():
    """info dict contains 'task_id' and 'ground_truth' keys."""
    env = ContextShieldEnv(seed=2)
    env.reset()
    _, _, _, info = env.step(make_action())
    assert "task_id" in info
    assert "ground_truth" in info


def test_step_observation_step_numbers():
    """Observations have incrementing step numbers."""
    env = ContextShieldEnv(seed=3)
    env.reset()
    for i in range(1, 6):
        obs, _, _, _ = env.step(make_action())
        assert obs.step_number == i


def test_reset_returns_observation_with_step_number_0():
    """reset() returns an Observation with step_number == 0."""
    env = ContextShieldEnv(seed=4)
    obs = env.reset()
    assert isinstance(obs, Observation)
    assert obs.step_number == 0
