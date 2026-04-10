"""
Tests for UnitForgeEnv.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from models.action import Action
from models.observation import Observation
from env.environment import ContextShieldEnv
from env.exceptions import EpisodeTerminatedError


def make_action(value: float = 1.0) -> Action:
    return Action(value=value)

def test_reset_clears_state():
    env = ContextShieldEnv(seed=42)
    env.reset()
    env.step(make_action(1.0))
    
    env.reset()
    state = env.state()
    assert state.step_number == 0
    assert state.history == []

def test_step_after_done_raises():
    env = ContextShieldEnv(seed=42)
    env.reset()
    for _ in range(5):
        env.step(make_action(1.0))

    with pytest.raises(EpisodeTerminatedError):
        env.step(make_action(1.0))

def test_step_returns_correct_types():
    env = ContextShieldEnv(seed=0)
    env.reset()
    action = make_action(1.0)
    result = env.step(action)

    obs, reward, done, info = result
    assert isinstance(obs, Observation)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)

def test_step_done_behavior():
    env = ContextShieldEnv(seed=1)
    env.reset()
    for _ in range(4):
        _, _, done, _ = env.step(make_action())
        assert done is False
    
    _, _, done, _ = env.step(make_action())
    assert done is True

def test_step_info_has_required_keys():
    env = ContextShieldEnv(seed=2)
    env.reset()
    _, _, _, info = env.step(make_action())
    assert "task_id" in info
    assert "difficulty" in info

def test_step_observation_step_numbers():
    env = ContextShieldEnv(seed=3)
    env.reset()
    for i in range(1, 6):
        obs, _, _, _ = env.step(make_action())
        assert obs.step_number == i

def test_reset_returns_observation_with_step_number_0():
    env = ContextShieldEnv(seed=4)
    obs = env.reset()
    assert isinstance(obs, Observation)
    assert obs.step_number == 0
