"""
Tests for PatchGymEnv.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from models.action import Action
from models.observation import Observation
from env.environment import PatchGymEnv
from env.exceptions import EpisodeTerminatedError


def make_action(command="list_packages", args=None) -> Action:
    return Action(command=command, args=args or {})


def test_reset_returns_observation():
    env = PatchGymEnv(seed=42)
    obs = env.reset()
    assert isinstance(obs, Observation)
    assert obs.step_number == 0
    assert obs.total_reward == 0.05


def test_list_packages_returns_dict():
    env = PatchGymEnv(seed=42)
    env.reset()
    obs, reward, done, info = env.step(make_action("list_packages"))
    assert isinstance(obs.result, dict)
    assert done is False


def test_show_cve_invalid_returns_error():
    env = PatchGymEnv(seed=42)
    env.reset()
    obs, reward, done, info = env.step(make_action("show_cve", {"cve_id": "CVE-FAKE-9999"}))
    assert obs.error is not None
    assert reward < 0  # penalty for wasted call


def test_check_imports_valid():
    env = PatchGymEnv(seed=42)
    env.reset()
    # list_packages first to know what's in requirements
    env.step(make_action("list_packages"))
    # pick a cve to find a valid package name
    task = env._task
    pkg = list(task.requirements.keys())[0]
    obs, reward, done, info = env.step(make_action("check_imports", {"package_name": pkg}))
    assert obs.error is None
    assert "imported" in obs.result


def test_step_after_done_raises():
    env = PatchGymEnv(difficulty="easy", seed=42)
    env.reset()
    # submit immediately to trigger done
    env.step(make_action("submit_plan", {"ranking": []}))
    with pytest.raises(EpisodeTerminatedError):
        env.step(make_action("list_packages"))


def test_submit_plan_ends_episode():
    env = PatchGymEnv(difficulty="easy", seed=42)
    obs = env.reset()
    task = env._task
    ranking = [c.cve_id for c in task.cves]
    obs, reward, done, info = env.step(make_action("submit_plan", {"ranking": ranking}))
    assert done is True


def test_state_reflects_steps():
    env = PatchGymEnv(seed=42)
    env.reset()
    env.step(make_action("list_packages"))
    env.step(make_action("list_packages"))
    state = env.state()
    assert state.step_number == 2


def test_check_conflicts_conflict_trap_rewards():
    env = PatchGymEnv(difficulty="hard", seed=42)
    env.reset()
    task = env._task
    trap = task.conflict_trap
    if trap:
        obs, reward, done, info = env.step(make_action(
            "check_conflicts",
            {"package": trap["package"], "version": trap["naive_fix"]}
        ))
        assert obs.result["conflict"] is True
        assert abs(reward - 0.10) < 1e-6  # correct: found the trap
