"""
OpenEnv adapter: wraps PatchGymEnv for HTTP serving.
"""
from __future__ import annotations
from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata, State as BaseState
from pydantic import Field

from env.environment import PatchGymEnv
from models.action import Action
from models.observation import Observation


class PatchGymState(BaseState):
    current_task_id: str | None = None
    done: bool = False
    history: list = Field(default_factory=list)
    total_score: float = 0.05


class PatchGymOpenEnv(Environment[Action, Observation, PatchGymState]):
    def __init__(self, difficulty: str | None = None, seed: int | None = None) -> None:
        super().__init__()
        self.difficulty = difficulty
        self.seed = seed
        self._episodes: dict[str, PatchGymEnv] = {}
        self._last_env: PatchGymEnv | None = None

    def _get_env(self, eid: str) -> PatchGymEnv:
        if eid not in self._episodes:
            env = PatchGymEnv(difficulty=self.difficulty, seed=self.seed)
            env.reset()
            self._episodes[eid] = env
        self._last_env = self._episodes[eid]
        return self._last_env

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any) -> Observation:
        eid = episode_id or "default"
        env = PatchGymEnv(difficulty=self.difficulty, seed=seed)
        self._episodes[eid] = env
        self._last_env = env
        return env.reset(seed=seed)

    def step(self, action: Action, timeout_s: Optional[float] = None, **kwargs: Any) -> Observation:
        eid = kwargs.get("episode_id", "default")
        env = self._get_env(eid)
        obs, _reward, _done, _info = env.step(action)
        return obs

    @property
    def state(self) -> PatchGymState:
        env = self._last_env or self._get_env("default")
        snap = env.state()
        return PatchGymState(
            episode_id=snap.episode_id,
            step_count=snap.step_number,
            current_task_id=snap.current_task_id,
            done=snap.done,
            history=list(snap.history),
            total_score=snap.total_score,
        )

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="patch-gym",
            description=(
                "Dependency vulnerability triage environment. "
                "Agents use tool calls to assess CVEs, plan safe fixes, and resolve dependency conflicts."
            ),
            version="1.0.0",
        )
