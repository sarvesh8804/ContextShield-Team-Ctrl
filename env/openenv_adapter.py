"""
OpenEnv `Environment` adapter wrapping ContextShieldEnv for HTTP/WebSocket serving.
"""
from __future__ import annotations

from typing import Any, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata, State as BaseState
from pydantic import Field

from env.environment import ContextShieldEnv
from models.action import Action
from models.observation import Observation


class ContextShieldState(BaseState):
    current_task_id: str | None = None
    items_in_episode: int = 0
    done: bool = False
    history: list = Field(default_factory=list)
    total_score: float = 0.0


class ContextShieldOpenEnv(Environment[Action, Observation, ContextShieldState]):
    """Thin wrapper exposing ContextShieldEnv through the OpenEnv server API."""

    def __init__(self, difficulty: str | None = None, seed: int | None = None) -> None:
        super().__init__()
        self.difficulty = difficulty
        self.seed = seed
        self._episodes: dict[str, ContextShieldEnv] = {}
        self._last_env: ContextShieldEnv | None = None

    def _get_env(self, episode_id: str | None) -> ContextShieldEnv:
        eid = episode_id or "default"
        if eid not in self._episodes:
            self._episodes[eid] = ContextShieldEnv(
                difficulty=self.difficulty,
                seed=self.seed,
            )
            self._episodes[eid].reset()
        env = self._episodes[eid]
        self._last_env = env
        return env

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        eid = episode_id or "default"
        env = ContextShieldEnv(difficulty=self.difficulty, seed=seed)
        self._episodes[eid] = env
        self._last_env = env
        return env.reset(seed=seed)

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        eid = kwargs.get("episode_id", "default")
        env = self._get_env(eid)
        obs, _reward, _done, _info = env.step(action)
        return obs

    @property
    def state(self) -> ContextShieldState:
        env = self._last_env
        if not env:
            env = self._get_env("default")
        snap = env.state()
        return ContextShieldState(
            episode_id=snap.episode_id,
            step_count=snap.step_number,
            current_task_id=snap.current_task_id,
            items_in_episode=snap.items_in_episode,
            done=snap.done,
            history=list(snap.history),
            total_score=snap.total_score,
        )

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="context-shield",
            description=(
                "Context-aware content moderation simulation. "
                "Agents choose allow, remove, or escalate with calibrated reasoning."
            ),
            version="1.0.0",
        )
