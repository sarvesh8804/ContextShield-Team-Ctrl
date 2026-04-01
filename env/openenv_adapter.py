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
    done: bool = False
    history: list = Field(default_factory=list)
    total_score: float = 0.0


class ContextShieldOpenEnv(Environment[Action, Observation, ContextShieldState]):
    """Thin wrapper exposing ContextShieldEnv through the OpenEnv server API."""

    def __init__(self, difficulty: str | None = None, seed: int | None = None) -> None:
        super().__init__()
        self._inner = ContextShieldEnv(difficulty=difficulty, seed=seed)

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        return self._inner.reset(seed=seed)

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        obs, _reward, _done, _info = self._inner.step(action)
        return obs

    @property
    def state(self) -> ContextShieldState:
        snap = self._inner.state()
        return ContextShieldState(
            episode_id=snap.episode_id,
            step_count=snap.step_number,
            current_task_id=snap.current_task_id,
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
