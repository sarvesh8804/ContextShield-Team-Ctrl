"""
ContextShieldEnv: OpenEnv-compliant content moderation simulation environment.
Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 8.1
"""
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.action import Action
from models.observation import Observation
from models.state import EpisodeState
from models.task import Task
from tasks.task_pool import TaskPool
from env.state import StateManager
from env.reward import RewardFunction
from env.exceptions import EpisodeTerminatedError
from graders.easy import EasyGrader
from graders.medium import MediumGrader
from graders.hard import HardGrader

_GRADERS = {
    "easy": EasyGrader,
    "medium": MediumGrader,
    "hard": HardGrader,
}


class ContextShieldEnv:
    def __init__(self, difficulty: str | None = None, seed: int | None = None) -> None:
        self.difficulty = difficulty
        if seed is not None:
            random.seed(seed)
        self._task_pool = TaskPool()
        self._state_manager = StateManager()
        self._reward_fn = RewardFunction()
        self._current_task: Task | None = None

    def reset(self) -> Observation:
        task = self._task_pool.sample(self.difficulty)
        self._current_task = task
        self._state_manager.reset(task)
        return Observation(
            content=task.content,
            platform=task.platform,
            region=task.region,
            user_history=task.user_history,
            task_id=task.task_id,
            difficulty=task.difficulty,
            step_number=0,
        )

    def step(self, action: Action) -> tuple[Observation, object, bool, dict]:
        if self._state_manager.done:
            raise EpisodeTerminatedError("Episode has already terminated. Call reset() to start a new episode.")

        task = self._current_task
        grader = _GRADERS[task.difficulty]()
        grader_score = grader.grade(action, task)
        reward = self._reward_fn.compute(grader_score, action, task)

        self._state_manager.record_step(action, reward)
        self._state_manager.mark_done()

        terminal_obs = Observation(
            content=task.content,
            platform=task.platform,
            region=task.region,
            user_history=task.user_history,
            task_id=task.task_id,
            difficulty=task.difficulty,
            step_number=1,
        )

        return (
            terminal_obs,
            reward,
            True,
            {"task_id": task.task_id, "ground_truth": task.ground_truth},
        )

    def state(self) -> EpisodeState:
        return self._state_manager.snapshot()
