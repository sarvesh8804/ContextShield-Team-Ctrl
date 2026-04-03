"""
ContextShieldEnv: OpenEnv-compliant content moderation simulation environment.
Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 8.1
"""
import os
import random
import sys

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

# Risk level derived from ground_truth + difficulty
_RISK_MAP = {
    ("allow",    "easy"):   "low",
    ("allow",    "medium"): "low",
    ("allow",    "hard"):   "medium",
    ("escalate", "easy"):   "medium",
    ("escalate", "medium"): "medium",
    ("escalate", "hard"):   "high",
    ("remove",   "easy"):   "medium",
    ("remove",   "medium"): "high",
    ("remove",   "hard"):   "high",
}


def _infer_risk_level(task: Task) -> str:
    return _RISK_MAP.get((task.ground_truth, task.difficulty), "medium")


def _context_factors_used(action: Action, task: Task) -> list[str]:
    """Return which context keywords from the task appear in the action reasoning."""
    lowered = action.reasoning.lower()
    return [kw for kw in task.context_keywords if kw.lower() in lowered]


def _default_episode_length() -> int:
    raw = os.environ.get("CONTEXT_SHIELD_EPISODE_LENGTH", "4")
    try:
        n = int(raw)
    except ValueError:
        return 4
    return max(1, min(n, 32))


class ContextShieldEnv:
    def __init__(
        self,
        difficulty: str | None = None,
        seed: int | None = None,
        episode_length: int | None = None,
    ) -> None:
        self.difficulty = difficulty
        if seed is not None:
            random.seed(seed)
        self.episode_length = (
            episode_length if episode_length is not None else _default_episode_length()
        )
        self._task_pool = TaskPool()
        self._state_manager = StateManager()
        self._reward_fn = RewardFunction()
        self._episode_tasks: list[Task] = []
        self._cursor: int = 0
        self._current_task: Task | None = None

    def reset(self, seed: int | None = None) -> Observation:
        tasks = self._task_pool.sample_episode(
            self.difficulty,
            seed=seed,
            length=self.episode_length,
        )
        self._episode_tasks = tasks
        self._cursor = 0
        self._current_task = tasks[0]
        self._state_manager.reset(tasks)
        return self._obs_from_task(
            tasks[0],
            step_number=0,
            items_in_episode=len(tasks),
            reward=None,
            done=False,
        )

    def _ensure_episode(self) -> None:
        if self._current_task is None:
            self.reset()

    def step(self, action: Action) -> tuple[Observation, object, bool, dict]:
        self._ensure_episode()

        if self._state_manager.done:
            raise EpisodeTerminatedError(
                "Episode has already terminated. Call reset() to start a new episode."
            )

        task = self._episode_tasks[self._cursor]
        grader = _GRADERS[task.difficulty]()
        grader_score = grader.grade(action, task)
        reward = self._reward_fn.compute(grader_score, action, task)

        self._state_manager.record_step(action, reward)

        n = len(self._episode_tasks)
        meta = {
            "task_id": task.task_id,
            "ground_truth": task.ground_truth,
            "policy_reason": task.explanation,
            "risk_level": _infer_risk_level(task),
            "context_factors_used": _context_factors_used(action, task),
            "episode_progress": f"{self._state_manager.step_number}/{n}",
        }
        info = {k: v for k, v in meta.items() if k != "episode_progress"}

        self._cursor += 1
        if self._cursor < n:
            self._current_task = self._episode_tasks[self._cursor]
            self._state_manager.current_task_id = self._current_task.task_id
            next_obs = self._obs_from_task(
                self._current_task,
                step_number=self._state_manager.step_number,
                items_in_episode=n,
                reward=reward.score,
                done=False,
                metadata=meta,
            )
            return (next_obs, reward, False, info)

        self._state_manager.mark_done()
        terminal_obs = self._obs_from_task(
            task,
            step_number=self._state_manager.step_number,
            items_in_episode=n,
            reward=reward.score,
            done=True,
            metadata=meta,
        )
        return (terminal_obs, reward, True, info)

    def state(self) -> EpisodeState:
        return self._state_manager.snapshot()

    # ------------------------------------------------------------------
    # What-if Context Simulation
    # ------------------------------------------------------------------

    def simulate_context_variation(
        self,
        task_id: str,
        new_context: dict,
    ) -> dict:
        """Re-evaluate a task under modified context without mutating episode state.

        Args:
            task_id: ID of the task to vary.
            new_context: dict with any subset of keys:
                         platform, region, user_history

        Returns:
            {
                "original_observation": Observation,
                "varied_observation": Observation,
                "original_risk_level": str,
                "varied_risk_level": str,
                "context_delta": dict,   # what changed
                "note": str,             # human-readable summary
            }
        """
        matches = [t for t in self._task_pool.get_all() if t.task_id == task_id]
        if not matches:
            raise ValueError(f"No task found with task_id: {task_id!r}")
        original_task = matches[0]

        varied_fields = original_task.model_dump()
        context_delta: dict = {}

        allowed_keys = {"platform", "region", "user_history"}
        for key, val in new_context.items():
            if key not in allowed_keys:
                raise ValueError(
                    f"simulate_context_variation only supports modifying: {allowed_keys}. Got: {key!r}"
                )
            context_delta[key] = {"from": varied_fields[key], "to": val}
            varied_fields[key] = val

        varied_task = Task(**varied_fields)

        original_obs = self._obs_from_task(
            original_task, step_number=0, items_in_episode=1, reward=None, done=False
        )
        varied_obs = self._obs_from_task(
            varied_task, step_number=0, items_in_episode=1, reward=None, done=False
        )

        original_risk = _infer_risk_level(original_task)
        varied_risk = _infer_risk_level(varied_task)

        note = (
            f"Task '{task_id}' re-evaluated under modified context. "
            f"Risk level: {original_risk} → {varied_risk}. "
            f"Changed fields: {list(context_delta.keys())}."
        )

        return {
            "original_observation": original_obs,
            "varied_observation": varied_obs,
            "original_risk_level": original_risk,
            "varied_risk_level": varied_risk,
            "context_delta": context_delta,
            "note": note,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _obs_from_task(
        task: Task,
        step_number: int,
        items_in_episode: int,
        *,
        reward: float | None = None,
        done: bool = False,
        metadata: dict | None = None,
    ) -> Observation:
        return Observation(
            content=task.content,
            platform=task.platform,
            region=task.region,
            user_history=task.user_history,
            task_id=task.task_id,
            difficulty=task.difficulty,
            step_number=step_number,
            items_in_episode=items_in_episode,
            reward=reward,
            done=done,
            metadata=dict(metadata or {}),
        )
