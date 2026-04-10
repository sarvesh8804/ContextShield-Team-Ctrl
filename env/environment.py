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
from env.exceptions import EpisodeTerminatedError
from graders.unit_grader import grade

class ContextShieldEnv:
    EPISODE_STEPS = 5

    def __init__(self, difficulty: str | None = None, seed: int | None = None) -> None:
        self.difficulty = difficulty
        if seed is not None:
            random.seed(seed)
        self._task_pool = TaskPool()
        self._state_manager = StateManager()
        self._current_task: Task | None = None
        self._task_queue: list[Task] = []

    def reset(self, seed: int | None = None) -> Observation:
        self._task_queue = self._task_pool.sample_sequence(
            self.EPISODE_STEPS, self.difficulty, seed=seed
        )
        self._current_task = self._task_queue[0]
        self._state_manager.reset(self._current_task)
        return self._obs_from_task(self._current_task, step_number=0, done=False)

    def _ensure_episode(self) -> None:
        if not self._task_queue:
            self.reset()

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        self._ensure_episode()

        if self._state_manager.done:
            raise EpisodeTerminatedError(
                "Episode has already terminated. Call reset() to start a new episode."
            )

        task = self._current_task
        score = grade(action.value, task.correct_answer)
        
        class MockReward:
            def __init__(self, s):
                self.score = s
            def model_dump(self):
                return {"score": self.score}
                
        fake_reward = MockReward(score)
        self._state_manager.record_step(action, fake_reward)
        
        step_index = self._state_manager.step_number
        is_done = step_index >= self.EPISODE_STEPS

        info = {
            "task_id": task.task_id,
            "difficulty": task.difficulty
        }

        if not is_done:
            self._current_task = self._task_queue[step_index]
            self._state_manager.set_current_task(self._current_task.task_id)
            next_obs = self._obs_from_task(
                self._current_task,
                step_number=step_index,
                done=False
            )
            return (next_obs, score, False, info)
        else:
            self._state_manager.mark_done()
            terminal_obs = self._obs_from_task(
                task,
                step_number=step_index,
                done=True
            )
            return (terminal_obs, score, True, info)

    def state(self) -> EpisodeState:
        return self._state_manager.snapshot()

    @staticmethod
    def _obs_from_task(
        task: Task,
        step_number: int,
        *,
        done: bool = False,
    ) -> Observation:
        return Observation(
            task_id=task.task_id,
            difficulty=task.difficulty,
            step_number=step_number,
            input_value=task.input_value,
            from_unit=task.from_unit,
            to_unit=task.to_unit
        )
