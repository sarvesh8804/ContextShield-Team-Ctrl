import uuid
from models.action import Action
from models.reward import Reward
from models.state import EpisodeState
from models.task import Task


class StateManager:
    def __init__(self) -> None:
        self.episode_id: str = ""
        self.current_task_id: str | None = None
        self.step_number: int = 0
        self.done: bool = False
        self.history: list[dict] = []
        self.total_score: float = 0.0

    def reset(self, task: Task) -> None:
        self.episode_id = str(uuid.uuid4())
        self.current_task_id = task.task_id
        self.step_number = 0
        self.done = False
        self.history = []
        self.total_score = 0.0

    def record_step(self, action: Action, reward: Reward) -> None:
        self.history.append({"action": action.model_dump(), "reward": reward.model_dump()})
        self.step_number += 1
        self.total_score += reward.score

    def snapshot(self) -> EpisodeState:
        return EpisodeState(
            episode_id=self.episode_id,
            current_task_id=self.current_task_id,
            step_number=self.step_number,
            done=self.done,
            history=list(self.history),
            total_score=self.total_score,
        )

    def mark_done(self) -> None:
        self.done = True
