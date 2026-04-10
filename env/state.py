import uuid
from models.action import Action
from models.state import EpisodeState
from models.task import Task


class StateManager:
    def __init__(self) -> None:
        self.episode_id: str = ""
        self.current_task_id: str | None = None
        self.step_number: int = 0
        self.done: bool = False

    def reset(self, task: Task) -> None:
        self.episode_id = str(uuid.uuid4())
        self.current_task_id = task.task_id
        self.step_number = 0
        self.done = False
