from typing import Any
from pydantic import BaseModel


class EpisodeState(BaseModel):
    episode_id: str
    current_task_id: str | None
    step_number: int
    done: bool
    history: list[Any]
    total_score: float
