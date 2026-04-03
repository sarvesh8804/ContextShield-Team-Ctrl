from pydantic import BaseModel


class EpisodeState(BaseModel):
    episode_id: str
    current_task_id: str | None
    step_number: int
    items_in_episode: int
    done: bool
    history: list[dict]
    total_score: float
