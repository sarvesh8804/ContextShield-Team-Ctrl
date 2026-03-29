from pydantic import BaseModel


class Observation(BaseModel):
    content: str
    platform: str
    region: str
    user_history: dict
    task_id: str
    difficulty: str
    step_number: int
