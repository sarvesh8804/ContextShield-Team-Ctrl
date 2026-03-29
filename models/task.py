from typing import Literal
from pydantic import BaseModel


class Task(BaseModel):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    content: str
    platform: Literal["social_media", "marketplace", "messaging"]
    region: Literal["US", "EU", "APAC"]
    user_history: dict
    ground_truth: Literal["allow", "remove", "escalate"]
    context_keywords: list[str]
    explanation: str
