from typing import Literal
from pydantic import BaseModel

class Task(BaseModel):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    input_value: float
    from_unit: str
    to_unit: str
    correct_answer: float
