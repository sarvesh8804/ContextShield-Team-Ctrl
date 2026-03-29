from typing import Literal
from pydantic import BaseModel, Field


class Action(BaseModel):
    decision: Literal["allow", "remove", "escalate"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
