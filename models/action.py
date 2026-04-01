from typing import Literal

from openenv.core.env_server.types import Action as BaseAction
from pydantic import Field


class Action(BaseAction):
    decision: Literal["allow", "remove", "escalate"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
