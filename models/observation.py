from typing import Optional, List, Any
from openenv.core.env_server.types import Observation as BaseObservation


class Observation(BaseObservation):
    """
    PatchGym observation returned after every reset() or step().

    task_id     : which of the 3 tasks is active
    step_number : current step within the episode (0 = after reset)
    result      : structured output of the last tool call (or None on reset)
    error       : error message if the tool call was invalid
    hint        : static contextual hint shown every step
    total_reward: cumulative episode score so far
    """
    task_id: str
    step_number: int
    result: Optional[Any] = None
    error: Optional[str] = None
    hint: str = ""
    total_reward: float = 0.05
