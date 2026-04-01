from openenv.core.env_server.types import Observation as BaseObservation


class Observation(BaseObservation):
    content: str
    platform: str
    region: str
    user_history: dict
    task_id: str
    difficulty: str
    step_number: int
