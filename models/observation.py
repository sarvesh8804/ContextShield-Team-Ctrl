from openenv.core.env_server.types import Observation as BaseObservation

class Observation(BaseObservation):
    task_id: str
    difficulty: str
    step_number: int
    input_value: float
    from_unit: str
    to_unit: str
