from pydantic import BaseModel


class Reward(BaseModel):
    score: float
    partial_credit: float
    penalty: float
    confidence_calibration: float
    feedback: str
