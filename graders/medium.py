from graders.base import BaseGrader
from graders.easy import EasyGrader
from models.action import Action
from models.task import Task


class MediumGrader(BaseGrader):
    def __init__(self):
        self._easy = EasyGrader()

    def grade(self, action: Action, task: Task) -> float:
        base = self._easy.grade(action, task)
        if base == 0.0:
            return 0.0
        context_bonus = (
            0.1
            if any(kw.lower() in action.reasoning.lower() for kw in task.context_keywords)
            else 0.0
        )
        return min(1.0, base + context_bonus)
