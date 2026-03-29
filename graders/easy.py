from graders.base import BaseGrader
from models.action import Action
from models.task import Task


class EasyGrader(BaseGrader):
    def grade(self, action: Action, task: Task) -> float:
        if action.decision == task.ground_truth:
            return 1.0
        if {action.decision, task.ground_truth} == {"remove", "escalate"}:
            return 0.5
        return 0.0
