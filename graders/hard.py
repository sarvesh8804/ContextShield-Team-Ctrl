from graders.base import BaseGrader
from models.action import Action
from models.task import Task


class HardGrader(BaseGrader):
    def grade(self, action: Action, task: Task) -> float:
        if action.decision != task.ground_truth:
            return 0.0
        keyword_hits = sum(
            1
            for kw in task.context_keywords
            if kw.lower() in action.reasoning.lower()
        )
        keyword_ratio = keyword_hits / max(len(task.context_keywords), 1)
        return 0.5 + 0.5 * keyword_ratio
