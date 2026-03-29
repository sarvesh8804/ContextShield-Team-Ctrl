from abc import ABC, abstractmethod
from models.action import Action
from models.task import Task


class BaseGrader(ABC):
    @abstractmethod
    def grade(self, action: Action, task: Task) -> float: ...
