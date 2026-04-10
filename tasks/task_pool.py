"""
TaskPool for PatchGym: loads Task objects from JSON data files.
"""
import json
import random
import pathlib
from typing import Optional
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.task import Task

DATA_DIR = pathlib.Path(__file__).parent / "data"


class TaskPool:
    def __init__(self) -> None:
        self._tasks: list[Task] = self._load_all()

    def _load_all(self) -> list[Task]:
        tasks: list[Task] = []
        for filename in ("easy.json", "medium.json", "hard.json"):
            path = DATA_DIR / filename
            with open(path) as f:
                raw = json.load(f)
            tasks.extend(Task(**item) for item in raw)
        return tasks

    def get_all(self) -> list[Task]:
        return self._tasks

    def sample(self, difficulty: Optional[str] = None, seed: Optional[int] = None) -> Task:
        pool = self._tasks
        if difficulty:
            pool = [t for t in pool if t.difficulty == difficulty]
        if not pool:
            raise ValueError(f"No tasks for difficulty={difficulty!r}")
        rng = random.Random(seed) if seed is not None else random
        return rng.choice(pool)
