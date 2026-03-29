"""
TaskPool: loads and samples Task objects from JSON data files.
Requirements: 3.1, 8.2
"""
import json
import random
import pathlib
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.task import Task  # noqa: E402

DATA_DIR = pathlib.Path(__file__).parent / "data"


class TaskPool:
    def __init__(self) -> None:
        self._tasks: list[Task] = self.load_all()

    def load_all(self) -> list[Task]:
        """Read and parse all three JSON files into a list of Task objects."""
        tasks: list[Task] = []
        for filename in ("easy.json", "medium.json", "hard.json"):
            path = DATA_DIR / filename
            with open(path) as f:
                raw = json.load(f)
            tasks.extend(Task(**item) for item in raw)
        return tasks

    def get_all(self) -> list[Task]:
        """Return the full list of tasks."""
        return self._tasks

    def sample(self, difficulty: Optional[str] = None, seed: Optional[int] = None) -> Task:
        """Return a random task, optionally filtered by difficulty.

        Args:
            difficulty: filter to a specific difficulty level, or None for any.
            seed: optional RNG seed for reproducible sampling.
        """
        pool = self._tasks
        if difficulty is not None:
            pool = [t for t in pool if t.difficulty == difficulty]
        if not pool:
            raise ValueError(f"No tasks available for difficulty: {difficulty}")
        if seed is not None:
            rng = random.Random(seed)
            return rng.choice(pool)
        return random.choice(pool)
