"""
Unit tests for TaskPool.
Requirements: 3.1, 8.2
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from tasks.task_pool import TaskPool
from models.task import Task


@pytest.fixture(scope="module")
def pool() -> TaskPool:
    return TaskPool()


def test_get_all_returns_at_least_15_tasks(pool):
    assert len(pool.get_all()) >= 15


def test_get_all_returns_task_instances(pool):
    for task in pool.get_all():
        assert isinstance(task, Task)


def test_sample_returns_a_task(pool):
    task = pool.sample()
    assert isinstance(task, Task)


def test_sample_easy_returns_easy_task(pool):
    task = pool.sample("easy")
    assert task.difficulty == "easy"


def test_sample_medium_returns_medium_task(pool):
    task = pool.sample("medium")
    assert task.difficulty == "medium"


def test_sample_hard_returns_hard_task(pool):
    task = pool.sample("hard")
    assert task.difficulty == "hard"


def test_sample_nonexistent_difficulty_raises_value_error(pool):
    with pytest.raises(ValueError, match="nonexistent"):
        pool.sample("nonexistent")


def test_sample_sequence_returns_requested_count(pool):
    tasks = pool.sample_sequence(5)
    assert len(tasks) == 5
    for t in tasks:
        assert isinstance(t, Task)


def test_sample_sequence_difficulty_filter(pool):
    tasks = pool.sample_sequence(3, difficulty="easy")
    assert len(tasks) == 3
    for t in tasks:
        assert t.difficulty == "easy"


def test_sample_sequence_seed_determinism(pool):
    seq1 = pool.sample_sequence(5, seed=42)
    seq2 = pool.sample_sequence(5, seed=42)
    assert [t.task_id for t in seq1] == [t.task_id for t in seq2]
