"""
Unit tests for task data coverage.
Requirements: 3.1, 3.5
"""
import json
import pathlib

import pytest

DATA_DIR = pathlib.Path(__file__).parent.parent / "tasks" / "data"


def load_json(filename: str) -> list[dict]:
    with open(DATA_DIR / filename) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def all_tasks() -> list[dict]:
    easy = load_json("easy.json")
    medium = load_json("medium.json")
    hard = load_json("hard.json")
    return easy + medium + hard


def test_total_tasks_at_least_15(all_tasks):
    assert len(all_tasks) >= 15, f"Expected >= 15 tasks, got {len(all_tasks)}"


def test_all_platforms_present(all_tasks):
    platforms = {t["platform"] for t in all_tasks}
    assert "social_media" in platforms
    assert "marketplace" in platforms
    assert "messaging" in platforms


def test_all_regions_present(all_tasks):
    regions = {t["region"] for t in all_tasks}
    assert "US" in regions
    assert "EU" in regions
    assert "APAC" in regions


def test_easy_tasks_count():
    easy = load_json("easy.json")
    assert len(easy) >= 5, f"Expected >= 5 easy tasks, got {len(easy)}"


def test_medium_tasks_count():
    medium = load_json("medium.json")
    assert len(medium) >= 5, f"Expected >= 5 medium tasks, got {len(medium)}"


def test_hard_tasks_count():
    hard = load_json("hard.json")
    assert len(hard) >= 5, f"Expected >= 5 hard tasks, got {len(hard)}"


def test_all_tasks_have_required_fields(all_tasks):
    required = {"task_id", "difficulty", "content", "platform", "region",
                "user_history", "ground_truth", "context_keywords", "explanation"}
    for task in all_tasks:
        missing = required - task.keys()
        assert not missing, f"Task {task.get('task_id')} missing fields: {missing}"


def test_ground_truth_values_valid(all_tasks):
    valid = {"allow", "remove", "escalate"}
    for task in all_tasks:
        assert task["ground_truth"] in valid, (
            f"Task {task['task_id']} has invalid ground_truth: {task['ground_truth']}"
        )


def test_difficulty_matches_file():
    for difficulty, filename in [("easy", "easy.json"), ("medium", "medium.json"), ("hard", "hard.json")]:
        tasks = load_json(filename)
        for task in tasks:
            assert task["difficulty"] == difficulty, (
                f"Task {task['task_id']} has difficulty '{task['difficulty']}', expected '{difficulty}'"
            )
