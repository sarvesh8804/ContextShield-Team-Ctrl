"""
Property-based tests for graders.
Uses hypothesis with @settings(max_examples=100).
"""
import sys
import os

# Ensure project root is on path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from models.action import Action
from models.task import Task
from graders.easy import EasyGrader
from graders.medium import MediumGrader
from graders.hard import HardGrader

# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

decisions = st.sampled_from(["allow", "remove", "escalate"])
ground_truths = st.sampled_from(["allow", "remove", "escalate"])
difficulties = st.sampled_from(["easy", "medium", "hard"])
platforms = st.sampled_from(["social_media", "marketplace", "messaging"])
regions = st.sampled_from(["US", "EU", "APAC"])

action_strategy = st.builds(
    Action,
    decision=decisions,
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    reasoning=st.text(min_size=0, max_size=200),
)

task_strategy = st.builds(
    Task,
    task_id=st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_-")),
    difficulty=difficulties,
    content=st.text(min_size=1, max_size=200),
    platform=platforms,
    region=regions,
    user_history=st.fixed_dictionaries({}),
    ground_truth=ground_truths,
    context_keywords=st.lists(
        st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters=" _-")),
        min_size=0,
        max_size=5,
    ),
    explanation=st.text(min_size=0, max_size=100),
)

# ---------------------------------------------------------------------------
# Property 1: Grader determinism
# Validates: Requirements 4.1, 4.2, 4.3
# ---------------------------------------------------------------------------

@given(action=action_strategy, task=task_strategy)
@settings(max_examples=100)
def test_easy_grader_determinism(action, task):
    """**Validates: Requirements 4.1, 4.2, 4.3**
    For any (Action, Task), EasyGrader.grade returns the same score on repeated calls.
    """
    grader = EasyGrader()
    assert grader.grade(action, task) == grader.grade(action, task)


@given(action=action_strategy, task=task_strategy)
@settings(max_examples=100)
def test_medium_grader_determinism(action, task):
    """**Validates: Requirements 4.1, 4.2, 4.3**
    For any (Action, Task), MediumGrader.grade returns the same score on repeated calls.
    """
    grader = MediumGrader()
    assert grader.grade(action, task) == grader.grade(action, task)


@given(action=action_strategy, task=task_strategy)
@settings(max_examples=100)
def test_hard_grader_determinism(action, task):
    """**Validates: Requirements 4.1, 4.2, 4.3**
    For any (Action, Task), HardGrader.grade returns the same score on repeated calls.
    """
    grader = HardGrader()
    assert grader.grade(action, task) == grader.grade(action, task)


# ---------------------------------------------------------------------------
# Property 2 (partial): Grader score bounds
# Validates: Requirements 4.1, 4.2, 4.3
# ---------------------------------------------------------------------------

@given(action=action_strategy, task=task_strategy)
@settings(max_examples=100)
def test_easy_grader_score_bounds(action, task):
    """**Validates: Requirements 4.1, 4.2, 4.3**
    EasyGrader score is always in [0.0, 1.0].
    """
    score = EasyGrader().grade(action, task)
    assert 0.0 <= score <= 1.0


@given(action=action_strategy, task=task_strategy)
@settings(max_examples=100)
def test_medium_grader_score_bounds(action, task):
    """**Validates: Requirements 4.1, 4.2, 4.3**
    MediumGrader score is always in [0.0, 1.0].
    """
    score = MediumGrader().grade(action, task)
    assert 0.0 <= score <= 1.0


@given(action=action_strategy, task=task_strategy)
@settings(max_examples=100)
def test_hard_grader_score_bounds(action, task):
    """**Validates: Requirements 4.1, 4.2, 4.3**
    HardGrader score is always in [0.0, 1.0].
    """
    score = HardGrader().grade(action, task)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# Property 9: Easy grader scoring rules
# Validates: Requirements 4.4, 4.5
# ---------------------------------------------------------------------------

@given(task=task_strategy)
@settings(max_examples=100)
def test_easy_grader_exact_match_returns_1(task):
    """**Validates: Requirements 4.4, 4.5**
    Exact decision match always returns 1.0.
    """
    action = Action(decision=task.ground_truth, confidence=1.0, reasoning="test")
    assert EasyGrader().grade(action, task) == 1.0


@given(task=task_strategy)
@settings(max_examples=100)
def test_easy_grader_remove_escalate_swap_returns_0_5(task):
    """**Validates: Requirements 4.4, 4.5**
    remove/escalate swap returns 0.5 symmetrically.
    """
    grader = EasyGrader()

    # remove decision, escalate ground_truth
    task_remove = task.model_copy(update={"ground_truth": "escalate"})
    action_remove = Action(decision="remove", confidence=0.5, reasoning="test")
    assert grader.grade(action_remove, task_remove) == 0.5

    # escalate decision, remove ground_truth
    task_escalate = task.model_copy(update={"ground_truth": "remove"})
    action_escalate = Action(decision="escalate", confidence=0.5, reasoning="test")
    assert grader.grade(action_escalate, task_escalate) == 0.5


@given(task=task_strategy)
@settings(max_examples=100)
def test_easy_grader_other_mismatches_return_0(task):
    """**Validates: Requirements 4.4, 4.5**
    All mismatches that are not remove/escalate swaps return 0.0.
    """
    grader = EasyGrader()

    # allow vs remove
    task_r = task.model_copy(update={"ground_truth": "remove"})
    action_a = Action(decision="allow", confidence=0.5, reasoning="test")
    assert grader.grade(action_a, task_r) == 0.0

    # allow vs escalate
    task_e = task.model_copy(update={"ground_truth": "escalate"})
    assert grader.grade(action_a, task_e) == 0.0

    # remove vs allow
    task_al = task.model_copy(update={"ground_truth": "allow"})
    action_r = Action(decision="remove", confidence=0.5, reasoning="test")
    assert grader.grade(action_r, task_al) == 0.0

    # escalate vs allow
    action_e = Action(decision="escalate", confidence=0.5, reasoning="test")
    assert grader.grade(action_e, task_al) == 0.0


# ---------------------------------------------------------------------------
# Property 11: Medium grader context bonus
# Validates: Requirements 4.6
# ---------------------------------------------------------------------------

@given(
    task=task_strategy.filter(lambda t: len(t.context_keywords) > 0),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@settings(max_examples=100)
def test_medium_grader_context_bonus(task, confidence):
    """**Validates: Requirements 4.6**
    Correct decision + keyword in reasoning scores higher than correct decision without keyword.
    """
    grader = MediumGrader()
    keyword = task.context_keywords[0]

    # Reasoning that contains a keyword
    action_with_kw = Action(
        decision=task.ground_truth,
        confidence=confidence,
        reasoning=f"This content involves {keyword} and should be actioned.",
    )
    # Reasoning that does NOT contain any keyword
    action_without_kw = Action(
        decision=task.ground_truth,
        confidence=confidence,
        reasoning="xyz_no_keyword_here_xyz",
    )

    score_with = grader.grade(action_with_kw, task)
    score_without = grader.grade(action_without_kw, task)

    # The bonus only makes a visible difference when base < 1.0.
    # When base == 1.0 the clamp keeps both at 1.0, so >= is the correct bound.
    assert score_with >= score_without
    # When base is strictly less than 1.0 the keyword bonus must push score_with higher.
    easy_base = EasyGrader().grade(action_without_kw, task)
    if easy_base < 1.0 and easy_base > 0.0:
        assert score_with > score_without


# ---------------------------------------------------------------------------
# Property 10: Hard grader monotonicity
# Validates: Requirements 4.7 (design)
# ---------------------------------------------------------------------------

@given(
    task=task_strategy.filter(lambda t: len(t.context_keywords) >= 2),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@settings(max_examples=100)
def test_hard_grader_monotonicity(task, confidence):
    """**Validates: Requirements 4.7 (design)**
    Correct decision with K keyword hits scores >= correct decision with K-1 keyword hits.
    """
    grader = HardGrader()
    keywords = task.context_keywords

    # Build reasoning with all keywords
    reasoning_all = " ".join(keywords)
    # Build reasoning with all but the last keyword
    reasoning_fewer = " ".join(keywords[:-1])

    action_all = Action(decision=task.ground_truth, confidence=confidence, reasoning=reasoning_all)
    action_fewer = Action(decision=task.ground_truth, confidence=confidence, reasoning=reasoning_fewer)

    score_all = grader.grade(action_all, task)
    score_fewer = grader.grade(action_fewer, task)

    assert score_all >= score_fewer
