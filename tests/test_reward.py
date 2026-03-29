"""
Property-based tests for RewardFunction.
Uses hypothesis with @settings(max_examples=100).
Validates: Requirements 5.2, 5.3, 5.4, 5.5
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hypothesis import given, settings, assume
from hypothesis import strategies as st

from models.action import Action
from models.task import Task
from env.reward import RewardFunction

# ---------------------------------------------------------------------------
# Shared strategies (mirrors test_graders.py)
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
    task_id=st.text(
        min_size=1,
        max_size=20,
        alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd"),
            whitelist_characters="_-",
        ),
    ),
    difficulty=difficulties,
    content=st.text(min_size=1, max_size=200),
    platform=platforms,
    region=regions,
    user_history=st.fixed_dictionaries({}),
    ground_truth=ground_truths,
    context_keywords=st.lists(
        st.text(
            min_size=1,
            max_size=20,
            alphabet=st.characters(
                whitelist_categories=("Lu", "Ll", "Nd"),
                whitelist_characters=" _-",
            ),
        ),
        min_size=0,
        max_size=5,
    ),
    explanation=st.text(min_size=0, max_size=100),
)

grader_score_strategy = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)

# ---------------------------------------------------------------------------
# Property 2: Reward score bounds
# Validates: Requirements 5.5
# ---------------------------------------------------------------------------

@given(action=action_strategy, task=task_strategy, grader_score=grader_score_strategy)
@settings(max_examples=100)
def test_reward_score_bounds(action, task, grader_score):
    """**Validates: Requirements 5.5**
    For any (Action, Task, grader_score in [0,1]), reward.score is in [0.0, 1.0].
    """
    rf = RewardFunction()
    reward = rf.compute(grader_score, action, task)
    assert 0.0 <= reward.score <= 1.0


# ---------------------------------------------------------------------------
# Property 3: Penalty only on false-allow
# Validates: Requirements 5.3
# ---------------------------------------------------------------------------

@given(action=action_strategy, task=task_strategy, grader_score=grader_score_strategy)
@settings(max_examples=100)
def test_penalty_only_on_false_allow(action, task, grader_score):
    """**Validates: Requirements 5.3**
    For any Action where decision != "allow" OR ground_truth != "remove", penalty == 0.0.
    """
    assume(action.decision != "allow" or task.ground_truth != "remove")
    rf = RewardFunction()
    reward = rf.compute(grader_score, action, task)
    assert reward.penalty == 0.0


# ---------------------------------------------------------------------------
# Property 4: Partial credit independence
# Validates: Requirements 5.2
# ---------------------------------------------------------------------------

@given(
    task=task_strategy,
    grader_score=grader_score_strategy,
    reasoning=st.text(min_size=0, max_size=200),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    decision_a=decisions,
    decision_b=decisions,
)
@settings(max_examples=100)
def test_partial_credit_independence(task, grader_score, reasoning, confidence, decision_a, decision_b):
    """**Validates: Requirements 5.2**
    Two Actions with identical reasoning but different decisions must have equal partial_credit.
    """
    rf = RewardFunction()
    action_a = Action(decision=decision_a, confidence=confidence, reasoning=reasoning)
    action_b = Action(decision=decision_b, confidence=confidence, reasoning=reasoning)
    reward_a = rf.compute(grader_score, action_a, task)
    reward_b = rf.compute(grader_score, action_b, task)
    assert reward_a.partial_credit == reward_b.partial_credit


# ---------------------------------------------------------------------------
# Property 12: Confidence calibration bonus
# Validates: Requirements 5.4
# ---------------------------------------------------------------------------

@given(action=action_strategy, task=task_strategy, grader_score=grader_score_strategy)
@settings(max_examples=100)
def test_confidence_calibration_bonus(action, task, grader_score):
    """**Validates: Requirements 5.4**
    When |confidence - grader_score| < 0.2, calibration == 0.1; else 0.0.
    """
    rf = RewardFunction()
    reward = rf.compute(grader_score, action, task)
    if abs(action.confidence - grader_score) < 0.2:
        assert reward.confidence_calibration == 0.1
    else:
        assert reward.confidence_calibration == 0.0
