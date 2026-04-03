"""
Tests for Pydantic data models.

Includes:
- Property 5: Invalid decision raises ValidationError (Validates: Requirements 2.4)
- Property 6: Out-of-range confidence raises ValidationError (Validates: Requirements 2.5)
- Unit tests for model field presence (Requirements 2.1, 2.2, 2.3)
"""

import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st
from pydantic import ValidationError

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.action import Action
from models.observation import Observation
from models.reward import Reward
from models.state import EpisodeState
from models.task import Task


# ---------------------------------------------------------------------------
# Property 5: Invalid decision raises ValidationError
# Validates: Requirements 2.4
# ---------------------------------------------------------------------------

VALID_DECISIONS = {"allow", "remove", "escalate"}


@given(
    decision=st.text().filter(lambda s: s not in VALID_DECISIONS),
    confidence=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    reasoning=st.text(min_size=1),
)
@settings(max_examples=100)
def test_property5_invalid_decision_raises_validation_error(decision, confidence, reasoning):
    """**Validates: Requirements 2.4**
    Property 5: Any string not in {"allow", "remove", "escalate"} must raise ValidationError.
    """
    with pytest.raises(ValidationError):
        Action(decision=decision, confidence=confidence, reasoning=reasoning)


# ---------------------------------------------------------------------------
# Property 6: Out-of-range confidence raises ValidationError
# Validates: Requirements 2.5
# ---------------------------------------------------------------------------

@given(
    confidence=st.one_of(
        st.floats(max_value=-0.0001, allow_nan=False, allow_infinity=False),
        st.floats(min_value=1.0001, allow_nan=False, allow_infinity=False),
    ),
    decision=st.sampled_from(["allow", "remove", "escalate"]),
    reasoning=st.text(min_size=1),
)
@settings(max_examples=100)
def test_property6_out_of_range_confidence_raises_validation_error(confidence, decision, reasoning):
    """**Validates: Requirements 2.5**
    Property 6: Any float outside [0.0, 1.0] must raise ValidationError.
    """
    assume(not (0.0 <= confidence <= 1.0))
    with pytest.raises(ValidationError):
        Action(decision=decision, confidence=confidence, reasoning=reasoning)


# ---------------------------------------------------------------------------
# Unit tests: model field presence
# Requirements: 2.1, 2.2, 2.3
# ---------------------------------------------------------------------------

def test_observation_fields_accessible():
    """All required Observation fields are present and accessible (Req 2.1)."""
    obs = Observation(
        content="Test content",
        platform="social_media",
        region="US",
        user_history={"prior_violations": 0},
        task_id="easy_001",
        difficulty="easy",
        step_number=0,
        items_in_episode=4,
    )
    assert obs.content == "Test content"
    assert obs.platform == "social_media"
    assert obs.region == "US"
    assert obs.user_history == {"prior_violations": 0}
    assert obs.task_id == "easy_001"
    assert obs.difficulty == "easy"
    assert obs.step_number == 0


def test_action_fields_accessible():
    """All required Action fields are present and accessible (Req 2.2)."""
    action = Action(decision="allow", confidence=0.9, reasoning="Looks fine.")
    assert action.decision == "allow"
    assert action.confidence == 0.9
    assert action.reasoning == "Looks fine."


def test_reward_fields_accessible():
    """All required Reward fields are present and accessible (Req 2.3)."""
    reward = Reward(
        score=0.8,
        partial_credit=0.2,
        penalty=0.0,
        confidence_calibration=0.1,
        feedback="Good job.",
    )
    assert reward.score == 0.8
    assert reward.partial_credit == 0.2
    assert reward.penalty == 0.0
    assert reward.confidence_calibration == 0.1
    assert reward.feedback == "Good job."


def test_episode_state_fields_accessible():
    """All required EpisodeState fields are present and accessible."""
    state = EpisodeState(
        episode_id="abc-123",
        current_task_id="easy_001",
        step_number=1,
        items_in_episode=4,
        done=False,
        history=[],
        total_score=0.0,
    )
    assert state.episode_id == "abc-123"
    assert state.current_task_id == "easy_001"
    assert state.step_number == 1
    assert state.items_in_episode == 4
    assert state.done is False
    assert state.history == []
    assert state.total_score == 0.0


def test_episode_state_current_task_id_nullable():
    """EpisodeState.current_task_id can be None."""
    state = EpisodeState(
        episode_id="abc-123",
        current_task_id=None,
        step_number=0,
        items_in_episode=4,
        done=False,
        history=[],
        total_score=0.0,
    )
    assert state.current_task_id is None


def test_task_fields_accessible():
    """All required Task fields are present and accessible."""
    task = Task(
        task_id="easy_001",
        difficulty="easy",
        content="Buy cheap Rolex watches!",
        platform="marketplace",
        region="US",
        user_history={"prior_violations": 0, "account_age_days": 10},
        ground_truth="remove",
        context_keywords=[],
        explanation="Obvious spam.",
    )
    assert task.task_id == "easy_001"
    assert task.difficulty == "easy"
    assert task.content == "Buy cheap Rolex watches!"
    assert task.platform == "marketplace"
    assert task.region == "US"
    assert task.user_history == {"prior_violations": 0, "account_age_days": 10}
    assert task.ground_truth == "remove"
    assert task.context_keywords == []
    assert task.explanation == "Obvious spam."


def test_action_valid_decisions():
    """All three valid decision values are accepted."""
    for decision in ["allow", "remove", "escalate"]:
        action = Action(decision=decision, confidence=0.5, reasoning="test")
        assert action.decision == decision


def test_action_confidence_boundary_values():
    """Confidence boundary values 0.0 and 1.0 are accepted."""
    a_min = Action(decision="allow", confidence=0.0, reasoning="test")
    a_max = Action(decision="allow", confidence=1.0, reasoning="test")
    assert a_min.confidence == 0.0
    assert a_max.confidence == 1.0
