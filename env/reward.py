"""
RewardFunction: computes dense reward signals for ContextShield.
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.action import Action
from models.task import Task
from models.reward import Reward

FILLER_PHRASES = ["i think", "maybe", "not sure", "i guess", "perhaps"]


class RewardFunction:
    def compute(self, grader_score: float, action: Action, task: Task) -> Reward:
        reasoning = action.reasoning
        lowered = reasoning.lower()

        # Partial credit: reasoning quality independent of decision
        has_filler = any(phrase in lowered for phrase in FILLER_PHRASES)
        if len(reasoning) > 50 and not has_filler:
            partial_credit = 0.3
        elif len(reasoning) > 20:
            partial_credit = 0.15
        else:
            partial_credit = 0.0

        # Penalty: false-allow (agent said allow but ground truth is remove)
        if action.decision == "allow" and task.ground_truth == "remove":
            penalty = 0.2
        else:
            penalty = 0.0

        # Confidence calibration bonus
        if abs(action.confidence - grader_score) < 0.2:
            confidence_calibration = 0.1
        else:
            confidence_calibration = 0.0

        raw_score = grader_score + partial_credit + confidence_calibration - penalty
        score = max(0.0, min(1.0, raw_score))

        feedback = (
            f"grader_score={grader_score:.2f}, "
            f"partial_credit={partial_credit:.2f}, "
            f"confidence_calibration={confidence_calibration:.2f}, "
            f"penalty={penalty:.2f}, "
            f"final_score={score:.2f}"
        )

        return Reward(
            score=score,
            partial_credit=partial_credit,
            penalty=penalty,
            confidence_calibration=confidence_calibration,
            feedback=feedback,
        )
