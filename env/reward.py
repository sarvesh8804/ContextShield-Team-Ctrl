"""
RewardFunction: computes dense reward signals for ContextShield.
Partial credit is now based on context-signal coverage in reasoning,
not just string length — making it a true measure of moderation quality.
"""
import sys
import os
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.action import Action
from models.task import Task
from models.reward import Reward

FILLER_PHRASES = ["i think", "maybe", "not sure", "i guess", "perhaps"]

# Context signals that indicate platform/region/user-history awareness
_PLATFORM_SIGNALS = ["social_media", "social media", "marketplace", "messaging", "platform"]
_REGION_SIGNALS = ["us", "eu", "apac", "region", "jurisdiction", "local law", "regulation"]
_USER_HISTORY_SIGNALS = [
    "prior violation", "account age", "flagged", "repeat offender",
    "new account", "user history", "violation history",
]


def _count_context_signals(reasoning: str, task: Task) -> int:
    """Count how many distinct context signal categories appear in the reasoning.

    Categories: platform (1 pt), region (1 pt), user_history (1 pt).
    Max = 3. Deterministic — no randomness.
    """
    lowered = reasoning.lower()
    score = 0
    
    def contains_any(terms, text):
        for t in terms:
            if re.search(r'\b' + re.escape(t) + r'\b', text):
                return True
        return False

    # Platform awareness: either the task's platform name or generic platform signals
    platform_terms = _PLATFORM_SIGNALS + [task.platform.replace("_", " "), task.platform]
    if contains_any(platform_terms, lowered):
        score += 1

    # Region awareness
    region_terms = _REGION_SIGNALS + [task.region.lower()]
    if contains_any(region_terms, lowered):
        score += 1

    # User history awareness
    if contains_any(_USER_HISTORY_SIGNALS, lowered):
        score += 1

    return score


class RewardFunction:
    def compute(self, grader_score: float, action: Action, task: Task) -> Reward:
        reasoning = action.reasoning
        lowered = reasoning.lower()

        # --- Partial credit: context-signal coverage (0.0 / 0.1 / 0.2 / 0.3) ---
        # Requires reasoning to be non-trivial (> 20 chars) and filler-free for full credit.
        has_filler = any(phrase in lowered for phrase in FILLER_PHRASES)
        signal_count = _count_context_signals(reasoning, task)

        if len(reasoning) <= 20 or has_filler:
            partial_credit = 0.0
        else:
            # 0.1 per context signal category referenced, max 0.3
            partial_credit = round(min(0.3, signal_count * 0.1), 10)

        # --- Penalty: false-allow (agent said allow but ground truth is remove) ---
        if action.decision == "allow" and task.ground_truth == "remove":
            penalty = 0.2
        else:
            penalty = 0.0

        # --- Confidence calibration bonus ---
        if abs(action.confidence - grader_score) < 0.2:
            confidence_calibration = 0.1
        else:
            confidence_calibration = 0.0

        raw_score = grader_score + partial_credit + confidence_calibration - penalty
        score = max(0.0, min(1.0, raw_score))

        feedback = (
            f"grader_score={grader_score:.2f}, "
            f"partial_credit={partial_credit:.2f} (context_signals={signal_count}/3), "
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
