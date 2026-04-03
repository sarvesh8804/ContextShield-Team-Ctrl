#!/usr/bin/env python3
"""Smoke test: oracle actions (ground-truth decisions + rich reasoning) should yield high mean rewards."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.environment import ContextShieldEnv
from models.action import Action


def _reasoning_for_task(task) -> str:
    """Include platform, region, user-history phrases, and all context keywords (for hard grader)."""
    parts = [
        f"platform {task.platform} social_media marketplace messaging",
        f"region {task.region} US EU APAC jurisdiction",
        "user history prior violations account age flagged",
    ]
    parts.extend(task.context_keywords)
    return "; ".join(parts)


def main() -> None:
    seed = int(os.environ.get("SEED", "42"))
    length = int(os.environ.get("CONTEXT_SHIELD_EPISODE_LENGTH", "4"))

    for difficulty in ("easy", "medium", "hard"):
        env = ContextShieldEnv(
            difficulty=difficulty,
            seed=seed,
            episode_length=length,
        )
        env.reset(seed=seed)
        rewards: list[float] = []
        while True:
            t = env._episode_tasks[env._cursor]
            action = Action(
                decision=t.ground_truth,
                confidence=1.0,
                reasoning=_reasoning_for_task(t),
            )
            _obs, reward_obj, done, _info = env.step(action)
            rewards.append(reward_obj.score)
            if done:
                break
        mean_r = sum(rewards) / len(rewards)
        print(
            f"{difficulty}: mean_reward={mean_r:.3f} steps={len(rewards)} "
            f"(oracle ground_truth + keywords)"
        )


if __name__ == "__main__":
    main()
