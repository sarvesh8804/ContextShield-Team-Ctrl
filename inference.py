import json
import os
import random

from openai import OpenAI
from env.environment import ContextShieldEnv
from models.action import Action

SYSTEM_PROMPT = """\
You are a Trust & Safety content moderator at a large social platform.
Your job is to review content and decide the correct moderation action.

DECISIONS:
- "allow"    — content is safe and complies with policy
- "remove"   — content clearly violates policy and must be taken down
- "escalate" — content is ambiguous, borderline, or requires human review

INSTRUCTIONS:
- Respond ONLY with a valid JSON object. No markdown, no explanation outside JSON.
- Required keys: decision (string), confidence (float 0.0-1.0), reasoning (string)
- reasoning MUST reference the platform, region, and/or user history signals you used.\
"""

from typing import Optional, List


def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None):
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_user_prompt(obs) -> str:
    return (
        f"Content: {obs.content}\n"
        f"Platform: {obs.platform}\n"
        f"Region: {obs.region}\n"
        f"User history: {obs.user_history}\n"
        f"Difficulty: {obs.difficulty}\n"
        f"Episode progress: item after {obs.step_number} completed / {obs.items_in_episode} total"
    )


def get_model_message(client: OpenAI, model: str, user_prompt: str) -> str:
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=256,
        )
        return (completion.choices[0].message.content or "").strip()
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return json.dumps(
            {"decision": "escalate", "confidence": 0.0, "reasoning": "error fallback"}
        )


def run_single_task(client: OpenAI, model_name: str, difficulty: str, seed: int):
    random.seed(seed)
    env_instance = ContextShieldEnv(difficulty=difficulty, seed=seed)
    obs = env_instance.reset(seed=seed)

    task_name = f"ContextShield-{difficulty}"
    log_start(task=task_name, env="context-shield", model=model_name)

    rewards: List[float] = []
    steps_taken = 0
    step_idx = 0

    try:
        while True:
            step_idx += 1
            user_prompt = build_user_prompt(obs)
            message = get_model_message(client, model_name, user_prompt)

            try:
                data = json.loads(message)
                action = Action(
                    decision=data.get("decision", "escalate"),
                    confidence=float(data.get("confidence", 0.0)),
                    reasoning=str(data.get("reasoning", "missing")),
                )
            except Exception:
                action = Action(
                    decision="escalate", confidence=0.0, reasoning="parse_error"
                )

            obs, reward_obj, done, _info = env_instance.step(action)
            reward_val = (
                getattr(reward_obj, "score", 0.0)
                if hasattr(reward_obj, "score")
                else float(reward_obj)
            )
            rewards.append(reward_val)
            steps_taken += 1

            log_step(
                step=steps_taken,
                action=message[:500] + ("..." if len(message) > 500 else ""),
                reward=reward_val,
                done=done,
                error=None,
            )

            if done:
                break
            if step_idx > 64:
                print("[WARN] max inner steps exceeded", flush=True)
                break

    finally:
        mean_score = sum(rewards) / max(len(rewards), 1) if rewards else 0.0
        success = mean_score >= 0.8
        log_end(
            success=success,
            steps=steps_taken,
            score=mean_score,
            rewards=rewards,
        )


def main():
    api_base = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("MODEL_NAME", "gpt-4o-mini")
    api_key = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("OPENAI_API_KEY")
        or "sk-mock"
    )

    client = OpenAI(base_url=api_base, api_key=api_key)
    seed = int(os.environ.get("SEED", "42"))

    for difficulty in ["easy", "medium", "hard"]:
        run_single_task(client, model_name, difficulty, seed)


if __name__ == "__main__":
    main()
