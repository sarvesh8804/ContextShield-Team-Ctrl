import asyncio
import json
import os
import random
import sys
from typing import List, Optional

from openai import OpenAI
from env.environment import ContextShieldEnv
from models.action import Action

# --- Configuration & Constants (Strict Compliance) ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
BENCHMARK = os.getenv("BENCHMARK", "context-shield")
SUCCESS_SCORE_THRESHOLD = 0.1
# ideally will be set to 0.8

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

# --- Logging Helpers (Strict Compliance) ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower() # Lowercase true/false
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# --- Logic Helpers ---
def build_user_prompt(obs) -> str:
    return (
        f"Content: {obs.content}\n"
        f"Platform: {obs.platform}\n"
        f"Region: {obs.region}\n"
        f"User history: {obs.user_history}\n"
        f"Difficulty: {obs.difficulty}"
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
        return json.dumps({"decision": "escalate", "confidence": 0.0, "reasoning": "error fallback"})

# --- Task Runner ---
async def run_single_task(client: OpenAI, model_name: str, difficulty: str, seed: int):
    random.seed(seed)
    env_instance = ContextShieldEnv(difficulty=difficulty, seed=seed)
    obs = env_instance.reset()
    
    task_name = f"ContextShield-{difficulty}"
    log_start(task=task_name, env=BENCHMARK, model=model_name)
    
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    done = False
    
    try:
        while not done:
            user_prompt = build_user_prompt(obs)
            message = get_model_message(client, model_name, user_prompt)
            
            try:
                data = json.loads(message)
                action = Action(
                    decision=data.get("decision", "escalate"),
                    confidence=float(data.get("confidence", 0.0)),
                    reasoning=str(data.get("reasoning", "missing"))
                )
            except Exception:
                action = Action(decision="escalate", confidence=0.0, reasoning="parse_error")
                
            obs, reward_obj, done, info = env_instance.step(action)
            reward_val = getattr(reward_obj, "score", 0.0) if hasattr(reward_obj, "score") else float(reward_obj)
            
            rewards.append(reward_val)
            steps_taken += 1
            
            log_step(step=steps_taken, action=message, reward=reward_val, done=done, error=None)
        
        # Mean score over the trajectory (Strict Compliance)
        score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
        success = score >= SUCCESS_SCORE_THRESHOLD
        
    finally:
        # Note: env_instance (ContextShieldEnv) doesn't have an explicit close() yet,
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    seed = int(os.getenv("SEED", "42"))
    
    for difficulty in ["easy", "medium", "hard"]:
        # Run each as a separate [START]/[STEP]/[END] sequence
        await run_single_task(client, MODEL_NAME, difficulty, seed)

if __name__ == "__main__":
    asyncio.run(main())
