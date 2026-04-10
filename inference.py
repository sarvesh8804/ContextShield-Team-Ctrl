import asyncio
import json
import os
import random
from typing import List, Optional

from openai import OpenAI
from env.environment import ContextShieldEnv
from models.action import Action

# --- Configuration & Constants ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
BENCHMARK = os.getenv("BENCHMARK", "unit-forge")
SUCCESS_SCORE_THRESHOLD = 0.5

SYSTEM_PROMPT = """\
You are UnitForge, an expert in scientific unit conversion.
You will be provided an input value, the source unit, and the target unit to convert to.
You must return the precise converted float value.

INSTRUCTIONS:
- Respond ONLY with a valid JSON object. No markdown, no explanation outside JSON.
- Required keys: value (float)
"""

def build_user_prompt(obs) -> str:
    return (
        f"Input Value: {obs.input_value}\n"
        f"From Unit: {obs.from_unit}\n"
        f"To Unit: {obs.to_unit}"
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
        return json.dumps({"value": 0.0})

async def run_single_task(client: OpenAI, model_name: str, difficulty: str, seed: int):
    random.seed(seed)
    env_instance = ContextShieldEnv(difficulty=difficulty, seed=seed)
    obs = env_instance.reset()
    
    task_name = f"UnitForge-{difficulty}"
    print(f"[START] task={task_name} env={BENCHMARK} model={model_name}", flush=True)
    
    rewards: List[float] = []
    steps_taken = 0
    done = False
    
    try:
        while not done:
            user_prompt = build_user_prompt(obs)
            message = get_model_message(client, model_name, user_prompt)
            
            try:
                data = json.loads(message)
                action = Action(value=float(data.get("value", 0.0)))
            except Exception:
                action = Action(value=0.0)
                
            obs, score, done, info = env_instance.step(action)
            rewards.append(score)
            steps_taken += 1
            
            print(f"[STEP] step={steps_taken} action={message} reward={score:.2f} done={done}", flush=True)
        
        final_score = sum(rewards) / len(rewards) if rewards else 0.0
        success = final_score >= SUCCESS_SCORE_THRESHOLD
    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(f"[END] success={success} steps={steps_taken} score={final_score:.3f} rewards={rewards_str}", flush=True)

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    seed = int(os.getenv("SEED", "42"))
    
    for difficulty in ["easy", "medium", "hard"]:
        await run_single_task(client, MODEL_NAME, difficulty, seed)

if __name__ == "__main__":
    asyncio.run(main())
