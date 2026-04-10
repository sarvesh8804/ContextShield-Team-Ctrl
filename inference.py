import asyncio
import json
import os
import random
from typing import List, Optional

from openai import OpenAI
from env.environment import PatchGymEnv
from models.action import Action

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
BENCHMARK = os.getenv("BENCHMARK", "patch-gym")
SUCCESS_SCORE_THRESHOLD = 0.50

SYSTEM_PROMPT = """\
You are a senior security engineer triaging dependency vulnerabilities.
You have access to these tools — call them as JSON:

  list_packages()                         → see requirements.txt
  show_cve(cve_id)                        → get CVE details
  check_imports(package_name)             → is this package actually used?
  get_fix_version(package, cve_id)        → safe version to upgrade to
  check_conflicts(package, version)       → does this upgrade break anything?
  submit_plan(ranking|plan|resolution)    → submit your final answer

ALWAYS check if a package is imported before prioritising its CVE.
For conflict-resolver tasks: ALWAYS run check_conflicts before submitting.

Respond ONLY with a single valid JSON object:
{
  "command": "<tool_name>",
  "args": { ... }
}

For submit_plan use:
  severity-ranker  → {"ranking": ["CVE-...", ...]}
  fix-planner      → {"plan": [{"package": "...", "version": "..."}, ...]}
  conflict-resolver→ {"resolution": [{"package": "...", "version": "..."}, ...]}
"""


def build_user_prompt(obs) -> str:
    lines = [
        f"Task: {obs.task_id}",
        f"Step: {obs.step_number}",
        f"Hint: {obs.hint}",
        f"Score so far: {obs.total_reward}",
    ]
    if obs.result is not None:
        lines.append(f"Last result: {json.dumps(obs.result)}")
    if obs.error:
        lines.append(f"Error: {obs.error}")
    return "\n".join(lines)


def get_model_action(client: OpenAI, model: str, prompt: str) -> dict:
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=512,
        )
        return json.loads(completion.choices[0].message.content or "{}")
    except Exception as exc:
        print(f"[DEBUG] Model error: {exc}", flush=True)
        return {"command": "list_packages", "args": {}}


async def run_episode(client: OpenAI, model_name: str, difficulty: str, seed: int):
    env = PatchGymEnv(difficulty=difficulty, seed=seed)
    obs = env.reset()
    task_name = f"PatchGym-{difficulty}"
    print(f"[START] task={task_name} env={BENCHMARK} model={model_name}", flush=True)

    rewards: List[float] = []
    steps = 0
    done = False
    final_score = 0.05
    success = False

    try:
        while not done:
            prompt = build_user_prompt(obs)
            raw = get_model_action(client, model_name, prompt)

            try:
                action = Action(
                    command=raw.get("command", "list_packages"),
                    args=raw.get("args", {}),
                )
            except Exception:
                action = Action(command="list_packages", args={})

            obs, reward, done, info = env.step(action)
            rewards.append(reward)
            steps += 1
            print(
                f"[STEP] step={steps} action={json.dumps(raw)} reward={reward:.2f} done={str(done).lower()}",
                flush=True,
            )

        final_score = obs.total_reward
        success = final_score >= SUCCESS_SCORE_THRESHOLD
    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={str(success).lower()} steps={steps} score={final_score:.3f} rewards={rewards_str}",
            flush=True,
        )


async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    seed = int(os.getenv("SEED", "42"))
    for difficulty in ["easy", "medium", "hard"]:
        await run_episode(client, MODEL_NAME, difficulty, seed)


if __name__ == "__main__":
    asyncio.run(main())
