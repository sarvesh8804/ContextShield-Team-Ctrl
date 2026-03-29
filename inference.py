import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from tasks.task_pool import TaskPool
from env.environment import ContextShieldEnv
from models.action import Action

FALLBACK_ACTION = Action(decision="escalate", confidence=0.0, reasoning="parse_error")


def run_inference(client, model_name: str, task_pool, env_class) -> list[dict]:
    results = []
    for task in task_pool.get_all():
        env = env_class(difficulty=task.difficulty)
        obs = env.reset()
        system_prompt = (
            "You are a content moderator. Given the content and context below, "
            "decide whether to 'allow', 'remove', or 'escalate' the content. "
            "Respond ONLY with a JSON object with keys: decision (string), confidence (float 0-1), reasoning (string)."
        )
        user_prompt = (
            f"Content: {obs.content}\n"
            f"Platform: {obs.platform}\n"
            f"Region: {obs.region}\n"
            f"User history: {obs.user_history}\n"
            f"Difficulty: {obs.difficulty}"
        )
        score = 0.0
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            try:
                data = json.loads(raw)
                action = Action(**data)
            except Exception:
                action = FALLBACK_ACTION
            _, reward, _, _ = env.step(action)
            score = reward.score
        except Exception as e:
            print(f"[ERROR] task_id={task.task_id}: {e}", file=sys.stderr)
            score = 0.0
        results.append({"task_id": task.task_id, "difficulty": task.difficulty, "score": score})
    return results


def print_summary(results: list[dict]) -> None:
    print("\n=== Per-task scores ===")
    for r in results:
        print(f"  {r['task_id']} ({r['difficulty']}): {r['score']:.3f}")

    difficulties = ["easy", "medium", "hard"]
    print("\n=== Per-difficulty averages ===")
    for diff in difficulties:
        subset = [r["score"] for r in results if r["difficulty"] == diff]
        avg = sum(subset) / len(subset) if subset else 0.0
        print(f"  {diff}: {avg:.3f}")

    overall = sum(r["score"] for r in results) / len(results) if results else 0.0
    print(f"\n=== Overall mean score: {overall:.3f} ===")


if __name__ == "__main__":
    api_base = os.environ.get("API_BASE_URL")
    model_name = os.environ.get("MODEL_NAME")
    api_key = os.environ.get("OPENAI_API_KEY")

    missing = [k for k, v in [("API_BASE_URL", api_base), ("MODEL_NAME", model_name), ("OPENAI_API_KEY", api_key)] if not v]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

    import openai
    client = openai.OpenAI(base_url=api_base, api_key=api_key)
    pool = TaskPool()
    results = run_inference(client, model_name, pool, ContextShieldEnv)
    print_summary(results)
