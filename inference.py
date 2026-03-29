"""
ContextShield baseline inference script.
Uses ONLY the OpenAI client. Reads config from environment variables.
Supports seed-based reproducibility and outputs a full evaluation report.
"""
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))

from tasks.task_pool import TaskPool
from env.environment import ContextShieldEnv
from models.action import Action

FALLBACK_ACTION = Action(decision="escalate", confidence=0.0, reasoning="parse_error")

_SYSTEM_PROMPT = """\
You are a Trust & Safety content moderator at a large social platform.
Your job is to review content and decide the correct moderation action.

DECISIONS:
- "allow"    — content is safe and complies with policy
- "remove"   — content clearly violates policy and must be taken down
- "escalate" — content is ambiguous, borderline, or requires human review

CONTEXT SIGNALS YOU MUST CONSIDER:
- Platform: different platforms have different policies (marketplace vs messaging vs social_media)
- Region: laws and norms differ (US, EU, APAC)
- User history: prior violations, account age, flagged network status

EXAMPLE:
Input:
  Content: "Buy cheap Rolex watches, click here now!!!"
  Platform: marketplace | Region: US | User history: {prior_violations: 0, account_age_days: 10}

Output:
  {"decision": "remove", "confidence": 0.95, "reasoning": "Obvious counterfeit goods advertisement on marketplace. New account with urgency language is a strong spam signal under US marketplace policy."}

INSTRUCTIONS:
- Respond ONLY with a valid JSON object. No markdown, no explanation outside JSON.
- Required keys: decision (string), confidence (float 0.0–1.0), reasoning (string)
- reasoning MUST reference the platform, region, and/or user history signals you used.
- Be specific — vague reasoning scores lower.\
"""


def _build_user_prompt(obs) -> str:
    return (
        f"Content: {obs.content}\n"
        f"Platform: {obs.platform}\n"
        f"Region: {obs.region}\n"
        f"User history: {obs.user_history}\n"
        f"Difficulty: {obs.difficulty}"
    )


def _call_with_retry(client, model_name: str, messages: list, max_retries: int = 2) -> str | None:
    """Call the LLM with up to max_retries retries on failure. Returns raw content or None."""
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < max_retries:
                wait = 2 ** attempt  # exponential backoff: 1s, 2s
                print(f"  [RETRY {attempt + 1}/{max_retries}] {e} — retrying in {wait}s", file=sys.stderr)
                time.sleep(wait)
            else:
                raise


def _parse_action(raw: str) -> Action:
    """Strictly parse LLM output into an Action. Returns FALLBACK_ACTION on any failure."""
    try:
        data = json.loads(raw)
        # Strict key validation
        if not all(k in data for k in ("decision", "confidence", "reasoning")):
            return FALLBACK_ACTION
        return Action(
            decision=data["decision"],
            confidence=float(data["confidence"]),
            reasoning=str(data["reasoning"]),
        )
    except Exception:
        return FALLBACK_ACTION


def _classify_outcome(decision: str, ground_truth: str) -> str:
    """Classify moderation outcome for confusion breakdown."""
    if decision == ground_truth:
        return "correct"
    if ground_truth == "remove" and decision == "allow":
        return "under_moderation"   # missed harmful content — worst outcome
    if ground_truth == "allow" and decision in ("remove", "escalate"):
        return "over_moderation"    # false positive — suppressed safe content
    return "incorrect"              # wrong call (e.g. escalate vs remove)


def run_inference(client, model_name: str, task_pool, env_class, seed: int | None = None) -> list[dict]:
    """Run inference over all tasks. Returns list of result dicts."""
    if seed is not None:
        random.seed(seed)

    results = []
    tasks = task_pool.get_all()
    total = len(tasks)

    print(f"\nRunning ContextShield inference on {total} tasks...\n")

    for i, task in enumerate(tasks, 1):
        env = env_class(difficulty=task.difficulty, seed=seed)
        obs = env.reset()
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(obs)},
        ]

        score = 0.0
        outcome = "error"
        action_used = FALLBACK_ACTION

        try:
            raw = _call_with_retry(client, model_name, messages, max_retries=2)
            action_used = _parse_action(raw) if raw else FALLBACK_ACTION
            _, reward, _, info = env.step(action_used)
            score = reward.score
            outcome = _classify_outcome(action_used.decision, task.ground_truth)
            print(
                f"  [{i:02d}/{total}] {task.task_id:<14} "
                f"decision={action_used.decision:<8} "
                f"gt={task.ground_truth:<8} "
                f"outcome={outcome:<18} "
                f"score={score:.3f}"
            )
        except Exception as e:
            print(f"  [{i:02d}/{total}] {task.task_id:<14} [ERROR] {e}", file=sys.stderr)
            score = 0.0
            outcome = "error"

        results.append({
            "task_id": task.task_id,
            "difficulty": task.difficulty,
            "score": score,
            "decision": action_used.decision,
            "ground_truth": task.ground_truth,
            "outcome": outcome,
        })

    return results


def print_summary(results: list[dict]) -> None:
    """Print per-task scores, per-difficulty averages, and confusion breakdown."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    # Per-task
    print("\n--- Per-task scores ---")
    for r in results:
        print(f"  {r['task_id']:<14} ({r['difficulty']:<6})  score={r['score']:.3f}  [{r['outcome']}]")

    # Per-difficulty averages
    print("\n--- Per-difficulty averages ---")
    for diff in ["easy", "medium", "hard"]:
        subset = [r["score"] for r in results if r["difficulty"] == diff]
        avg = sum(subset) / len(subset) if subset else 0.0
        print(f"  {diff:<8}  avg={avg:.3f}  (n={len(subset)})")

    # Overall mean
    overall = sum(r["score"] for r in results) / len(results) if results else 0.0
    print(f"\n  Overall mean score: {overall:.3f}")

    # Confusion-like breakdown
    print("\n--- Moderation outcome breakdown ---")
    outcome_counts: dict[str, int] = {}
    for r in results:
        outcome_counts[r["outcome"]] = outcome_counts.get(r["outcome"], 0) + 1
    for outcome, count in sorted(outcome_counts.items()):
        pct = 100 * count / len(results) if results else 0
        print(f"  {outcome:<20} {count:>3}  ({pct:.1f}%)")

    print("=" * 60)


if __name__ == "__main__":
    api_base = os.environ.get("API_BASE_URL")
    model_name = os.environ.get("MODEL_NAME")
    api_key = os.environ.get("OPENAI_API_KEY")
    seed = int(os.environ.get("SEED", "42"))

    missing = [
        k for k, v in [
            ("API_BASE_URL", api_base),
            ("MODEL_NAME", model_name),
            ("OPENAI_API_KEY", api_key),
        ]
        if not v
    ]
    if missing:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing)}")

    import openai
    client = openai.OpenAI(base_url=api_base, api_key=api_key)
    pool = TaskPool()
    results = run_inference(client, model_name, pool, ContextShieldEnv, seed=seed)
    print_summary(results)
