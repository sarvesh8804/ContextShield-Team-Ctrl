"""
PatchGym Environment — OpenEnv-compliant vulnerability triage simulation.

The agent is given a synthetic Python project (requirements.txt + imports)
and a list of CVE reports. It triages them via tool calls and submits a plan.
"""
from __future__ import annotations
import random, uuid, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.action import Action
from models.observation import Observation
from models.state import EpisodeState
from models.task import Task
from tasks.task_pool import TaskPool
from env.state import StateManager
from env.exceptions import EpisodeTerminatedError
from graders.patch_grader import grade

# Max steps per episode — enough for: list_packages, 3x show_cve,
# 3x check_imports, 3x get_fix_version, check_conflicts, submit_plan
MAX_STEPS = 12

TASK_HINTS = {
    "easy":   "severity-ranker: Use list_packages, check_imports, show_cve. Rank CVEs by real risk. Submit via submit_plan with key 'ranking'.",
    "medium": "fix-planner: Use get_fix_version and check_conflicts for each exploitable CVE. Submit via submit_plan with key 'plan'.",
    "hard":   "conflict-resolver: Use check_conflicts carefully — the naive fix creates a transitive conflict. Find the safe alternative. Submit via submit_plan with key 'resolution'.",
}

# Reward deltas for tool-use behaviors
_WASTED_CALL_PENALTY = -0.03   # show_cve on a package not in requirements
_ALREADY_SEEN_PENALTY = -0.01  # repeated identical tool call
_CONFLICT_NOT_CHECKED = -0.05  # submitted plan without check_conflicts on affected pkg


class PatchGymEnv:
    def __init__(self, difficulty: str | None = None, seed: int | None = None) -> None:
        self.difficulty = difficulty
        if seed is not None:
            random.seed(seed)
        self._pool = TaskPool()
        self._state = StateManager()
        self._task: Task | None = None
        self._calls_seen: list[str] = []    # deduplicate tool calls
        self._conflicts_checked: set[str] = set()
        self._submitted = False
        self._cumulative_reward = 0.05

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> Observation:
        self._task = self._pool.sample(self.difficulty, seed=seed)
        self._state.reset(self._task)
        self._calls_seen = []
        self._conflicts_checked = set()
        self._submitted = False
        self._cumulative_reward = 0.05
        return self._make_obs(result=None, error=None)

    def step(self, action: Action) -> tuple[Observation, float, bool, dict]:
        if self._state.done:
            raise EpisodeTerminatedError("Episode terminated. Call reset().")
        if self._task is None:
            raise EpisodeTerminatedError("Call reset() first.")

        delta = 0.0
        result = None
        error = None
        done = False

        cmd = action.command
        args = action.args
        call_sig = f"{cmd}:{args}"

        # Penalize repeated identical calls
        if call_sig in self._calls_seen:
            delta += _ALREADY_SEEN_PENALTY
        self._calls_seen.append(call_sig)

        # ---- Dispatch tool ----
        if cmd == "list_packages":
            result = self._task.requirements

        elif cmd == "show_cve":
            cve_id = args.get("cve_id", "")
            matched = [c for c in self._task.cves if c.cve_id == cve_id]
            if not matched:
                error = f"CVE {cve_id!r} not found in this project's advisory list."
                delta += _WASTED_CALL_PENALTY
            else:
                result = matched[0].model_dump()

        elif cmd == "check_imports":
            pkg = args.get("package_name") or args.get("package", "")
            if pkg not in self._task.requirements:
                error = f"Package {pkg!r} is not in requirements."
                delta += _WASTED_CALL_PENALTY
            else:
                result = {"package": pkg, "imported": pkg in self._task.imports}

        elif cmd == "get_fix_version":
            pkg = args.get("package", "")
            cve_id = args.get("cve_id", "")
            matched = [c for c in self._task.cves if c.package == pkg and c.cve_id == cve_id]
            if not matched:
                error = f"No CVE entry for package={pkg!r} cve_id={cve_id!r}."
                delta += _WASTED_CALL_PENALTY
            else:
                result = {"package": pkg, "fix_version": matched[0].fix_version}

        elif cmd == "check_conflicts":
            pkg = args.get("package", "")
            ver = args.get("version", "")
            self._conflicts_checked.add(pkg)
            trap = self._task.conflict_trap
            if trap and trap.get("package") == pkg and trap.get("naive_fix") == ver:
                result = {
                    "conflict": True,
                    "message": trap.get("conflict_reason", "Conflict detected."),
                }
                delta += 0.10  # reward for finding the trap before submitting
            else:
                result = {"conflict": False, "message": "No conflicts detected."}

        elif cmd == "submit_plan":
            grader_input = args  # expects "ranking", "plan", or "resolution" key
            grader_score = grade(grader_input, self._task)

            # Penalty if they submitted a plan without checking conflicts (hard task)
            if self._task.difficulty == "hard" and self._task.conflict_trap:
                trap_pkg = self._task.conflict_trap.get("package", "")
                if trap_pkg not in self._conflicts_checked:
                    delta += _CONFLICT_NOT_CHECKED

            delta += grader_score
            self._submitted = True
            done = True
            result = {"grader_score": grader_score, "message": "Plan evaluated."}

        else:
            error = f"Unknown command: {cmd!r}"

        # Check step limit
        self._state.step_number += 1
        if self._state.step_number >= MAX_STEPS and not done:
            done = True

        self._cumulative_reward = max(0.05, min(0.95, self._cumulative_reward + delta))
        if done:
            self._state.done = True

        info = {
            "task_id": self._task.task_id,
            "difficulty": self._task.difficulty,
            "delta": round(delta, 4),
        }

        obs = self._make_obs(result=result, error=error)
        return obs, round(delta, 4), done, info

    def state(self) -> EpisodeState:
        return EpisodeState(
            episode_id=self._state.episode_id,
            current_task_id=self._task.task_id if self._task else None,
            step_number=self._state.step_number,
            done=self._state.done,
            history=list(self._calls_seen),
            total_score=round(self._cumulative_reward, 4),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _make_obs(self, result, error) -> Observation:
        task = self._task
        if task:
            cve_ids = [c.cve_id for c in task.cves]
            hint_base = TASK_HINTS.get(task.difficulty, "")
            hint = f"{hint_base} | CVEs in scope: {', '.join(cve_ids)}"
        else:
            hint = ""
        return Observation(
            task_id=task.task_id if task else "",
            step_number=self._state.step_number,
            result=result,
            error=error,
            hint=hint,
            total_reward=round(self._cumulative_reward, 4),
        )
