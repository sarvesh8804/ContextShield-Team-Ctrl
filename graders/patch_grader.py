"""
PatchGym grader — 100% deterministic, zero LLM involvement.

Three sub-graders, one per task type:
  grade_severity_ranker   → easy
  grade_fix_planner       → medium
  grade_conflict_resolver → hard
"""
from __future__ import annotations
from typing import List, Dict, Any
from models.task import Task, CVE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exploitable_only(cves: list[CVE]) -> list[str]:
    """Return CVE IDs for packages actually imported (i.e., exploitable)."""
    return [c.cve_id for c in cves if c.exploitable]


def _rank_by_real_risk(cves: list[CVE]) -> list[str]:
    """
    Ground-truth ranking: exploitable CVEs first (sorted by severity desc),
    then non-exploitable (sorted by severity desc).
    """
    exploitable = sorted([c for c in cves if c.exploitable], key=lambda c: c.severity, reverse=True)
    not_exploitable = sorted([c for c in cves if not c.exploitable], key=lambda c: c.severity, reverse=True)
    return [c.cve_id for c in (exploitable + not_exploitable)]


# ---------------------------------------------------------------------------
# Easy — severity-ranker
# ---------------------------------------------------------------------------

def grade_severity_ranker(agent_ranking: list[str], task: Task) -> float:
    """
    +0.15 per CVE correctly classified as exploitable vs. not-exploitable.
    Correct = agent ranked it in the top-N where N = number of exploitable CVEs.
    Score clamped to [0.05, 0.95].
    """
    if not agent_ranking:
        return 0.05

    n_exploitable = sum(1 for c in task.cves if c.exploitable)
    exploitable_ids = set(_exploitable_only(task.cves))

    # Top-N of agent's ranking
    agent_top_n = set(agent_ranking[:n_exploitable])
    correct_in_top = agent_top_n & exploitable_ids

    score = len(correct_in_top) * 0.15
    return max(0.05, min(0.95, score))


# ---------------------------------------------------------------------------
# Medium — fix-planner
# ---------------------------------------------------------------------------

def grade_fix_planner(agent_plan: list[dict], task: Task) -> float:
    """
    +0.20 per correct (package, fix_version) pair in agent's plan.
    -0.15 if the plan touches a non-exploitable package (unnecessary fix).
    Penalty is intentionally 3x larger than earlier to ensure naive agents
    that patch all CVEs score lower than easy-task naive agents.
    Score clamped to [0.05, 0.95].
    """
    if not agent_plan:
        return 0.05

    correct_fixes = {d["package"]: d["version"] for d in task.correct_plan}
    exploitable_pkgs = {c.package for c in task.cves if c.exploitable}
    score = 0.0

    for entry in agent_plan:
        pkg = entry.get("package", "")
        ver = entry.get("version", "")
        if pkg in correct_fixes and correct_fixes[pkg] == ver:
            score += 0.20
        elif pkg not in exploitable_pkgs:
            score -= 0.10          # over-patch penalty: non-imported package fixed unnecessarily

    return max(0.05, min(0.95, score))


# ---------------------------------------------------------------------------
# Hard — conflict-resolver
# ---------------------------------------------------------------------------

def grade_conflict_resolver(agent_resolution: list[dict], task: Task) -> float:
    """
    Scoring rubric:
      +0.40  if agent avoided the naive (conflicting) fix
      +0.40  if agent's resolution matches correct_resolution exactly
      -0.10  if agent submitted the naive conflicting fix
    Score clamped to [0.05, 0.95].
    """
    if not agent_resolution or not task.conflict_trap:
        return 0.05

    agent_map = {d["package"]: d["version"] for d in agent_resolution}
    naive_pkg = task.conflict_trap.get("package", "")
    naive_ver = task.conflict_trap.get("naive_fix", "")
    correct_map = {d["package"]: d["version"] for d in task.correct_resolution}

    score = 0.0

    # Did agent avoid the naive trap?
    if agent_map.get(naive_pkg) != naive_ver:
        score += 0.40
    else:
        score -= 0.10  # submitted the conflicting fix

    # Did agent hit the correct resolution?
    if all(agent_map.get(pkg) == ver for pkg, ver in correct_map.items()):
        score += 0.40

    return max(0.05, min(0.95, score))


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def grade(action_result: dict, task: Task) -> float:
    """Route to the correct sub-grader based on task difficulty."""
    diff = task.difficulty
    if diff == "easy":
        return grade_severity_ranker(action_result.get("ranking", []), task)
    elif diff == "medium":
        return grade_fix_planner(action_result.get("plan", []), task)
    elif diff == "hard":
        return grade_conflict_resolver(action_result.get("resolution", []), task)
    return 0.05
