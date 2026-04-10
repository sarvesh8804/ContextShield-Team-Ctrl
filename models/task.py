from typing import Literal, List, Dict, Any
from pydantic import BaseModel


class CVE(BaseModel):
    cve_id: str
    package: str
    severity: float          # CVSS score 0-10
    fix_version: str
    description: str
    exploitable: bool        # is the package actually imported?


class DependencyConflict(BaseModel):
    package: str
    version: str
    breaks: List[str]        # list of packages that break


class Task(BaseModel):
    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    # The synthetic project
    requirements: Dict[str, str]          # package -> current version
    imports: List[str]                    # packages actually used in code
    cves: List[CVE]
    # For conflict-resolver: which naive fix causes a chain conflict
    conflict_trap: Dict[str, Any] = {}    # {package, naive_fix, conflict_with, safe_fix}
    # Ground truth for grading
    correct_ranking: List[str] = []       # ordered CVE IDs by real risk (severity-ranker)
    correct_plan: List[Dict[str, str]] = []  # [{package, version}] (fix-planner)
    correct_resolution: List[Dict[str, str]] = []  # (conflict-resolver)
