from typing import Literal, Optional, List
from openenv.core.env_server.types import Action as BaseAction
from pydantic import BaseModel


class PackageFix(BaseModel):
    package: str
    version: str


class Action(BaseAction):
    """
    PatchGym action — one tool call per step.

    command: one of the 6 tool names exposed by the environment.
    args: free-form dict of arguments for that command.
          e.g. {"cve_id": "CVE-2023-1234"} or {"plan": [...]}
    """
    command: Literal[
        "list_packages",
        "show_cve",
        "check_imports",
        "get_fix_version",
        "check_conflicts",
        "submit_plan",
    ]
    args: dict = {}
