"""
FastAPI application for ContextShield (OpenEnv HTTP API).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openenv.core.env_server.http_server import create_fastapi_app

from env.openenv_adapter import ContextShieldOpenEnv
from models.action import Action
from models.observation import Observation

app = create_fastapi_app(
    ContextShieldOpenEnv,
    Action,
    Observation,
)


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
