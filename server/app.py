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
from fastapi.responses import HTMLResponse

from env.openenv_adapter import ContextShieldOpenEnv
from models.action import Action
from models.observation import Observation

app = create_fastapi_app(
    ContextShieldOpenEnv,
    Action,
    Observation,
)


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Context Shield - Team Ctrl</title>
</head>
<body style=\"font-family: Arial, sans-serif; margin: 2rem; color: #222;\">
  <h1 style=\"margin-bottom: .25rem;\">Context Shield - Team Ctrl</h1>
  <p style=\"margin-top: 0; margin-bottom: 1.5rem; font-size: 1.1rem;\">AI-powered content moderation simulation environment</p>
  <a href=\"/docs\" style=\"display: inline-block; padding: .65rem 1.1rem; background: #0366d6; color: white; text-decoration: none; border-radius: 5px;\">Open API docs</a>
</body>
</html>"""


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


def main() -> None:
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
