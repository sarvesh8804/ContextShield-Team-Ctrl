"""
FastAPI server for PatchGym (OpenEnv HTTP API).
"""
from __future__ import annotations
import os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openenv.core.env_server.http_server import create_fastapi_app
from fastapi.responses import HTMLResponse

from env.openenv_adapter import PatchGymOpenEnv
from models.action import Action
from models.observation import Observation

app = create_fastapi_app(PatchGymOpenEnv, Action, Observation)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def root() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>PatchGym — Dependency Vulnerability Triage</title>
</head>
<body style="font-family: Arial, sans-serif; margin: 2rem; color: #222;">
  <h1>PatchGym ⚙️</h1>
  <p>AI agent environment for CVE triage, fix planning, and dependency conflict resolution.</p>
  <a href="/docs" style="padding:.65rem 1.1rem; background:#0366d6; color:white; text-decoration:none; border-radius:5px;">Open API docs</a>
</body>
</html>"""


def main() -> None:
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
