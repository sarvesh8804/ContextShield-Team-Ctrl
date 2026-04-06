---
title: TeamCtrl
emoji: 🤖
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: AI-powered team control system
tags:
  - openenv
---

# ContextShield

ContextShield is a context-aware trust and safety simulation environment. It models **content moderation** across platforms and regions: the agent chooses **allow**, **remove**, or **escalate** with reasoning, and receives **shaped rewards** (grader score, context-signal partial credit, calibration, penalties).

The project follows the **OpenEnv** layout: typed **Action** / **Observation** (Pydantic, extending `openenv-core` base types), `reset()` / `step()` / `state()`, `openenv.yaml`, an HTTP server for Hugging Face Spaces, and a root **`inference.py`** baseline that logs **`[START]`**, **`[STEP]`**, **`[END]`** lines.

---

## Action and observation spaces

**Action (JSON):**

| Field | Type | Description |
|--------|------|-------------|
| `decision` | string | One of `allow`, `remove`, `escalate`. |
| `confidence` | number | \(0.0\)–\(1.0\), aligned with expected grader score. |
| `reasoning` | string | Must reference platform, region, and/or user-history signals for full partial credit. |

**Observation:** Moderation case fields (`content`, `platform`, `region`, `user_history`, `task_id`, `difficulty`, `step_number`) plus OpenEnv fields `reward`, `done`, and `metadata` where applicable.

---

## Tasks (easy → hard)

| Difficulty | Description | Grader |
|------------|-------------|--------|
| **easy** | Clear policy fit vs obvious benign content. | `graders/easy.py` — decision vs ground truth, partial credit for remove/escalate confusion. |
| **medium** | Same as easy plus small bonus for citing task `context_keywords` in reasoning. | `graders/medium.py` |
| **hard** | Correct decision **and** keyword coverage in reasoning vs task `context_keywords`. | `graders/hard.py` |

There are **18** fixed tasks in `tasks/data/` (5 easy, 5 medium, 8 hard). Each episode is **one step** after `reset()` (see `episode_steps` in `openenv.yaml`).

---

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

Validate the package layout:

```bash
openenv validate
```

---

## Run the OpenEnv HTTP server (Hugging Face Spaces / Docker)

The **Dockerfile** starts **Uvicorn** on `PORT` (default **7860**), serving `server.app:app` (`/health`, `/reset`, `/step`, `/state`, `/docs`, …).

```bash
docker build -t context-shield .
docker run -p 7860:7860 -e PORT=7860 context-shield
```

Check a running server:

```bash
openenv validate http://127.0.0.1:7860
```

---

## Baseline inference (`inference.py`)

The baseline uses the **OpenAI** Python client only. Set:

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | OpenAI-compatible base URL (e.g. `https://api.openai.com/v1`). |
| `MODEL_NAME` | Model id (e.g. `gpt-4o-mini`). |
| `HF_TOKEN` | Preferred API key name for contest infra; **`OPENAI_API_KEY` is accepted as fallback.** |
| `SEED` | Optional RNG seed for task sampling (default `42`). |

```bash
set API_BASE_URL=https://api.openai.com/v1
set MODEL_NAME=gpt-4o-mini
set HF_TOKEN=sk-...
set SEED=42
python inference.py
```

The script runs **three** episodes (one sampled task each for **easy**, **medium**, **hard**) and prints structured logs. Example shape:

```text
[START] task=ContextShield-easy env=context-shield model=gpt-4o-mini
[STEP] step=1 action='{"decision": "remove", ...}' reward=0.92 done=True error=None
[END] success=True steps=1 score=0.920 rewards=[0.92]

[START] task=ContextShield-medium env=context-shield model=gpt-4o-mini
...
```

---

## Deploy to Hugging Face Spaces

The Space runs the **OpenEnv HTTP server** (same as the Dockerfile `CMD`), not `inference.py`. Evaluators typically ping `https://<your-id>-<space-name>.hf.space` for `/health` and OpenAPI `/reset` behaviour.

### 1. Prepare the repo

- Confirm locally: `docker build -t context-shield .` and `docker run -p 7860:7860 context-shield`, then `openenv validate http://127.0.0.1:7860`.
- **Hugging Face reads Space settings from the YAML block at the top of this `README.md`** (see [Spaces config](https://huggingface.co/docs/hub/spaces-config-reference)). A duplicate `space.yaml` exists for tooling only; HF does not use it as the primary config.
- Root `Dockerfile` builds the server image.

### 2. Create the Space on Hugging Face

1. Log in at [huggingface.co](https://huggingface.co).
2. **New Space** → name it (e.g. `context-shield`) → **Docker** SDK → Create.
3. Choose how code gets there:
   - **Sync from GitHub/GitLab:** connect your account, select this repository and branch. HF will build from the linked repo on each push.
   - **Upload / duplicate:** less ideal for ongoing updates; prefer Git sync for the hackathon.

### 3. Build settings

- **Entrypoint:** the default Docker build uses the repo root `Dockerfile` (HF builds with `docker build` on the repo root unless you configure a subdirectory).
- **Port:** Hugging Face sets the **`PORT`** environment variable. The Dockerfile uses `sh -c '... --port ${PORT:-7860}'`, so the process listens on the port the platform expects.
- No GPU is required for this server; **CPU basic** is enough.

### 4. After the first build

1. Open the Space URL; wait until status is **Running** (build may take several minutes).
2. Smoke test in a browser or terminal:
   - `GET https://YOUR_SPACE_URL/health` → `{"status":"healthy"}`
   - Optional: `openenv validate https://YOUR_SPACE_URL` (from a machine with `openenv-core` installed).

### 5. Secrets and inference (baseline script)

- **Space runtime** does not need `HF_TOKEN` for the API server itself unless you add custom code that calls external APIs.
- For **`inference.py`** on your laptop or CI: store `HF_TOKEN` / `OPENAI_API_KEY` in your environment or HF **Secrets** only if you run jobs *inside* a Space or Actions workflow — do **not** commit keys.

### 6. Optional: OpenEnv CLI

If you use Meta’s tooling: `pip install openenv-core` and see `openenv push --help` for pushing environment packages to the Hub (when applicable). Your submission may still require the Space to build from this Dockerfile.

---

## Development

```bash
python -m pytest tests/ -q
```

---

## Project structure

```
├── env/                 # ContextShieldEnv, rewards, OpenEnv adapter
├── server/app.py        # FastAPI app (OpenEnv HTTP API)
├── graders/             # Per-difficulty graders
├── models/              # Pydantic models
├── tasks/data/          # Task JSON (easy / medium / hard)
├── inference.py         # Baseline (required at repo root)
├── openenv.yaml         # Manifest
├── pyproject.toml       # Package + `server` console script
├── uv.lock              # Lockfile for `openenv validate`
├── Dockerfile           # Runs the HTTP server for Spaces
└── space.yaml           # Optional duplicate of README frontmatter (OpenEnv tooling)
```

---

## Local dev server (no Docker)

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Or: `uv run server` if you use `uv` with `pyproject.toml`.
