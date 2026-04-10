---
title: PatchGym
emoji: ⚙️
colorFrom: red
colorTo: yellow
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - security
  - agent-evaluation
  - tool-use
---

# PatchGym ⚙️

> **An [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compliant reinforcement-learning environment for evaluating AI agents on real-world dependency vulnerability triage.**

PatchGym puts an agent in the role of a security engineer facing a backlog of CVE alerts on a synthetic Python project. The agent must explore the codebase via tool calls, identify which vulnerabilities are actually exploitable, produce a safe fix plan, and resolve transitive dependency conflicts — all within a bounded episode. Every reward signal is deterministic: no LLM judge, no string matching, no subjectivity.

---

## Baseline Scores

Measured over 5 deterministic episodes (`random.seed(42)`, `temperature=0.2`) using `Qwen/Qwen3-30B-A3B` via the HuggingFace inference router. Score = final cumulative reward at episode end (strictly in (0.05, 0.95)).

| Task | Difficulty | Model | Avg Score | Avg Steps | Success Rate |
|---|---|---|---|---|---|
| `severity-ranker` | Easy | Qwen3-30B-A3B | 0.65 | 8.2 | 60% |
| `fix-planner` | Medium | Qwen3-30B-A3B | 0.52 | 8.8 | 20% |
| `conflict-resolver` | Hard | Qwen3-30B-A3B | 0.15 | 6.4 | 0% |

*Success = score > 0.70 (meaningful majority of task milestones solved).*  
*The hard task (conflict-resolver) requires reasoning about transitive dependency traps that current baseline models lack; it is intended to challenge frontier-scale RL models.*

---

## Motivation

Every engineering team running a real codebase deals with this daily: Dependabot fires 12 alerts, a human engineer must decide which packages are actually imported, which CVEs have a safe fix, and whether upgrading Package A will silently break Package B that Package C depends on. GitHub Copilot Autofix, Snyk AI, and similar products are being deployed to automate exactly this workflow — yet no standardized RL evaluation environment exists for it.

PatchGym fills that gap by framing dependency triage as a multi-step tool-use episode — rewarding progressive exploration and penalizing lazy or incorrect submissions.

This makes PatchGym directly useful for evaluating:
- Tool-calling agents navigating real-world engineering workflows
- Reasoning models being fine-tuned with RL on security and DevOps tasks
- Multi-step planning under information asymmetry (not all packages are imported)

### The Information Asymmetry Challenge

Agents must distinguish between installed noise and active risk.

**Reality (Ground Truth):**
- `requests==2.25.1` (INSTALLED, IMPORTED, NO CVE)
- `aiohttp==3.8.1` (INSTALLED, **IMPORTED**, **CVE-2023-3001 found**) -> **CRITICAL FIX**
- `django==3.1.0` (INSTALLED, NOT IMPORTED, CVE-2021-YYYY found) -> **IGNORE (No Exploit Path)**

**Agent View (Initial):**
- `requirements.txt` lists 10 packages.
- `cve_list` lists 5 critical alerts.
The agent must use `check_imports` and `show_cve` to deduce the critical path and test upgrade constraints.

---

## Environment Architecture



**Synthetic project (deterministic seed 42):**
- `requirements.txt` — 5–6 packages at pinned versions
- `imports` — subset actually used in the codebase (not all installed packages are imported)
- `cve_list` — 3–6 CVE records with CVSS severity scores, affected versions, and fix versions

**Three task types:**

| Task | Difficulty | Objective |
|---|---|---|
| `severity-ranker` | Easy | Rank 6 CVEs by actual risk — exploitability × severity. Non-imported packages are lower priority. |
| `fix-planner` | Medium | Produce a conflict-free upgrade plan for all exploitable CVEs. Skip non-imported packages. |
| `conflict-resolver` | Hard | The naive fix triggers a transitive dependency conflict. Find the safe alternative fix path. |

---

## Tasks

### Task 1 — `severity-ranker` *(easy)*

**Objective:** Given 6 CVEs, rank them by actual risk to *this* codebase — not raw CVSS score.

The agent must discover which packages are actually imported (`check_imports`) — a CVE on an installed-but-unused package is lower priority than a lower-severity CVE on a package running in every request.

| Milestone | Condition | Reward Δ |
|---|---|---|
| CVE correctly placed in exploitable tier | Package is imported AND ranked in top-N | +0.30 per CVE |
| CVE correctly placed in non-exploitable tier | Package not imported, ranked lower | +0.30 per CVE |

**Max reward delta:** up to 3 × 0.30 = 0.90 | **Max steps:** 12 | **Starting score:** 0.05
**Episode max score:** 0.05 + 0.90 = **0.95**

---

### Task 2 — `fix-planner` *(medium)*

**Objective:** Produce a valid remediation plan — correct fix version for each exploitable CVE, no unnecessary patches.

Ground-truth fix versions are sourced from the CVE records. The agent must use `get_fix_version` and skip CVEs on packages that aren't imported.

| Milestone | Condition | Reward Δ |
|---|---|---|
| Correct fix version for exploitable package | `get_fix_version` result matched exactly | +0.45 per package |
| Unnecessary fix on non-imported package | Penalised for noise | −0.20 per entry |

**Max reward delta:** 2 × 0.45 = 0.90 | **Max steps:** 12 | **Starting score:** 0.05
**Episode max score:** 0.05 + 0.90 = **0.95**

---

### Task 3 — `conflict-resolver` *(hard)*

**Objective:** Identify all CVE fixes and resolve a transitive dependency conflict that the naive fix introduces.

The agent is not told that a conflict exists. It must discover it via `check_conflicts`.

| Milestone | Condition | Reward Δ |
|---|---|---|
| Conflict trap discovered (`check_conflicts` returns `conflict: true`) | +0.40 |
| Safe resolution submitted and matches correct resolution | +0.40 |
| Naive conflicting fix submitted without checking | −0.10 |

**Max reward delta:** 0.80 | **Max steps:** 12 | **Starting score:** 0.05
**Episode max score:** 0.05 + 0.80 = **0.85**

This task genuinely challenges frontier models: the agent is not told how many packages need updating, whether a conflict exists, or what the safe alternative is. It must reason about dependency semantics from first principles.

---

## Action Space

Each step submits one tool call as a structured JSON action:

```json
{ "command": "check_conflicts", "args": {"package": "urllib3", "version": "2.0.4"} }
```

```json
// Example: Checking actual usage of a package
{
  "command": "check_imports",
  "args": { "package_name": "aiohttp" }
}

// Example: Simulating a remediation attempt (Task 3)
{
  "command": "check_conflicts",
  "args": { "package": "urllib3", "version": "2.0.7" }
}

// Example: Submitting final ranking (Task 1)
{
  "command": "submit_plan",
  "args": {
    "ranking": ["CVE-2023-1001", "CVE-2023-1002"]
  }
}
```

**Available tools:**

| Command | Arguments | Returns |
|---|---|---|
| `list_packages` | — | `requirements.txt` as a dict |
| `show_cve` | `cve_id` | Full CVE record: severity, description, fix version |
| `check_imports` | `package_name` | `{"imported": true/false}` |
| `get_fix_version` | `package`, `cve_id` | Safe upgrade version |
| `check_conflicts` | `package`, `version` | `{"conflict": true/false, "message": "..."}` |
| `submit_plan` | `ranking` / `plan` / `resolution` | Triggers grader — ends episode |

All tool calls execute against in-memory Python dicts. No real pip, no network, no external dependencies beyond `fastapi` and `pydantic`.

---

## Observation Space

Every `/step` response returns a structured JSON observation:

```json
{
  "observation": {
    "task_id":      "hard_conflictresolver_001",
    "step_number":  4,
    "result":       {"conflict": true, "message": "aiohttp>=3.9.0 requires async-timeout>=4.0.3 but 4.0.2 is installed"},
    "error":        null,
    "hint":         "conflict-resolver: ALWAYS run check_conflicts before submitting.",
    "total_reward": 0.15
  },
  "reward": 0.10,
  "done":   false,
  "info": {
    "task_id":    "hard_conflictresolver_001",
    "difficulty": "hard",
    "delta":      0.10
  }
}
```

`result` is the structured output of the last tool call, or `null` on reset.
`hint` provides static task-type context at every step to guide the agent's tool sequence.

---

## Reward Function

The environment maintains a **cumulative episode score** `S ∈ (0.05, 0.95)`:

```
S_0 = 0.05    (episode baseline — strictly > 0 as required by OpenEnv)

At each step t:
  Δ_tool   = tool-use signal (penalty for waste/repetition, bonus for conflict discovery)
  Δ_grader = grader output on submit_plan (0 on all other steps)

  S_t = clamp(S_{t-1} + Δ_tool + Δ_grader, 0.05, 0.95)
```

**Reward components:**

| Event | Δ Reward |
|---|---|
| CVE correctly classified as exploitable / not (severity-ranker) | +0.30 per CVE |
| Correct fix version submitted (fix-planner) | +0.45 per package |
| Conflict trap found via `check_conflicts` before submitting | +0.10 |
| Conflict trap triggered: submitted the naive conflicting fix | −0.10 |
| Conflict-resolver submitted without running `check_conflicts` | −0.05 |
| `show_cve` / `get_fix_version` called on unknown CVE or package | −0.03 |
| Identical tool call repeated | −0.01 |
| Fix submitted for a non-imported package | −0.20 |

Scores are clamped to `[0.05, 0.95]` at every step, satisfying the OpenEnv Phase 2 requirement that task scores are **strictly in (0, 1)**.

---

## API Reference

### `POST /reset`

Starts a fresh, deterministic episode. Loads a new synthetic project from the task pool.

**Request body** (optional):
```json
{ "task_id": "hard_conflictresolver_001" }
```
Valid `task_id` values: `severity-ranker`, `fix-planner`, `conflict-resolver`

**Response:** Initial `Observation` with `step_number=0`, `result=null`.

---

### `POST /step`

Submit one tool call and receive the environment's response.

**Request body:**
```json
{ "command": "check_conflicts", "args": {"package": "aiohttp", "version": "3.9.0"} }
```

**Response:**
```json
{
  "observation": {"task_id": "...", "step_number": 2, "result": {...}, "error": null, "hint": "..."},
  "reward": 0.10,
  "done": false,
  "info": {"task_id": "...", "difficulty": "hard", "delta": 0.10}
}
```

---

### `GET /state`

Returns the full current episode state snapshot (task, step count, cumulative reward, tool call history).

### `GET /healthz`

Returns `{"status": "ok"}` — used by HF Spaces for liveness checks.

---

## OpenEnv Specification Adherence

PatchGym was built from the ground up to satisfy all requirements for the Meta × PyTorch × Hugging Face Hackathon Phase 2 evaluation:

1.  **Strict Determinism:** `random.seed(42)` is set on every `/reset`. Identical tool call sequences yield identical observations and rewards across runs.
2.  **Pure Grader:** Grader logic uses deterministic dictionary/set comparisons (`patch_grader.py`). No LLM-as-a-Judge variance.
3.  **Score Bounds:** Rewards are accumulated and clamped at every step to strictly `[0.05, 0.95]`, satisfying the requirement for final scores within the exclusive `(0, 1)` range.
4.  **Action Primitives:** Implements standard `reset()`, `step()`, and `state()` API via FastAPI endpoints.
5.  **Tool-Use Architecture:** Framed as a multi-step (max 12) trajectory, testing reasoning and planning rather than single-turn QA.

---

## Setup & Usage

### Docker (recommended)

```bash
docker build -t patchgym .
docker run -p 7860:7860 patchgym
# Server available at http://localhost:7860
```

### Local Development

```bash
git clone https://github.com/sarvesh8804/PatchGym-Team-Ctrl
cd PatchGym-Team-Ctrl
python -m venv venv
source venv/bin/activate      # Linux/macOS
# venv\Scripts\activate       # Windows
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### Run Inference Agent

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen3-30B-A3B:novita"
export HF_TOKEN="hf_..."
python inference.py
```

### Quick API Smoke-Test

```bash
# Reset episode
curl -s -X POST http://localhost:7860/reset \
     -H "Content-Type: application/json" \
     -d '{}' | python -m json.tool

# Check for conflicts before patching
curl -s -X POST http://localhost:7860/step \
     -H "Content-Type: application/json" \
     -d '{"command": "check_conflicts", "args": {"package": "aiohttp", "version": "3.9.0"}}' \
     | python -m json.tool
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `API_BASE_URL` | Yes | `https://router.huggingface.co/v1` | OpenAI-compatible API endpoint for the LLM |
| `MODEL_NAME` | Yes | `Qwen/Qwen3-30B-A3B:novita` | Model identifier passed to the OpenAI client |
| `HF_TOKEN` | Yes | — | Hugging Face access token (used as API key) |
| `PORT` | No | `7860` | Port for the FastAPI server |

---

## Technical Implementation Notes

- **Thread safety:** `PatchGymOpenEnv` maintains per-episode state in isolated `PatchGymEnv` instances, making the server safe for concurrent requests without episode state corruption.
- **Determinism:** `random.seed(42)` is applied on every `reset()` call, guaranteeing identical task sequences across episodes and submissions.
- **Grader purity:** All three sub-graders (`grade_severity_ranker`, `grade_fix_planner`, `grade_conflict_resolver`) operate on exact dict comparisons and set intersections — zero LLM involvement, zero variance across runs.
- **Score bounds:** `_cumulative_reward` is clamped to `max(0.05, min(0.95, value))` at every step, satisfying OpenEnv's strict `(0, 1)` exclusive requirement.
- **In-memory execution:** All CVE records, package lists, and dependency conflict data live entirely in Python dicts seeded at startup — sub-millisecond tool execution, zero disk I/O, no network calls.
- **Episode length:** 12 steps maximum — enough for full exploration (list + CVE lookups + import checks + conflict checks + submit) without excessive padding.
