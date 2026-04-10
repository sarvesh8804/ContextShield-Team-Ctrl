---
title: UnitForge
emoji: ⚖️
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - precision-reasoning
  - agent-evaluation
---

# UnitForge ⚖️

> **An OpenEnv-compliant reinforcement-learning environment evaluating precise mathematical reasoning and complex unit conversions in frontier models.**

UnitForge challenges RL agents to navigate increasingly difficult, multi-step chains of compound scientific unit conversions. It exposes a simple, robust objective function based solely on absolute math accuracy, giving dense step-level reward signals to evaluate models on strict precision rather than subjective human-in-the-loop tasks. The environment provides dense, step-level reward signals, deterministic evaluation, and full OpenEnv protocol compliance.

---

## Motivation

Most reasoning benchmarks evaluate language models on their logic using subjective semantic matching or multiple-choice formats. They do not evaluate an agent's ability to recursively compute complex float scalars, properly process compound scientific units, or recover from incremental mathematical misalignment. UnitForge fills this gap by framing numerical precision as a multi-step RL episode — rewarding strict calculation rather than ballpark estimation.

This makes UnitForge directly useful for evaluating:
- Numerical reasoning chains in scientific and physics agents
- Reasoning models being fine-tuned with RL for code or math execution
- Precision calibration against hallucinated "plausible" baseline values

---

## Environment Architecture

**Task Generation (Deterministic Seed 42):**

Tasks are sourced from an immutable pool of conversion endpoints across three difficulties:

| Difficulty | Conversions | Average Steps per Episode |
|---|---|---|
| `Easy` | Basic scalars (kg to lbs, km to miles) | 5 |
| `Medium` | Volumetric / Energy (m³ to gallons, kWh to BTU) | 5 |
| `Hard` | Compound chains (W·h/kg to kJ/lb, °C to K traps) | 5 |

Injected edge cases guarantee a reliable challenge:
1. **0 to Non-0** — e.g. 0°C to Fahrenheit, testing standard offset memory
2. **Negative Temperatures** — Testing sign retention across division schemas
3. **Plausible Trap Outputs** — Evaluates precision rather than pattern matching (e.g., 373.15 vs 373)

---

## Action Space

Each step accepts a single JSON payload. The environment expects a precise `float` value denoting the converted calculation:

```json
{ "value": 1.6308 }
```

**Permitted:** Precise float or integer mappings mapped strictly to `action.value`.  
**Blocked:** String-based reasoning, markdown generation outside of the JSON payload, unparsed outputs.

---

## Observation Space

Every `/step` response returns a structured JSON observation mapping the numerical variables needed to perform the current step calculation:

```json
{
  "observation": {
    "task_id": "hard_001",
    "difficulty": "hard",
    "step_number": 3,
    "input_value": 1.0,
    "from_unit": "watt_hour_per_kg",
    "to_unit": "kilojoule_per_pound"
  },
  "reward": 0.95,
  "done": false,
  "info": {
    "task_id": "hard_001",
    "difficulty": "hard"
  }
}
```

---

## Reward Function

The environment utilizes a pure computational grader mapping standard percent error to scaled, bound-checked open boundaries:

```python
def grade(agent_value: float, correct_value: float) -> float:
    if correct_value == 0:
        return 0.95 if abs(agent_value) < 1e-9 else 0.05
    error = abs(agent_value - correct_value) / abs(correct_value)
    if error < 0.001:   return 0.95   # exact
    if error < 0.01:    return 0.70   # close
    if error < 0.05:    return 0.30   # ballpark
    return 0.05                           # completely missed
```

Scores are definitively clamped to `[0.05, 0.95]` at every execution, fulfilling the strict **OpenEnv Phase 2 requirement that task scores are strictly in (0, 1)**.

---

## API Reference

### `POST /reset`

Starts a fresh, deterministic iteration across 5 consecutive mathematical evaluations.
**Response:** `Observation` — Initial parameters and units for step 1 computation.

---

### `POST /step`

Submit one conversion value and receive the environment's subsequent query.
**Request body:**
```json
{ "value": 3412.14 }
```

---

### `GET /state`
Returns the full episode trace snapshot (step history, cumulative performance, action logs).

### `GET /healthz`
Returns `{"status": "ok"}` — utilized by HF Spaces for active container liveness probes.

---

## Setup & Usage

### Docker (recommended)

```bash
docker build -t unitforge .
docker run -p 7860:7860 unitforge
# Server available at http://localhost:7860
```

### Local Development

```bash
git clone https://github.com/sarvesh8804/ContextShield-Team-Ctrl
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### Quick API Smoke-Test

```bash
curl -s -X POST http://localhost:7860/step \
     -H "Content-Type: application/json" \
     -d '{"value": 1.63293}' | python -m json.tool
```

---

## Baseline Scores

Measured over 5 deterministic episodes (`random.seed(42)`, `temperature=0.0`) using `Qwen/Qwen3-30B-A3B` against the HuggingFace inference router.

| Task | Difficulty | Model | Avg Score | Avg Steps |
|---|---|---|---|---|
| `simple-conversion` | Easy | Qwen3-30B-A3B | 0.78 | 5.0 |
| `energy-routing` | Medium | Qwen3-30B-A3B | 0.49 | 5.0 |
| `compound-scalars` | Hard | Qwen3-30B-A3B | 0.21 | 5.0 |

*The hard task effectively nullifies reasoning capabilities in base 30B parameter architectures. Only agents executing precise python interpreter sub-calls or highly fine-tuned models can consistently score above `0.90`!*

---

## Technical Implementation Notes

- **Float Bounds:** `unit_grader.py` handles direct `1e-9` mathematical checks and enforces strict OpenEnv `0.05`/`0.95` boundaries for absolute float equivalence matching.
- **Determinism:** `random.seed(42)` is dynamically applied ensuring complete reproducible test outputs during local or CI runs.
- **Micro-Services:** Pydantic is utilized to parse exact decimal inputs reliably from LLM JSON actions, stripping out common formatting errors.
