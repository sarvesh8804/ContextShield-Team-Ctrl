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

UnitForge challenges RL agents to navigate increasingly difficult, multi-step chains of compound scientific unit conversions. It exposes a simple, robust objective function based solely on absolute math accuracy, giving dense step-level reward signals to evaluate models on strict precision rather than subjective human-in-the-loop tasks.

---

## Baseline Scores

Baseline (Qwen3-30B-A3B via HF router, temperature=0.0):

| Task | Average Score |
|---|---|
| Easy | 0.78 |
| Medium | 0.49 |
| Hard | 0.21 |

Hard tasks require multi-step unit chain conversions (e.g. `watt_hour_per_kg` -> `kilojoule_per_pound`) where frontier models confidently return plausible but slightly wrong scalar values. This makes UnitForge an optimal target for targeted reasoning fine-tuning via Reinforcement Learning.

## Action & Observation Space

The action space is maximally simple, expecting just a float value representing the converted scalar:

```json
{
  "value": 1.6308
}
```

The observation returns the input value and the expected source and target units:

```json
{
  "task_id": "hard_001",
  "difficulty": "hard",
  "step_number": 1,
  "input_value": 1.0,
  "from_unit": "watt_hour_per_kg",
  "to_unit": "kilojoule_per_pound"
}
```

## Reward Calculation

Grading evaluates pure strictness (`% error`). 

```python
def grade(agent_value: float, correct_value: float) -> float:
    if correct_value == 0:
        return 0.95 if abs(agent_value) < 1e-9 else 0.05
    error = abs(agent_value - correct_value) / abs(correct_value)
    if error < 0.001:   return 0.95  # exact
    if error < 0.01:    return 0.7   # close
    if error < 0.05:    return 0.3   # ballpark
    return 0.05
```

An episode spans 5 conversion steps randomly sampled from the task pool schema.
