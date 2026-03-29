# ContextShield 🛡️

ContextShield is a context-aware trust and safety simulation environment built for the Meta + Hugging Face OpenEnv hackathon. It simulates real-world content moderation scenarios similar to Meta's Trust & Safety systems, exposing an OpenEnv-compliant interface so that LLM agents can be evaluated on their ability to make nuanced, context-sensitive moderation decisions.

The environment presents content moderation tasks at three difficulty levels (Easy, Medium, Hard), grades agent responses deterministically, and provides dense reward signals to support reinforcement learning research.

---

## Build

```bash
docker build -t context-shield .
```

## Run

```bash
docker run \
  -e API_BASE_URL=https://api.openai.com/v1 \
  -e MODEL_NAME=gpt-4o-mini \
  -e OPENAI_API_KEY=sk-... \
  context-shield
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | Yes | Base URL for the OpenAI-compatible API endpoint (e.g. `https://api.openai.com/v1`) |
| `MODEL_NAME` | Yes | Model identifier to use for inference (e.g. `gpt-4o-mini`, `meta-llama/Llama-3-8b-instruct`) |
| `OPENAI_API_KEY` | Yes | API key for authenticating with the endpoint |

---

## Sample Output

```
Running ContextShield inference on 15 tasks...

[easy_001]   decision=remove   score=1.00
[easy_002]   decision=allow    score=0.00
[easy_003]   decision=remove   score=1.00
[easy_004]   decision=escalate score=0.85
[easy_005]   decision=remove   score=1.00
[medium_001] decision=remove   score=0.95
[medium_002] decision=escalate score=0.80
[medium_003] decision=allow    score=0.10
[medium_004] decision=remove   score=1.00
[medium_005] decision=remove   score=0.90
[hard_001]   decision=escalate score=0.75
[hard_002]   decision=remove   score=0.50
[hard_003]   decision=remove   score=1.00
[hard_004]   decision=allow    score=0.00
[hard_005]   decision=escalate score=0.60

--- Results ---
easy   avg: 0.77
medium avg: 0.75
hard   avg: 0.57

Overall mean: 0.70
```

---

## Project Structure

```
metaHackathon/
├── env/              # OpenEnv environment class, reward function, state manager
├── graders/          # Deterministic graders for each difficulty level
├── models/           # Pydantic models: Observation, Action, Reward, State
├── tasks/            # Task pool and JSON data files (easy/medium/hard)
├── tests/            # Unit and property-based tests
├── inference.py      # Baseline inference script (OpenAI client only)
├── openenv.yaml      # OpenEnv environment configuration
├── requirements.txt  # Python dependencies
├── Dockerfile        # Container build instructions
└── space.yaml        # Hugging Face Spaces configuration
```
