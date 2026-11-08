# backend/

FastAPI server for the live show loop. It speaks fixed lines for now, through Kokoro on the CPU;
it grows into the director → generator → safety → TTS loop described in
[docs/design.md](../docs/design.md#architecture).

## Setup

```sh
cd backend
uv python install 3.12  # or any Python 3.12
uv venv --seed --python 3.12 .venv
.venv/Scripts/python -m pip install -r requirements.txt
cp .env.example .env
```

`.env` points at any OpenAI-compatible chat endpoint (see `.env.example`):

| Variable | Example |
|---|---|
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1`, `http://localhost:11434/v1` (Ollama) |
| `LLM_API_KEY` | provider key (empty for Ollama) |
| `LLM_MODEL` | e.g. `llama-3.1-8b-instant` |

## Run

```sh
cd backend
.venv/Scripts/python -m uvicorn app.main:app --reload --port 8000
```

- `GET /health` → `{"status": "ok"}`
- `WS /ws`: currently echoes JSON. Will carry operator events (context, laugh button) in, and
  joke text + audio chunks out.

## Pick a provider first

Comedy timing needs the next line within ~1 s of a laugh ending, so measure time-to-first-token:

```sh
backend/.venv/Scripts/python scripts/latency_probe.py --runs 5   # from repo root
```

## Planned modules

| Module | Role | Week |
|---|---|---|
| `sim/` | simulated audience env + bandit/MDP policies | 2 |
| `show/` | director (RL planner), generator, safety filter, memorization guard | 3–5 |
| `audio/` | VAD, laughter detection, laugh-aware turn-taking | 5 |
| `tts/` | streaming TTS with timing marks | 3–4 |

Python note: audio/ML wheels (faster-whisper, torch) may lag on 3.14. If installs fail in
week 5, recreate this venv with 3.12 or 3.13.

`app/bank/` holds `bank-v1.yaml`, 120 hand-written lines (6 modes x 4 bits x 5 jokes), and `BankGenerator`: model-free material for tests, calibration and outages.
