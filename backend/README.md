# backend/

FastAPI server for the live show loop: director, bank generator, safety filter, Kokoro voice on the CPU and a
reward source (operator keys or the simulated room), as described in [docs/design.md](../docs/design.md#architecture).

## Setup

```sh
cd backend
uv python install 3.12  # or any Python 3.12
uv venv --seed --python 3.12 .venv
.venv/Scripts/python -m pip install -r requirements.txt
cp sample.env .env
```

`.env` points at any OpenAI-compatible chat endpoint (see `sample.env`):

| Variable | Example |
|---|---|
| `GEMINI_BASE_URL` | `https://api.groq.com/openai/v1`, `http://localhost:11434/v1` (Ollama) |
| `GEMINI_API_KEY` | provider key (empty for Ollama) |
| `JOKE_WRITER_MODEL` | e.g. `llama-3.1-8b-instant` |
| `JOKE_WRITER_PROVIDER` | `openai` (default, the three above) or `anthropic` (official SDK, key in `ANTHROPIC_API_KEY`, `JOKE_WRITER_MODEL=claude-haiku-5-5`) |

## Run

```sh
cd backend
.venv/Scripts/python -m uvicorn app.main:app --reload --port 8000
```

- `GET /health` → `{"status": "ok"}`
- `WS /ws`, JSON text frames plus binary WAV frames from the server. A show runs beside the receive loop.

| Client sends | Fields | Meaning |
|---|---|---|
| `start` | `context` (`venue`, `energy`, `age`), `room` (`sim`\|`live`), `operator`, `consent` (bool, default false) | Start a show. One at a time. `consent: true` means the operator confirmed the room agreed to be recorded |
| binary frame | 16-bit mono 16 kHz PCM, under 64 KiB | Room audio, only after `record`. Refused with an `error` when no recording was ever agreed; dropped silently once it ended |
| `audio_start` | `wall_ms` (epoch ms) | Sent before the first audio frame: when the first sample was captured. Sets `start_at`; the arrival-time estimate is the fallback |
| `audio_health` | `produced_s`, `elapsed_s`, `gap_ms`, `wall_s`, `rate` (numbers) | Every ~5 s of recording: audio the microphone delivered against capture time, silence filled for gaps, wall time, device rate. Warns below 99% of either; diagnostics only, not in the session log |
| `audio_failed` | `reason` (optional text) | The browser could not open the microphone; the show runs unrecorded and the reason comes back in `record_stop` |
| `played` | none | The browser finished playing the last audio |
| `pause` / `resume` | none | The operator paused or resumed (the browser has already suspended or resumed playback). Written as a `pause` / `resume` row on the show clock. Refused with an `error`, the show going on, when no show is running (also during the microphone prompt), on `pause` while paused, and on `resume` while not paused. While paused the show holds: no line, bridge or cached line starts (the server waits for `played` with no timeout, and awaits the resume before the next one), and the reaction window and the hold stop their timers and resume with the time left; keys pressed meanwhile still count. A pause left open when the show ends runs to the end |
| `reaction` | `bit`, `joke`, `key` (`W`, `K`, `Space`, `D`) | A key press. Counts only while that joke's window is open |
| `rating` | `value` (integer 1 to 5) | After `show_end`, finished or aborted, on the same connection: appends one `rating` row to that show's log. One per show; refused (an `error`, the show's log untouched) when there is no finished show, the show is still running, or it was already rated |

| Server sends | Fields | Meaning |
|---|---|---|
| `line` | `text` | The next line; the WAV follows as a binary frame. Send `played` when it ends |
| `reaction_window` | `state` (`open`\|`closed`), `bit`, `joke` | Opens when audio ends, closes at the end of the hold (or on timeout) |
| `sim_reaction` | `bit`, `joke`, `key` | `sim` room only: the key the simulated room "pressed" |
| `record` | none | Sent after a `start` with `consent`: begin sending audio. The show waits for the first frame (15 s) |
| `record_stop` | `reason` (text or null) | Stop the microphone, sent once when the recording ends for any reason: the show ended, no audio arrived in time, the browser failed, or the disk refused a write (the show goes on) |
| `show_end` | `status` (`finished`\|`aborted`), `ratable` (bool) | The show is over. `ratable` is false when it died before writing its log (nothing to rate; the `error` before it says why). The connection stays open for a `rating` |
| `rating_saved` | `value` | The rating row was written |
| `pause_state` | `paused` (bool) | The pause or resume row was written; the browser keeps (or corrects itself to) this state. A refused one comes back as an `error` instead |
| `error` | `message`, optional `about` | A bad client message or a failed start or show. `about: "pause"` marks the refusal of a `pause` or `resume`. A bad message never ends the show |

**What a pause does to the data (D-92):** every pause and resume is a log row on the show clock, and the recording keeps running, so `paused_at - show_start.audio.start_at` is the offset in the WAV. Only a joke whose playback was interrupted loses its label: it keeps its `reaction` row but counts as unlabeled in the bit's reward, in the live show and in replay (the rest of the bit still counts if 3 of 5 labels remain), and its audio should not be used to train the laugh detector (`app.pauses.paused_jokes`). A pause in the reaction window or between lines or bits costs nothing, apart from a pause pressed within milliseconds of a line ending, which can drop that line. Replay assumes the server clock does not step backwards. A show that was paused or cut short keeps its finished bits for the director, but its rating is left out of the reported rating (`python -m sim.acceptance` lists it as not counted). Pause only when you must: each pause costs the label of the joke it lands on.

The mode the director picked is never sent. Presses outside their window are not logged as reactions (replay counts every
row), only noted by the Python logger. Within a hold the last press wins and each change is a `reaction` row.
By default the show uses the bank and a bank-only filter (`BankOnlySafety`), which needs no key. `JOKE_SOURCE=llm` switches to the
real stack (`app/generate.py`): one model call per bit for 5 jokes x 3 candidates, the safety filter's rules plus one LLM verdict call
per bit, the next line rendered while the current one plays, and filler while a bit loads: cached lines from the best bits of earlier live shows (`app/cache.py`, rules re-checked for this room's age, a teen room only from teen shows, never rated, logged in a `cache` row) at the show open and between bits, then in-character bridge lines. A failed call is retried
twice inside a deadline; a failed generation or a bit whose candidates are all rejected falls back to the hand-checked bank (rules only, no API call), logged in
`candidates`. A `live` room refuses to start without `app/blocklist.local.txt`.

```sh
.venv/Scripts/python -m uvicorn app.main:app --port 8000 --env-file .env   # with JOKE_SOURCE=llm in .env
```

## Acceptance run (T11, the Stage 0 exit)

The user runs the shows; `sim/acceptance.py` scores them. Needed: 3 simulated shows in a row, then 1 live show.

**Before the live show (setup rules; they cost a run in Exp 21 when skipped):**

- No Bluetooth headset: use the laptop microphone and speakers (a Bluetooth headset drops the mic to a phone-call codec and shifts the audio clock).
- Windows Sound settings, Communications tab: **Do nothing** (otherwise Windows ducks the robot's voice when it hears a "call").
- Microphone properties, Advanced: **Audio enhancements off** (with them on, echo cancellation removed the laptop's own output from the recording).
- The room is told the show is recorded and agrees, then tick the consent box. Without it the show runs and nothing is recorded.
- `backend/.env` has the Gemini key and `backend/app/blocklist.local.txt` exists (a live room refuses to start without it).

**Start the backend from the main checkout** (`D:/vs-code-workspace/humor-me`, not a worktree: the key and the blocklist are there), in a Python 3.12 venv as in Setup (on the operator laptop the working one is `D:/vs-code-workspace/humor-me-demo/backend/.venv`; the main checkout's `backend/.venv` is a stale 3.14 venv):

```sh
cd backend
.venv/Scripts/python -m uvicorn app.main:app --port 8000 --env-file .env   # with JOKE_SOURCE=llm in .env
cd ../frontend && npm run dev   # then open http://localhost:3000
```

Set `JOKE_SOURCE=llm` in `backend/.env` (a one-command prefix differs between Bash and PowerShell, and `--env-file` reads it either way). Simulated shows can run on `JOKE_SOURCE=bank` (no quota) or `llm`; the live show uses `llm` (decisions-detail.md 2026-11-15).

**Run:** (1) three Simulated shows back to back, untouched; (2) one Live show, you press W / K / Space / D for every joke, consent ticked; (3) after each show answer the rating prompt (keys 1 to 5, or Skip; a show that failed before it began shows its error and no prompt). Then score them. The session ids are the file names in `data/sessions/` without `.jsonl`:

```sh
# from the repo root, root venv. Plain: prints the report and saves nothing, so try ids freely.
python -m sim.acceptance <sim id 1> <sim id 2> <sim id 3> <live id>
# When the right shows are in the list, keep it as the Exp 22 evidence (replaces the last record):
python -m sim.acceptance <sim id 1> <sim id 2> <sim id 3> <live id> --record
```

Pass only the shows meant to count: an unreadable log among the ids fails check (a).

It prints a table per show (valid rows, status, bank share, bridges, label counts, pauses with paused seconds and jokes dropped, rating, audio coverage), a reported rating that leaves out paused and aborted shows (they stay listed as not counted), and the four verdicts: (a) 3 simulated in a row, (b) a valid finished live show, (c) label spread (one key over 60% of presses is flagged, reported only), (d) per-bit reward noise against the sim's 0.15, with "RERUN Exp 18" when it is more than half away. With one live show the noise rests on 8 bits and the interval is wide; the output says when the call is a guess. `--record` saves numbers and the session ids only, in `reports/metrics/acceptance.json` (no text, no audio, no paths); then run `python scripts/run_notebook.py` to refresh the Exp 22 section, which lists the recorded ids. Logs and recordings in `data/sessions/` are gitignored: never commit them.

## Comparing generator models (Exp 24)

Compares Gemini Flash-Lite (today's), Gemini Flash and Claude Haiku 5.5 on 60 bits (30 are voted on; all 60 count toward failures and speed). The bar is fixed in [the plan](../docs/plans/live-loop.md) (T13). Run from the repo root, in the root venv, in the main checkout (the keys are there).

1. **Keys in `backend/.env`** (names as in `backend/sample.env`): `GEMINI_BASE_URL`, `GEMINI_API_KEY` and `ANTHROPIC_API_KEY` (a paid key; Claude is not free). The models are named by the harness, not by `JOKE_WRITER_MODEL`. If `app/blocklist.local.txt` exists the filter's rules use it, as in a live show.
2. **Generate (about 50 minutes unattended: 60 bits per model, paced for the free tier, resumable):** `.venv/Scripts/python -m sim.compare_generators run --env-file backend/.env`. Stop and rerun any time; finished calls are not repeated. A bad key, a wrong model name or any unexpected error on a model's first call stops the run and saves nothing. `--redo-failed` adds a new attempt where the last one failed for a key, setting or provider-availability reason (every attempt is kept and counts toward the failure rate); `--models gemini-flash-lite,claude-haiku-5-5` picks models, `--n 72` adds prompts, `--ballot` sets how many of the first prompts are voted on (30).
3. **Ballot:** `python -m sim.compare_generators export` (the first 30 prompts) (after `run` has finished) writes `ballot.md` and `votes.csv` under `data/model_comparison/gen-<digest>/` and the key apart in `data/model_comparison/key/`. A model with no safe thread for a prompt is not shown and is ranked last for it.
4. **Vote (about 40 minutes), then judge:** `python -m sim.compare_generators vote` shows each prompt and asks for the order, best first (`BAC`). It saves after every answer, `s` skips, `q` quits; run it again to continue. Don't open the `key` folder. Or fill `votes.csv` by hand from `ballot.md`.
5. **Judge (runs on Explorer, Qwen2.5-7B-Instruct, a few hours at most, resumable; the default judge, no API key):**
   - `python -m sim.compare_generators judge --export-bundle data/model_comparison/judge` after you have voted on every ballot prompt and `run` has finished. It writes `calls.jsonl` (our own jokes, your rankings, no model names). **Smoke test first:** add `--smoke 3` (9 calls), run it on the cluster with `LIMIT=3 sbatch ...` or the full smoke job, and check `replies.jsonl` (see `explorer/README.md`, "Exp 24 judge").
   - Upload `calls.jsonl` to the cluster (OOD Files app) and run `sbatch explorer/judge_generators.sbatch`; resubmit the same command if it stops, and again if it exits with "calls were not rankings".
   - Download `replies.jsonl` and run `python -m sim.compare_generators judge --import-replies <file>`. It scores the judge leave-one-out and prints the verdict; only if it passes top pick, pairwise order and per-model bias do its rankings of prompts 31 to 60 join yours. A run with unanswered calls is marked incomplete and joins nothing.
   - **First model to try: Ollama's `qwen3.8:27b`.** Same bundle, but export with `--judge-provider ollama --export-bundle data/model_comparison/judge-ollama` (smoke: add `--smoke 3`), upload `calls.jsonl` to `data/model_comparison/judge-ollama/` on the cluster, run `sbatch explorer/judge_ollama.sbatch` (smoke: `LIMIT=3 sbatch ...`), download `replies.jsonl` and run `python -m sim.compare_generators judge --judge-provider ollama --import-replies <file>`. If you can reach an Ollama server directly (started with `OLLAMA_CONTEXT_LENGTH=32768`), `judge --judge-provider ollama` with no bundle talks to it at `OLLAMA_URL` (default `http://127.0.0.1:11434/v1`). Either way the judge refuses prompts that do not fit the context, and a reply that shows Ollama read far fewer tokens than sent is treated as truncation, never as an answer. Qwen2.5-7B (`judge_generators.sbatch`) is the fallback.
   - `--judge-model claude-haiku-5-5` (needs `ANTHROPIC_API_KEY`, about $0.30) or `gemini-flash-lite` runs a judge through an API instead, with no bundle.
6. **Score:** `python -m sim.compare_generators score` prints the table and the pick, and saves numbers only to `reports/metrics/model_comparison.json`. Then `python scripts/run_notebook.py` to refresh the Exp 24 section and commit the metrics file.

Everything under `data/model_comparison/` (joke text, ballot, key, votes) is gitignored: never commit it. To run a show on Claude, set `JOKE_WRITER_PROVIDER=anthropic` (the model defaults to `claude-haiku-5-5`); the safety check stays on Gemini unless `SAFETY_CHECK_PROVIDER=anthropic` is set too, so leave it for now.

## Cached bits run (Exp 23)

After a live show, run three simulated shows on `GENERATOR=llm`, then score them from the repo root. Plain, it prints and saves nothing; `--record` keeps numbers and ids only (`reports/metrics/cached_bits.json`):

```sh
python -m sim.evaluate_cached <sim id 1> <sim id 2> <sim id 3> [--record]
```

Bar: show-open silence at most 2 s on each of the 3. A show whose bit 0 had no cached bit does not count. Between-bit silence is printed beside it, with no bar.

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
