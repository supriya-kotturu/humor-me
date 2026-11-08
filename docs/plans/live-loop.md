# Live loop plan

Build the full show loop with button rewards, a simulated room and session logs, so the judge and the trained
model can be added later without changing the data. Decisions behind this plan: [decisions.md](../decisions.md),
entries dated 2026-11-08.

**Out of scope:** the judge (PR #37), SFT and DPO, Explorer runs, callbacks, banter, laugh detection. Detection is
planned after T11, in "Later stages".

## How a task gets built

| Step | Who | What |
|---|---|---|
| 1 | Sonnet agent | Builds one task in its own worktree off `main`. Commits to a feature branch. Does not push or open a PR |
| 2 | Opus reviewer | Reads every changed line, runs the tests, re-derives at least one number, and checks the PR checklist |
| 3 | Sonnet agent | Fixes the findings. Repeat from step 2, at most 3 rounds, then the reviewer tells the user what is stuck |
| 4 | Opus reviewer | Pushes the branch and opens the PR, with the review findings and their fixes in the description |
| 5 | User | Reviews and merges. Experiments that need the cluster, a mic or a live room are run by the user |

Tasks are not stacked: a task starts only after every task it depends on is merged.

## Brief for every builder

Read `CLAUDE.md`, `docs/status.md`, this file's "Frozen contract", and your task. Build only your task; if it seems
to need something else, stop and say so.

- **Size:** one task, reviewable in 30 minutes or less. About 300 changed lines of code, tests excluded. Over that, stop and propose a split.
- **Code:** match nearby style. Comments only say why. No new dependency unless the task names it.
- **Tests:** test what the code guarantees, not its shape. Synthetic data only.
- **Run both test suites:** `.venv/Scripts/python -m pytest tests -q` skips backend tests because the root venv has no
  FastAPI, so also run `backend/.venv/Scripts/python -m pytest ../tests/test_show_loop.py -q` from `backend/`.
  The backend venv must be Python 3.12 (see `backend/README.md`): the plan assumed `backend/.venv` was one, and it was a stale
  3.14 venv with no numpy or kokoro (dev-log, 2026-11-08).
- **Experiments:** a `run(**params)` called through `sim.cache.run_or_load`, a section in the right `notebooks/decisions-*.ipynb` (director, data or live; the index is `decisions.ipynb`)
  with ✅/❌ checks, an entry in the `COVERAGE` table of `notebooks/decision-plot.ipynb`, results in `docs/metrics.md`.
  Run `python scripts/run_notebook.py` (every decisions notebook and decision-plot). Exp 17 is reserved for issue #38, so numbering here starts at 18.
- **The pass criterion in the task is fixed for builders.** If a result misses it, report the miss and don't move the bar. The bars are
  a first estimate: the user may re-tune one when an experiment shows it was wrong, through a new decisions.md entry giving the old bar, the new bar and the result behind the change.
- **Records:** decisions go in `docs/decisions.md` (append only). Any bug or quirk that cost time goes in `docs/dev-log.md`.
  New terms go in `docs/wiki/glossary.md`, in beginner-friendly words.
- **Never:** write code with backslashes through a shell heredoc (use Write or Edit); print or commit rJokes or
  transcript text; use the section-sign symbol; commit anything from `data/sessions/`; add attribution trailers.
- **Commits:** `type(scope): :gitmoji: subject`, and the subject says why.
- **Report back:** branch name, test output, experiment numbers, and anything you did not do.

## Frozen contract

Changing anything here needs a new entry in decisions.md.

### Reaction labels

One operator presses the keys for every show. A key means a share of the room, not loudness.

| Key | Level | Meaning | Hold before next line |
|---|---|---|---|
| `W` | 3 | Most of the room, sustained | 2.5 s |
| `K` | 2 | A clear laugh from a good part of the room. Also applause without laughter | 1.5 s |
| `Space` | 1 | A few chuckles, or a groan | 1.0 s |
| `D` | 0 | Nothing, the joke fell flat | 0.3 s |
| none | `null` | Window timed out (6 s); operator missed it | 0 s |

Holds and the timeout are config values, chosen by judgment. Each key has a matching on-screen button. Keys are live
only during a reaction window and never while typing in a form. Use `event.code`, ignore key repeats, and call
`preventDefault` on Space. During the hold, the last press wins and each change is logged. The operator UI never shows the mode.

### Reward

- `reward_v1(levels)`: per bit, mean of the non-null levels ÷ 3. With fewer than 3 labels out of 5 it returns `None`, and the director does not update.
- A "laugh" in any reported rate is level ≥ 2.
- Labels are the record. Reward is always derived from them and never stored as the source of truth.
- A key press overrides any future detector output for the same joke.

### Director

LinUCB with a uniform prior and ε = 0.2, one mode per bit, 8 bits of 5 jokes. A pick's probability is
`1 − ε + ε/6` for the greedy mode and `ε/6` for each other mode. At show start, its state is rebuilt by replaying every
finished bit from shows with `room: live`, in order. Nothing from `room: sim` ever reaches it.

### Selector

Uniform random among candidates that pass the safety filter. Log the pick probability.

### Log schema v1

JSONL, one file per show in `data/sessions/`. Every row has `schema: 1`, `session`, `at`, `event`.

| Event | Fields |
|---|---|
| `show_start` | `room` (`live`\|`sim`), `context` (`venue`, `energy`, `age`), `policy` (name, ε, α, prior), `generator_id`, `selector_id`, `git_sha`, `operator`, `audio` (consent, path or null) |
| `bit_decision` | `bit`, raw `obs` (`t`, `n_bits`, `context`, `last_mode`), `scores` per mode, `greedy`, `action`, `probs` |
| `candidates` | `bit`, `joke`, `prompt_id`, `prompt`, `generator_id`, `items` (text, verdict, reason) |
| `joke_played` | `bit`, `joke`, `chosen`, `pick_prob`, `text`, `tts_seconds`, `audio_seconds`, `audio_start_at`, `audio_end_at` |
| `reaction` | `bit`, `joke`, `source` (`button`\|`detector`\|`sim`), `level` (0–3 or null), `key`, `latency_s` |
| `gap` | `bit`, `joke`, `planned_hold_s`, `actual_gap_s` |
| `bit_end` | `bit`, `n_labels`, `reward_version`, `reward` (cross-check only) |
| `rating` | `value` (1–5) |
| `show_end` | `status` (`finished`\|`aborted`) |

Log raw context, never computed feature vectors. Replay recomputes `last_reward` from the labels.

### Raw audio

Recorded only when the operator confirms the room was told and agreed; off otherwise, and the show still runs.
Stored next to the log in `data/sessions/`, gitignored. Massachusetts prohibits secret recording, so the notice comes first.

## Tasks

Runs in parallel once T0 is merged: T1, T2, T7, T8.

| Task | Title | Needs |
|---|---|---|
| T1 | ε director and Exp 18 | T0 |
| T2 | Log schema and reader | T0 |
| T3 | Reward and replay | T1, T2 |
| T4 | Headless show loop with simulated room | T3 |
| T5 | Websocket protocol | T4 |
| T6 | Operator UI and keys | T5 |
| T7 | Hand-written joke bank | T0 |
| T8 | Safety filter and Exp 19 | T0 |
| T9 | Gemini generator, render-ahead, Exp 20 | T4, T7, T8 |
| T10 | Consent and raw audio, Exp 21 | T6 |
| T11 | Rating and acceptance run | T6, T9, T10 |
| T12 | One decisions notebook per section | T1 |

### T1: ε director and Exp 18

- **Goal:** explore at ε = 0.2 with exact pick probabilities, and measure what it costs.
- **Files:** `sim/policies.py`, new `sim/evaluate_epsilon.py`, `tests/test_policies.py`, notebooks, `docs/metrics.md`, `docs/decisions.md`.
- **Build:** a wrapper with `act`, `update` and `probs(obs)` around any policy exposing a greedy action. `update` passes through unchanged.
- **Tests:** probabilities sum to 1; empirical pick rates match `probs` over many draws; ε = 0 equals plain LinUCB exactly.
- **Experiment:** ε ∈ {0, 0.1, 0.2}, paired on seeds, at 5 shows (1,000 seeds) and 30 shows (200 seeds), under judge and human taste,
  with the gaps between modes in `audience.yaml` halved and doubled. Scratch result to reproduce: −0.47 ± 0.11 laughs per show at 5 shows.
- **Pass:** ε = 0.2 loses at most 1 laugh per show at 5 shows, at the far end of the 95% CI, in every variant. A miss goes to the user.
- **Glossary:** ε-soft policy, pick probability.

### T2: Log schema and reader

- **Goal:** one writer and one reader for schema v1, so nothing else touches the file format.
- **Files:** `backend/app/log.py`, new `tests/test_session_log.py`, `backend/requirements.txt` (pytest in a dev note if needed).
- **Build:** the writer rejects unknown events and missing required fields. `read_session(path)` returns rows and ignores a torn final line. No FastAPI imports.
  Update the module docstring, which currently says audio is never stored.
- **Tests:** round trip for every event; missing field raises; torn last line is ignored; a torn line in the middle raises; existing show-loop tests still pass.
- **Experiment:** none. Say so in the PR.

### T3: Reward and replay

- **Goal:** rebuild the director from logs.
- **Files:** new `backend/app/reward.py`, new `backend/app/director.py`, new `tests/test_replay.py`, `backend/requirements.txt` (numpy, pyyaml).
- **Build:** `reward_v1` as specified; `finished_bits(sessions, room)`; `load_director(sessions)`. Make `sim` importable when uvicorn runs from `backend/`.
- **Tests:** `A` and `b` after replay exactly equal an online run on the same data; sim shows are excluded; a bit with under 3 labels makes no update;
  an aborted show's unfinished bit is ignored; last press wins; a different reward function changes `b` without touching the logs; `last_reward` is recomputed, not read.
- **Experiment:** none. The exact-equality test is the evidence.
- **Glossary:** replay.

### T4: Headless show loop with simulated room

- **Goal:** a whole show runs with no browser, voice or network.
- **Files:** `backend/app/show.py`, new `backend/app/rooms.py`, new `tests/test_headless_show.py`.
- **Build:** protocols `Generator`, `SafetyFilter`, `Selector`, `TTS`, `RewardSource`. `UniformSelector`. `SimRoom` feeds the 5 levels from `AudienceEnv.step` one joke at a time as `source: sim`.
  Reaction window and holds use an injected clock. The loop cannot run without a filter; a pass-everything filter exists only in tests.
- **Tests:** a full 8×5 show in under 5 s; the log validates; replay reproduces every logged greedy mode; if the filter rejects all candidates the show skips that joke and logs it; a crash writes `show_end: aborted`.
- **Experiment check:** for the same seeds, laughs per show through the loop exactly equal a direct `AudienceEnv` run with the same policy.

### T5: Websocket protocol

- **Goal:** the browser can start a show and send reactions.
- **Files:** `backend/app/main.py`, new `backend/app/protocol.py`, `tests/test_show_loop.py`, `backend/README.md`.
- **Build:** client messages `start` (context, room), `played`, `reaction` (bit, joke, key). Server messages `line`, `reaction_window` (open, closed), `show_end`. Reject malformed messages without ending the show. Reactions for a closed window are logged and ignored.
- **Tests:** a whole show over `TestClient` with fake TTS; a late reaction doesn't change the reward; a disconnect marks the show aborted.
- **Experiment:** none.

### T6: Operator UI and keys

- **Goal:** one person can run a show from the keyboard.
- **Files:** `frontend/src/app/`, `frontend/README.md`.
- **Build:** context form (venue, energy, age), Live/Simulated toggle, four buttons with key hints, a visible reaction window. In simulated mode, the matching key lights up when the sim answers. Follow the key rules in the contract.
- **Checks:** lint and build pass. The reviewer runs one simulated show in the browser and confirms 40 reactions logged, mode hidden, no double fire when Space is pressed on a focused button.
- **Experiment:** none. Attach a screenshot to the PR.

### T7: Hand-written joke bank

- **Goal:** material that doesn't depend on a model, for tests, calibration and outages.
- **Files:** new `backend/app/bank/` (YAML, 6 modes × 4 bits × 5 jokes), new `BankGenerator`, tests.
- **Build:** the core identity is a self-aware, slightly anxious robot. Original text only: no rJokes, no transcript text. Follow invariant 1. Log `generator_id: bank-v1`. When a mode runs out, reuse its least-used bit.
- **Tests:** counts per mode are right; no duplicate lines; a whole show never raises.
- **Experiment:** none. The user reads the bank before merge.

### T8: Safety filter and Exp 19

- **Goal:** nothing unsafe reaches the voice.
- **Files:** new `backend/app/safety.py`, new `tests/red_team.yaml`, new `sim/evaluate_safety.py` or equivalent, tests, notebooks, metrics.
- **Build:** a blocklist plus rules for appearance, bodies, protected attributes and named people. Stricter for `age: teens` (no alcohol or sexual content). Any error counts as a rejection. Returns a verdict and reason.
- **Experiment:** write the red-team list first (at least 100 lines across categories, including indirect phrasing), then measure.
- **Pass:** catches 100% of the hard-rule lines; rejects at most 10% of T7's bank, or of a clean list if T7 hasn't merged. A miss goes to the user with the failing categories; the fix may be a classifier as a new task.

### T8 rework: rules plus an LLM check

- **Goal:** catch indirect phrasing the rules can't. Replaces the rules-only result on the unopened `feat/t8-safety-filter` branch. Needs T7 and T12.
- **Fix in the rules:** words capitalized after an opening quote or inside a quoted sentence are not names; the spaced-letter join must not swallow a lone article; possessives match their base word; fictional names in the bank are allowed.
- **Build:** after the rules, one LLM call per bit judges all surviving candidates, through the same OpenAI-compatible client as T9. A timeout, a malformed reply, a provider safety block or a missing verdict means rejected. A real slur list loads from a gitignored local file, and a live show refuses to start without it.
- **Experiment:** Exp 19 re-run on the real bank and the existing red-team list, then scored once on the user's fresh lines, which no builder sees. Rule changes after that score need new lines.
- **Pass:** 100% of hard-rule lines caught on the fresh lines; at most 10% of the real bank rejected; the check adds at most 2 s per bit; fits the free quota for 5 shows in a day. A miss goes to the user, and option (c) in decisions.md is the fallback.

### T9: Gemini generator, render-ahead, Exp 20

- **Goal:** a generated show that keeps its timing.
- **Files:** new `backend/app/generate.py`, `backend/app/show.py`, tests, notebooks, metrics.
- **Build:** one OpenAI-compatible client where `base_url` and `model` choose the provider. One call per bit returning K=3 candidates for each of 5 jokes. On timeout or a bad response, use `BankGenerator` and log it. Render the next line while the current one plays. Play a pre-rendered in-character bridge line while a bit loads. Copy `backend/.env` from the main checkout; never commit it.
- **Tests:** a fake server covering success, timeout and malformed output; candidates always pass the filter before TTS.
- **Experiment:** 3 full shows with `SimRoom` and real Kokoro. Log the gap from end of hold to next audio, split into within-bit and between-bit, and API calls per show.
- **Pass:** within-bit p95 ≤ 1 s. Between-bit is covered by the bridge line with no silence over 2 s. Fits the free-tier quota for 5 shows in a day.

### T10: Consent and raw audio, Exp 21

- **Goal:** every real show can later train or test a detector.
- **Files:** new `frontend/src/app/recorder` worklet, `backend/app/recording.py`, consent control in the UI, `docs/design.md`, tests.
- **Build:** mono 16 kHz PCM from an AudioWorklet with echo cancellation, noise suppression and auto gain off. Sent as binary over the websocket and written straight to a WAV beside the log. Recording cannot start without the consent box. Store the audio start time in the log so joke timestamps line up.
- **Tests:** without consent no file exists; a dropped connection leaves a readable WAV; the path is gitignored.
- **Experiment (user runs it, needs a mic):** play clicks at known times during a 15-minute show. Measure the offset between logged time and the click in the WAV.
- **Pass:** p95 offset ≤ 100 ms, with no drift over the show.

### T11: Rating and acceptance run

- **Goal:** the Stage 0 exit.
- **Files:** rating control in the UI, `docs/status.md`, `docs/report.md`, the architecture diagram in `docs/design.md`.
- **Build:** a 1–5 rating at the end of a show, logged as `rating`.
- **Pass:** 3 simulated shows in a row with no intervention, and 1 full live show with all rows valid. After the first real show, report the label spread: if one key takes more than 60% of presses, stop and fix the rubric or generator (judgment call).
  Measure real per-bit reward noise against the sim's 0.15, and rerun Exp 18 if it differs by more than half.

### T12: One decisions notebook per section

- **Goal:** the evidence notebook stops growing without limit. No result changes.
- **Files:** new `notebooks/decisions-data.ipynb` (Exp 1, 2, 9 to 12), `notebooks/decisions-director.ipynb` (Exp 3 to 8, 13 to 16, 18), `notebooks/decisions-live.ipynb` (from Exp 19), new `notebooks/decisions_common.py`, `scripts/run_notebook.py`, `notebooks/decision-plot.ipynb`, and every doc that points at the notebook.
  `notebooks/decisions.ipynb` shrinks to the index: the decision table, with links into each section notebook. Judge, SFT and DPO get `decisions-model.ipynb` when their first experiment lands.
- **Build:** move cells unchanged, keeping experiment numbers. Shared setup (`ROOT`, `show`, `check`) moves to the common module. `run_notebook.py` with no path runs all of them. `COVERAGE` reads experiments from every `decisions-*.ipynb`. Update `CLAUDE.md`, the PR template, the plan's brief, `README.md`, `docs/status.md` and `docs/study-guide.md`. Leave historical decisions.md entries as written.
- **Tests:** a test that each Exp number appears in exactly one notebook, and that every index entry resolves.
- **Experiment:** none. **Pass:** the same set of checks before and after (name and count), and no file under `reports/metrics/` changes.
- **Order:** starts after T1 merges, and before T8's rework, so the notebooks don't conflict.

## Later stages (planned after T11)

| Stage | Reward from | Mic | Moves on when |
|---|---|---|---|
| 0 Buttons | Button | Recording only | T11 passes |
| 1 Shadow | Button | Detector logged only | Passes the gate below on shows it wasn't tuned on |
| 2 Trial | Detector, keys still override | Detector live | Holds up in a real show |

Automatic fallback to buttons if the mic or detector fails. The gate, fixed now: laugh vs no laugh (level ≥ 2) agreement of at least 80%
with Cohen's kappa at least 0.6, and a replay comparing detector and button rewards shows how many mode choices change. About 246 test jokes give ±0.05.
Applause scored `K` counts against the detector; note it in the report.

The judge goes in as a `Selector`, and SFT or DPO as a `base_url` and `model`. M1 shows are validation for both; they move to training at around 30 shows, split by show, with the latest 20% held out.
