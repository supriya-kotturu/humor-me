# Status (read this first in a new session)

Last updated: 2026-11-15 (Exp 22 recorded). Update this file whenever the state below changes.

Learning the material from scratch (RL foundations → our experiments → Explorer runbook):
[study-guide.md](study-guide.md).

## Where we are

| Area | State |
|---|---|
| Milestone | M1 target = **Dec 8, 2026**, spillover by Dec 12. **Dec 14:** final presentation (≤ 15 min, team) and final report due. RL course final project. Plan: [roadmap.md](roadmap.md) |
| Workflow | One feature per branch off `main`, PR reviewable in ≤ 30 min, backed by a metric; **no stacked PRs**. Every experiment is a `run()` + a section in a section notebook ([data](../notebooks/decisions-data.ipynb), [director](../notebooks/decisions-director.ipynb), [live](../notebooks/decisions-live.ipynb)), indexed in [notebooks/decisions.ipynb](../notebooks/decisions.ipynb) |
| **RL simulator** | Done: `sim/env.py` (room's truth in `sim/audience.yaml`), director decides **per bit** (8 bits × 5 jokes) |
| **Director policy** | **LinUCB + uniform prior**: 97% of oracle, about 95% of the true best (Exp 16), +18% vs always-Neurotic (notebook Exp 4). Holds under human taste too: 96% of oracle (Exp 13) |
| Within-show adaptation | Open problem: the director reads rooms, not a room *turning* (Exp 5: 95% → 80% of oracle after a mid-set shift; Exp 6: longer sets don't help) |
| Show state (A → B → C) | **A** (tonight's record as features): not adopted, hedges (Exp 7). **B** (career + tonight's correction): not adopted, no speed passes (Exp 8). **C1** (MDP with in-show exploration and fatigue planning): negative result: all three runs fail, including after fixing a bug; it reads a turn better but loses in steady shows, so the demo keeps LinUCB (Exp 14, 15). **C2** (callbacks): after the demo and DPO |
| Decision notebooks | Split by section so no notebook grows without limit: `decisions-data` (Exp 1, 2, 9–12), `decisions-director` (Exp 3–8, 13–16, 18), `decisions-live` (Exp 19 onward: 19, 20, 21, 22); `decisions.ipynb` is the index. `decisions-model` comes with the first judge/SFT/DPO experiment (Exp 17 is reserved for the judge). All 94 claim checks ✅. Run `python scripts/run_notebook.py` (all notebooks) before committing |
| Decision plots | [notebooks/decision-plot.ipynb](../notebooks/decision-plot.ipynb): the deck's results redrawn from `reports/metrics`, with checks that each chart still matches its slide and that every experiment has a chart or a stated reason (15 charted, 5 with a stated reason). PNGs in `reports/figures/decision-plot/`, swapped into slides 14, 18, 20, 29, 30, 31 and 33 of `docs/slides/humor-me.pptx` and `slides.md`, and added to the Google Slides copy along with slides 15, 16 and 25 |
| Tests | `.venv/Scripts/python -m pytest tests -q`: 415 passing, 3 skipped, synthetic data only |
| rJokes | R1 prep done (403k jokes). **R2 done: 90k labeled**, `modes.jsonl` on the laptop (30k copy kept as `modes_30k.jsonl`). **Precision rule met:** every mode within ±0.02 (optimist 1,170 jokes, ±0.016). **Human vs judge ranking:** agree only in part (neurotic up, wordsmith down, optimist clearly disagrees; Spearman ρ +0.20 to +0.37), and mode makes a small difference in upvotes (win rates 0.480–0.527, Exp 10, [metrics.md](metrics.md#human-vs-judge-mode-ranking-2026-11-01)). **Correlation map:** no flag or safety label needs an extra adjustment, so Exp 10 stands (Exp 12). **Hand check (105 jokes):** mode labels 92% precise at the labeled mode mix, misses lean to wordsmith; judge safety flags caught 0 of 16 human-flagged rows (Exp 9) |
| Explorer (NEU HPC) | Working env: Python 3.13 module, `torch==2.5.1+cu121`, `transformers` 4.57, V100 in float16, 5.2 jokes/s in batch ([explorer/README.md](../explorer/README.md)) |
| Transcript pipeline | **Paused 2026-10-26**. Bits F1 0.80 on Bargatze; kept for structure stats + report appendix |
| Report | [report.md](report.md): overview written (user + AI), results table filled for Exp 1–7 |
| Slides | [Story deck](https://docs.google.com/presentation/d/1xfH1YYanHXuAsJoo5ZYPF2OOkr8EKGbYLoE5kKo_UE0/edit) (Google Slides, owner-only): 59 slides through Exp 16, all visible for a detailed two-presenter talk, with native result tables and the plot graphs; no live demo yet. `docs/slides/humor-me.pptx` has the same story in 39 slides with the plot graphs and full Q&A notes, restyled to match the Google deck. **Update deck (Nov 15):** `docs/slides/humor-me-update-2026-11-15.pptx`, 13 slides on what changed since Nov 1 (live loop T1 to T10, Exp 18 to 21), monologue and Q&A notes in `docs/slides/slides-update-2026-11-15.md`; not yet in the Google copy (user) |
| Wiki | [docs/wiki/](wiki/README.md) + [glossary](wiki/glossary.md), mirrored to the GitHub wiki by `scripts/sync_wiki.py` after merges |
| **Backend / frontend (demo)** | **The full loop runs headless** against the simulated room (`run_show`), with the director rebuilt from logs. The websocket carries it (T5: `start`, `played`, `reaction` in; `line`, `reaction_window`, `sim_reaction`, `show_end`, `error` out). The operator UI (T6) runs simulated and live shows from the keyboard; **revamped (#53):** a full-width mirrored wave (lilac to electric blue) with the line as a subtitle, a lilac-tinted theme in JetBrains Mono ([screenshot](../reports/figures/operator-ui.png)). **Safety filter (T8):** rules, then one LLM call per bit, failing closed; 29/30 on the user's fresh lines against a 75% bar (Exp 19); mild indirect digs can slip, and nobody pre-approves scripts. **Generator (T9):** `GENERATOR=llm` runs the Gemini generator (one call per bit, 3 candidate threads of 5 jokes), the safety check, render-ahead, bridge lines between bits and the bank as fallback (rules-checked); retries are bounded by a deadline. A bit plays one candidate thread (user decision). Exp 20 (3 shows, sim room): within-bit dead air p95 0.002 s, longest between-bit 0.004 s, about 83 requests for 5 shows against an assumed quota of 500 a day (about 140 requests in one day hit no daily limit); the bank supplied 0% of the jokes (12.5%, 12.5% and 0% in the run before). The session log is schema v2: bridge lines are `bridge` rows, and v1 logs still read ([metrics.md](metrics.md#generated-show-timing-exp-20-2026-11-14)). **Raw audio (T10) is built:** a consent box, mono 16 kHz capture over the websocket into a WAV beside the log, `audio.start_at` in `show_start`; T10 is merged (#51), and recording runs on the mic's capture clock (100% of real time with Kokoro at full speed; the first AudioWorklet capture lost a third). **Exp 21 (click alignment) met its re-tuned bar:** 30/30 beeps found, a constant offset of +241 ms (includes the speaker's output latency, reported not barred), spread 20 ms, drift -2 ms over 294 s; the first bar (absolute offset at most 100 ms) is missed at 261 ms, and the user re-tuned it ([metrics.md](metrics.md#click-alignment-exp-21-2026-11-15)). Setup for any recorded show: no Bluetooth headset, Windows Communications "Do nothing", mic Audio enhancements off. **T11 is merged and the acceptance run passed (Exp 22, Stage 0 done):** 3 simulated shows clean, 1 live show clean (40 jokes, audio 100%, rated 3), no key over 60%, real per-bit reward noise 0.121 vs the sim's 0.15, so no Exp 18 rerun. One simulated show was 75% bank because the LLM safety check timed out on 5 bits; its timeout is fixed in a follow-up ([metrics.md](metrics.md#acceptance-run-exp-22-2026-11-15)). T11 built the rating prompt after `show_end` writes a `rating` row to the same log (replay ignores it), and `sim/acceptance.py` scores 3 simulated shows in a row, 1 live show, label spread and real per-bit reward noise against the sim's 0.15 (Exp 22, run instructions in [backend/README.md](../backend/README.md#acceptance-run-t11-the-stage-0-exit)). The first live show uses Gemini (user, decisions.md 2026-11-15) and needs `backend/app/blocklist.local.txt`. No live room before the mic was proven (user); it now is. Rewards come from operator keys first; a laugh detector follows in stages, with keys as the fallback. Biggest risk to Dec 8 |

## Open PRs

| PR | What |
|---|---|
| #37 | Judge API (Tingyu, Exp 17), overdue since Nov 8 |

## Models and providers

| Use | Provider / model | Notes |
|---|---|---|
| Live show loop | Gemini `gemini-flash-lite-latest` (`LLM_MODEL`) | TTFT ~0.6 s measured |
| Offline pipeline (S4/S5) | Gemini `gemini-flash-latest` (`PIPELINE_MODEL`) | Paused with the transcript pipeline |
| rJokes mode/safety labels | `Qwen/Qwen2.5-7B-Instruct` via `transformers`, greedy, on Explorer V100 | `explorer/label_modes.py`; vLLM can't run on Explorer's driver/GPUs |
| Config | `backend/.env` (gitignored), template `backend/.env.example` | Gemini's OpenAI-compatible endpoint |
| Considered | Claude API (Haiku 5.5 live, Sonnet/Opus 5.5 offline), local 8B via Ollama on the RTX 4060 | decisions.md 2026-10-21 |

## Evaluation protocol (don't break this)

- **Sim:** pass criteria for a new director variant are fixed **before** running; notebook checks
  assert the decision (adoptions *and* rejections), so all ✅ = every decision still holds.
- **Transcripts:** tuning set Nate Bargatze *Hello World*; held-out Mike Birbiglia *The Old Man
  and the Pool*, scored once with the final prompt, never tuned on.
- **rJokes precision rule:** if any mode's 95% CI on mean percentile is wider than ±0.02, extend
  the labeled sample by 30k (`explorer/mode_precision.py`).

## Next steps

1. **Model track (Explorer):** own premises → 6-mode candidates → judge comparison → SFT → DPO, latest
   Nov 22 (decisions.md 2026-11-01). The trained model speaks in the Dec 8 demo.
2. **Demo track:** tasks T1–T11 in [plans/live-loop.md](plans/live-loop.md), each PR built by a Sonnet agent and reviewed before
   it reaches the user. T1–T10 merged, Exp 21 met, operator UI revamped; T11 merged and the acceptance run passed (Exp 22). The safety check's budget is lengthened (12 s / 25 s, 3 retries, one re-ask on a bad reply, 5 bridge lines; Exp 20 rerun met its bars with the filter at 0 failures, but the provider was healthy, so the stall handling rests on tests; decisions.md 2026-11-15). Next: real live shows to collect DPO and detector data. Laugh detection follows T11.
3. Week 6: final sim figures for the report.
4. Callbacks (C2) after the demo and DPO. C1 is closed as a negative result unless the user reopens it.

Parked: transcript callback pass, Birbiglia held-out run, `pipeline.bits --all`.
Backlog: [explore-next.md](explore-next.md).

## Open questions

- Paid Gemini tier or Claude API for the live loop, to get past free-tier limits?
- Local 8B (Ollama) as an offline fallback for the live demo?
