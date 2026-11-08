# Status (read this first in a new session)

Last updated: 2026-11-08. Update this file whenever the state below changes.

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
| Decision notebooks | Split by section so no notebook grows without limit: `decisions-data` (Exp 1, 2, 9–12), `decisions-director` (Exp 3–8, 13–16, 18), `decisions-live` (Exp 19 onward, empty so far); `decisions.ipynb` is the index. `decisions-model` comes with the first judge/SFT/DPO experiment (Exp 17 is reserved for the judge). All 70 claim checks ✅. Run `python scripts/run_notebook.py` (all notebooks) before committing |
| Decision plots | [notebooks/decision-plot.ipynb](../notebooks/decision-plot.ipynb): the deck's results redrawn from `reports/metrics`, with checks that each chart still matches its slide and that every experiment has a chart or a stated reason (14 charted, 3 not). PNGs in `reports/figures/decision-plot/`, swapped into slides 14, 18, 20, 29, 30, 31 and 33 of `docs/slides/humor-me.pptx` and `slides.md`, and added to the Google Slides copy along with slides 15, 16 and 25 |
| Tests | `.venv/Scripts/python -m pytest tests -q`: 153 passing, 1 skipped, synthetic data only |
| rJokes | R1 prep done (403k jokes). **R2 done: 90k labeled**, `modes.jsonl` on the laptop (30k copy kept as `modes_30k.jsonl`). **Precision rule met:** every mode within ±0.02 (optimist 1,170 jokes, ±0.016). **Human vs judge ranking:** agree only in part (neurotic up, wordsmith down, optimist clearly disagrees; Spearman ρ +0.20 to +0.37), and mode makes a small difference in upvotes (win rates 0.480–0.527, Exp 10, [metrics.md](metrics.md#human-vs-judge-mode-ranking-2026-11-01)). **Correlation map:** no flag or safety label needs an extra adjustment, so Exp 10 stands (Exp 12). **Hand check (105 jokes):** mode labels 92% precise at the labeled mode mix, misses lean to wordsmith; judge safety flags caught 0 of 16 human-flagged rows (Exp 9) |
| Explorer (NEU HPC) | Working env: Python 3.13 module, `torch==2.5.1+cu121`, `transformers` 4.57, V100 in float16, 5.2 jokes/s in batch ([explorer/README.md](../explorer/README.md)) |
| Transcript pipeline | **Paused 2026-10-26**. Bits F1 0.80 on Bargatze; kept for structure stats + report appendix |
| Report | [report.md](report.md): overview written (user + AI), results table filled for Exp 1–7 |
| Slides | [Story deck](https://docs.google.com/presentation/d/1xfH1YYanHXuAsJoo5ZYPF2OOkr8EKGbYLoE5kKo_UE0/edit) (Google Slides, owner-only): 59 slides through Exp 16, all visible for a detailed two-presenter talk, with native result tables and the plot graphs; no live demo yet. `docs/slides/humor-me.pptx` has the same story in 39 slides with the plot graphs and full Q&A notes |
| Wiki | [docs/wiki/](wiki/README.md) + [glossary](wiki/glossary.md), mirrored to the GitHub wiki by `scripts/sync_wiki.py` after merges |
| **Backend / frontend (demo)** | **First slice built:** fixed lines → Kokoro on the CPU → browser audio and frequency display, with session logs. **The full loop runs headless** against the simulated room (`run_show`), with the director rebuilt from logs. Not connected to the browser yet; generator, safety filter and reaction keys are next, as tasks in [plans/live-loop.md](plans/live-loop.md). Rewards come from operator keys first; a laugh detector follows in stages, with keys as the fallback. Biggest risk to Dec 8 |

## Open PRs

| PR | What |
|---|---|
| T0 | Live loop plan, PR standard and the Nov 8 decisions |

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
   it reaches the user. Raw audio (T10) must merge before the first real show. Laugh detection follows T11.
3. Week 6: final sim figures for the report.
4. Callbacks (C2) after the demo and DPO. C1 is closed as a negative result unless the user reopens it.

Parked: transcript callback pass, Birbiglia held-out run, `pipeline.bits --all`.
Backlog: [explore-next.md](explore-next.md).

## Open questions

- Paid Gemini tier or Claude API for the live loop, to get past free-tier limits?
- Local 8B (Ollama) as an offline fallback for the live demo?
