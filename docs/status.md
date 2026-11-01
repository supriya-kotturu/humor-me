# Status (read this first in a new session)

Last updated: 2026-10-31. Update this file whenever the state below changes.

Learning the material from scratch (RL foundations → our experiments → Explorer runbook):
[study-guide.md](study-guide.md).

## Where we are

| Area | State |
|---|---|
| Milestone | M1 = Dec 15, 2026 (RL course final project, school demo). Plan: [roadmap.md](roadmap.md) |
| Workflow | One feature per branch off `main`, PR reviewable in ≤ 30 min, backed by a metric; **no stacked PRs**. Every experiment is a `run()` + a section in [notebooks/decisions.ipynb](../notebooks/decisions.ipynb) |
| **RL simulator** | Done: `sim/env.py` (room's truth in `sim/audience.yaml`), director decides **per bit** (8 bits × 5 jokes) |
| **Director policy** | **LinUCB + uniform prior**: 97% of oracle, +18% vs always-Neurotic (notebook Exp 4) |
| Within-show adaptation | Open problem: the director reads rooms, not a room *turning* (Exp 5: 95% → 80% of oracle after a mid-set shift; Exp 6: longer sets don't help) |
| Show state (A → B → C) | **A** (tonight's record as features): not adopted, hedges (Exp 7). **B** (career + tonight's correction): not adopted, no speed passes (Exp 8). **C** (MDP + callbacks, with in-show exploration and fatigue planning): next |
| Decision notebook | Exp 1–9, all claim checks ✅. Run `python scripts/run_notebook.py` before committing it |
| Tests | `.venv/Scripts/python -m pytest tests -q`: 112 passing, synthetic data only |
| rJokes | R1 prep done (403k jokes). **R2 first 30k labeled** (job 11534420, ~1.6 h, 11 parse failures); `modes.jsonl` copied to the laptop. **Precision rule not met:** optimist has only 399 jokes (±0.029) → extend ([metrics.md](metrics.md#pr-20-rjokes-mode-labels-r2-first-30k-2026-10-31)). **Hand check (105 jokes):** mode labels 92% precise at the 30k mix, misses lean to wordsmith; judge safety flags caught 0 of 16 human-flagged rows (Exp 9) |
| Explorer (NEU HPC) | Working env: Python 3.13 module, `torch==2.5.1+cu121`, `transformers` 4.57, V100 in float16, 5.2 jokes/s in batch ([explorer/README.md](../explorer/README.md)) |
| Transcript pipeline | **Paused 2026-10-26**. Bits F1 0.80 on Bargatze; kept for structure stats + report appendix |
| Report | [report.md](report.md): overview written (user + AI), results table filled for Exp 1–7 |
| Wiki | [docs/wiki/](wiki/README.md) + [glossary](wiki/glossary.md), mirrored to the GitHub wiki by `scripts/sync_wiki.py` after merges |
| **Backend / frontend (demo)** | **Scaffold only.** Voice loop, waveform UI, laugh detection, safety filter, logging: not started. Biggest risk to Dec 15 |

## Open PRs

| PR | What |
|---|---|
| – | none (#16–#20 merged) |

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

1. **Show state step C** (MDP over bits + callbacks): step B showed a per-bit correction learns
   what bombed but never tries untried modes, and trusting tonight repeats modes into fatigue.
   C needs in-show exploration and fatigue planning (decisions.md 2026-10-31). Notebook Exp 10.
2. **Extend R2** (precision rule): `N=60000 sbatch explorer/label_modes.sbatch` resumes and
   labels the next 30k. At optimist's 1.3% share, 60k gives ~800 optimist jokes, right at the
   ±0.02 edge, so a third extension may follow. Hand check done (Exp 9): mode labels usable,
   safety flags not. Next: the human-vs-judge mode ranking (explore-next #1).
3. **Demo track from Monday Nov 2:** voice loop → waveform UI → operator laugh button (roadmap
   weeks 3–4), then laugh detection (week 5).
4. Week 6: final sim figures for the report.

Parked: transcript callback pass, Birbiglia held-out run, `pipeline.bits --all`.
Backlog: [explore-next.md](explore-next.md).

## Open questions

- Paid Gemini tier or Claude API for the live loop, to get past free-tier limits?
- Local 8B (Ollama) as an offline fallback for the live demo?
