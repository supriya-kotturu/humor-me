# Roadmap

## M1 — Target Dec 8, 2026 (RL course final project)

Everything is finished by **Dec 8**. Spillover is handled by **Dec 12**. **Dec 14:** final presentation
(15 min at most, team) and final report due (decisions.md 2026-11-01).

Budget: 8 weeks × 12–15 h ≈ **100–110 h**. Two audiences: course graders (need RL rigor) and
class demo (needs a live run that is fast, safe, funny).

**Principle:** the RL contribution is proven in **simulation** (reproducible, thousands of
episodes); the live demo shows the same policy acting in the real world. Build the sim first
so the grade isn't hostage to audio/latency problems.

| Tier | Scope |
|---|---|
| Must | Simulated audience env; bandit algos + baselines + regret curves; voice loop + waveform UI; trained joke model (SFT, then DPO) with core identity × 6 modes generation; safety filter (teens hard rule); laughter detection with operator-button fallback; laugh-aware turn-taking; episode logging |
| Should | MDP show state (step C1, sim only); training-mode episode rating; 3–5 pilot sessions |
| Could | LLM-judge rerank of K=3 candidates; pairwise vote UI |
| Cut (post-M1) | Callbacks (step C2, after the demo and DPO); banter/roast handling via ASR; VLM snapshot (operator sets context), commute venue, reward model from logged shows, H1 live experiment |

| Wk | Dates | Deliverable | Exit criterion |
|---|---|---|---|
| 1 | Oct 20–26 | Stack scaffold (FastAPI + Next.js), log schema, **latency spike** (streaming LLM → streaming TTS), sim env spec | Measured end-to-end latency; pick providers |
| 2 | Oct 27–Nov 2 | **Sim env** (gym-style, parametric audience) + ε-greedy, LinUCB, Thompson + fixed/random baselines | Regret curves for all policies across 3 venue contexts |
| Model track | Nov 1–22 | Own premises → 6-mode candidates → judges compared on rJokes pairs and a small human vote → SFT → DPO, on Explorer. Starts tonight; latest SFT Nov 15, DPO Nov 22 | Trained model beats the base model on a human vote; runs locally within the latency budget, or Gemini is the fallback |
| 3–4 | Nov 3–16 | Voice loop, waveform UI, operator panel (venue/energy/age + laugh button), mode-conditioned generation, safety filter | 10-min generated set plays end-to-end, logged |
| 5 | Nov 17–23 | VAD + laughter detector + hold-during-laugh; bandit wired to live reward | Detector vs operator button agreement measured; no talking over laughs |
| 6 | Nov 24–30 | Episode rating UI; final sim experiments + plots (switching penalty ablation) | Report-ready figures |
| 7 | Dec 1–8 | Pilot sessions (friends/classmates), fix top issues. **Feature freeze:** report and slides finished, demo rehearsed in a noisy room | ≥ 3 logged live sessions; demo runs 3× in a row without intervention |
| 8 | Dec 9–12 | Spillover only: nothing new starts here | Everything from week 7 closed |
| – | Dec 13–14 | Rehearse the talk. **Dec 14: final presentation (≤ 15 min, team); final report due** | Talk fits in 15 min, demo included if there is one |

### Sprint Oct 26 – Nov 2: reach ~40% of M1 "Must"

40% means: the RL result exists end to end in simulation, on priors grounded in real joke data,
and a long-form set can be generated as text. Voice, laughter and live wiring are the other 60%.

| # | Step | Where | Est. | Exit criterion |
|---|---|---|---|---|
| 1 | rJokes prep: dedupe, length filter, blocklist, per-year score percentile; stats report | L | 3 h | `data/processed/rjokes/` + stats in dev-log; tests pass |
| 2 | Explorer setup: JupyterLab on GPU partition, venv, vLLM smoke test | E | 2 h | 7–8B model answers a prompt; throughput noted |
| 3 | Mode + safety labeling of ~30k stratified jokes; 100 spot-checked by hand | E | 4 h | Per-mode score distributions; labeler agreement with hand checks reported |
| 4 | Sim env (gym-style): contexts, hidden mode payoffs, noise, fatigue, novelty, mid-set shift | L | 4 h | Unit tests; seeded episodes reproducible |
| 5 | Policies: fixed, random, ε-greedy, LinUCB, Thompson, ± switching penalty; regret plots | L | 4 h | Regret curves for all policies × 3 venues |
| 6 | Set planner + bit writer prototype (text only); one 10 min set per venue logged | L/API | 4 h | 3 sets, each in the word budget, each with a closer and ≥ 1 planned callback |
| 7 | Docs: status, dev-log, figures into `reports/` | L | 1 h | status.md says where we are |

Order: 1 → (2, 4 in parallel) → 3 → 5 → 6 → 7. Steps 4–5 only need rJokes priors as an input
file; they start on placeholder priors so Explorer delays don't block them.
Stretch, only after this sprint: joke scorer on Explorer; judge-as-audience (idea B).

Fallbacks: detector unreliable → operator button is the reward. Bandit live-wiring slips →
live demo uses operator-chosen mode; RL results stand on sim.

---

## Long-term plan (post-M1)

Estimates assume **one developer, part-time (~15 h/week)**. Full-time ≈ divide by 2.5.
Stage 5 is calendar-bound (depends on getting audiences), not effort-bound.

| # | Stage | Deliverable | Exit criteria | Est. |
|---|---|---|---|---|
| 0 | Foundations | Repo docs, stack choice, license checks | Stack + log schema draft agreed in decisions.md | 1 wk |
| 1 | Voice loop MVP | Prepared set → TTS → waveform UI; human "laughed" button; episode logging | 10-min set plays end-to-end; every joke logged with timestamps | 3 wks |
| 2 | Hearing the room | VAD + laughter detector + laugh-aware turn-taking | Detector agrees with human button ≥ 80% on recorded sessions; bot never talks over a laugh | 3 wks |
| 3 | Generation | Core identity + 6 modes, K-candidate gen, LLM pairwise judge, safety filter, venue snapshot, ASR banter + roast handling | Judge rerank beats random pick on rJokes-style held-out pairs; safety filter passes red-team list; next line ready ≤ 1s after laugh decay | 4 wks |
| 4 | Director | Contextual bandit (Thompson/LinUCB) + switching penalty; simulated-audience harness | Bandit converges to the right mode for each simulated venue; mode switches stay bounded | 3 wks |
| 5 | Training mode + collection | End-of-episode rating + pairwise vote UI; run sessions across venues | ≥ 30 sessions across ≥ 3 venue classes | 6–10 wks calendar (overlaps 6) |
| 6 | Experiments (H1) | Run the condition table in design.md; analysis | Pre-registered metrics reported with CIs, win or lose | 4 wks |
| 7 | Learned reward + DPO (H2) | RM from logs/votes; compare to LLM judge; DPO small generator | RM vs judge on held-out live reactions; distilled model meets latency budget | 5 wks |

**Total: ~6–8 months part-time** (~29 wks effort; Stage 5 overlaps Stage 6 prep).

## Biggest schedule risks

- Latency (Stage 3) — may force switching TTS/LLM providers or early distillation.
- Laugh detector in noisy venues (bars) (Stage 2) — may need venue-specific thresholds.
- Session count (Stage 5) — H1 needs enough sessions per condition; walk-up audiences are noisy.

## Open questions

- Stack: Python backend + web UI? Which LLM/TTS providers?
- Per-condition session count needed for H1 (do a power estimate after first 10 sessions).
- Consent flow for recording in training mode.
- Venue taxonomy: fixed list vs free-text from VLM mapped to classes.
