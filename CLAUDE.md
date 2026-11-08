# humor-me

Research project: a live, voice-only stand-up comedy agent that reads the room via laughter,
adapts its material, and learns from audience feedback across sessions.

Read before working:
- [docs/status.md](docs/status.md): **current state, models in use, next steps. Read first; keep it updated.**
- [README.md](README.md): repo map, setup, run order per phase
- [docs/wiki/](docs/wiki/README.md): project concepts (stand-up structure, bandits, sim, rJokes, HPC).
  **Add or update a page whenever a new topic is explored**; each opens with where it fits and is used.
  New terms go in [docs/wiki/glossary.md](docs/wiki/glossary.md) in the same PR; use them as defined there.
- [pipeline/wiki/](pipeline/wiki/README.md): concepts behind each pipeline stage
- [docs/design.md](docs/design.md) — architecture, reward design, research hypothesis
- [docs/data-sources.md](docs/data-sources.md) — what data we use and for what role
- [docs/pipeline.md](docs/pipeline.md) — transcript → beats/bits/sets pipeline, labels, schemas
- [docs/roadmap.md](docs/roadmap.md) — stages, exit criteria, estimates
- [docs/decisions.md](docs/decisions.md) — decision log; append, don't rewrite history
- [notebooks/decisions.ipynb](notebooks/decisions.ipynb) — **index of the evidence for every decision**, linking each
  to a section notebook: `decisions-data.ipynb` (transcripts, rJokes), `decisions-director.ipynb` (sim, director),
  `decisions-live.ipynb` (live loop, Exp 19 onward); judge, SFT and DPO get `decisions-model.ipynb` with their first
  experiment. One section per experiment-backed decision, re-runnable, with ✅/❌ claim checks; Exp numbers are global.
  **Every experiment is a `run(**params)` function called via `sim.cache.run_or_load`, and gets a section in the
  right notebook in the same PR.** Run `python scripts/run_notebook.py` (all notebooks) before committing.
- [notebooks/decision-plot.ipynb](notebooks/decision-plot.ipynb) — **the decisions as charts.** Every experiment gets an
  entry in its `COVERAGE` table in the same PR: its figures, or the reason it has none. A missing entry fails the
  notebook's check. Run `python scripts/run_notebook.py notebooks/decision-plot.ipynb` before committing it.
- [docs/dev-log.md](docs/dev-log.md) — problems hit (symptom → cause → caught by → fix).
  **Add an entry whenever a bug, data quirk or tooling issue costs real time.**
- [references/md/](references/md/) — markdown versions of the reference papers

## Measure before cut (every decision, every agent)

Applies to adopting, rejecting, keeping or cutting anything. If a choice has no measurement, write
"judgment call, not measured" in decisions.md.

1. **Name the number that would change the decision** before running. If no result would change it, don't run it.
2. **Size the prize first.** Compute the ceiling (oracle, planning optimum, ideal observer) in laughs before
   building a variant, and compare it with the cost in hours.
3. **Check the code before believing the result.** Test what the method guarantees, not only its shape. A failed
   run is a negative result only after the implementation is checked; a bug goes in dev-log.md.
4. **Decide on paired CIs.** Compare policies on the same seeds. A gate smaller than the CI's half-width can't
   decide anything, so add seeds or say it's inconclusive. Fix sample sizes for human votes with a power estimate.
5. **Measure at the horizon that gets used.** The demo has about 5 shows, not 300.
6. **Say what is made up.** State which `audience.yaml` values a sim result depends on, and vary them when the
   decision rests on them. Real logs outrank the sim.

Each session uses its own git worktree, and each PR targets `main` unless the user approves stacking.

## Invariants (do not violate without a new entry in decisions.md)

1. **Never joke about bodies, appearance, or protected attributes** of people present.
   Punch at self, the situation, and what people *say* — never at who they *are*.
2. **No face/video in the realtime loop.** Audio only. One optional async scene snapshot at
   session start, converted to text context, then discarded. Never persist images or faces.
3. **One core identity.** A self-aware, slightly anxious robot. HumorGen personas are *modes*
   the identity draws on, not separate characters.
4. **Can take a roast.** Roasts aimed at the bot get concede-and-escalate, never hostility.
5. **Per-joke laugh signal is the training reward; episode rating is the reported outcome.**
   Don't optimize raw laugh counts (gameable by joke spam, confounded by room size).
6. **Log everything** needed to replay an episode (context, candidates, chosen joke, mode,
   timing, laugh features, ratings). Logged sessions are the primary dataset.

## Paper roles (common mix-up)

- **HumorGen** = six cognitive personas → candidate generation, LLM-judge Elo, distillation.
- **LoL (Innovative Thinking, Infinite Humor)** = thought-leap reasoning + frozen pairwise
  judge for online DPO. It does *not* classify personas.
