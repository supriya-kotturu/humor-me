# Decision Log

Append-only. Format: date — decision — why — alternatives rejected.

## 2026-10-20 — Audio-only realtime loop
Why: latency, and removes appearance-based joke risk by construction.
Rejected: continuous facial-expression tracking (noisy, slow, privacy cost).
Keep: one optional async scene snapshot at session start → text only.

## 2026-10-20 — Per-joke laughter is the reward; episode rating is the outcome
Why: episode rating is sparse (1 scalar / ~20 jokes) and peak-end biased; raw laugh counts
are gameable. Normalize laugh features to room baseline.

## 2026-10-20 — Contextual bandit over HumorGen persona modes, fixed core identity
Why: consistent personality with adaptive material; venue is a context feature/prior.
Rejected: switching full personas (incoherent); full RL online (too little signal per show).

## 2026-10-20 — LLM-as-judge first, learned reward model later
Why: no data at start. Training mode collects ratings + pairwise votes to replace the judge (H2).

## 2026-10-20 — Data sources
Primary: own logs. Cold start: rJokes, HumorGen/LoL methods. Deferred: scrapsfromtheloft
(copyright), HumorGen dataset (anonymous repo, no license yet). See data-sources.md.

## 2026-10-20 — Research framing
H1: adaptive bandit beats best fixed mode and random mode live. H2: learned RM beats LLM judge.

## 2026-10-20 — M1 = Dec 15 (RL course final project, school demo)
Budget ~100–110 h. RL contribution proven in a simulated audience env (regret curves vs
baselines); live demo shows the same policy acting live. Cut for M1: VLM snapshot, commute,
learned RM, DPO, live H1 experiment. See roadmap.md.

## 2026-10-20 — Context = three operator-set features, not one venue label
venue (bar, company_event, showcase, commute) × energy (low, normal, high) × age (teens,
young_adults, mixed). Why: original list mixed place, mood, and demographic; audio-only can't
infer age. Commute deferred (transit noise, unclear audience).

## 2026-10-20 — Teens = stricter safety filter (hard rule)
Why: minors. Not left to the bandit.

## 2026-10-20 — Model output = next bit, conditioned on room state
State: context (venue/energy/age), bits so far (mode, premise, laugh outcome), open
callbacks, laugh trend, elapsed time. RL planner picks the next bit spec; generator writes it.
Why: fits a sequential RL framing (MDP over bits), and keeps the set coherent.

## 2026-10-20 — Transcripts are for structure + reward, never as imitation text
Uses: (1) laugh predictor from normalized audience markers (reward model / sim reward),
(2) set structure (bit order, segues, callbacks) as planner demonstrations, (3) bit skeletons
→ synthetic sets in the robot's voice → SFT. Why: SFT on raw transcripts blends 19 voices
into a generic one and memorizes lines. Corpus stays local in data/ (gitignored).
Crowd-work is its own bit type.

## 2026-10-20 — Laugh labels: lexicon-mapped levels, binary primary
Levels 0–4 (none/chuckle/laugh/big laugh/applause break) from a versioned markers.yaml;
unknown marker strings stop the pipeline. Unattributed chuckles/sighs = performer, not
audience. Binary (≥2) is the primary target; ordinal only within-set. Groan/gasp are a
separate channel; groan is mode-dependent reward (positive for Wordsmith).
Why: transcribers differ in vocabulary and density; most never write audience chuckles.

## 2026-10-20 — Opener/closer are positions, not bit types
Bit types: prepared, crowd_work, callback, segue. Position assigned by index in set.

## 2026-10-20 — Density gate for laugh labels; Birbiglia replaces Bamford in held-out
Only 11/20 sets are densely tagged (≤150 words per audience laugh). Sparse/unmarked sets
are structure-only, since absent markers would be read as "no laugh". Bamford is sparse, so
Birbiglia (also Neurotic) takes her held-out slot. 8 training sets remain for the laugh
predictor: thin; more *densely tagged* transcripts are the highest-value data to add.

## 2026-10-21 — Applause alone is not a laugh
Applause/cheer markers without a laugh get level 0 + `applause` flag; level 4 requires a laugh
(same marker or merged with adjacent applause). Why: eyeball check found mid-set cheers for
hometown mentions labeled as applause breaks. Level-4 counts dropped (e.g. Notaro 23 → 0).

## 2026-10-21 — Two datasets, split logically at the output
`laugh_labeled` vs `structure_only`, decided by the density gate (overridable per set via
`dataset:` in body_overrides.yaml) and written to `data/processed/datasets.json`. Stages 3+
write each to its own folder. Why: lets each grow independently and makes it impossible for
the laugh predictor to train on sparse sets. Rejected: separate raw/ folders, since density
is only known after markers run.

## 2026-10-21 — Callbacks are first-class: hooks in set memory, callback(k) planner action
Each bit stores a paraphrased `hook` and how hard it landed; the planner can call back a landed
hook ≥ 2 bits later. Sim callback payoff parameters are fit from transcript callback stats
(S4–S5). Why: callbacks are the delayed-reward decision that justifies sequential RL over a
bandit, and the cross-bit context the user wants the agent to use.

## 2026-10-21 — Laugh predictor trains on lines, not beats
`lines.jsonl`: last line before a reaction gets its level, setup lines 0, each with 150 words of
preceding context (bit-scoped after S4). Beats stay for structure. Why: beat length alone
predicted the beat label (77% positive; 23 vs 74 median words).

## 2026-10-21 — Gemini (free tier) for S4–S5, with minimal excerpts
User's call: small, non-commercial corpus; annotation quality outweighs the exposure. Mitigations:
TextTiling runs locally and only windows around uncertain boundaries are sent; responses must
paraphrase (premises, hooks), never quote; small batches with backoff for free-tier rate limits.
Considered: paid Gemini / Claude API (no training on API data, ~$3–6 batched), local 8B via
Ollama on the RTX 4060 (private, weaker annotation). Revisit local 8B as an offline fallback
for the live demo.

## 2026-10-21 — S4: Gemini directly on whole sets (supersedes "minimal excerpts" above)
One Gemini call per set returns bit starts (as line_idx) plus annotations (premise paraphrase,
mode, type, callback_to, hook) in one pass, halving calls against free-tier limits. Output
references lines by index only, never quotes. TextTiling deferred: add later only as a report
baseline. Hand labels for Bargatze + Birbiglia via `pipeline.labels` (indices committed, text local).

## 2026-10-21 — Bit = smallest standalone unit; same-topic bits form a thread
From hand-labeling Bargatze: a bit is the smallest stretch that works on its own without earlier
setup. Consecutive bits on one topic are marked `continues` and form a thread. A callback must
return to an earlier bit after other material (≥ 2 bits back); a "callback" to the previous bit
is converted to `continues`. Gemini is prompted with the same definitions; S6 scores both levels.
Bargatze: 52 bits, 38 threads, 3 callbacks.

## 2026-10-26 — Pause the transcript pipeline; rJokes + simulator first
S0–S6 are frozen as they stand (bits F1 0.80 vs 0.21 baseline on Bargatze, callbacks 0/3).
They stay as a report appendix and as a source of **aggregate** set-structure stats (bits per
set, words per bit, tags, callback rate), which feed the long-form set planner and the sim's
timing priors. No more Gemini quota goes to S4 on new sets; the Birbiglia held-out run waits
until labels exist and there is spare time. Why: transcripts were never on the M1 critical
path (data-sources.md), the week-2 sim + bandit deliverable is behind, and the legal exposure
of copyrighted specials is the weakest part of the project. rJokes becomes the working joke
dataset: per-joke scores, 432k rows, local use under Reddit's terms. HumorGen's
`africa_headlines.tsv` is prompts only (no jokes, no ratings): at most a source of topic seeds.
Judge-as-audience (idea B) is a stretch goal after the sim + bandits ship.

## 2026-10-26 — Long-form sets are planned hierarchically, not generated in one pass
A 10–15 min set is a plan of bits, not a 1,500-word completion. Set planner → bit writer →
live director (details in design.md, "Long-form set generation"). Length is controlled by the
number and size of bits, not by tokens. We imitate stand-up *structure* (setup–punch–tags,
rule of three, act-outs, callbacks, closer), never a comedian's voice; the voice is the fixed
core identity. Why: one-pass long generation drifts, repeats and loses callbacks, and a plan is
what lets the director reorder, cut or extend bits from live laughs.

## 2026-10-26 — Northeastern Explorer for GPU work; laptop for everything else
Explorer (OOD JupyterLab, course GPU partition) runs batch jobs: local-LLM mode/safety labeling
of rJokes (vLLM + a 7–8B instruct model) and a small learned joke scorer (encoder regressor on
rJokes scores). Sim, bandits, pipeline and the live loop stay on the laptop and must not depend
on Explorer. Why: removes the Gemini free-tier bottleneck for bulk labeling. Jobs are written as
scripts first and notebooks second, so they also run on the 4060 at smaller scale.

## 2026-10-27 — The director picks a mode per bit, not per joke
Options: (a) per joke, weaker fatigue; (b) per joke + recent modes in context; (c) per bit.
Chose (c): a comedian commits to one premise and attitude for a bit and varies between bits.
Per joke, the sim rewarded rotation (random beat every fixed mode; metrics.md v1), and a
history-aware per-joke bandit would learn to rotate, which is a sim artifact, not comedy.
Per bit, coherence is built in (no switching penalty needed), each decision's reward averages
5 jokes (less noise), and the last bit's mode and reward go in the context. Cost: only 8
decisions per show. Evidence: metrics.md PR #1 v2 (best fixed mode now beats random by 3–9
laughs per set; oracle beats best fixed by 4–5).

## 2026-10-27 — Director policy: LinUCB with a uniform prior
Options: ε-greedy (tabular per context), LinUCB, Thompson (LinTS), each with a HumorGen,
uniform, or wrong (shuffled) prior. Chose LinUCB + uniform: lowest laughs lost over 300 shows
(728 ± 64 vs 868–926 for the others with linear models, 2,457+ for ε-greedy), 97% of the oracle
once experienced. The prior made no significant difference at weight 1, so the uniform prior
wins on keeping the LLM judge's taste out. Rejected ε-greedy: no pooling across contexts, 3.4×
the loss. Thompson stays as the comparison in the report (near-identical once experienced).
Evidence: metrics.md PR #4.

## 2026-10-28 — Keep the director's features; within-show adaptation needs show state
Tested a reaction feature (same mode × last reward) against a mid-set taste shift. It added
0.2–0.4 laughs after the shift, within the CIs, so it isn't adopted. All bandit policies
recover no faster than random (2.1–2.5 of 4 bits) because they carry no memory of *this*
show beyond the last bit. Consequence: the next step is a stateful director (show-so-far
features / MDP), built together with callbacks, not more per-bit features.
Evidence: metrics.md "mid-set preference shift".

## 2026-10-28 — Keep 8-bit sets; longer sets don't let the director read a turn
Question (from the user): would more bits per set let the director recover after a mid-set
shift? Tested 8/16/32/52-bit sets with the shift at the midpoint and the same total experience
(3,200 bits per career). After the shift, LinUCB's choice quality drops from ~0.93 to ~0.73 and
climbs only to ~0.81 even 26 bits later (52-bit set); after-shift laughs stay at 80–82% of the
oracle at every length. Cause: the director's estimates pool thousands of bits across hundreds
of shows; it has no notion of "this room, tonight". Keep 8-bit (10-minute) sets, the demo
length. Also learned: "bits to recover" rewards luck (random scores well, a sticky deterministic
policy scores badly), so recovery is judged by choice-quality curves from now on.
Evidence: notebooks/decisions.ipynb Exp 6; metrics.md "set length vs shift".

## 2026-10-28 — Show state step A (tonight's record as features): not adopted; go to B
Plan (user): add show state incrementally, A → B → C. A gives LinUCB/LinTS two more inputs per
mode: tried tonight, and mean reward tonight; the career model learns how much to trust them.
Pass criteria were fixed before running: shifted-show choice quality +≥ 0.05, steady-show
change ≥ −0.01. Result: gains +0.008 to +0.046, steady changes −0.011 to −0.030, so it **fails
both in every cell**. It hedges (less committed to the old favorite at the moment of the shift)
rather than reads the turn: the curve doesn't climb afterwards, and steady shows pay for it.
Why: one linear weight on "tonight's mean" has to serve both shifted and steady rooms, so it
settles on a compromise. Next: step B, where tonight's correction is explicit and per show.
Evidence: notebooks/decisions.ipynb Exp 7; metrics.md "show state step A".

## 2026-10-31 — Show state step B (career + tonight's correction): not adopted; C needs exploration and fatigue planning
Step B kept LinUCB as the career model and added a per-show correction per mode: tonight's
surprise (reward − career prediction) ÷ (bits tonight + κ), reset each show. Swept κ ∈ {0.5,
1, 2, 4, 8} against the same pre-set criteria as A (shifted gain ≥ 0.05, steady change ≥ −0.01,
at 8 and 16 bits). **No κ passed**; most made shifted shows *worse* (−0.01 to −0.03) and steady
shows lost up to −0.043. Diagnosis (tracked, not guessed): (1) **win-stay meets fatigue**:
trusting tonight fast nearly doubles repeats of the last mode (29% → 54%) and the room tires of
repeats; (2) **it learns what bombed, not what's now good**: untried modes get no correction, so
after a turn it still picks a hurt mode about as often (51% vs 48%). Some signal exists (16-bit
κ = 8 climbs past LinUCB late in the set) but is outweighed. Consequence for C: the director must
*value trying an untried mode* after a turn (exploration within the show) and *plan around
fatigue*, which is sequential decision-making, not a better per-bit correction.
Evidence: notebooks/decisions.ipynb Exp 8; metrics.md "show state step B".
