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

## 2026-10-31 — Sim `base`: HumorGen win rates mapped linearly (retroactive record)
Formula: `base = 0.3 + 2 * (win_rate - 0.5)` (sim/audience.yaml), turning Table 4's six
LLM-judge win rates into the room's general mode strengths (neurotic 0.57 ... wordsmith 0.00).
Picked by hand by the AI (Claude) when writing PR #1 (2026-10-27); the user approved "start by
hand, swap in rJokes in PR 3". No options were compared then; this entry records it after the
fact. What each constant does (sim: laugh when latent > 0, noise sd 1, so P(laugh) ~ Phi(mean)):
- `- 0.5`: centres an average persona at 0; a win rate is only relative to the field.
- `* 2`: spread best-worst = 0.57, about half the noise sd and the same size as the venue
  offsets (+-0.4), so a venue can flip the best mode. A much larger slope (e.g. 10: spread 2.8)
  makes neurotic best everywhere, so fixed-neurotic is unbeatable and nothing is learned; a
  slope near 0 leaves only the made-up venue offsets.
- `+ 0.3`: an average mode gets Phi(0.3) ~ 62% laugh chance fresh. It also happens to cancel
  the steady-state fatigue of -0.3, but that fatigue came later (per-bit decisions, same day);
  it was not the reason.
Linear because six judge numbers only support "order and rough gaps". Alternative not taken:
the Thurstone mapping `sqrt(2) * sigma * Phi^-1(win_rate)` (sigma = 1: neurotic +0.48,
wordsmith -0.55, about twice the spread), principled for pairwise data but its sigma is the
judge's noise, not the room's, so it adds rigor the inputs can't back. Superseded when PR 3
replaces `base` with rJokes per-mode upvote percentiles, which need their own mapping (decide
centre, scale and offset again; percentiles are not win rates).
Evidence: PR #1 transcript and code; metrics.md PR #1 (best vs runner-up gaps 0.015-0.052).

## 2026-10-31 — rJokes is for mode statistics only, never joke material
A hand check of 105 R2 labels (15 per judge mode) found the judge's **mode** labels 92% precise at
the 30k mode mix (misses lean to wordsmith; absurdist often unplaceable), but its **safety** flags
caught 0 of 16 jokes the user flagged (sexual 8, group 3, bodies 1, out-of-schema 4). Options:
(a) extend the safety schema and re-label; (b) a separate, stricter safety classifier; (c) use
rJokes only for per-mode statistics (the human mode ranking, the sim's `base`) and never as a
source of jokes, so its safety labels are never relied on. **Chose (c)** (user): the ranking needs
only mode labels, and invariant 1 already puts safety in the live loop. Rows 75 and 83 (judge wrong,
true mode unknown) stay unfilled; no re-labeling.
Evidence: notebooks/decisions.ipynb Exp 9; metrics.md "Hand check of R2 labels".

## 2026-10-31 — Quote a few rJokes rows in docs; never the dataset
The data rule was "Reddit terms: local use, never redistribute". The report needs examples to
show what the hand check found. Options: paraphrase (no Reddit text, loses the wording) or quote a
handful of rows verbatim with attribution. **Chose quoting** (user): a few illustrative rows are
not the dataset. Limits: a small table in docs only (metrics.md hand check: 8 rows), credited to
r/Jokes via rJokes, nothing NSFW quoted; the full hand-check file with text stays local
(`data/processed/rjokes/hand_check_with_text.csv`, gitignored).

## 2026-11-01 — Human vs judge mode ranking: design and pass criteria (fixed before running)
**Why:** HumorGen's LLM judge ranks the six modes (Table 4 win rates: neurotic 63.4, absurdist 55.8,
cynic 55.2, observer 49.1, optimist 41.1, wordsmith 34.9); the sim's `base` comes from it. Does a
human signal (rJokes upvotes on the 90k R2 labels) rank them the same way?
**Confound found first:** joke length predicts upvote percentile more than mode does (shortest to
longest quartile: 0.465 → 0.550, larger than the whole between-mode spread of raw means, 0.469–0.531),
and modes differ in length (median words: wordsmith 14, absurdist 26, others ~20). Year is already
controlled (`pct` is within-year). So raw means per mode are not used.

**D1 Estimator.** Options: (a) raw mean pct: rejected, length-confounded. (b) win rate vs the field
within year × length-bin cells (ties half, pooled by pair count): same unit as HumorGen's Table 4.
(c) Bradley–Terry on sampled matched pairs: rejected, with 6 modes it adds a mode-vs-mode matrix but
little else, and pair sampling adds noise that (b)'s exact counting avoids. (d) OLS pct ~ mode
(sum-to-zero) + year + length bin: most efficient, effects in pct units. **Chose (b) and (d)**
(user): compare the two; a ranking that depends on the estimator isn't a finding.
**Length bins:** quartiles and deciles of `n_words` (user: "decide from outcomes"). With D1 that is
4 variants: WR-q4, WR-d10, REG-q4, REG-d10.
**Sample:** 90k labels joined to `train.jsonl`; parse failures and `none` dropped (HumorGen has no
`none`), so the field is the six modes.
**Uncertainty:** 1,000 bootstrap resamples of jokes (seed 0), shared by all variants. Judge-side
uncertainty (Table 4's CIs) is not modelled: a stated limitation.

**D2 Comparison and pass criteria.** Options: (a) Spearman ρ only: with 6 modes ρ needs ~0.89 for
significance, so alone it mostly says "inconclusive". (b) per-mode side of the field only: says
where, not how much. **Chose (c) both, plus one prediction** (user). Per variant:
- **C1** ρ(human, judge) with a bootstrap 95% CI: *agree* if the lower bound > 0, *disagree* if the
  upper bound < 0, else *inconclusive*.
- **C2** per mode, side of the field: human above/below (WR vs 0.5, REG effect vs 0; *undetermined*
  if the bootstrap CI crosses it) vs judge above/below 50%. Report matches out of 6.
- **C3** prediction: *wordsmith is last for humans too*, holds if it has the lowest human estimate
  in ≥ 95% of resamples.
A claim is reported as a finding only if it holds in **all 4 variants**; otherwise "depends on the
estimator". Raw vs adjusted estimates are reported side by side to show the confound's size.

**D3 Sim `base` (next PR).** Options: (a) replace with human values: likely flattens mode differences
(human spread ~±0.05 → `base` spread ~0.2 vs 0.57), so policies would barely differ; (b) keep the
judge's, human ranking as validation only; **chose (c)** (user): re-run the policy experiment (Exp 4)
under both tastes. Its pass criteria are fixed in that PR before it runs.

## 2026-11-01 — Human `base` for the sim: decile win rates
The two-tastes run needs one human estimate. Options: quartile or decile win rates (regression gives
the same order but isn't in HumorGen's unit). The two bin sizes disagree only on absurdist, the
longest mode (0.505 vs 0.493), where coarse bins leave more of the length confound in. **Chose
deciles** (user).
Evidence: notebooks/decisions.ipynb Exp 10.

## 2026-11-01 — Human vs judge mode ranking: outcome
Against the criteria fixed above, in all 4 variants: C1 "agree" but weak (Spearman ρ +0.20 to +0.37);
C2 matches neurotic and wordsmith, clearly disagrees on optimist; C3 holds (wordsmith last).
**Conclusion:** the judge is a partial stand-in for human taste, so keep both tastes for the sim as
D3 planned. We do **not** conclude that people care less about mode: the judge's spread is about 6×
wider, but upvote noise, different joke sources and label errors all shrink the human number.
The D3 estimate of a human `base` spread (~0.2) was too high: decile win rates give 0.094.
Evidence: notebooks/decisions.ipynb Exp 10; metrics.md "Human vs judge mode ranking".

## 2026-11-01 — Joke length vs upvotes: weak correlation, clear trend
Raised by the user: is length correlated with upvotes, given that we adjust every mode comparison for
it? A one-off check on all of `train.jsonl` with a known year (322,731 jokes): Spearman ρ +0.13 between
`n_words` and `pct` (Pearson +0.15); within a single year, +0.09 to +0.28 (2010–2019). Weak, because
length explains little about any one joke, but the trend is steady: mean `pct` 0.46 at ≤13 words, 0.51
at 19–50, 0.55 at 51–100, 0.63 at 101–300.
**Decision:** keep comparing modes among same-year, similar-length jokes. A weak correlation is enough
to need this, since mode effects are smaller still and modes differ in length.
**Limits:** not an experiment yet: this ran outside the notebook, so it has no `run()` or claim check.
Correlation doesn't show that length causes upvotes.
**Open (user's call):** a correlation map of every rJokes feature against upvotes, as its own experiment.
`n_posts` and `score_spread` would need a caveat, because both are computed from the score.

## 2026-11-01 — rJokes correlation map: design and rule (fixed before running)
The user chose to run the map before the two-tastes sim run, to check that Exp 10 adjusted for the
right things. Already seen before this entry (the one-off check above): length, year, `n_posts` and
`score_spread` against `pct`. Flags, modes and safety labels against `pct` have not been looked at.
**Measure:** Spearman ρ with midranks for ties, because upvotes are heavily skewed; each mode, flag and
safety label is a yes/no column. Each feature gets ρ with `pct` twice: raw, and within the same year
and length group (deciles, as in Exp 10). 95% CIs from 200 bootstrap resamples, seed 0.
**Data:** text features and flags on all of `train.jsonl` with a known year; modes and safety labels on
the 90k labeled sample, where the full feature-by-feature map is also computed.
**Rule:** Exp 10 is re-run with an extra adjustment for a feature only if, within the same year and
length group, its |ρ| is at least 0.05 with `pct` **and** at least 0.05 with some mode. A confound
needs both links. Otherwise Exp 10 stands as it is. The threshold 0.05 is a judgment call: about half
of the 0.10 usually called a small correlation, because the mode effects at stake are themselves small.
**Excluded from the rule:** `n_posts` and `score_spread` (computed from the score), `year` and length
(already adjusted for), and modes themselves. Safety labels are reported, but the hand check found them
unreliable (Exp 9), so a hit on one is a reason to look closer, not to re-run. `mode2` is not used.

## 2026-11-01 — rJokes correlation map: outcome, Exp 10 stands
No feature met the rule. Within the same year and length group, every flag and safety label stays under
0.05 with `pct` (largest: `safety_targets_group` −0.026), so **no extra adjustment and no re-run**.
Length itself still leaks through the groups: +0.033 within deciles, as large as the biggest mode link
(wordsmith −0.032). It wasn't a rule candidate because it's already adjusted for, so this is a stated
limit of Exp 10, not a trigger. It also limits how fine a ranking the middle modes can support.
The map only covers features we have: it can't rule out anything unmeasured, like topic or joke format.
Evidence: notebooks/decisions.ipynb Exp 12; metrics.md "rJokes correlation map".

## 2026-11-01 — Correlation-map rule misses a weak link paired with a strong one
Asked by the user why the threshold was 0.05, I found it was a judgment call with a flaw. A feature can
shift a mode's result by about (its link to `pct`) × (its link to the mode), but the rule tests each link
separately. `safety_targets_group` (−0.026 with `pct`, +0.325 with cynic) fails the rule, yet its estimated
shift (0.0085) is over 3× what the rule's minimum implies (0.0025). The rule's outcome stands, since changing
a rule after seeing results would fit it to them.
**Extra check (user), fixed before running, not a rule trigger:** re-run all 4 Exp 10 variants, same
resamples, also adjusting for `safety_targets_group`. **Main findings hold** if C1, C3 and the sides of
neurotic, optimist and wordsmith all stay the same. Report how far cynic moves (estimate: its ρ with `pct`
goes from about +0.002 to +0.011). This safety label is unreliable (Exp 9), so a change would be a reason
to doubt cynic's place, not to adopt the adjustment.
**Outcome: main findings hold.** C1, C3 and the sides of neurotic, optimist and wordsmith are unchanged.
Cynic moves: decile win rate 0.501 → 0.510 (CI [0.505, 0.514], now above 50%), regression −0.003 → +0.002.
Its side still depends on the estimator, but it moves from fourth to third, passing observer. **Decision:**
the sim's human `base` keeps the Exp 10 decile values, without this adjustment, and the middle-mode order
(observer, cynic, absurdist) is reported as unsettled, not as a ranking.
Evidence: notebooks/decisions.ipynb Exp 12.

## 2026-11-01 — Two tastes in the sim: pass criteria (fixed before running)
Does the Exp 4 choice (LinUCB + uniform prior) still hold if the room's taste comes from people, not the
judge? Same setup as Exp 4: 300 shows, 10 seeds. Human `base` = `0.3 + 2*(win_rate − 0.5)` from the Exp 10
decile win rates, without the extra adjustment. Only `base` changes.
Under human taste, LinUCB + uniform must:
- **P1** reach at least 95% of oracle laughs in the last 50 shows (Exp 4: 97%);
- **P2** beat the best single fixed mode in the last 50 shows, with CIs that don't overlap. The best of
  all six, since neurotic isn't the best under human taste;
- **P3** not lose to ε-greedy or Thompson (uniform prior) with CIs that don't overlap.
**All pass:** the policy choice doesn't depend on taste, so keep the judge `base` as the default and report
both. **Any fail:** the choice depends on taste, and the user decides before anything changes.
No third taste in between: the judge's spread is wide and upvote noise shrinks the human one, so a real room
probably sits between them. Holding at both ends is good evidence for the middle, not proof.
Also reported, not a criterion: LinUCB with the judge's values as its prior in a human room.

## 2026-11-01 — Two tastes in the sim: outcome, the director choice holds
All three pass under human taste. P1: 96.2% of oracle (97.0% under judge taste). P2: 45.3 ± 1.0 laughs per
show vs the best fixed mode, absurdist, at 36.5 ± 1.1. P3: not beaten by ε-greedy (42.5 ± 0.8) or Thompson
(45.6 ± 1.1). **Decision:** keep LinUCB + uniform prior and the judge `base` as the default, and report both
tastes. A judge prior in a human room makes no clear difference (45.0 ± 1.0).
**Limits:** venue and age offsets are still made up, and with a weak `base` they mostly decide the best mode,
so part of learning's advantage comes from those made-up values. This shows the choice doesn't depend on
`base`, not how much learning would help in a real room. Live pilots have to answer that.
Evidence: notebooks/decisions.ipynb Exp 13; metrics.md "Two tastes in the sim".

## 2026-11-01 — M1 target moves to Dec 8; presentation and report on Dec 14
From the user. Replaces the Dec 15 date (2026-10-20 entry). Target Dec 8, spillover by Dec 12, and on Dec 14
the final presentation (15 min at most, team) and final report are due. The old week 8 (freeze, rehearse,
finalize the report) now has to fit into Dec 1–8, so the demo track has about a week less of slack.

## 2026-11-01 — Cut banter and roast handling from M1
From the user, to make room for the trained joke model and show state. The agent won't hear or answer what
people say in the demo, so nothing can trigger a roast response. Invariant 4 (concede and escalate, never
hostile) still stands and applies again if banter returns after M1. Laugh detection is unaffected.

## 2026-11-01 — Split show state step C: keep the MDP, defer callbacks
Step C was "MDP + callbacks". **C1** = MDP over the 8 bits with show state, in-show exploration and fatigue
planning, in the simulator. **C2** = callbacks. Keep C1 for its RL value; callbacks come back after the demo
and DPO are done (user). They depend on a model that can write them and would add more made-up rewards to
the simulator. C1's pass criteria get fixed in its own PR, before it runs.

## 2026-11-01 — rJokes may also test judges, still never joke material
Extends the 2026-10-31 rule (user). To choose the judge behind the DPO preference pairs, compare a plain judge,
a "human taste" prompt, and that prompt with example pairs, on how often each picks the joke with more upvotes
among same-year, similar-length pairs. Asking a judge to have human taste doesn't give it that taste, so this
measures it. Examples shown to a judge must pass a hand check first. rJokes text is still never used to train
or feed the generator, and never committed. Upvote noise caps every judge's score, so small gaps won't count.

## 2026-11-01 — Trained joke model and DPO move into M1, from our own premises
Replaces DPO's "cut until after M1" (user): the Dec 8 demo is the agent on screen with its waveform and audio,
speaking jokes from a model we trained. DPO should finish well before that. Training order follows HumorGen:
SFT first, then DPO on top. HumorGen found DPO about equal to SFT, so SFT is the safe checkpoint.
**Preference pairs:** there are no logged shows yet and rJokes is not joke material, so they come from a judge
on jokes generated per mode, chosen by the rJokes comparison above, plus a small human vote. The model learns
that judge's taste, which Exp 10 shows matches people only in part. The report must say so.
**Prompts.** Options: (a) SemEval 2026 Task 1 headlines and word pairs, as HumorGen did: directly comparable,
but we found no license or download link and news can be sensitive; **(b) chose our own premises** (user), which
give control over the data and the bit and scene structure we want. Cost: not directly comparable to HumorGen.
Limits: our 7B generator and judge are weaker than HumorGen's 32B and 70B. The reward model from logged shows
(H2) stays after M1.
