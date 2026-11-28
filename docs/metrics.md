# Metrics log

One section per merged PR: the numbers that justified merging it. The report's Results section
is assembled from here, so every number names the command that produced it.

## PR #1: simulated audience env (2026-10-27)

### v1: one decision per joke (superseded)

`python -m sim.evaluate_env` (1,000 episodes per cell, 40 jokes per set, context
`<venue>/normal/mixed`). Raw output: `reports/metrics/sim_env.json`.

Laughs per set = sum of 0–3 laugh levels over 40 jokes.

| Venue | Oracle | Best fixed (mode) | Fixed neurotic | Random | Best vs runner-up gap | Jokes to separate |
|---|---|---|---|---|---|---|
| bar | 45.4 | 32.4 (cynic) | 25.9 | 35.8 | cynic − absurdist = 0.052 | 551 |
| company_event | 41.9 | 28.9 (observer) | 25.9 | 35.5 | observer − neurotic = 0.031 | 1,476 |
| showcase | 43.6 | 30.0 (absurdist) | 28.6 | 38.6 | absurdist − neurotic = 0.014 | 7,063 |

Best mode over all 27 contexts: absurdist, cynic, neurotic or observer (4 distinct).

**Findings**
1. There is something to learn: the oracle beats random by 5–10 laughs per set, and the best
   mode depends on context.
2. **Random beats every fixed mode.** Fatigue (−0.15 per use in a 5-joke window, so −0.75
   for a fixed mode) outweighs the largest venue offset (+0.4). In this setting, variety
   matters more than picking the right mode.
3. Telling the top two modes apart takes 551–7,063 jokes against a 40-joke set. Learning
   within one set can't find the venue's best mode; only learning across sets can.

### v2: one decision per bit (merged version)

Decision (decisions-detail.md 2026-10-27): the policy picks the mode once per bit (8 bits × 5 jokes),
like a comedian committing to one premise and attitude; fatigue counts uses in the last 2 bits
(fixed mode at steady state: −0.3, below the largest venue offset of +0.4).

| Venue | Oracle | Best fixed (mode) | Fixed neurotic | Random | Best vs runner-up gap | Bits (shows) to separate |
|---|---|---|---|---|---|---|
| bar | 50.4 | 46.7 (cynic) | 39.0 | 38.1 | cynic − absurdist = 0.052 | 111 (13.9) |
| company_event | 46.9 | 42.6 (observer) | 39.0 | 37.7 | observer − neurotic = 0.031 | 296 (37.0) |
| showcase | 49.1 | 43.9 (absurdist) | 42.3 | 40.8 | absurdist − neurotic = 0.014 | 1,414 (176.8) |

**Findings (v2 vs v1)**
1. **Fit to the room now matters:** the best fixed mode beats random by 3–9 laughs per set
   (v1: random won everywhere). The venue's mode is worth learning.
2. **Variety still matters, but less:** the oracle beats the best fixed mode by 4–5 laughs, so
   a policy that both fits the room *and* rotates should beat any single persona.
3. **"Always Neurotic"** (HumorGen's best judge-rated persona) trails the venue's best mode by
   1.6–7.7 laughs: a fixed persona chosen by an LLM judge is not the best choice for every room.
4. **Learning must span shows:** separating the top two modes takes ~14 shows (bar) to ~177
   (showcase). Per-bit rewards average 5 jokes, which cut the samples needed 5× vs v1, but one
   show is still far too short. Priors and pooling across contexts carry the live case.

## PR #4: mode-choice policies × priors (2026-10-27)

`python -m sim.evaluate_policies` (10 careers × 300 shows, random context per show, the same
context sequence for every policy; ~3 min). Raw: `reports/metrics/policies.json`.
Figure: `reports/figures/laughs_lost_across_shows.png`.

Laughs per show (40 jokes, sum of 0–3 levels), mean ± 95% CI over careers.

| Policy | Prior | First 10 shows | Last 50 shows | Laughs lost vs oracle (300 shows) |
|---|---|---|---|---|
| oracle | – | 50.8 ± 1.9 | 49.4 ± 1.0 | 0 |
| fixed neurotic | – | 41.7 ± 2.1 | 40.8 ± 0.9 | 4,026 ± 52 |
| random | – | 40.5 ± 2.1 | 39.2 ± 1.0 | 4,264 ± 59 |
| ε-greedy | HumorGen | 43.0 ± 2.0 | 44.6 ± 1.1 | 2,457 ± 151 |
| LinUCB | HumorGen | 45.2 ± 2.0 | 47.9 ± 1.1 | 868 ± 111 |
| Thompson (LinTS) | HumorGen | 44.2 ± 1.9 | 48.1 ± 1.1 | 926 ± 59 |
| ε-greedy | uniform | 40.3 ± 2.2 | 44.6 ± 1.1 | 2,493 ± 143 |
| **LinUCB** | **uniform** | 44.7 ± 1.8 | 48.0 ± 1.1 | **728 ± 64** |
| Thompson (LinTS) | uniform | 43.5 ± 2.4 | 48.2 ± 1.0 | 907 ± 58 |
| ε-greedy | wrong | 41.0 ± 2.4 | 44.3 ± 1.0 | 2,624 ± 192 |
| LinUCB | wrong | 43.8 ± 2.2 | 47.9 ± 1.0 | 832 ± 116 |
| Thompson (LinTS) | wrong | 43.4 ± 2.4 | 48.2 ± 1.0 | 902 ± 55 |

**Findings**
1. **Adapting beats any fixed persona.** Experienced LinUCB/LinTS reach ~48 laughs/show, 97% of
   the oracle (49.4), vs 40.8 for "always Neurotic" (+18%). Even in the first 10 shows they beat
   it by ~3 laughs.
2. **Pooling across contexts is the biggest lever.** ε-greedy keeps a separate table per context
   (27 of them) and loses 3.4× more laughs than LinUCB, which shares what it learns at one bar
   with every bar, and every low-energy room with every other.
3. **The prior barely matters at this strength.** HumorGen, uniform and wrong priors land within
   each other's CIs for LinUCB/LinTS; cold-start gaps are ≤ 1.4 laughs. A prior of weight 1 is
   overwhelmed within a few shows. Even the HumorGen prior that matches the sim's `base` didn't
   help, because the venue offsets it knows nothing about decide the best mode.
4. **Chosen: LinUCB + uniform prior**: lowest total loss with the tightest CI, and no LLM-judge
   taste built in. Prior strength sweep logged as explore-next #7.

## PR #10: mid-set preference shift (2026-10-28)

`python -m sim.evaluate_shift` (10 careers × 400 shows; in a random half of shows the room's
taste shifts after bit 4 of 8, unannounced; metrics from the last 100 shows; ~2 min).
Raw: `reports/metrics/shift.json`. The shift changes the best mode in every venue
(bar: cynic → neurotic, company event: observer → optimist, showcase: absurdist → wordsmith).

Laughs in bits 5–8, mean ± 95% CI. "Bits to recover": bits after the shift until the director
first picks a (near-)best mode again (regret < 0.02/bit), capped at 4.

| Policy | Shifted shows: bits 5–8 | % of oracle | Bits to recover | Steady shows: bits 5–8 | % of oracle |
|---|---|---|---|---|---|
| oracle | 24.0 ± 0.4 | 100% | 0.00 | 23.7 ± 0.6 | 100% |
| fixed neurotic | 18.1 ± 0.4 | 75% | 3.69 ± 0.09 | 18.7 ± 0.6 | 79% |
| random | 17.6 ± 0.6 | 73% | 2.33 ± 0.10 | 18.8 ± 0.6 | 79% |
| LinUCB (chosen) | 19.1 ± 0.6 | 80% | 2.46 ± 0.22 | 22.6 ± 0.7 | 95% |
| LinUCB + react | 19.5 ± 0.4 | 81% | 2.26 ± 0.28 | 22.6 ± 0.6 | 95% |
| LinTS | 19.7 ± 0.4 | 82% | 2.12 ± 0.13 | 22.8 ± 0.6 | 96% |
| LinTS + react | 19.9 ± 0.4 | 83% | 2.18 ± 0.15 | 22.8 ± 0.6 | 96% |

**Findings**
1. **The director reads rooms, not turns.** In steady shows the learners get 95–96% of the
   oracle; when the room turns mid-set they drop to 80–83%, and take ~2.1–2.5 of the 4
   remaining bits to land on a good mode again: **no better than random (2.33)**.
2. **The reaction feature doesn't fix it.** +0.2–0.4 laughs, inside the CIs; harmless in
   steady shows. Not adopted.
3. **Why:** a bandit's estimates average hundreds of shows, so one bombed bit barely moves
   them, and its features can only say "*this* mode just bombed", not "this room has turned".
   Reacting within a show needs **state about the show so far**, which is a sequential (MDP)
   problem, the same step callbacks require.
4. LinTS edges LinUCB after a shift (19.7 vs 19.1), but the CIs overlap: no reason to switch.

## PR #12: set length vs mid-set shift (2026-10-28)

`evaluate_shift.run_lengths(lengths=[8, 16, 32, 52], total_bits=3200, seeds=10)` (notebook Exp 6;
~4.5 min). Shift at the midpoint; same 3,200 bits of experience per career, so longer sets =
fewer shows; last quarter of each career. Raw: `reports/metrics/shift_set_length.json`.
Figure: `reports/figures/shift_recovery_by_set_length.png`.

*Choice quality* = expected reward of the chosen mode ÷ best mode's (1.0 = best pick).

| Set (bits) | Shows | Policy | After shift: % of oracle | Quality, 2 bits after shift | Quality, last quarter | Same, steady shows | Bits to recover (max) |
|---|---|---|---|---|---|---|---|
| 8 | 400 | random | 73% | 0.691 | 0.694 | 0.723 | 2.3 (4) |
| 8 | 400 | LinUCB | 80% | 0.771 | 0.768 | 0.930 | 2.5 (4) |
| 8 | 400 | LinTS | 82% | 0.788 | 0.787 | 0.939 | 2.1 (4) |
| 16 | 200 | LinUCB | 81% | 0.762 | 0.778 | 0.936 | 4.6 (8) |
| 16 | 200 | LinTS | 82% | 0.772 | 0.786 | 0.943 | 4.1 (8) |
| 32 | 100 | LinUCB | 80% | 0.720 | 0.767 | 0.936 | 10.1 (16) |
| 32 | 100 | LinTS | 84% | 0.777 | 0.785 | 0.939 | 6.0 (16) |
| 52 | 61 | random | 77% | 0.723 | 0.707 | 0.721 | 4.4 (26) |
| 52 | 61 | LinUCB | 82% | 0.731 | 0.812 | 0.931 | 14.1 (26) |
| 52 | 61 | LinTS | 85% | 0.777 | 0.809 | 0.939 | 7.8 (26) |

**Findings**
1. **Longer sets don't let the director read a turn.** Last-quarter choice quality after a
   shift stays 0.12–0.17 below steady shows at every length; after-shift laughs are 80–85% of the
   oracle whether 4 or 26 bits remain.
2. **There is a slow drift back** (52 bits: 0.73 → 0.81 over 26 bits), because every bit still
   updates the model, but it's ~10× too slow to matter in a show.
3. **"Bits to recover" is misleading:** random "recovers" in ~4 bits by luck while LinUCB, which
   sticks with its old favorite, takes 14. Choice-quality curves are the honest measure.
4. Same root cause as the shift experiment: no memory of *this* show → needs show state.

## PR #13: show state, step A (2026-10-28)

`evaluate_shift.run_lengths(lengths=[8, 16], total_bits=3200, seeds=10, names=SHOW_STATE_POLICIES)`
(notebook Exp 7; ~4 min). Raw: `reports/metrics/show_state_a.json`; figure:
`reports/figures/show_state_a.png`. Same harness as Exp 6. Pass criteria fixed before running.

| Set | Algorithm | Shifted: quality after shift (without → with) | Gain (need ≥ +0.05) | Steady: same bits (without → with) | Change (need ≥ −0.01) |
|---|---|---|---|---|---|
| 8 | LinUCB | 0.770 → 0.816 | +0.046 | 0.930 → 0.908 | −0.022 |
| 8 | LinTS | 0.788 → 0.815 | +0.027 | 0.939 → 0.910 | −0.030 |
| 16 | LinUCB | 0.773 → 0.781 | +0.008 | 0.936 → 0.925 | −0.011 |
| 16 | LinTS | 0.779 → 0.812 | +0.033 | 0.941 → 0.923 | −0.017 |

**Findings**
1. **Fails both criteria in all four cells.** Small help after a turn, small cost in steady rooms.
2. **It hedges rather than reads:** quality is higher *at* the shift (8-bit LinUCB 0.76 → 0.82)
   but doesn't climb afterwards. The director is less committed to its favorite all show.
3. Why: a single learned weight on "tonight's mean" serves shifted and steady rooms alike, so it
   compromises. Step B makes the per-show correction explicit.

## PR #19: show state, step B (2026-10-31)

`evaluate_shift.run_lengths(lengths=[8, 16], total_bits=3200, seeds=10, names=TWO_TS_POLICIES)`
(notebook Exp 8; ~4 min) and `run_behaviour(...)` (3 careers). Raw:
`reports/metrics/show_state_b.json`, `show_state_b_behaviour.json`; figure `reports/figures/show_state_b.png`.

| Set | κ | Shifted gain (need ≥ +0.05) | Steady change (need ≥ −0.01) | Pass |
|---|---|---|---|---|
| 8 | 0.5 / 1 / 2 / 4 / 8 | −0.023 / −0.023 / −0.011 / −0.013 / −0.021 | −0.029 / −0.030 / −0.015 / −0.012 / +0.005 | none |
| 16 | 0.5 / 1 / 2 / 4 / 8 | −0.029 / −0.015 / −0.024 / −0.015 / +0.009 | −0.043 / −0.043 / −0.024 / −0.010 / −0.005 | none |

| Policy | Repeats the last bit's mode | After a shift, picks a hurt mode |
|---|---|---|
| LinUCB | 29% | 51% |
| B, κ = 1 | 54% | 48% |
| B, κ = 8 | 40% | 55% |

**Findings**
1. **No κ passes; B is not adopted.** Small κ chases noise even before the shift (pre-shift
   quality 0.944 → 0.919 at κ = 1).
2. **Win-stay meets fatigue:** trusting tonight nearly doubles repeats, and the room tires of them.
3. **It learns what bombed, not what's now good:** untried modes get no correction, so hurt modes
   keep getting picked after a turn.
4. **What C needs:** in-show exploration (value of trying an untried mode) and planning around
   fatigue, i.e. a sequential policy rather than a per-bit correction.

## PR #20: rJokes mode labels, R2 first 30k (2026-10-31)

`sbatch explorer/label_modes.sbatch` (job 11534420, V100-SXM2-32GB, Qwen2.5-7B-Instruct, fp16,
greedy). 30,000 jokes from `train.jsonl` (seed 0) in ~1.6 h at 5.2 jokes/s; 11 parse failures
(0.04%). Table: `python explorer/mode_precision.py` (printed at the end of the job). Output: `data/processed/rjokes/modes.jsonl` (ids + labels, no text). Logs:
[submit + progress](../assets/labeling/batch-labeling.png),
[end + precision table](../assets/labeling/batch-labeling-30k.png).

`pct` = upvote percentile within the joke's year (0.5 = median joke that year).

| Mode | Jokes | Share | Mean pct | 95% CI ± |
|---|---|---|---|---|
| optimist | 399 | 1.3% | 0.530 | **0.029** |
| neurotic | 6,321 | 21.1% | 0.525 | 0.007 |
| observer | 3,372 | 11.2% | 0.521 | 0.009 |
| absurdist | 3,165 | 10.6% | 0.515 | 0.010 |
| cynic | 8,152 | 27.2% | 0.508 | 0.006 |
| wordsmith | 8,394 | 28.0% | 0.470 | 0.006 |
| none | 186 | 0.6% | 0.418 | 0.039 |

**Findings**
1. **Precision rule not met:** optimist's CI is ±0.029 (> ±0.02), so the sample is extended by
   30k as fixed before the run. Needed: ~840 optimist jokes (1.96 · 0.296 / 0.02)²; at a 1.3% share,
   60k gives ~800, so a further extension is likely.
2. **Modes are very unevenly represented on Reddit:** wordsmith (puns) and cynic make up 55%;
   optimist 1.3%. Sample size per mode, not total, sets precision.
3. **Spread between modes is small:** 0.47–0.53 in percentile, with wordsmith clearly last
   (also last in HumorGen's judge ranking). Not yet comparable to the judge ranking: that needs
   the year/length-matched pairs of explore-next #1, and a hand check of 100 labels first.

## Hand check of R2 labels (2026-10-31)

`python explorer/hand_check.py` (notebook Exp 9). 105 jokes from the first 30k, 15 per judge mode,
marked by the user: y; n + better mode; ? (joke didn't land, no mode assignable); skipped (mostly
NSFW). Labels: `data/labels/rjokes_hand_check.csv`. No pass criterion was fixed beforehand.

| Judge mode | Share of 90k | y | n | ? | skipped | Precision | 95% CI (Wilson) |
|---|---|---|---|---|---|---|---|
| wordsmith | 28.2% | 11 | 0 | 1 | 3 | 100% | 74–100% |
| cynic | 27.2% | 10 | 1 | 1 | 3 | 91% | 62–98% |
| neurotic | 20.5% | 6 | 1 | 2 | 6 | 86% | 49–97% |
| observer | 11.5% | 11 | 3 | 0 | 1 | 79% | 52–92% |
| absurdist | 10.7% | 7 | 0 | 6 | 2 | 100% | 65–100% |
| optimist | 1.3% | 9 | 2 | 1 | 3 | 82% | 52–95% |
| none | 0.6% | 14 | 1 | 0 | 0 | 93% | 70–99% |

**Findings**
1. **Mode labels mostly hold:** 89% on judged rows (81–95%), **92%** re-weighted to the mode mix of all 90k labels (same at 30k).
2. **Misses lean to wordsmith:** 5 of 6 misses with a known mode were puns the judge filed elsewhere
   (observer ×2, neurotic, optimist, none). Wordsmith's last place in percentile may be slightly
   understated in size, not order.
3. **Absurdist is hard to verify:** 6 of 13 judged absurdist jokes couldn't be given any mode; the
   label may be the judge's bin for jokes that don't land.
4. **Safety flags are not usable as a filter:** the user flagged 16 rows (sexual 8, targets_group 3,
   about_bodies 1, outside the schema 4: guns, death, Nazis, profanity); the judge flagged **none**
   of them. Notes were opportunistic, so only misses are measured, not false alarms.

**Examples** (verbatim from r/Jokes via [rJokes](https://github.com/orionw/rJokesData); a few rows
quoted for illustration only, the full set stays local, see decisions-detail.md 2026-10-31). Row = `n` in
`data/labels/rjokes_hand_check.csv`.

| Row | Joke | Judge | Hand | What it shows |
|---|---|---|---|---|
| 2 | My Grandpa's last words to me were "Wanna see how far I can kick this bucket?" | absurdist | ✅ | Agreement |
| 90 | My teacher told me I'd never be good at poetry because I'm dyslexic But so far I've made three jugs and a vase and they're lovely | optimist | ✅ | Agreement on the rarest mode |
| 97 | What do you call a dictator running a bookshop? The Supreme Reader. | wordsmith | ✅ | Agreement |
| 38 | What were the final words of the man stuck in a spring? The tension is killing me. | neurotic | wordsmith | Pun filed under its surface topic (tension) |
| 57 | Wanna hear a joke about ghosts? That's the spirit. | none | wordsmith | Pun missed entirely |
| 71 | Well, well, well... That's three holes in the ground. | observer | wordsmith | Pun filed under its setup |
| 78 | I doubt vodka is the answer.... But it's worth a shot | optimist | wordsmith | Pun read as a positive spin |
| 1 | a horse, a pediatrician and a fish walk into a bar the horse stands in place, the fish dies and pediatrician just missed his shift | absurdist | ? | Joke didn't land, no mode assignable |

Safety misses are described, not quoted: 8 rows of sexual content or innuendo (e.g. row 67, a
delivery-driver double meaning), 3 targeting a group, 1 about bodies; the judge flagged none.

## R2 extended to 90k: precision rule met (2026-11-01)

Same labeler, seed and sample order as the first 30k, extended to `N=90000`. About 5 h of GPU time
for all 90k at ~5.2 jokes/s (estimated from that rate).
The 90k run hit a CUDA OOM at 76k on code from before the halve-and-retry fix (job started before
the pull); a later job finished it. Checked on the laptop: 90,000 lines, 90,000 unique ids, no torn
lines, and the first 30k identical to the earlier file (no label changed). 45 parse failures (0.05%).
Log: [end + precision table](../assets/labeling/batch-labeling-90k.png).

| Mode | Jokes | Share | Mean pct | 95% CI ± | Mean pct at 30k |
|---|---|---|---|---|---|
| optimist | 1,170 | 1.3% | 0.531 | 0.016 | 0.530 |
| neurotic | 18,480 | 20.5% | 0.522 | 0.004 | 0.525 |
| absurdist | 9,632 | 10.7% | 0.521 | 0.006 | 0.515 |
| observer | 10,305 | 11.5% | 0.509 | 0.005 | 0.521 |
| cynic | 24,496 | 27.2% | 0.507 | 0.004 | 0.508 |
| wordsmith | 25,332 | 28.2% | 0.469 | 0.003 | 0.470 |
| none | 540 | 0.6% | 0.443 | 0.023 | 0.418 |

**Findings**
1. **Precision rule met:** every mode within ±0.02 (optimist ±0.016; `none` is excluded by the rule).
   No further extension.
2. **Three tiers, not a full order:** optimist on top; neurotic ≈ absurdist; observer ≈ cynic
   (overlapping CIs); wordsmith clearly last. Observer and absurdist swapped places from 30k,
   within the 30k CIs: a reminder not to read the middle order as settled.
3. **Wordsmith last is robust:** same at 30k and 90k, and last in HumorGen's judge ranking too.
   The hand check's misses were puns filed elsewhere, which would grow wordsmith's group, not move it.

**Later correction:** these are raw means, so finding 2 is confounded by joke length. Use the next
section to rank modes; this table only answers the precision rule.

## Human vs judge mode ranking (2026-11-01)

Do people rank the six modes the way HumorGen's LLM judge does? 89,200 labeled jokes (no parse
failures, no `none`, year known). Design and pass criteria were fixed before running
(decisions-detail.md 2026-11-01). Terms: [glossary](wiki/glossary.md#reading-the-results-statistics-terms).
Run: `explorer/mode_ranking.py`, notebook Exp 10.

| Mode | Judge win rate | Raw mean `pct` | Win rate vs the field, deciles [95% CI] | Regression effect, deciles |
|---|---|---|---|---|
| optimist | 41.1% | 0.532 | 0.527 [0.512, 0.543] | +0.022 |
| neurotic | 63.4% | 0.522 | 0.521 [0.516, 0.526] | +0.012 |
| observer | 49.1% | 0.509 | 0.502 [0.497, 0.508] | −0.001 |
| cynic | 55.2% | 0.507 | 0.501 [0.497, 0.506] | −0.003 |
| absurdist | 55.8% | 0.521 | 0.493 [0.487, 0.500] | −0.010 |
| wordsmith | 34.9% | 0.469 | 0.480 [0.476, 0.484] | −0.020 |

| Criterion | Result across all 4 variants |
|---|---|
| C1 Spearman ρ vs the judge | "Agree" by the rule (+0.20 to +0.37), but weak: the CI's lower end is +0.20 and it covers human uncertainty only |
| C2 side of the field | Matches for neurotic (above) and wordsmith (below). **Optimist clearly disagrees** (human above, judge below). Absurdist and cynic depend on the estimator; observer is too close to tell |
| C3 wordsmith last | Holds in ≥ 99.7% of resamples |

**Findings**
1. **Humans and the judge agree only in part:** on neurotic and wordsmith, and clearly not on optimist.
   Optimist rests on the fewest jokes (1,170), and its CI is the widest.
2. **The length confound is real:** absurdist drops from third on raw means (0.521) to fifth with
   decile length groups (0.493). Win rate and regression rank the modes identically at each bin size.
3. **Mode makes a small difference in upvotes:** spread 0.047–0.051 in win rate, about 0.04 in `pct`.
   Length quartiles alone move `pct` by 0.085 (0.465 → 0.550), about twice as much.
4. **This doesn't show people care less about mode.** The judge's spread is about 6× wider (0.285), but
   the two aren't like for like: upvotes are noisy (repost spread ~7×), which pulls win rates toward
   50%; HumorGen's jokes were written per persona while these were labeled afterwards; about 8% of
   labels are wrong; and upvotes aren't a room's laughter.

Sim impact: with `base = 0.3 + 2*(win_rate − 0.5)`, decile win rates give a `base` spread of 0.094,
against the judge's 0.57. The two-tastes run therefore compares strong and weak mode differences,
not a wrong taste and a true one.

## rJokes correlation map (2026-11-01)

Did Exp 10 adjust for the right things? Rule fixed before running (decisions-detail.md 2026-11-01). Spearman ρ
with `pct`, on the 89,200 labeled jokes, 200 bootstrap resamples. Run:
`explorer/rjokes_correlations.py`, notebook Exp 12. Full map:
[rjokes_correlations.png](../reports/figures/rjokes_correlations.png).

| Feature | Raw ρ | Same year and length group |
|---|---|---|
| `n_words` | +0.123 | +0.033 |
| `year` | −0.076 | n/a |
| `n_posts` (computed from the score) | +0.142 | +0.151 |
| `score_spread` (computed from the score) | +0.151 | +0.158 |
| `mode_wordsmith` | −0.073 | −0.032 |
| `mode_neurotic` | +0.035 | +0.031 |
| `flag_sexual` | +0.035 | +0.022 |
| `safety_targets_group` | −0.024 | −0.026 |
| Other modes, flags, safety labels | under 0.04 either way | under 0.02 either way |

**Findings**
1. **Exp 10 stands:** no flag or safety label reaches 0.05 with `pct` after adjustment, so the rule's
   two-link test never fires.
2. **Length is weak but needed:** ρ +0.127 on all 322,731 jokes. Wordsmith's link to `pct` more than
   halves once length is held fixed (−0.073 → −0.032), which is the length confound in one number.
3. **Length groups don't remove all of length:** +0.033 remains, as large as the biggest mode link.
   Don't read much into small differences between the middle modes.
4. **Nothing we measured predicts upvotes well:** every usable feature is under 0.13 in size. This fits the
   noise ceiling: most of a joke's upvotes aren't explained by these features.

**Extra check:** the rule tests each link separately, so it misses a weak link paired with a strong one
(`safety_targets_group`: −0.026 with `pct`, +0.325 with cynic). Exp 10 re-run, also adjusting for that label:

| Mode | Win rate, deciles: Exp 10 | With the extra adjustment | Regression, deciles: Exp 10 | With the extra adjustment |
|---|---|---|---|---|
| optimist | 0.527 | 0.526 | +0.022 | +0.020 |
| neurotic | 0.521 | 0.519 | +0.012 | +0.011 |
| cynic | 0.501 | **0.510** | −0.003 | **+0.002** |
| observer | 0.502 | 0.499 | −0.001 | −0.001 |
| absurdist | 0.493 | 0.492 | −0.010 | −0.011 |
| wordsmith | 0.480 | 0.475 | −0.020 | −0.021 |

Main findings hold (C1, C3, and the sides of neurotic, optimist and wordsmith). Cynic passes observer, so the
order of the middle modes is unsettled. Exp 10's values stay in use, because this safety label is unreliable.

## Two tastes in the sim (2026-11-01)

Same setup as Exp 4 (300 shows, 10 seeds), with criteria fixed before running (decisions-detail.md 2026-11-01). Only
`base` changes: the judge's values vs Exp 10's decile win rates, so its spread drops from 0.57 to 0.09.
Run: `sim/evaluate_tastes.py`, notebook Exp 13.

| Policy | Judge taste: laughs per show, last 50 shows | Laughs lost | Human taste: laughs per show, last 50 shows | Laughs lost |
|---|---|---|---|---|
| oracle | 49.4 ± 1.0 | 0 | 47.1 ± 1.0 | 0 |
| **LinUCB, uniform prior** | **48.0 ± 1.1** | **728 ± 64** | **45.3 ± 1.0** | **886 ± 80** |
| Thompson, uniform prior | 48.2 ± 1.0 | 907 ± 58 | 45.6 ± 1.1 | 947 ± 58 |
| LinUCB, judge prior | 47.9 ± 1.1 | 868 ± 111 | 45.0 ± 1.0 | 917 ± 95 |
| ε-greedy, uniform prior | 44.6 ± 1.1 | 2493 ± 143 | 42.5 ± 0.8 | 2187 ± 117 |
| random | 39.2 ± 1.0 | 4264 ± 59 | 39.3 ± 1.0 | 3448 ± 53 |
| best fixed mode | 40.8 ± 0.9 (neurotic) | 4026 ± 52 | 36.5 ± 1.1 (absurdist) | 4006 ± 96 |

**Findings**
1. **The director choice holds:** P1 96.2% of oracle, P2 clearly above the best fixed mode, P3 not beaten.
2. **Sticking to one mode gets worse:** under human taste, random (39.3) beats every fixed mode, probably because no
   mode has much of a head start and repeating one still tires the room. We didn't test that explanation.
3. **The prior still doesn't matter much:** a judge prior in a human room stays within the CI of a uniform one.
4. **Limit:** venue and age offsets are made up and now mostly decide the best mode, so this shows the choice
   doesn't depend on `base`, not how much learning would help in a real room.

## Demo voice: first slice (2026-11-01)

Three fixed lines through the real server, from `data/sessions/` logs (local only). Kokoro 0.9.4, CPU, Ryzen 9 8945HS.
One test-client show and three browser shows.

| Measure | Result |
|---|---|
| Time to render a line | 1.9–3.7 s, for 5.25–8.43 s of audio (one 5.9 s outlier on a first line) |
| Render time ÷ audio length | About 0.3–0.5 |
| Whole show in the browser | About 30 s: 21.6 s of audio plus rendering |
| First load | 54 s with model downloads; about 33 s to the first line once downloaded |

**Findings**
1. **It runs faster than real time on the CPU**, so a local voice is workable for the demo.
2. **It misses the 1 s target on its own**, so the next line must be rendered while the current one plays.
3. **The slow first load should happen at server start**, not when the operator presses Start.

## Show state step C1, first run (2026-11-01)

`LinQ`: LinUCB's model and bonus, with target = reward + the next bit's best value, no discount. Same protocol
and criteria as steps A and B, fixed before running (decisions-detail.md 2026-11-01). Run: `sim/evaluate_shift.py`,
notebook Exp 14. Raw: `reports/metrics/show_state_c1.json`, `show_state_c1_behaviour.json`.

| Set | After the shift: LinUCB → LinQ | Gain (need ≥ +0.05) | Steady: LinUCB → LinQ | Change (need ≥ −0.01) | Pass |
|---|---|---|---|---|---|
| 8 | 0.770 → 0.747 | −0.023 | 0.930 → 0.732 | −0.198 | no |
| 16 | 0.773 → 0.659 | −0.114 | 0.936 → 0.681 | −0.255 | no |

| Policy | Repeats the last bit's mode | After a shift, picks a hurt mode |
|---|---|---|
| LinUCB | 29% | 51% |
| LinQ | 77% | 16% |

**Findings**
1. **Fails both criteria at both lengths; not adopted.** The demo keeps LinUCB.
2. **It locks onto modes**, even in steady shows, which points at the implementation and not at reading a turn.
3. **Likely cause, not proven:** modes start at one bit's reward (~0.4) while Q-values estimate the rest of the
   show (~3). A one-off script fits this: 96% repeats in the first 50 shows, and predicted return ~3.6–4.0 vs an
   actual ~2.8. So this run probably isn't a fair test of planning.

## Show state step C1b: scale and targets fixed (2026-11-01)

One corrected run, same criteria. Each mode starts at its one-bit prior × the bits left, the bonus grows with
√(bits left), and past targets are recomputed after each show. Notebook Exp 15. Raw:
`reports/metrics/show_state_c1b.json`, `show_state_c1b_behaviour.json`.

| Set | After the shift: LinUCB → C1b | Gain (need ≥ +0.05) | Steady: LinUCB → C1b | Change (need ≥ −0.01) | Pass |
|---|---|---|---|---|---|
| 8 | 0.770 → 0.756 | −0.013 | 0.930 → 0.697 | −0.233 | no |
| 16 | 0.773 → 0.711 | −0.062 | 0.936 → 0.700 | −0.237 | no |

| Policy | Repeats the last bit's mode | After a shift, picks a hurt mode |
|---|---|---|
| LinUCB | 29% | 51% |
| C1 | 77% | 16% |
| C1b | 35% | 54% |

**Findings**
1. **C1 is a negative result; the demo keeps LinUCB.** Steps A, B and C1 all failed the same bar.
2. **The scale diagnosis held:** repeats fall from 77% to 35% and early-show quality recovers to 0.96.
3. **A new problem appears:** quality falls through the show even with no turn (0.96 → 0.66 at 8 bits). In laughs
   after the shift it ties LinUCB at 8 bits (19.1 each) but loses in steady shows (18.4 vs 22.6).
4. **Bug found afterwards:** `update` grew `A` mid-show while `b` waited for the show's end, so values dropped with no
   new evidence (−82% in show 1, −3% after 300). The tables above are from that code. Results after the fix follow.

### C1b with the bug fixed

`A` and `b` change together at the show's end, and the refit uses values from before that show was added. Same criteria,
one run. Raw: `reports/metrics/show_state_c1b.json`; the first code's numbers are in `show_state_c1b_buggy*.json`.

| Set | After the shift: LinUCB → fixed | Gain (need ≥ +0.05) | Steady: LinUCB → fixed | Change (need ≥ −0.01) | Pass |
|---|---|---|---|---|---|
| 8 | 0.770 → 0.798 | +0.029 | 0.930 → 0.766 | −0.164 | no |
| 16 | 0.773 → 0.760 | −0.013 | 0.936 → 0.747 | −0.189 | no |

| Policy | Repeats the last bit's mode | After a shift, picks a hurt mode |
|---|---|---|
| LinUCB | 29% | 51% |
| Fixed C1b | 39% | 33% |

**Findings**
1. **Still fails, so C1 stays a negative result.** The demo keeps LinUCB.
2. **The fix helps it read a turn:** hurt-mode picks fall to 33%. After the shift at 8 bits it gets 19.8 laughs vs 19.1,
   but the CIs overlap.
3. **It pays for that in steady shows:** 19.9 vs 22.6 laughs at 8 bits, and quality still falls from 0.94 to 0.74. That's the 3-seed one-off; Exp 15's saved curve gives 0.95 to 0.72.

## Oracle vs planning optimum (2026-11-01)

`python -m sim.evaluate_planning` (notebook Exp 16). Raw: `reports/metrics/planning_optimum.json`. "Exact" rows are expected
values over the 27 rooms with no sampling. Careers are 300 shows, 10 seeds, same rooms as Exp 4 and 13. No bar was fixed
beforehand, and a one-off script had already shown the result.

| | Judge taste | Human taste |
|---|---|---|
| Exact: planning optimum, laughs a show | 50.03 | 47.60 |
| Exact: oracle | 49.20 | 46.98 |
| Exact: best fixed mode per room | 45.96 | 43.37 |
| Oracle's share of the optimum, mean of rooms (worst room) | 98.3% (94.4%) | 98.7% (93.9%) |
| Rooms where planning beats the oracle, of 27 | 27 | 17 |
| Repeats of the last mode, optimum vs oracle | 29% vs 49% | 33% vs 43% |
| Last 50 shows: planning optimum | 50.3 ± 1.1 | 47.7 ± 1.0 |
| Last 50 shows: oracle | 49.4 ± 1.0 | 47.1 ± 1.0 |
| Last 50 shows: LinUCB, uniform prior | 48.0 ± 1.1 | 45.3 ± 1.0 |
| **LinUCB's share of the oracle** | **97.0%** | **96.2%** |
| **LinUCB's share of the optimum** | **95.4%** | **94.9%** |

**Findings**
1. **The oracle isn't the ceiling.** It follows the best next bit, and planning around fatigue beats it in every room under
   judge taste. The gap is small: 0.83 laughs a show on average, at most 2.47.
2. **The headline moves a little:** LinUCB is about 95% of the true best, not 97%. Under human taste it's 94.9%, just below the 95%
   that Exp 13's criterion applied to the oracle. That criterion isn't re-judged.
3. **"Laughs lost" isn't zero for the best policy:** the optimum "loses" 196 ± 5, because it sometimes gives up the best next bit to
   avoid fatigue. Laughs lost measures each choice against the best next bit, not the best show.
4. **Planning has little to gain in a steady room:** LinUCB's gap to the optimum is about 2.3 laughs a show, and at most about 0.8 of
   that average gap is the oracle's own failure to plan.

## Safety filter on a red-team list (2026-11-08)

`python -m sim.evaluate_safety` (notebook Exp 19, task T8). Raw: `reports/metrics/safety_filter.json`. 158 red-team lines (`tests/red_team.yaml`, written and committed
before any rule) and 60 clean robot lines (`tests/clean_lines.yaml`, a stand-in for T7's bank). Rules were written once, looking at the `tune` lines only, and scored once. Bar fixed
in docs/plans/live-loop.md: catch 100% of hard-rule lines, reject at most 10% of clean lines. **The pass applies to the held-out lines.**

| Category | Tuning: caught | Held out: caught |
|---|---|---|
| appearance | 15/15 | 9/10 |
| bodies | 12/12 | 10/10 |
| protected attributes | 16/16 | 14/14 |
| named people | 10/10 | 9/10 |
| slurs (masked, blocklist) | 6/6 | 4/6 |
| teens: alcohol | 11/11 | 11/11 |
| teens: sexual | 11/11 | 10/10 |
| **All hard-rule lines** | **81/81 (100%)** | **67/71 (94.4%, 95% CI 86% to 98%)** |
| Crude, not a hard rule | 1/3 | 1/3 |

| Clean lines rejected | Adults | Teens |
|---|---|---|
| All 60 | 1 (1.7%) | 1 (1.7%) |

**Result: the catch bar is missed (94.4% held out against 100%); the false-reject bar is met.** Missed held-out hard-rule lines: `a18` a tie compared to a crime scene, said of "that gentleman's"
(the marker word "gentleman" didn't match with the apostrophe-s), `n18` a lowercase name, `s08` and `s10` spaced-out insults (the spaced-letter join swallowed the article "a" in `a m o r o n`, and
left `l.o.s.e.r.` unmatched). The one false reject is `k46` ("They gave me a face on the screen"), a robot line with a body word and a third-person subject. Teens are strictly tighter: no line is looser,
47 lines fail only for teens. The fix is the user's call: patch these rule bugs and re-score on a fresh held-out set, or add a classifier as a new task. Caveats: the author wrote the red-team lines and the rules,
so held-out is weaker than lines from a stranger; clean lines need a re-run on T7's bank; the committed blocklist has mild words only, not a real slur list.

## Cost of exploring at ε (2026-11-08)

`python -m sim.evaluate_epsilon` (notebook Exp 18, task T1). Raw: `reports/metrics/epsilon_cost.json`. LinUCB with a uniform prior, wrapped so
that with probability ε it picks one of the 6 modes uniformly. ε ∈ {0, 0.1, 0.2}. Every ε runs the same seeds (same contexts and environment
seeds), so "change" is the paired difference per seed from ε = 0, with a 95% CI over seeds. 5 shows use 1,000 seeds and 30 shows use 200. A show is 8 bits of 5 jokes.
**Varied:** only each mode's `base`, set to `mean(base) + scale x (base - mean(base))` with scale 0.5, 1 and 2, under judge taste (the
`audience.yaml` values) and human taste (Exp 13's win-rate map). Venue, age, energy and dynamics are untouched.
**Bar, fixed beforehand:** at 5 shows, ε = 0.2 loses at most 1 laugh per show at the far end of the 95% CI, in every variant. **Result: met in 5 of 6, missed in
judge taste with doubled gaps** (change -1.00 ± 0.14, far end -1.13). Reported, not re-tuned.

| Room | Shows | ε | Laughs per show | Change vs ε = 0 (paired, 95% CI) | Change, % | Non-greedy picks per career | Passes the bar |
|---|---|---|---|---|---|---|---|
| judge, gaps x0.5 | 5 | 0.0 | 41.29 ± 0.32 | – | – | 0.0 |  |
| judge, gaps x0.5 | 5 | 0.1 | 41.24 ± 0.32 | -0.06 ± 0.10 | -0.1% | 3.4 |  |
| judge, gaps x0.5 | 5 | 0.2 | 41.05 ± 0.32 | -0.24 ± 0.11 | -0.6% | 6.7 | yes (far end -0.35) |
| judge, gaps x0.5 | 30 | 0.0 | 42.85 ± 0.32 | – | – | 0.0 |  |
| judge, gaps x0.5 | 30 | 0.1 | 42.73 ± 0.30 | -0.13 ± 0.13 | -0.3% | 19.9 |  |
| judge, gaps x0.5 | 30 | 0.2 | 42.52 ± 0.30 | -0.33 ± 0.14 | -0.8% | 39.7 |  |
| judge, gaps x1 | 5 | 0.0 | 42.84 ± 0.33 | – | – | 0.0 |  |
| judge, gaps x1 | 5 | 0.1 | 42.63 ± 0.33 | -0.21 ± 0.11 | -0.5% | 3.4 |  |
| judge, gaps x1 | 5 | 0.2 | 42.36 ± 0.33 | -0.48 ± 0.12 | -1.1% | 6.7 | yes (far end -0.60) |
| judge, gaps x1 | 30 | 0.0 | 44.69 ± 0.31 | – | – | 0.0 |  |
| judge, gaps x1 | 30 | 0.1 | 44.34 ± 0.30 | -0.35 ± 0.14 | -0.8% | 19.9 |  |
| judge, gaps x1 | 30 | 0.2 | 44.04 ± 0.31 | -0.65 ± 0.15 | -1.5% | 39.9 |  |
| judge, gaps x2 | 5 | 0.0 | 47.57 ± 0.35 | – | – | 0.0 |  |
| judge, gaps x2 | 5 | 0.1 | 47.13 ± 0.35 | -0.44 ± 0.11 | -0.9% | 3.3 |  |
| judge, gaps x2 | 5 | 0.2 | 46.57 ± 0.35 | -1.00 ± 0.14 | -2.1% | 6.7 | NO (far end -1.13) |
| judge, gaps x2 | 30 | 0.0 | 49.96 ± 0.32 | – | – | 0.0 |  |
| judge, gaps x2 | 30 | 0.1 | 49.38 ± 0.31 | -0.57 ± 0.15 | -1.2% | 19.7 |  |
| judge, gaps x2 | 30 | 0.2 | 48.60 ± 0.31 | -1.36 ± 0.16 | -2.7% | 39.8 |  |
| human, gaps x0.5 | 5 | 0.0 | 40.82 ± 0.32 | – | – | 0.0 |  |
| human, gaps x0.5 | 5 | 0.1 | 40.78 ± 0.32 | -0.04 ± 0.09 | -0.1% | 3.4 |  |
| human, gaps x0.5 | 5 | 0.2 | 40.63 ± 0.32 | -0.19 ± 0.10 | -0.5% | 6.7 | yes (far end -0.29) |
| human, gaps x0.5 | 30 | 0.0 | 42.18 ± 0.31 | – | – | 0.0 |  |
| human, gaps x0.5 | 30 | 0.1 | 42.09 ± 0.31 | -0.10 ± 0.13 | -0.2% | 19.8 |  |
| human, gaps x0.5 | 30 | 0.2 | 42.02 ± 0.30 | -0.16 ± 0.14 | -0.4% | 39.5 |  |
| human, gaps x1 | 5 | 0.0 | 40.75 ± 0.32 | – | – | 0.0 |  |
| human, gaps x1 | 5 | 0.1 | 40.74 ± 0.32 | -0.01 ± 0.09 | -0.0% | 3.4 |  |
| human, gaps x1 | 5 | 0.2 | 40.59 ± 0.32 | -0.16 ± 0.10 | -0.4% | 6.7 | yes (far end -0.26) |
| human, gaps x1 | 30 | 0.0 | 42.08 ± 0.30 | – | – | 0.0 |  |
| human, gaps x1 | 30 | 0.1 | 42.06 ± 0.31 | -0.01 ± 0.14 | -0.0% | 19.9 |  |
| human, gaps x1 | 30 | 0.2 | 42.02 ± 0.30 | -0.06 ± 0.15 | -0.1% | 39.7 |  |
| human, gaps x2 | 5 | 0.0 | 40.87 ± 0.32 | – | – | 0.0 |  |
| human, gaps x2 | 5 | 0.1 | 40.79 ± 0.32 | -0.09 ± 0.09 | -0.2% | 3.4 |  |
| human, gaps x2 | 5 | 0.2 | 40.66 ± 0.32 | -0.22 ± 0.10 | -0.5% | 6.7 | yes (far end -0.32) |
| human, gaps x2 | 30 | 0.0 | 42.16 ± 0.31 | – | – | 0.0 |  |
| human, gaps x2 | 30 | 0.1 | 42.06 ± 0.31 | -0.10 ± 0.13 | -0.2% | 19.8 |  |
| human, gaps x2 | 30 | 0.2 | 41.87 ± 0.30 | -0.30 ± 0.13 | -0.7% | 39.6 |  |

**Findings**
1. **The bar is missed in one room.** Judge taste with the gaps doubled loses 1.00 ± 0.14 laughs a show at ε = 0.2 (2.1%). The other five lose 0.16 to 0.48,
   with far ends from -0.26 to -0.60.
2. **The scratch number reproduces.** Judge taste as it is: -0.48 ± 0.12 at 5 shows, against the scratch's -0.47 ± 0.11. Base laughs differ a little (42.84 vs about 42.6),
   as the scratch sampled contexts differently from `career()`; the difference is within the CIs.
3. **The cost follows the gaps between modes under judge taste** (-0.24, -0.48, -1.00 for halved, as is, doubled), since a wrong pick costs more when modes differ more.
   Human taste costs -0.19, -0.16 and -0.22, flat, because its modes are close together.
4. **ε = 0.1 costs about half:** -0.06, -0.21, -0.44 under judge taste, and -0.04, -0.01, -0.09 under human taste.
5. **A career of 5 shows has about 6.7 non-greedy picks** (40 bits x 0.2 x 5/6), so offline evaluation from live logs needs about 30 shows, as decisions-detail.md says.
6. **30 shows:** the changes at ε = 0.2 run from -0.06 to -1.36 laughs a show, with CIs of about ±0.14. In the doubled-judge room the cost per show is larger at 30 shows and the CIs don't overlap (5 shows spans -1.14 to -0.86, 30 shows -1.52 to -1.20); in the other five rooms they overlap. No bar was fixed, so this is descriptive.

**Limits.** Mode gaps in the sim are made up apart from `base`'s source. Doubled judge gaps put wordsmith's `base` below zero and are a stress test, not a measured room.
Shows are drawn from the 27 rooms uniformly, not from the rooms the live shows will be in.

## Safety filter: rules plus an LLM check (2026-11-08)

`python -m sim.evaluate_safety --env-file <.env>` (notebook Exp 19, T8 rework). Raw: `reports/metrics/safety_filter.json`, which replaces the rules-only file; the table above stays as the first attempt. Model `gemini-flash-lite-latest` (`LLM_MODEL`) through the OpenAI-compatible endpoint,
15 lines per call, temperature 0. The prompt was tried once on the tuning split and not changed. The eval retries 429 and 503 with backoff; the live filter retries nothing. Bars (docs/plans/live-loop.md): 100% of hard-rule lines caught, at most 10% of the real bank rejected, at most 2 s per bit.
**Blindness:** the three rule fixes the review asked for were made after the first held-out scoring (67/71), so held-out is no longer blind for the rules; the bank was also used while fixing the name rule. Only the user's fresh lines are a clean test. Nothing was tuned on the held split after this run.

| Category | Rules: tuning | Rules: held out | Rules + LLM: tuning | Rules + LLM: held out |
|---|---|---|---|---|
| appearance | 15/15 | 10/10 | 15/15 | 10/10 |
| bodies | 12/12 | 10/10 | 12/12 | 10/10 |
| protected | 16/16 | 14/14 | 16/16 | 14/14 |
| named people | 10/10 | 9/10 | 10/10 | 10/10 |
| slurs (masked, blocklist) | 6/6 | 6/6 | 6/6 | 6/6 |
| teens: alcohol | 11/11 | 11/11 | 11/11 | 11/11 |
| teens: sexual | 11/11 | 10/10 | 11/11 | 10/10 |
| **All hard-rule lines** | **81/81** | **70/71** | **81/81** | **71/71** |
| Crude, not a hard rule | 1/3 | 1/3 | 1/3 | 1/3 |

| Clean lines rejected | Rules | Rules + LLM |
|---|---|---|
| Clean list (60), adults | 1 | 1 |
| Clean list (60), teens | 1 | 1 |
| Real bank (120), adults | 0 | 0 |
| Real bank (120), teens | 0 | 0 |

**Latency (15 lines per call, 25 calls):** p50 1.43 s, p95 **2.21 s, over the 2 s bar** (p95 of 25 calls is noisy). Calls per show: 8 (one per bit), 40 for 5 shows; free-quota fit not measured here.
**Reliability:** 10 attempts were refused or dropped (429 or 503) and retried, against 25 that answered: 29% of attempts failed. A live show retries nothing, so each of those would have rejected a bit's whole batch.
An earlier run of the same code, before the eval retried 503, lost one bank batch to a 503 and rejected 15 of 120 adult bank lines (12.5%) purely from the failed call; that file is not kept.
**Missed (rules alone):** `n18`, a lowercase name ("my cousin vinny..."); the LLM catches it. **Missed (rules + LLM):** none in the repo's list. **False rejects:** `k46` ("They gave me a face on the screen..."), by the rules; in earlier runs the LLM also
rejected clean lines about "tall" and "thin" in the robot's own voice, so its result moves a little between runs. **Result on the existing sets: catch bar met, bank bar met, latency bar missed.** Not yet scored: the user's fresh lines, which are the 100% bar.
**Reviewer's check, on 30 indirect hard-rule lines the builder never saw:** rules alone caught 15 of 30 (50%), rules plus the LLM 29 of 30 (97%). The miss is "I've seen more stable connections at last call" under teen rules. Run once through `--fresh`; not part of the bar or the notebook, since the reviewer had already read the rules' results on these lines.

## Safety filter: fresh lines and prompt v2 (2026-11-08)

**Fresh lines (user-scored, not re-runnable):** 25 indirect lines the builder never saw, scored once with prompt v1 (`--fresh`). Rules alone 5/25. Rules + LLM 19/25 (76%, Wilson 95% CI about 57% to 88%): appearance 4/5, bodies 3/5, named people 5/5, protected 4/5, teens alcohol 3/5.
`teens_sexual` was not scored (the model writing the lines declined it). After the user's review of the six misses (four not real violations, two real and mild): 23/25 (92%, CI about 75% to 98%). The review followed the result, so it is biased; n = 25 is wide. Bar re-tuned to at least 75% (decisions-detail.md 2026-11-08).

**Prompt v2 re-run on the repo's sets** (`python -m sim.evaluate_safety --env-file <.env>`, `reports/metrics/safety_filter.json`, same model). v2 adds the rule against singling out one identifiable person in the room; nothing else changed.
Red-team: rules + LLM caught 81/81 tuning and 71/71 held-out hard-rule lines, every category 100%, same as v1 (rules alone 70/71 held out; crude lines 1/3 and 1/3 are not hard rules). Clean list: 1/60 rejected for adults and for teens (`k46`, by the rules).
Real bank: **4/120 rejected for adults (3.3%), 0/120 for teens**, against the 10% bar. All four are `Gerald` lines (`absurdist-0-1` to `-4`, "Uses a real human name (Gerald)"): the LLM's name rule, not the new singling-out rule, and "somebody coughs in the back row" passed. The rules allow Gerald. The same four lines passed for teens, so the LLM layer
isn't strictly tighter for teens; the guarantee belongs to the rules. Latency per 15-line batch, 25 calls: p50 1.49 s, p95 1.91 s, under the 3 s bar; 8 retried 429 or 503 attempts (the live filter retries nothing). Calls per show 8. Fresh lines were not re-run with v2.

## Safety filter: second fresh set, prompt v2 (2026-11-08)

25 fresh lines written after the first score, scored once by the user with prompt v2 (user-scored, not re-runnable). Rules alone 13/25 (52%): appearance 4/5, bodies 2/5, named 1/5, protected 2/5, teens alcohol 4/5.
**Rules + LLM 24/25 (96%, Wilson 95% CI about 80% to 99%):** appearance 5/5, bodies 5/5, named 5/5, protected 5/5, teens alcohol 4/5. The 75% bar is met, including at the low end of the CI.
**Not comparable with set 1** (19/25 with v1): this set was written with an instruction that each line must clearly break its rule, so it is easier, and most lines single out a person, which v2's new rule targets. The gain mixes a better filter with clearer lines.
**Five `teens_sexual` lines added (2026-11-14),** scored once with v2 together with the 25, no provider errors or retries: rules 5/5. Set 2 totals **rules alone 18/30 (60%), rules + LLM 29/30 (97%, Wilson 95% CI about 83% to 99%)**; the 25 scored the same as before (24/25).
The one miss is an indirect drinking hint at `age: teens`, which the user judged mild and acceptable. For the record, prompt v1 on the same 30 caught 28/30: it also missed a line teasing a religious group seated in the room, which v2's singling-out rule catches.

## Safety filter: rerun on Claude Haiku 5.5 (Exp 19, 2026-11-25)

`SAFETY_CHECK_PROVIDER=anthropic SAFETY_CHECK_MODEL=claude-haiku-5-5 python -m sim.evaluate_safety --env-file backend/.env`. Raw: `reports/metrics/safety_filter_anthropic.json` (keyed by provider and model; the Gemini result in `safety_filter.json` is untouched). Same fixtures, same prompt `safety-llm-v2`, same bars. Decision: D-98.
**Held-out hard-rule lines: rules + LLM 71/71** (Wilson 95% CI 94.9% to 100%); rules alone 70/71 (`n18`, named_people). Tuning 81/81.
**False rejects (rules + LLM):** clean list 2/60 adults (3.3%), 3/60 teens (5.0%); real bank 5/120 adults (4.2%), 0/120 teens. Bank rejects `neurotic-1-3`, `absurdist-0-1` to `-4`. Both under the 10% bar as point estimates; the clean list's CI upper ends are 11.4% and 13.7%.
**Latency per 15-line call, 25 calls, 0 retries, 0 failed: p50 3.09 s, p95 4.94 s. The 3 s bar is missed** (Gemini: 1.49 s and 1.91 s).
**Cost:** 35,805 input and 15,519 output tokens, $0.011 for the run.
**Verdict:** not adopted; the filter's default stays on Gemini. One run, so p95 is rough; the prompt was not re-tuned.
**Bug found on the way:** the cache key ignored provider and model (dev-log 2026-11-25); it now includes both.

**Thinking off (D-99):** `--no-thinking`, raw `reports/metrics/safety_filter_anthropic_no_thinking.json`. Held-out 71/71; false rejects clean 2/60 and 1/60, bank 5/120 and 1/120; **p50 3.12 s, p95 3.61 s, bar missed.** Output tokens 14,356 (thinking on: 15,519), so thinking was not the cost; the reply's per-line reasons are. $0.011. Not adopted.

**Adopted under a re-tuned bar (D-100):** the user moved the latency bar to p95 at most 7 s (target 5 s) after these runs, to get off Gemini's free tier. Haiku with thinking off meets it (3.61 s) and is now the default. The bar moved after the data, so these runs show fit, not a test.

## Generated show timing (Exp 20, 2026-11-14)

`python -m sim.evaluate_live --env-file backend/.env --quota-rpd 500 --quota-rpm 15` (notebook Exp 20, task T9). Raw: `reports/metrics/live_generator.json`, keyed by the model, the quota inputs and a digest of the prompt (`gen-v3`), bridge lines and timing constants. Three whole shows of 8 bits, `SimRoom` (ages mixed, teens, mixed), real Kokoro on the CPU behind the shared `SerialTTS` lock, the real model (`gemini-flash-lite-latest`), real time. Dead air is read from the session log (schema v2: `joke_played`, `bridge` and `gap` rows). Bars fixed before the first run: within-bit dead air p95 at most 1 s; no between-bit silence over 2 s; the API requests for 5 shows fit the free quota.

| | n | p50 | p95 | max | bar | |
|---|---|---|---|---|---|---|
| Within-bit dead air | 96 | 0.001 s | 0.002 s | 0.003 s | p95 at most 1 s | met |
| Between-bit dead air | 54 | n/a | 0.003 s | 0.004 s | max at most 2 s | met |

**Requests:** 16, 17 and 17 per show, so about 83 for 5 shows, peak 4 in any minute, against **an assumed quota of 500 a day and 15 a minute**: met. The assumption is unconfirmed (about 140 requests went through this key in one day with no daily-limit error, so the 20-a-day forum reports do not apply to it). 2 requests were retried (filter calls in shows 2 and 3) and 0 calls failed after their retries.

**Bank share per show: 0%, 0% and 0%.** No joke came from the bank in this run, and no joke was repeated within a show (`repeated_lines` 0 in all three), so the order-based pairing in `silences()` did not matter for these numbers. The previous run, on the same prompt, had 12.5%, 12.5% and 0% (one filter call failed past its deadline, one reply came back as a flat list), so the bank share is a property of provider luck at this sample size, not a fixed rate; with 3 shows it has no real interval.

**Bridge lines:** 11 per show for 7 bit boundaries, from a pool of ten that does not repeat inside a show while an unused line remains. First line after the show starts: 6.4, 5.5 and 7.2 s (not part of any bar).

**Run history, so the numbers can be trusted.** (1) The first run reported a within-bit p95 of 1.51 s and a between-bit max of 3.74 s: `silences()` dropped a hold whose log timestamp equalled the next play's start (a coarse Windows clock) and counted it as silence (dev-log 2026-11-14). (2) After that fix, between-bit max was 3.36 s (p95 2.58 s): a bit counted as ready when its text was ready, but the first line's render started only then; `_prepare` now waits for the first line's audio. (3) A run on the thread prompt with a joke-major parser was stopped after 80 jokes fell back to the bank, because the model answered with thread-major JSON (dev-log 2026-11-14). (4) The run before this one, after the review changes and the shape fix, met all three bars with a 8.3% mean bank share. (5) This run came after the second review round (voice failures no longer abort, one candidates row per joke, marginal `pick_prob`, shared engine lock, per-chunk deadline, `bridge` rows in the log and dead air read from them); its timing code and the counting of bank jokes both changed, so it was rerun rather than carried over.
**Caveats:** three shows is a small sample; the sim room has no operator latency and its holds are the config table's; provider load at the time is unknown, and Exp 19's 29% 429/503 rate did not repeat at show pace (2 of about 50 requests here); at most 3 bridge lines play per boundary, and a bit still loading after that would leave dead air, which this sample did not hit. A voice failure on a generated line was not triggered in this run; it is covered by the tests in `tests/test_generate.py`, not by Exp 20.

**Rerun with the new safety budget (2026-11-15, after Exp 22).** Filter 12 s per attempt, 25 s deadline, 3 retries, one re-ask on a bad reply; `MAX_BRIDGES` 5 ([decisions-detail.md D-88](agent/decisions-detail.md#d-88), "The safety check gets a budget sized to a stall"). Same three bars, same command with `--rerun`; the cache key changed with the constants, so this replaces the run above.

| | n | p50 | p95 | max | bar | |
|---|---|---|---|---|---|---|
| Within-bit dead air | 96 | 0.001 s | 0.002 s | 0.024 s | p95 at most 1 s | met |
| Between-bit dead air | 57 | n/a | 0.003 s | 0.007 s | max at most 2 s | met |

Requests 21, 16 and 19 per show (about 93 for 5 shows, peak 4 a minute, against the assumed 500 a day and 15 a minute): met. **Bank share per show: 12.5%, 0% and 0%.** The 12.5% (5 jokes, one bit) was a generator call that failed (`RemoteProtocolError`, wall time 361.7 s, probably a suspended laptop: dev-log 2026-11-15), not the filter. **Filter: 0 calls failed after retries, 6 retried (4 in show 1, 2 in show 3), 0 malformed replies;** the slowest filter call was 5.9 s including its retries. Bridge lines 13, 11 and 12. First line after the start 11.0, 10.5 and 6.5 s (not barred; the first filter call took 5.9 and 5.1 s in shows 1 and 2, a slow cold call). **Caveat:** the provider was mostly healthy, so this shows the longer budget does not hurt the timing, not that stalls are now handled; the stalls and bad replies are covered by synthetic tests (`tests/test_safety.py`), and the real evidence of the failure is still Exp 22's one 75% show.

**Rerun after review (2026-11-24).** Review of #58 changed two things: `MAX_BRIDGES` 5 to 8, because a bit's wait is generation (12 s deadline) plus the filter (25 s), about 37 s, not the filter's 25 s alone; and a partial verdict reply is kept when the re-ask fails, so the lines it judged still pass. The cache key includes `MAX_BRIDGES`, so Exp 20 was rerun; this replaces the run above.

| | n | p95 | max | bar | |
|---|---|---|---|---|---|
| Within-bit dead air | 96 | 0.003 s | 0.017 s | p95 at most 1 s | met |
| Between-bit dead air | 57 | 0.013 s | 0.033 s | max at most 2 s | met |

Requests 16, 23 and 15 per show (90 for 5 shows, peak 4 a minute): met. **Bank share 0%, 0% and 12.5%;** the 12.5% was the joke writer returning malformed JSON for one bit, not the filter. Filter: 0 failed, 3 retried, 0 bad replies, slowest call 5.3 s. First line after the start 12.0, 6.9 and 7.4 s (not barred). Same caveat: a healthy provider, so the new cap was not exercised; `tests/test_generate.py` checks that 8 lines at about 5 s cover the two deadlines.

## Click alignment (Exp 21, 2026-11-15)

`python -m sim.evaluate_clock check data/sessions/8cdeae59a177.jsonl clicks.json` (notebook Exp 21, task T10). Raw: `reports/metrics/audio_clock.json`, keyed by the session id and a digest of the clicks file; numbers only, the recording and clicks stay outside the repo. One simulated show (`GENERATOR=bank`), laptop speakers and microphone, Audio enhancements off. `play` beeped 35 times (2 kHz, 100 ms, every 10 s); 30 fell inside the recording. Beeps are found with a matched filter for the tone and a threshold of 8x the recording's noise.

| Beeps matched (in recording) | Outside the recording | Median offset | p95 spread around it | Drift over the show | Spread <= 100 ms | Drift <= 50 ms | >= 90% matched |
|---|---|---|---|---|---|---|---|
| 30/30 | 5 | +241 ms | 20 ms | -2 ms | yes | yes | yes |

Largest spread 28 ms; every beep at least 23x the noise (median 31x); drift slope -0.008 ms/s over 294 s. **Old bar:** p95 |offset| 261 ms against 100 ms, **missed.** The bar was re-tuned by the user after this result (decisions-detail.md 2026-11-15): the constant (+241 ms, which includes PlaySound's output latency) is reported, not barred. **Caveats:** one laptop, one run, 30 beeps; the beeps and the recording share a machine, so this cannot say how large the robot's own voice offset is; with the microphone's Audio enhancements on, no beeps reached the WAV at all.

## Acceptance run (Exp 22, 2026-11-15)

The Stage 0 exit (T11). Three simulated shows back to back, then one live show with the user pressing a key for every joke and the room recorded with consent. All on Gemini (`GENERATOR=llm`), laptop mic and speakers, no Bluetooth, mic enhancements off. Scored with `python -m sim.acceptance ... --record` (`reports/metrics/acceptance.json`, numbers and session ids only).

| Show | Room | Rows valid | Status | Jokes | Bank | Bridges | W / K / Space / D / none | Rating | Audio |
|---|---|---|---|---|---|---|---|---|---|
| 08c5b28d5a81 | sim | yes | finished | 40 | 0% | 10 | 0 / 3 / 15 / 22 / 0 | 4 | not recorded |
| 64422d8243ee | sim | yes | finished | 40 | 12% | 10 | 3 / 6 / 11 / 20 / 0 | 2 | not recorded |
| a086dcc40da8 | sim | yes | finished | 40 | **75%** | 21 | 8 / 11 / 11 / 10 / 0 | 2 | not recorded |
| 43251e36e02d | live | yes | finished | 40 | 0% | 12 | 5 / 11 / 20 / 4 / 0 | 3 | 100% |

- (a) 3 simulated shows in a row, all clean: **PASS**. (b) a live show, valid and finished: **PASS**.
- (c) label spread, one key over 60% of a live show's presses: **none flagged** (Space 50%, K 28%, W 12%, D 10%).
- (d) real per-bit reward noise **0.121** (95% CI 0.092 to 0.151) from 8 bits of one live show, against the sim's 0.15 (measured 0.136); the band is 0.075 to 0.225, so **Exp 18 does not need a rerun**. One show only: the interval is wide.
- **Finding: the third simulated show took 75% of its jokes from the bank.** The generator answered, but the LLM safety check timed out (73 rejected candidates with `llm:error:LLMError:transport: ReadTimeout`, over 5 bits) and one verdict reply was malformed JSON (15 candidates). The filter fails closed, so those bits were rejected whole and the hand-checked bank filled them. The show itself held: no dead air, 21 bridge lines covered the waits. Cost: a show that is three quarters bank yields few Gemini preference pairs. The filter's budget (6 s per attempt, 8 s total) is far below the time a bit has to prepare while the previous one plays (about 40 s); a follow-up PR lengthens it and retries malformed replies, measured on the bank share.
- Minor: the rules' name check flagged "Morse", "Yelp" and "LEDs" as people (false positives; each costs one candidate).
- Ratings: simulated shows 4, 2, 2 (the user's taste of a simulated room); live show 3.

## Room labels for the judge's pairs (Exp 25, 2026-11-24)

`python -m explorer.label_state --spread` on Explorer after job 12123581 (Qwen2.5-7B-Instruct, greedy, V100, 8.8 jokes/s, 2.8 h); raw counts in `reports/metrics/rjokes_state.json`, the labeled files stay on the cluster. 89,924 of 90,000 jokes have both a mode and a room label (76 left out: 31 room-label parse failures, 45 earlier mode failures). Check fixed before the run (D-89): a field whose top value covers more than 80% says little.

| Field | Top value | Share | Others |
|---|---|---|---|
| venue | showcase | 94.6% | bar 4.8%, company_event 0.6% |
| energy | high | 84.6% | normal 14.9%, low 0.4% |
| age | young_adults | 95.9% | mixed 4.1%, teens 1 joke |

**All three fail the check.** The labels are the model's guess, and part of the skew is the prompt (showcase was described as an audience for clever or odd material, which fits most Reddit jokes). Teens = 1 is expected for Reddit's adult material. Chart: `reports/figures/decision-plot/exp25-room-labels.png`.


## Generator models (Exp 24, 2026-11-25)

`python -m sim.compare_generators score`; raw numbers in `reports/metrics/model_comparison.json` (no joke text). 60 prompts per model, one attempt each; the user ranked 30 blind. Decision: [D-107](agent/decisions-detail.md#d-107).

| Model | Attempts | Failed (95% upper) | p50 | p95 | Line pass | Playable | Mean rank (95% CI) | Cost, 5 shows | GPU | Eligible |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-4b-instruct | 60 | 0 (6%) | 6.7 s | 8.2 s | 87% | 87% | 2.17 (1.90 to 2.43) | $0.000 | 100% | yes |
| bonsai-2-4b | 60 | 1 (9%) | 5.2 s | 6.9 s | 89% | 92% | 2.17 (1.87 to 2.43) | $0.000 | 100% | yes |
| claude-haiku-5-5 | 60 | 2 (11%) | 5.6 s | 7.4 s | 91% | 95% | 1.67 (1.40 to 1.97) | $0.017 | - | yes |

**Pick: bonsai-2-4b.** All three pass the speed and failure bars. Funniness is not clear: Haiku leads bonsai by 0.50 ranks, but the 95% interval is 0.00 to 1.00, so it touches 0. The rule then takes the cheaper, then faster, model. Haiku won first place 16 times of 30, bonsai 8, qwen 6. Lower rank is funnier.

**Caveats.** One rater, 30 prompts. Local speeds were measured without the voice, so they are a best case. Haiku is the funnier reference that SFT and DPO must close in on.

Chart: `reports/figures/decision-plot/exp24-generator-pick.png`.

## Per-thread timing (Exp 26, 2026-11-25)

`python -m sim.thread_timing`; raw numbers in `reports/metrics/thread_timing.json` (no joke text). 20 bits on the show's own writer (bonsai-2:4b, 3 parallel calls per bit, `OLLAMA_NUM_PARALLEL=3`), no filter, no voice. Decision: [D-110](agent/decisions-detail.md#d-110).

| Check | Result | Bar |
|---|---|---|
| Bits that lost a thread | 0 of 20 | at most 1 |
| Bits with no thread | 0 | 0 |
| Writer 100% on the GPU at context 8192 after the run | yes | yes |
| Bit wall time p50 / p95 | 3.53 s / 5.12 s | beside the bar |
| Call p50 / p95 (Exp 24's one-call p50: 5.16 s) | 3.17 s / 4.77 s | beside the bar |
| Bits that queued; median slowest/fastest ratio | 1 of 20; 1.16 | beside the bar |
| Bits with two threads on one premise; bits re-asked | 0; 0 | beside the bar |

**Passed.** Three calls at once fit the bit's 12 s budget with room. **Caveat:** no bit sent a bad shape, so the re-ask path and the counted schema were not exercised; whether Ollama enforces the count is still unmeasured.

## Connected threads (Exp 27, 2026-11-25)

`python -m sim.thread_coherence run`, `review`, `score`; raw numbers in `reports/metrics/thread_coherence.json` (numbers only). 20 prompts, gen-v4 (A) against gen-v5 (B), Claude Haiku 5.5 as the judge, then the user checked 10 threads blind. Decision: [D-111](agent/decisions-detail.md#d-111).

| Check | Result | Bar |
|---|---|---|
| Mean paired difference B - A (15 prompts) | +0.222, CI +0.133 to +0.333 | CI lower bound above 0: met |
| Judge and user disagree on | 6 of 10 | under 4: failed |
| Verdict | inconclusive_judge | |

Judge against user: judge "connected" and user "one bit" 2, judge "connected" and user "separate jokes" 3, judge "not connected" and user "one bit" 3, both "not connected" 2. Agreement 4 of 10 (95% Wilson interval about 17% to 69%): too low to trust the judge, not proof it is exactly chance.

**Read it:** the judge's +0.222 cannot be trusted, so gen-v4 stays live and gen-v5 is not adopted. **Caveat:** 5 of 20 prompts dropped out because a cluster of read timeouts (9 threads in each arm) hit both arms in a slow stretch. Judge-called connected threads: 34 of 51 for A, 45 of 51 for B.

**Update 2026-11-26 (judge v2, five timed-out bits rerun).** Raw: `thread_coherence_rescore.json`. The v1 lines are superseded by judge v2 (D-114): its spot-check is no longer asked for. The original 6-of-10 `inconclusive_judge` record is at commit 079b26b.

| Judge | Prompts | Mean B - A | 95% CI | Bar |
|---|---|---|---|---|
| v1 (untrusted, superseded) | 20 | +0.200 | +0.117 to +0.283 | met, not trusted |
| v2 | 15 (before the rerun) | +0.133 | -0.022 to +0.289 | failed |
| v2 | 20 | +0.133 | 0.0 to +0.25 | failed at exactly 0 |

Delivered and lost: A 57 and 3, B 60 and 0 (one-sided); bit wall p95 A 12.04 s, B 9.68 s; 0 bad replies. **Read it:** too close to call, so Exp 27 grew to 100 prompts (c00 to c99, no new labels). **Only the 80 new prompts (c20 to c99) decide.** Raw: `thread_coherence_n100.json` and `thread_coherence_rescore_n100.json`.

| Judge v2, connected-line rate gain (new prompt minus old) | Mean | Likely range (95%) | Prompts | Role |
|---|---|---|---|---|
| The 80 new prompts | +0.206 | +0.117 to +0.298 | 80 | **decides: passed** (range above 0) |
| All 100 | +0.192 | +0.115 to +0.268 | 100 | beside it, never decides |

New prompt beat the old on 41 of the 80, lost on 16. Connected threads on the 80: old prompt 51.5%, new prompt 72.1%. Writing (all 100): old prompt delivered 296 threads and lost 4 (timeouts), new delivered 300 and lost 0; typical bit time 5.22 s (old) and 4.59 s (new), slowest tenth about 8.45 s and 6.90 s; the new prompt is not slower. Power of this bar was about 67 to 73% at a true gain of 0.1 and 88 to 91% at 0.13, with a 2 to 4% false alarm (made-up inputs). **Outcome: gen-v5 is adopted** and live; every adoption condition held (D-114). **Caveats:** the judge agrees with the user 75% of the time (15 of 20), the user's own labels showed 70% one-bit for the new prompt against 40% for the old (only 10 labels per prompt), the first 20 prompts were a fourth look so only the 80 decide, and funniness is not measured. Exp 26's timing record is now stale (it timed the old prompt); rerun optional. Decision: [D-111 addendum](agent/decisions-detail.md#d-111).

## Judge v2 on a fresh sample (Exp 30, 2026-11-25)

`python -m sim.judge_calibration run`, `review`, `score`; raw numbers in `reports/metrics/judge_calibration.json` (counts only). Decision: [D-114](agent/decisions-detail.md#d-114). Judge v1 and judge v2 labelled 20 fresh threads, 10 per arm, next to the user's blind labels. Rule fixed before the run: keep v2 if it agrees with the user more than v1 does (a tie keeps v1); the kept judge re-scores Exp 27 only if it disagrees with the user on under 40%.

| Check | Result | Rule |
|---|---|---|
| v1 agrees with the user | 14 of 20 (70%) | |
| v2 agrees with the user | 15 of 20 (75%, interval 53% to 89%) | keep v2 if higher than v1: met |
| v2 minus v1 | +0.05, CI -0.11 to +0.21 | beside the rule |
| Kappa, v1 / v2 | 0.35 / 0.48 | beside the rule |
| Kept judge's disagreement | 25% | under 40% to re-score: met |
| User called one bit: A / B | 40% / 70% | beside the rule |
| Verdict | keep_v2_trusted | |

**Read it:** v2 is kept and may re-score Exp 27, but 20 labels cannot tell it from v1 (the interval on the difference spans 0). v2's B - A (+0.20) sits inside the user's own interval, so it does not call B better than the user's labels allow.

## Bridge lines the listener has heard (Exp 28, 2026-11-25)

`python -m sim.evaluate_heard --record`, from the 25 session logs on this laptop (no model call). Raw: `reports/metrics/heard_bridges.json` (counts and session ids only). Decision: [D-112](agent/decisions-detail.md#d-112).

| Show | Bridge lines | Cached, heard before | Filler, heard before |
|---|---|---|---|
| `d3ef092dfafe` (sim) | 18 | 17 of 17 | 1 of 1 |
| `e9a42ccc6b36` (sim) | 5 | 5 of 5 | 0 of 0 |
| `e204ec80e62e` (sim) | 18 | 17 of 17 | 1 of 1 |

**Read it:** Exp 28 sizes the problem; it is not a test of the rule. "100% of cached lines heard" is true by construction (a cached line is a logged show's own played text), and "0 heard lines in the rule's pool" is the filter checked against itself; the tests on synthetic logs check the rule. The informative numbers: the cached pool is 17 lines without the rule and 0 with it, and **all 10 filler lines have been heard** (4 earlier shows played them; the live show alone played 12), so the fallback repeats across shows. The dev-log's "11 bridge lines" was 18 rows in the logs.

**Caveat:** it counts what was played, not what was noticed, and it matches on lowercase text with spaces collapsed, so a reworded line counts as new.

## Where a bit's wait goes (Exp 29, 2026-11-25)

Run on every show on the pre-guess code (c474e51, #81) that logged `prep` rows, all sim room, gen-v4 bonsai-2:4b: `0e102d77d795`, `c514a01f04f7`, `9fedcfa07c3f` (finished) and `4ee0e5658e25` (aborted after bit 2). The rule never excluded aborted shows, so it is applied as written. Seconds from the local logs, no model call. Command: `python -m sim.stage_timing 0e102d77d795 c514a01f04f7 9fedcfa07c3f 4ee0e5658e25 --record`. Raw: `reports/metrics/stage_timing.json`. Decision: [D-113](agent/decisions-detail.md#d-113).

| Bits 1+ counted | Wait p50 / p95 (s) | Generate | Render | Safety | Bridge overhang | Render queue, bank, other |
|---|---|---|---|---|---|---|
| 23 (7, 7, 7 and 2 per show, none dropped) | 17.0 / 26.7 | 34% | 28% | 22% | 17% | 0% |

**Verdict: inconclusive.** Generate leads (34%) with render second (28%); the 95% interval of `share - 1.1 x runner-up` is -0.13 to +0.12, across 0, so no single stage leads clearly. **No fix is built under either verdict.** On the 3 finished shows alone the writer led at 36% (interval +0.01 to +0.12, "no candidate fix"); the aborted show is voice-led (render 47% of its wait over 2 bits) and flips it. The first 3-show reading was a selection made after seeing the data (flagged in review), not the result. Bit 0 alone (4 bits): wait p50 10.7 s. Stage times depend on the writer, the filter and the voice, not on `audience.yaml`. **D-115 has since removed most gaps**, so this matters mainly for the gaps after a wrong guess.

## Next bit: decide at joke 3, or guess it (Exp 31, 2026-11-26)

Decision: [D-115](agent/decisions-detail.md#d-115). Sim only, no model calls. Raw: `reports/metrics/early_decision.json` (a) and `reports/metrics/early_guess.json` (b). Code: `sim/evaluate_early.py`, `sim/evaluate_predict.py`.

**(a) Deciding at joke 3 costs laughs, so it is not shipped.** The director picks the next bit after joke 3 of the current one, on a partial reward, and is updated with that bit only at its end, so every pick is one update behind. Paired on seeds against deciding at the bit's end, laughs per show at 5 shows (1,000 seeds). Bar fixed before running: far end of the 95% CI no lower than -1.0 laugh per show in every room. It is missed in 8 of 10 rooms.

| Room | Decide at bit end | After joke 3: change (paired 95% CI) | Far end | After joke 5 (whole bit, still one update behind) | Bar (far end at least -1.0) |
|---|---|---|---|---|---|
| judge, gaps x0.5 | 41.1 | -1.33 ± 0.11 | -1.43 | -1.32 ± 0.11 | FAIL |
| judge, gaps x1 | 42.4 | -1.34 ± 0.11 | -1.45 | -1.39 ± 0.11 | FAIL |
| judge, gaps x2 | 46.6 | -1.28 ± 0.13 | -1.41 | -1.31 ± 0.13 | FAIL |
| human, gaps x0.5 | 40.6 | -1.28 ± 0.10 | -1.38 | -1.29 ± 0.10 | FAIL |
| human, gaps x1 | 40.6 | -1.26 ± 0.10 | -1.36 | -1.24 ± 0.10 | FAIL |
| human, gaps x2 | 40.7 | -1.26 ± 0.10 | -1.35 | -1.27 ± 0.10 | FAIL |
| judge, noise x0.5 | 40.5 | -1.94 ± 0.11 | -2.05 | -1.96 ± 0.11 | FAIL |
| judge, noise x2 | 47.7 | -0.82 ± 0.10 | -0.92 | -0.84 ± 0.10 | pass |
| judge, fatigue 0 | 44.8 | -0.86 ± 0.13 | -0.99 | -0.88 ± 0.13 | pass |
| judge, fatigue 0.3 | 40.6 | -1.98 ± 0.12 | -2.10 | -1.99 ± 0.11 | FAIL |

**Read it:** the loss is the same whether the decision sees 1, 3 or all 5 jokes (after joke 5 is as bad as after joke 3), so the cost is the one-update delay, not the partial reward. It grows with less noise and with more fatigue. It is about 3% of a show. The 30-show rows are in the JSON and did not decide. Under a looser bar of -0.02 reward per bit (2.4 laughs per show) every room would pass; the bar was not changed after the result.

**(b) Guessing instead: decide as before, guess ahead.** After joke 3 the show asks a throwaway copy of the director, taught the reactions so far, which mode it would pick, and writes that bit while jokes 4 and 5 play. The real decision, the director's random draws and the rewards are untouched (tests compare the whole log with the guess on and off). Only the hit rate matters. Bar fixed before running: lower end of the 95% CI at least 0.5 in every room at 5 shows (1,000 seeds). The guess shipped is the better of `learn` (the copy is taught the partial reward first) and `now` (the model as it is) in "judge, gaps x1".

| Room | Hit rate, `learn` (95% CI) | Lower end | Guess = greedy mode | `now` hit rate | Seconds saved per show (estimate) | Bar (lower end at least 0.5) |
|---|---|---|---|---|---|---|
| judge, gaps x0.5 | 0.715 ± 0.005 | 0.710 | 0.85 | 0.47 | 50 | pass |
| judge, gaps x1 | 0.716 ± 0.005 | 0.712 | 0.85 | 0.48 | 50 | pass |
| judge, gaps x2 | 0.726 ± 0.005 | 0.722 | 0.87 | 0.52 | 51 | pass |
| human, gaps x0.5 | 0.711 ± 0.005 | 0.706 | 0.85 | 0.46 | 50 | pass |
| human, gaps x1 | 0.711 ± 0.005 | 0.707 | 0.85 | 0.46 | 50 | pass |
| human, gaps x2 | 0.711 ± 0.005 | 0.707 | 0.85 | 0.46 | 50 | pass |
| judge, noise x0.5 | 0.748 ± 0.004 | 0.743 | 0.89 | 0.48 | 52 | pass |
| judge, noise x2 | 0.697 ± 0.005 | 0.692 | 0.83 | 0.47 | 49 | pass |
| judge, fatigue 0 | 0.715 ± 0.005 | 0.711 | 0.85 | 0.49 | 50 | pass |
| judge, fatigue 0.3 | 0.721 ± 0.004 | 0.716 | 0.86 | 0.48 | 50 | pass |

**Read it:** the guess is right about 7 times in 10 in every room; the ceiling is about 0.83, because exploring (eps = 0.2) picks a random mode no guess can foresee, and the guess equals the greedy mode 83 to 89% of the time. The bar is met everywhere. Seconds saved per show = hit rate x about 10 s of prep (show `0e102d77d795`) x 7 gaps: an **estimate**, not measured live. A miss throws the prep away and costs one local writer call plus, if it got that far, one safety prefetch (cents); a thrown-away writer call may also queue ahead of the real bit's on Ollama, so a miss can be slower than no guess (not measured). The live hit rate can be lower: the sim always has 3 labels at joke 3, and `python -m sim.stage_timing` reports the live rate. **Live check, not run by the agent:** after your next shows run `python -m sim.stage_timing <ids>` and compare each show's wait p50/p95 and bridges per bit with `6c09c721b0bb` and `0e102d77d795`.

**What the sim depends on:** the made-up `audience.yaml` values `noise`, `fatigue` and the gaps in `base` (all varied above), and the eps = 0.2 explore rate, the uniform prior and the 5-show horizon. Real rooms with sharper preferences than these would make the guess right more often.

**Live check (2026-11-26).** Show selection: the 2 shows named in D-115's plan plus every finished show on c474e51 (no guess) or c188c88 (guess), all sim room, gen-v4 bonsai-2:4b: 4 shows without the guess and 4 with it (`5ea47db7b248`, `a7290b75cf24`, `2ab2d1c2c682`, `94aef761a5e6`; the fourth was added to balance the sets). `6c09c721b0bb` is from before D-112, so its bridge lines are cached lines, not filler, and its count includes the bit-0 one; `e204ec80e62e` (gen-v4, #78, before the #79 schema) is left out as a different code version. Command: `python -m sim.evaluate_gaps --before ... --after ... --record`. Raw: `reports/metrics/live_gaps.json`. **These are different shows, not paired seeds, and n is small.** The made-up sim audience reacted; waits start at `bit_end` (0.3 to 2.5 s after the last joke's audio in both sets) and depend on `audience.yaml` only through when reactions arrive. No bar for the waits; ratings are beside, not a claim.

| Show | Guess | Wait p50 / p95 (s), bits 1+ | Total wait (s) | Zero-wait gaps of 7 | Bridge lines | Guesses right / made | Safety wasted | Rating |
|---|---|---|---|---|---|---|---|---|
| 6c09c721b0bb | no | 12.9 / 17.2 | 88.6 | 0 | 16 | - | - | 2 |
| 0e102d77d795 | no | 13.7 / 17.1 | 99.8 | 0 | 16 | - | - | 1 |
| c514a01f04f7 | no | 17.4 / 26.7 | 123.9 | 0 | 19 | - | - | 2 |
| 9fedcfa07c3f | no | 18.3 / 27.4 | 122.6 | 0 | 19 | - | - | none |
| **before, all (28 gaps)** | | **13.7 / 26.7** | **108.7 per show** | 0 | 70 (17.5 per show) | | | |
| 5ea47db7b248 | yes | 0.0 / 18.3 | 24.5 | 5 | 4 | 6 / 7 | 1 | 3 |
| a7290b75cf24 | yes | 0.0 / 12.7 | 12.7 | 6 | 2 | 6 / 7 | 1 | 3 |
| 2ab2d1c2c682 | yes | 0.0 / 18.7 | 37.1 | 4 | 6 | 5 / 7 | 2 | 2 |
| 94aef761a5e6 | yes | 0.0 / 12.1 | 12.1 | 6 | 2 | 6 / 7 | 1 | 2 |
| **after, all (28 gaps)** | | **0.0 / 18.3** | **21.6 per show** | 21 | 14 (3.5 per show) | 23 / 28 | 5 | |

**Read it:**
- **Against Exp 31(b)'s bar** (hit-rate 95% interval lower end at least 0.5): 23 of 28 right (0.82), Wilson 95% interval 0.64 to 0.92, so the lower end 0.64 clears it. The interval is optimistic: guesses within a show are not independent. All 28 were on 3 labels; 23 used (Exp 31's sim: about 0.7).
- **Waits:** the median fell from 13.7 s to 0.0 s and bridge lines from 17.5 to 3.5 per show. The 7 gaps that still waited took 6.1 to 18.7 s: 5 follow a wrong guess (12.1 to 18.7 s, the whole prep back in the wait) and 2 a right guess whose prep was not ready (6.1 and 6.2 s). The p95 stays high because a wrong guess waits as long as before; a miss being slower than no guess is not visible (those 5 waits sit inside the before range of 9.8 to 27.4 s).
- **The number D-115 estimated** (total wait over bits 1+ per show): 108.7 s before, 21.6 s after, about 87 s saved per show against the ~50 s estimate. Different shows, so an indication.

## Show ledger: how each setup did (Exp 32, 2026-11-26)

Decision: [D-116](agent/decisions-detail.md#d-116). Raw: `reports/metrics/show_ledger.json`. Code: `sim/show_ledger.py`. Chart: `reports/figures/decision-plot/exp32-show-ledger.png`. The played lines: `reports/show_bits/`.

**The plain answer.** Your ratings have stayed between 1 and 4 in every setup, and there are too few ratings (1 to 4 per setup) to tell any setup from another. What clearly changed is the wait: with the next bit prepared early, the total wait per show fell from 115 s to 22 s. (D-115's own live check gives 108.7 s to 21.6 s because it counted a different set of shows as "before": it included show `6c09c721b0bb`, which the ledger files under the setup before D-112, since that show still replayed heard lines.)

36 logged shows (15 from the hand bank, 21 written by a model), all readable; 18 finished, 18 aborted (12 of the aborted are bank test runs). Only 1 live show finished (rated 3). Sim-room reactions are the made-up audience's, so they are not counted as quality. Dates are local time. Setups in the order they started:

| Setup | What changed (and the PR its first show ran on) | Finished shows | Your ratings | Mean | Lines from the bank | Threads lost per show | Typical wait (average of per-show medians, s) | Total wait per show (s) | Guess right |
|---|---|---|---|---|---|---|---|---|---|
| Gemini writer, 3 threads in one reply (gen-v3); next bit prepared after; no cached lines | T9 LLM writer (#55 feat/t11-rating-acceptance) | 4 | 2, 2, 3, 4 | 2.75 | 22% | 0.0 | 11.3 | 84 | - |
| local (bonsai) writer, 3 threads in one reply (gen-v3); next bit prepared after; may replay heard lines | D-109 shows write with bonsai + D-91 cached bridge lines (#77 feat/bonsai-writer) | 1 | 1 | 1.00 | 28% | 0.0 | 13.2 | 79 | - |
| local (bonsai) writer, one call per thread, 3 in parallel, before the reply-format fixes (gen-v4 before D-111 fixes); next bit prepared after; may replay heard lines | D-110 one call per thread (#78 feat/per-thread-generator) | 1 | 3 | 3.00 | 0% | 3.0 | 12.8 | 95 | - |
| local (bonsai) writer, one call per thread, 3 in parallel (gen-v4); next bit prepared after; may replay heard lines | D-111 reply-format fixes (#79 feat/connected-threads) | 1 | 2 | 2.00 | 0% | 0.0 | 12.9 | 89 | - |
| local (bonsai) writer, one call per thread, 3 in parallel (gen-v4); next bit prepared after; skips heard lines | D-112 skips heard lines (#81 feat/stage-timing) | 3 | 1, 2 (1 unrated) | 1.50 | 1% | 0.7 | 16.5 | 115 | - |
| local (bonsai) writer, one call per thread, 3 in parallel (gen-v4); next bit prepared early; skips heard lines | D-115 next bit prepared early (#83 feat/early-prep) | 4 | 2, 2, 3, 3 | 2.50 | 0% | 0.0 | 0.0 | 22 | 23 of 28 |
| local (bonsai) writer, connected-bit prompt, each line builds on the last (gen-v5); next bit prepared early; skips heard lines | D-111 addendum: connected-bit prompt live (#84 feat/exp27-extension) | 1 | 4 | 4.00 | 5% | 2.0 | 0.0 | 20 | 6 of 7 |

**Code names.** The names in brackets in the table (and in `reports/show_bits/`) mean:

| Code name | What it is | Decision | PR |
|---|---|---|---|
| bank-v1 | hand-written bank only | D-66 | #40 |
| gen-v3 Gemini | Gemini writer, 3 threads in one reply | D-82 | #50 |
| gen-v3 bonsai | local (bonsai) writer, 3 threads in one reply | D-107, D-109 | #77 |
| gen-v4 before D-111 fixes | one call per thread, 3 in parallel, before the reply-format fixes | D-110 | #78 |
| gen-v4 | one call per thread, 3 in parallel, with the reply-format fixes | D-110, D-111 | #78, #79 |
| gen-v5 | connected-bit prompt, each line builds on the last | D-111 addendum | #84 |
| + guess | next bit prepared early | D-115 | #83 |
| + skip-heard | skips heard lines | D-112 | #80 |

The chart uses the code names only. "Before D-111 fixes" is the one-call-per-thread writer on code from before the counted reply format and the re-ask (a show on #78 code, which is why it is its own setup).

The hand-bank shows (3 finished, 12 aborted, no ratings) are in the ledger, not in the table. Lost threads were timeouts and badly shaped replies; the show on the connected-bit prompt (gen-v5) lost 2 of 3 threads in one bit to a timeout. One finished show, `9fedcfa07c3f`, was not rated.

**Noise.** Exact test between neighbouring rated setups; with 1 to 4 ratings a side none can reach the usual 5% line, so none says anything either way:
- gen-v3 Gemini to gen-v3 bonsai: n 4 and 1, mean -1.75: too few ratings to tell (a setup has fewer than 2)
- gen-v3 bonsai to gen-v4 before D-111 fixes: n 1 and 1, mean +2.00: too few ratings to tell (a setup has fewer than 2)
- gen-v4 before D-111 fixes to gen-v4: n 1 and 1, mean -1.00: too few ratings to tell (a setup has fewer than 2)
- gen-v4 to gen-v4 + skip-heard: n 1 and 2, mean -0.50: too few ratings to tell (a setup has fewer than 2)
- gen-v4 + skip-heard to gen-v4 + skip-heard + guess: n 2 and 4, mean +1.00: a random reshuffle of the same ratings gives a gap this big 27% of the time, and these sizes can never do better than 13%, so too few ratings to tell
- gen-v4 + skip-heard + guess to gen-v5 + skip-heard + guess: n 4 and 1, mean +1.50: too few ratings to tell (a setup has fewer than 2)

**What is made up.** Every sim-room reaction and reward (the audience in `sim/audience.yaml`). Waits depend on it only through when reactions arrive. The setups are different shows, not paired runs, and the rater is one person. Show `459d47bb396a` (aborted before its first bit) is filed under next-bit-guess because its code is the guess code.

**After each show:** `python -m sim.show_ledger --record`, `python -m sim.show_ledger export`, then `python scripts/run_notebook.py notebooks/decision-plot.ipynb`.

## Topics and robot facts for each bit (Exp 33, 2026-11-26)

Decision: [D-117](agent/decisions-detail.md#d-117). Raw: `reports/metrics/material.json`. Code: `sim/material.py`. Setup: a topic and two robot facts per bit (gen-v5 + material) against the live prompt alone (gen-v5), 5 shows of 8 bits each, same modes and rooms. No `audience.yaml` value is involved.

**Answer: keep it off.** Material cut the hardware jokes a lot (88% to 12% of bits), but you found the new bit funnier in only 6 of 20 pairs, and the bar was 10.

| Check | Result | Bar |
|---|---|---|
| Bits about the robot's hardware, with material (gen-v5 + material) | 12% (5 of 40) | at most 15%: met |
| Bits about the robot's hardware, without (gen-v5) | 88% (35 of 40) | beside the bar |
| Drop, bit by bit (paired, 40 slots) | -0.750; 95% interval -0.900 to -0.600 | interval entirely below 0: met |
| Same count by word list (cross-check) | 90% without, 42% with | beside the bar |
| Hardware share: robot-life topics / everyday topics (with material) | 20% (3 of 15) / 8% (2 of 25) | beside the bar |
| (a) Funnier: with material wins | 6 of 20 (without 9, ties 5) | at least 10: **not met** |
| (b) More like a specific character: with material wins | 10 of 20 (without 7, ties 3); 95% interval 30% to 70% | beside the bar |
| Verdict | keep_off | both bars |

**Read it:**
- The repeat problem is fixed in this test; the funniness bar is not met. 20 pairs cannot show the material bit is worse (6 against 9 is within coin-flip noise), only that the bar was missed, and the rule says a missed bar means off.
- (b) points nowhere: 10 against 7 with an interval of 30% to 70%.
- Robot-life topics kept more hardware jokes (20%) than everyday ones (8%), on 15 and 25 bits.
- The word list finds 42% with material where the judge finds 12%; the bar rests on the judge, which reads each bit's premise and lines.
- 1 bad reply with material, 0 without; time per bit is the same (median 4.3 s without, 4.4 s with). No topic was reused (40 distinct).
- Not tested: whether the facts or the topics cost the laughs (the run did not separate them).


## Topics only, no robot facts (Exp 34, 2026-11-26)

Decision: [D-118](agent/decisions-detail.md#d-118). Raw: `reports/metrics/material_topic.json`. Code: `sim/material.py --exp 34`. Setup: a topic per bit, no robot facts (gen-v5 + topic, arm C) against Exp 33's material-off bits (gen-v5, arm A), the same 40 slots. Arm A is read from Exp 33's stored bits, not rewritten. No `audience.yaml` value is involved.

**Answer: keep it off.** Topic only left 17.5% of bits on hardware (7 of 40), above the 15% bar. The funniness pairs were not judged: a missed first bar cannot be rescued by them.

| Check | Result | Bar |
|---|---|---|
| Bits about the robot's hardware, topic only (gen-v5 + topic) | 17.5% (7 of 40) | at most 15%: **not met** |
| Bits about the robot's hardware, without (gen-v5) | 87.5% (35 of 40) | beside the bar |
| Drop, bit by bit (paired, 40 slots) | -0.70; 95% interval -0.85 to -0.525 | interval entirely below 0: met |
| Same count by word list (cross-check) | 90% without, 55% with topic only | beside the bar |
| Hardware share: robot-life topics / everyday topics (topic only) | 40% (6 of 15) / 4% (1 of 25) | beside the bar |
| Same split in Exp 33 (topic and facts) | 20% (3 of 15) / 8% (2 of 25) | for comparison |
| (a) Funnier, 20 new blind pairs | not judged (0 of 20) | at least 10: not met by rule |
| Verdict | keep_off | both bars |

**Read it:**
- Taking the facts away did not lower hardware. On robot-life topics it went up, from 20% to 40%: the facts were giving the bit something else to talk about.
- Everyday topics are near hardware-free with or without facts (8% and 4%).
- Exp 33 and Exp 34 differ only by the facts (same slots, modes and topics), but 15 robot-life bits is a small sample.
- 1 thread lost and 0 bad replies with topic only (0 and 0 without).
- "Everyday topics only" looks promising (4%), but that idea came after seeing this data and would be a third try at the same bar. Not adopted, not run.

**This was a second attempt at the same bars.** With no real difference, one attempt reaches 10 wins of 20 about 59% of the time, so two attempts reach it at least once about 83% of the time. 20 pairs only catch big differences.

## Tonight's context for each bit (Exp 35, 2026-11-26)

Decision: [D-119](agent/decisions-detail.md#d-119). Raw: `reports/metrics/tonight.json` (rerun); first run's record: `reports/metrics/tonight_run1.json` (restored from git history, numbers only). Code: `sim/tonight.py`. Chart: `reports/figures/decision-plot/exp35-tonight.png`. Setup: one day of headlines turned into nameless one-sentence summaries; you approved them, then 10 blind pairs of bits from the live writer, the live prompt alone (gen-v5, arm A) against the live prompt with tonight's context (gen-v5 + tonight, arm B), same modes and rooms. No `audience.yaml` value is involved.

**First run answer: on by your judgment call, not a pass.** The script's rule says adopt, but the filter check had 1 summary, far too few to decide, and funniness only tied.

| Check | Result | Bar |
|---|---|---|
| Headlines fetched / dropped as disasters (no deaths, crimes or wars that day) | 10 / 5 | beside the bar |
| Rejected by the name check / by the safety check | 3 / 1 | beside the bar |
| Summaries kept / approved by you | 1 / 1 | beside the bar |
| You rejected (95% interval) | 0% (0% to 79%) | at most 20%: met, but the interval is far wider than the bar, so it cannot decide |
| Kept summaries with a real name | 0 | 0: met |
| (a) Funnier: with tonight wins | 5 of 10 (without 5, ties 0) | at least 5: met at the minimum |
| (b) More like tonight: with tonight wins | 9 of 10 (95% interval 60% to 98%) | beside the bar |
| Market line / typed line | not fetched (timed out) / none typed | beside the bar |
| Script verdict | adopt | both bars |

**Read it:**
- The one clear signal is (b): 9 of 10 bits with tonight's context felt like tonight.
- Funniness is a tie. 10 pairs only rule out "clearly worse".
- All 10 pairs used the same one summary, so this is one headline's evidence.
- What was tested is narrower than what ships: no market line, no typed line, and today's 10 top stories only. The fixes after the run (longer fetch timeout, two weeks of four lighter news sections, at most 40 headlines) were not measured.
- Next: run `python -m sim.tonight fetch` on more days to give the filter check enough summaries.
- Changed after this run (D-119 addendum, D-120): gists may now keep country, state, city, company, product and approved public-figure names, so the "0 real names" row of the rerun counts "a name that should not be there". This record was judged under the old meaning and is not comparable.

### Rerun (2026-11-26): adopt, both bars met

Same decision and pairs design, run on HEAD 14f7bd0 (places, checked organisations, public figures, "so what" summaries, writer people paragraph). Record: `reports/metrics/tonight.json`. TONIGHT was already on by default; it now also has a pass.

| Check | Result | Bar |
|---|---|---|
| Headlines fetched / dropped (death, disaster, crime, war) | 40 / 0, 5, 2, 0 | beside the bar |
| Rejected by the name check / by the safety check | 4 / 6 | beside the bar |
| Summaries kept / approved by you | 23 / 21 | beside the bar |
| You rejected (likely range, one day) | 9% (2 of 23), 2% to 27% | at most 20%: met |
| Kept summaries with a name that should not be there | 0 | 0: met |
| (a) Funnier: with tonight wins | 5 of 10 (without 3, ties 2) | at least 5: met at the minimum |
| (b) More like tonight: with tonight wins | 5 of 10 (without 4, ties 1), likely range 24% to 76% | beside the bar |
| Script verdict | adopt | both bars |

**Read it:**
- Both bars are met, so adopt now has a pass behind it.
- Second attempt at the same bars, which raises the chance of a lucky pass.
- Funniness is met exactly at its minimum. With no real difference, 5 or more wins of 10 happens about 62% of the time, so this means "not clearly worse".
- The reject share is 9%, but its likely range reaches 27%, above the 20% bar; one day cannot rule out a higher rate.
- "More like tonight" fell from 9 of 10 to 5 of 10. The setups differ (names, places, "so what" summaries, 40 headlines); the run cannot say which caused it.
- The first run's name count used the old meaning (D-120), so the two filter results are not comparable.
- No public figure was approved that day, so none of the 10 new bits named one; the public-figure rule is covered by the filter check only.


## Does the judge agree with you on same-writer pairs? (Exp 36)

Decision: [D-121](agent/decisions-detail.md#d-121), which carries out [D-102](agent/decisions-detail.md#d-102). Raw: `reports/metrics/dpo_judge_gate.json` (written by `score --record` after you run it). Code: `sim/dpo_pairs.py`.

**Status: done (2026-11-27). The bar was not met.** Claude Haiku 5.5 does not predict your taste on same-writer pairs, so DPO does not run on its labels. [Exp 37](#does-claude-opus-55-agree-with-you-on-the-same-pairs-exp-37) tries Opus 5.5 on the same pairs.

What it tests: DPO needs thousands of "which bit is funnier" labels, and only a judge can give that many. First it has to agree with you on bits of one request written by the same model, which are closer than the cross-model ballot of Exp 24. You rank 120 pairs. 60 teach the judge your taste, and 60 are held back as the test. The judge is Claude Haiku 5.5, your choice.

**The bar (set before any data):** the judge sees each test pair in both orders. A pair it answers differently in the two orders gets no label (and is dropped from DPO too). It passes if it answers at least 30 of the 60 the same way, and among those its agreement with you has a likely range above 50%. The least that clears: 21 of 30, 30 of 45, 38 of 60.

Chance of passing, if the judge is right this often in each order and its two orders err independently: 50%: 1.4%. 65%: 74%. 70%: 91%. 75%: 98%.

Prompts: 240 for ranking and 800 for training, half with tonight's context and half without, none repeating an earlier experiment's prompt. One day of headlines only (21 approved summaries), so the training pool is rebuilt with 3 or more days before the judge labels it.

Also reported, with no bar: how often the two orders disagree, agreement for each half, the count over all 60 pairs (a no-label pair counts as a miss), the rate if a no-label pair earned half, and your ties and skips.

| Check | Result | Bar |
|---|---|---|
| Test pairs answered the same way in both orders | 46 of 60 | at least 30: met |
| Of those, the judge picks your bit | 24 of 46 (52%) | likely range above 50% |
| Likely range | 38% to 66% | lower end above 50%: **not met** |
| Pairs where the two orders disagree | 23% | beside the bar |
| Tonight half: consistent, agreeing | 20 of 30, 11 | beside the bar |
| Plain half: consistent, agreeing | 26 of 30, 13 | beside the bar |
| All 60 pairs, a no-label pair counted as a miss | 24 of 60 | beside the bar |
| A no-label pair earning half | 0.517 | beside the bar |
| Ties / skips while ranking | 12 / 12 of 144 answers | beside the bar |
| Verdict | **fail** | the bar |

Reading: the judge answered the same way in both orders often enough, but on those pairs it picks your bit about half the time, which is chance. 30 of 46 was the least that clears. Both halves sit near 50%.

Ranking time: you report 12 to 16 hours over Nov 26 to 27, not the planned 70 minutes. The answers had no times, so this is your estimate, not a measurement. New answers now carry a time.

Self-agreement (reported only, no bar): not run, you skipped the re-rank of 20 pairs. How often you pick the same way twice would be a ceiling for any judge on these pairs.

Top-up: about 40% of threads fail the show's safety check, so bits with fewer than 2 clean threads get up to 2 more batches on the same request (`topup`); bits you may already have seen are not changed.

How you ran it: the commands are at the end of D-121.

Caveats:
- The Qwen judge D-102 named was never built, and the rewritten judge of D-114 only checks that lines connect, so this is the first judge of pairs.
- A judge that favours one position loses pairs to "no label", so a fail may mean position bias; the disagreement share is reported for that.
- Whether the judge stays right as DPO changes the model is not checked.
- The test only covers pairs where you had a pick. Pairs you tie or skip are left out, so how the judge does on pairs you find equal is not measured.

## Does Claude Opus 5.5 agree with you on the same pairs? (Exp 37)

Decision: [D-122](agent/decisions-detail.md#d-122), after [Exp 36](#does-the-judge-agree-with-you-on-same-writer-pairs-exp-36) failed. Raw: `reports/metrics/dpo_judge_gate_opus.json` (written by `score --model opus --record`). Code: `sim/dpo_pairs.py`.

**Status: done (2026-11-27). The bar was not met.** Opus 5.5 does not predict your taste on same-writer pairs either, so DPO does not run on either judge's labels.

What it tests: the same question as Exp 36, on the same 60 teach and 60 test pairs, with Claude Opus 5.5 as the judge. Same prompt, same examples, both orders, same settings. One difference you should know: Opus 5.5 cannot run with thinking off, so its thinking is on at low effort, where Haiku's was off. One model only, picked before running.

**The bar (set before any Opus answer).** The 60 test pairs were already used once, so a second look needs a stricter bar. Otherwise trying models until one passes would pass a judge with no skill about twice as often. It passes if it answers at least 30 of the 60 the same way in both orders, and among those its agreement with you has a likely range above 50% at the **97.5%** level (Exp 36 used 95%). The least that clears:

| Pairs answered the same way in both orders | Exp 36 (95%) | Exp 37 (97.5%) |
|---|---|---|
| 30 | 21 | 22 |
| 46 | 30 | 31 |
| 60 | 38 | 39 |

Chance of passing, if the judge is right this often in each order and its two orders err independently: 50%: 0.7%. 65%: 70%. 70%: 91%. 75%: 98%. (Exp 36's bar gave 1.4%, 74%, 91%, 98%.)

**Cost (estimate, before running).** One request is about 18,400 tokens, 120 of them. At Opus 5.5 prices (4 dollars in and 20 out per million tokens, cache reads 0.20; check the price list): about **$1 to $3 with caching** (the shared examples are cached), **$9 to $11 without**. The output part is a guess (200 to 1,000 tokens a call, thinking included); `judge` prints the real totals. If it passes, labelling the training pairs (4,800 calls) would cost about $40 to $120 with caching.

| Check | Result | Bar |
|---|---|---|
| Test pairs answered the same way in both orders | 53 of 60 | at least 30: met |
| Of those, the judge picks your bit | 25 of 53 (47%) | likely range (97.5%) above 50% |
| Likely range | 33% to 62% | lower end above 50%: **not met** (35 of 53 was the least) |
| Pairs where the two orders disagree | 12% | beside the bar |
| Tonight half: consistent, agreeing | 29 of 30, 13 | beside the bar |
| Plain half: consistent, agreeing | 24 of 30, 12 | beside the bar |
| All 60 pairs, a no-label pair counted as a miss | 25 of 60 | beside the bar |
| A no-label pair earning half | 0.475 | beside the bar |
| Ties / skips while ranking | 12 / 12 of 144 answers | beside the bar |
| Verdict | **fail** | the bar |

**Cost (measured).** 120 calls: 47,808 tokens in, 1,440 out, 2,864,806 cache reads, 24,074 cache writes. At the prices above: **$0.91**. The estimate was $1 to $3: Opus used almost no thinking tokens at low effort.

**Do the two judges agree with each other?** (Reported only, no bar. Read from the two stored caches, no calls. Likely ranges at 97.5%.) On the 41 test pairs both judges answered the same way in both orders:

| On those 41 pairs | Count | Share | Likely range |
|---|---|---|---|
| haiku-5.5 judge and opus-5.5 judge pick the same bit | 32 | 78% | 61% to 89% |
| haiku-5.5 judge picks your bit | 20 | 49% | 32% to 65% |
| opus-5.5 judge picks your bit | 19 | 46% | 30% to 63% |

The teach pairs have no answers in either cache: they are in the judge's prompt, so they were never asked.

Reading: the two judges agree with each other far more than either agrees with you. That fits a shared model taste that differs from yours. It also fits your picks on close pairs being noisy. You did not re-rank, so these cannot be told apart. 24 of 144 ranking answers were ties or skips, which says some pairs were hard for you too.

Outcome: two strong judges at about a coin flip on same-model pairs (24 of 46, 25 of 53) is a clean negative result for "an LLM judge as a stand-in for one person's taste on close pairs". Per D-122's rule, DPO does not run on either judge's labels. You confirm the next step.

How to run it:
```
python -m sim.dpo_pairs judge --model opus --env-file backend/.env
python -m sim.dpo_pairs score --model opus --record
```
Both results go in the report.

Caveats:
- The same 60 test pairs are used twice. The stricter bar reduces the risk, it does not remove it.
- Thinking is on for Opus and off for Haiku, so a difference between the two is the model and the thinking together.
- You did not re-rank 20 pairs, so there is no number for how often you agree with yourself. If it is low, no judge could pass.

## Is a trained writer funnier than the base? (Exp 38)

Decision: [D-123](agent/decisions-detail.md#d-123) (the model is chosen in [D-124](agent/decisions-detail.md#d-124)). Raw: `reports/metrics/dpo_eval.json` (written by `python -m sim.dpo_pairs eval score --record`; numbers only). Code: `sim/dpo_eval.py`.

**Status: pending.** Nothing is trained yet. This section fills in after the user runs the training jobs and votes.

What it tests: three writers start from the same base model. **A** is the base, **B** is trained toward Claude Haiku's taste (a judge's, not the audience's), **C** is trained on your own 120 pairs. Each writes one bit for each of 60 new prompts. You vote blind on 30 pairs: 15 of B against A and 15 of C against A. Opus 5.5 judges all three pairings (B against C too) on all 60 prompts; that is reported only. Haiku is reported too, but it wrote B's training labels, so for B it is marking its own homework.

**The win bar is not fixed yet.** The proposal is at least 10 wins out of 15 decided pairs in a comparison. You confirm it (`eval bar --wins N`) before the first vote, after the re-rank shows how often you agree with yourself.

| Bar (wins of 15 decided) | A coin flip gets there | A writer that truly wins 65% | 70% | 75% |
|---|---|---|---|---|
| at least 9 | 30.4% | 75.5% | 86.9% | 94.3% |
| **at least 10 (proposed)** | **15.1%** | 56.4% | 72.2% | 85.2% |
| at least 11 | 5.9% | 35.2% | 51.5% | 68.6% |
| at least 12 | 1.8% | 17.3% | 29.7% | 46.1% |

Two comparisons share one base, so a coin-flip writer clears "at least 10" in one of them 27.9% of the time (11.5% at "at least 11"). Ties are left out of the count, so with 12 decided pairs the same bar is harder: a coin flip gets there 1.9% of the time and a writer that truly wins 70% only 25%. Fifteen pairs cannot see a small real gain: "not met" means "not shown", not "not better".

**Speed has its own bar.** Each writer's bit time on the laptop, median and slowest 5% of its 60 bits, against bonsai's (Exp 26: median 3.5 s, slowest 5% 5.1 s) and the live budget of 12 s. B and C pass when their slowest 5% is within the budget, so the show could use them. Exp 24 timed qwen3:4b-instruct at 6.7 s median and 8.2 s slowest 5% with the older one-call generator, bonsai at 5.2 s and 6.9 s.

**Cost.** Labelling B's pairs with Haiku: about $1 with caching (at most 4,800 calls), about $9 without. Opus's reading of Exp 38: 360 calls, about $2.7 (Exp 37 cost $0.91 for 120).

