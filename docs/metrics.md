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

Decision (decisions.md 2026-10-27): the policy picks the mode once per bit (8 bits × 5 jokes),
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

| Judge mode | Share of 30k | y | n | ? | skipped | Precision | 95% CI (Wilson) |
|---|---|---|---|---|---|---|---|
| wordsmith | 28.0% | 11 | 0 | 1 | 3 | 100% | 74–100% |
| cynic | 27.2% | 10 | 1 | 1 | 3 | 91% | 62–98% |
| neurotic | 21.1% | 6 | 1 | 2 | 6 | 86% | 49–97% |
| observer | 11.2% | 11 | 3 | 0 | 1 | 79% | 52–92% |
| absurdist | 10.6% | 7 | 0 | 6 | 2 | 100% | 65–100% |
| optimist | 1.3% | 9 | 2 | 1 | 3 | 82% | 52–95% |
| none | 0.6% | 14 | 1 | 0 | 0 | 93% | 70–99% |

**Findings**
1. **Mode labels mostly hold:** 89% on judged rows (81–95%), **92%** re-weighted to the 30k mode mix.
2. **Misses lean to wordsmith:** 5 of 6 misses with a known mode were puns the judge filed elsewhere
   (observer ×2, neurotic, optimist, none). Wordsmith's last place in percentile may be slightly
   understated in size, not order.
3. **Absurdist is hard to verify:** 6 of 13 judged absurdist jokes couldn't be given any mode; the
   label may be the judge's bin for jokes that don't land.
4. **Safety flags are not usable as a filter:** the user flagged 16 rows (sexual 8, targets_group 3,
   about_bodies 1, outside the schema 4: guns, death, Nazis, profanity); the judge flagged **none**
   of them. Notes were opportunistic, so only misses are measured, not false alarms.

**Examples** (verbatim from r/Jokes via [rJokes](https://github.com/orionw/rJokesData); a few rows
quoted for illustration only, the full set stays local, see decisions.md 2026-10-31). Row = `n` in
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
