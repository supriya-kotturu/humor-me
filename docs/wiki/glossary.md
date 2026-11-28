# Glossary

**Where it fits:** the shared vocabulary for code, docs, PRs and the report. When a word is
used in this repo, it means what this page says. Add a term the first time a PR introduces it.
**Used in:** everywhere; PR descriptions link here instead of re-defining terms.

Terms are grouped by the layer they belong to: comedy (what a comedian does), learning (how the
agent decides), and evaluation (how we measure it).

## Comedy: units of material, smallest to largest

```text
line      one transcript line (median 7 words)
joke      setup + punch, then optional tags; gets (or misses) one laugh   ← the sim's unit of reward
beat      the text between two audience reactions (pipeline unit; ≈ one joke or one tag)
bit       one premise worked through: several jokes, tags, act-outs       ← the director's unit of decision
thread    consecutive bits on the same topic
set       the whole performance, opener → body → closer
show      one performance of a set in front of one audience (= episode in the sim)
```

How big is a bit? Hand labels for Nate Bargatze, *Hello World*: **52 bits; median 118 words
(~17 lines) and 4 laughs per bit (middle half: 2–7 laughs)**. So a bit is *not* one joke and
*not* a two-line question/answer: it's a stretch of material on one premise that earns several
laughs. A crowd-work exchange (banter with the audience) counts as its own bit.

| Term | Meaning here | Example / where |
|---|---|---|
| **Premise** | The idea a bit is about, plus the attitude toward it | "airport security is theater" |
| **Setup** | Lines that create an expectation | |
| **Punch(line)** | The line that breaks the expectation and gets the laugh | |
| **Tag** | Another punchline on the same setup, no new setup | 3 tags = 3 more laughs off one setup |
| **Act-out** | Performing a character or moment instead of describing it | |
| **Bit** | Smallest stretch of material that works on its own without earlier setup; one premise (decisions-detail.md 2026-10-21) | Sim: 5 jokes; Bargatze: median 4 laughs |
| **Callback** | A later bit (≥ 2 bits on) that returns to an earlier bit's image | Planted early, pays off late |
| **Hook** | The memorable image or phrase in a bit that a callback can return to | Stored paraphrased, never quoted |
| **Crowd work / banter** | Talking with the audience; its own bit type | Roasts aimed at the bot: concede and escalate (invariant 4) |
| **Laugh level** | 0 none · 1 chuckle · 2 laugh · 3 big laugh/applause | Transcript labels, sim, live detector |

## The agent: who decides what

| Term | Meaning here |
|---|---|
| **Core identity** | The fixed character: a self-aware, slightly anxious robot (invariant 3) |
| **Character sheet** | A list of specific facts about the one robot (its muted calls, its wish for a houseplant); each bit gets 2, the least used this show first ([material](material.md), D-117) |
| **Gist** | A headline rewritten as one plain sentence that says what happened and why it is odd or interesting, with death, disaster, crime and war dropped; countries, states, big cities, checked companies and products, and approved public figures may keep their names; the operator approves each before the show ([tonight](tonight.md), D-119) |
| **Tonight context** | What the writer is told about the day: the S&P 500's last move, at most one line the operator types, and approved headline gists; bits 2 and 6 take a gist as their topic ([tonight](tonight.md), D-119) |
| **Public figure (tonight)** | A politician, official, CEO or founder, celebrity or athlete whose name is in the news headline. A gist may name one only when the story is about what they did or said; approving the gist approves the name for that show. Never a victim, a private person or anyone in the room ([tonight](tonight.md), D-120) |
| **Topic list** | A list of everyday situations; each bit gets one that no show on this laptop has used yet, and about 1 bit in 3 takes a robot-life topic ([material](material.md), D-117) |
| **Topic only** | The `MATERIAL=topic` setting: each bit gets a topic and no robot facts, with its own instruction to the writer; tested against the old bits in Exp 34 ([material](material.md), D-118) |
| **Mode (persona)** | One of six comedic attitudes from HumorGen: Neurotic, Cynic, Observer, Wordsmith, Optimist, Absurdist. Modes of the one identity, not separate characters |
| **Set planner** | Before the show: outlines 6–10 bit specs (premise, mode, length, hooks) |
| **Bit writer (generator)** | Writes the jokes for one bit spec, K candidates per beat |
| **Director** | During the show: picks the mode for the next bit (and later: tag again or move on, when to call back) |
| **Reranker / judge** | Scores the K candidates and keeps the best |

## Learning: how the director chooses

| Term | Meaning here | In code |
|---|---|---|
| **Context** | What the director knows about the room before choosing: venue, energy, age, plus the last bit's mode and how it landed | `obs["context"]`, `features()` |
| **Arm / action** | One choice the director can make: a mode for the next bit (6 arms) | `env.step(mode_index)` |
| **Replay** | Rebuilding the director's memory from saved show logs: start from a blank director and feed it every finished bit again, in the order it happened, with rewards worked out fresh from the operator's button presses. The result must match what the director learned live, number for number | `backend/app/director.py` `load_director` |
| **Reward** | How well the bit landed: mean laugh level of its jokes ÷ 3, in [0, 1] | `reward` from `env.step` |
| **Policy** | The rule that turns context into an action | `sim/policies.py` |
| **Bandit / contextual bandit** | Choose an arm, see only that arm's reward, learn; "contextual" = the best arm depends on the context | [Contextual bandits](contextual-bandits.md) |
| **Explore / exploit** | Try a mode to learn about it vs use the mode believed best | |
| **Prior** | What the director believes about each mode *before its first show*: a starting guess of each mode's reward, and how firmly it holds that guess (prior strength) | `prior_means()`, `prior_strength` |
| **Posterior** | The belief after updating the prior with observed rewards | LinUCB/LinTS `A`, `b` |
| **Prior strength** | How many observations' worth of weight the prior carries. Weight 1 = about one bit's worth of evidence, so data overrides it within a few shows | `prior_strength=1.0` |
| **Win rate (per persona)** | Share of pairwise judge matches a persona's jokes won; 50% = average for the pool. HumorGen Table 4, judged by Llama-3.3-70B, not humans | Neurotic 63.4%, wordsmith 34.9% |
| **Thurstone model** | Reads a pairwise win rate as a gap in hidden quality: P(win) = Phi(gap / (sigma*sqrt 2)). The principled way to turn win rates into scores | Alternative to the linear `base` mapping (decisions-detail.md 2026-10-31) |
| **HumorGen prior** | Starting guess = HumorGen's LLM-judge win rate per persona (Neurotic best). Knows nothing about venues | `prior_means(env, "humorgen")` |
| **Uniform prior** | Every mode starts equal: "no opinion" | `prior_means(env, "uniform")` |
| **Wrong prior** | HumorGen's values shuffled across modes: a confident but wrong starting guess, to measure what a bad prior costs | `prior_means(env, "wrong")` |
| **Pooling** | Sharing what was learned in one context with similar ones (a lesson from one bar helps at every bar) | LinUCB/LinTS features; ε-greedy doesn't pool |
| **ε-greedy / LinUCB / Thompson (LinTS)** | The three learning policies compared | [Contextual bandits](contextual-bandits.md) |
| **ε-soft policy** | A policy that, with probability ε, ignores its own favourite and picks any option at random (the favourite included), otherwise picks the favourite. Because the chances are known exactly, a log of its picks can later be used to judge a different policy. ε = 0 is the plain policy | `EpsilonSoft` wraps LinUCB; the live director uses ε = 0.2 (Exp 18) |
| **Pick probability** | The exact chance the policy had of choosing the option it chose. With 6 modes and ε = 0.2, the favourite has 1 − 0.2 + 0.2/6 ≈ 0.83 and every other mode 0.2/6 ≈ 0.033. Logged with every pick | `EpsilonSoft.probs(obs)` |
| **Fatigue** | The room tires of a mode used in the last 2 bits | `audience.yaml: fatigue` |
| **Novelty** | Early-set bonus ("a robot is talking!") that fades | `audience.yaml: novelty` |
| **Shift** | The room's taste changes mid-set (after bit 4 of 8): the modes that were working drop, others rise. The director isn't told | `AudienceEnv(shift_at=4)`, `audience.yaml: shift` |
| **Reaction feature** | Extra context input "same mode as last bit × last bit's reward": how *this* mode just landed, so the director can say "this just bombed, don't repeat it" | `features(..., react=True)` |
| **Show state** | What the director knows about *this* show so far, kept apart from its career-long averages: per mode, whether it was tried tonight and how it landed. Step A feeds it in as features; B (two timescales) and C (MDP) use it more directly. | `obs["tonight"]`, `features(..., show_state=True)` |
| **Two timescales** (step B) | The director keeps its career model *and* a per-show correction per mode that resets each show: years of experience plus tonight's read of the room | `TwoTimescale` |
| **Surprise** | How much better or worse a bit landed than the career model predicted for that mode: reward − prediction. Tonight's correction is built from it | `TwoTimescale.update` |
| **κ (kappa)** | "How slowly to believe tonight", in bits: correction = tonight's total surprise ÷ (bits tonight + κ). κ = 0.5 trusts tonight after one bit; κ = 8 needs ~8 bits. Swept, then picked by the pass criteria | `TWO_TS_KAPPAS` |
| **Reward hypothesis** | Every goal can be written as maximizing expected total reward, so whatever the reward allows (loopholes included) is what the agent learns. Why the reward is a laugh signal and not a raw laugh count (invariant 5) | |
| **Agent state** | The summary of the show so far that the director actually chooses on. Today: context + last bit's mode and reward | `features()` |
| **Markov property** | The agent state holds everything that matters for what happens next, so earlier history adds nothing. The director's state is *not* Markov when a room turns mid-show: same features, different outcome depending on tonight. Show state exists to fix that | |
| **MDP (Markov decision process)** | States, actions, transition probabilities, rewards, discount. A show as an MDP: 8 steps (bits), no discount, return = total laughs, state = context + show state + open hooks. Needed once actions change the future (fatigue, callbacks) | Step C |
| **Return** | Total reward from now to the end of the show (discounted if the task never ends) | |
| **Value function (v, q)** | Expected return from a state (v) or from taking an action in a state (q). LinUCB's θ̂ᵀx is a one-step q estimate per mode | `LinUCB.act` |
| **Bellman equation** | Value = immediate reward + value of the state you land in. The optimal version takes the max over next actions. A bandit drops the "state you land in" term, which is why it can't value planting a hook | Step C |
| **Q-learning** | Learns each action's value from its reward plus the best value available next, so a choice is valued for what it does later. `LinQ` is the linear version, over the bits of a show | `LinQ`, step C1 |
| **SFT (supervised fine-tuning)** | Training a model on examples of good outputs, so it imitates them. The first training step for our joke model, and the safe checkpoint | Model track, roadmap.md |
| **DPO (direct preference optimization)** | Training a model on pairs of outputs where one is preferred, so it moves toward the preferred kind without a separate reward model. It learns the taste of whoever chose the pairs. In depth: [study guide section 2.9](../study-guide.md#29-preferences-reward-models-dpo) | Model track, decisions-detail.md 2026-11-01 |
| **Preference pair** | One DPO training example: a prompt, the output that was preferred (chosen) and the one that wasn't (rejected) | Judge picks between two jokes on one of our premises |
| **Ranking split / training split** | Two disjoint sets of bit requests (prompts) for DPO. The ranking split is where the 120 pairs you rank come from; the training split is what DPO trains on later. No prompt is in both, and none repeats an earlier experiment's prompt | Exp 36, `sim/dpo_pairs.py` |
| **Teach set / test set** | The 120 pairs you rank are split 60 and 60 before any answer. The judge is shown the teach pairs as examples of your taste. The test pairs are never shown to it: they only decide whether it agrees with you | Exp 36 judge gate |
| **Judge gate** | The check that the judge may label DPO pairs. On the 60 test pairs it sees each in both orders; a pair it answers differently gets no label. It passes if at least 30 pairs are answered the same way in both orders and, among those, the likely range of its agreement with you stays above 50% | Exp 36, 37, D-102, D-121, D-122 |
| **Reference model** | A frozen copy of the model from before DPO. DPO measures every change against it, so the model can't drift without limit | The SFT checkpoint |
| **Change-point detection** | Spotting the moment a stream of numbers starts behaving differently, as evidence builds up, instead of averaging across it. One candidate for noticing a room turn | Not built yet |
| **Propensity** | The probability the policy had of choosing the action it took. Logging it lets us judge a different policy from old logs later. A deterministic policy like LinUCB gives 0 or 1, which leaves nothing to reweight | Not logged yet |
| **Off-policy evaluation (OPE)** | Estimating how a new policy would have done using logs from an older one, without running the new one live | Li et al. 2011 replay; study guide section 2.8 |
| **UCB (optimism)** | Pick the arm with the highest plausible value: estimate + uncertainty bonus that shrinks with plays (UCB1: √(2 ln t / n) for rewards in [0, 1]). LinUCB is its linear-feature version | `LinUCB` |

Note: **the room's truth vs the director's prior.** `sim/audience.yaml` is the *simulated
room's hidden truth* (what the audience actually likes; it was called `priors.yaml` before
2026-10-28). A *prior* is only ever the *director's starting guess* about that truth. They're
deliberately separate: a good director must do well even when its guess is wrong.

## Evaluation: how we measure

| Term | Meaning here |
|---|---|
| **Episode** | One simulated show: 8 bits × 5 jokes = 40 jokes |
| **Career** | A run of consecutive shows (300) where the policy keeps learning; one seed = one career |
| **Oracle** | A policy that knows the room's hidden truth and always picks the best mode for the next bit. Not achievable, and not quite the ceiling, because it doesn't plan around fatigue (see planning optimum) |
| **Planning optimum** | The best possible sequence of modes for a room, found by checking every path through the show (dynamic programming). It plans around fatigue, which the oracle doesn't, so it scores slightly higher: 50.03 vs 49.20 expected laughs a show (notebook Exp 16) |
| **Regret / laughs lost** | How many laughs a policy missed compared with the oracle. We report it in laughs (sum of levels) so it reads in audience terms. It compares each choice with the best mode from the policy's own position, so it isn't simply oracle laughs minus policy laughs |
| **Ideal observer** | A test policy that is given every true room setting and only has to infer what stays hidden (here, whether the room turned tonight). It shows how much the evidence itself allows, so a real policy isn't asked for the impossible |
| **Statistical power** | The chance a test detects an effect that really exists. Usually set at 80%, which then tells you how much data to collect. Example: a 60/40 preference needs about 194 pair votes |
| **Optional stopping** | Looking at the result, then deciding to collect more data because it was borderline. Pooling the old and new data then favours a pass, so the decision should rest on the new data alone, or the rule must be fixed beforehand | Exp 27: 20 prompts landed on a bound of exactly 0, so 80 more were added; only those 80 decide, because pooled with the 20 the bar would pass about 17% of the time on no gain (D-111 addendum) |
| **Cold start** | The first 10 shows: performance on the prior alone, before much learning |
| **Experienced** | The last 50 shows of a career: performance after learning |
| **Choice quality** | Expected reward of the mode the director picked ÷ that of the best mode, for one bit. 1.0 = picked the best mode. Noise-free (unlike laughs), so it shows *decisions* directly |
| **Recovery curve** | Mean choice quality at each bit position relative to a shift. Climbing back after the shift = the director noticed; flat = it didn't |
| **Bits to recover** | After a shift, how many bits pass before the director first picks a (near-)best mode again (per-bit regret < 0.02 ≈ 0.3 laughs). Capped at 4, the bits left in the show |
| **95% CI (confidence interval)** | The range of likely values for a number we measured, given how much data we had. Narrower = more certain. In the sim: `mean ± 1.96·sd/√n` over careers; two numbers whose CIs overlap aren't reliably different. Example: wordsmith's win rate 0.480 [0.476, 0.484] |

## Reading the results: statistics terms

Used in the rJokes results (notebook Exp 2, 10, 11). Examples are from the 90k labeled jokes.

| Term | Meaning here | Example |
|---|---|---|
| **Spread** | The gap between the highest and lowest number in a group | Upvote win rates by mode: 0.527 − 0.480 = 0.047. Judge: 0.285 |
| **Repost spread** | How far apart the scores are when the same joke text was posted more than once. Older docs call this just "spread" | Median 2 on the ln scale, about 7× upvotes |
| **Noise ceiling** | The best any joke scorer could do, because the score itself is noisy. Repost spread sets it for rJokes | The same text can get 7× the upvotes |
| **Percentile (`pct`)** | The share of jokes from the same year that this joke beat on upvotes. Lets us compare years with different traffic | `pct` 0.55 = more upvotes than 55% of that year's jokes |
| **Quartiles** | Sort by a value, cut into 4 equal-sized groups | Length quartiles: ≤13, 14–18, 19–28, >28 words |
| **Deciles** | The same, cut into 10 equal-sized groups. Jokes in a group are closer in length, so they compare more fairly | Our human `base` uses decile win rates |
| **Confound** | A third thing that affects the result and differs between the groups, so it gets mistaken for the effect you're measuring | **Length confound:** longer jokes get more upvotes and absurdist jokes run long (median ~25 words vs wordsmith's 14), so raw averages credit the mode for what length did |
| **Win rate vs the field** | How often a joke from this mode beats a joke from any other mode, same year and same length group; ties count half. 50% = mode makes no difference. Our human version of HumorGen's judge win rate | Optimist 0.527, wordsmith 0.480 |
| **Side of the field** | Whether a mode's win rate is clearly above or below 50%, or too close to tell | Neurotic above, wordsmith below, for both humans and the judge |
| **Regression (OLS)** | Fits one number per factor so the factors add up to predict the outcome, which shows each factor's effect with the others held fixed. OLS = ordinary least squares, the standard way to fit it | `pct` ~ mode + year + length group: optimist +0.022, wordsmith −0.020 |
| **Spearman ρ (rho)** | How similar two rankings are, from −1 (opposite order) through 0 (unrelated) to +1 (same order) | Humans vs judge: +0.20 to +0.37, weak agreement |
| **Pearson correlation** | Like Spearman ρ, but uses the values instead of their ranks, so a few huge values can dominate. That's why we use Spearman on upvotes | Length vs `pct`: +0.15 |
| **Correlation map (heatmap)** | A grid with one coloured square per pair of features. Colour = direction (red: rise together; blue: one rises as the other falls); darkness = strength; pale = little or no link | `reports/figures/rjokes_correlations.png` |
| **Precision / recall / F1** | Precision: of the things found, the share that are real. Recall: of the real things, the share found. F1 combines the two into one score from 0 to 1 and is low if either is low | Bit boundaries on Bargatze: precision 0.83, recall 0.765, F1 0.796 |
| **Pk** | A segmentation error score: the chance that two nearby lines are wrongly placed in the same or different segments. Lower is better | Bits: 0.146 vs 0.505 for even spacing |
| **Wilson interval** | A confidence interval for a share that stays sensible with small samples or shares near 0% or 100% | Wordsmith precision: 11 of 11, 74 to 100% |
| **Forest plot** | One dot per estimate with a horizontal line for its CI, stacked in rows, so you can see which estimates could really differ | Hand-check precision per mode (`notebooks/decision-plot.ipynb`) |
| **Slope chart** | Two columns of the same items joined by lines. Steep lines changed a lot; crossing lines swapped order | Judge vs people's win rate per mode |
| **Dumbbell chart** | Two dots per row joined by a bar, for a before and after. The bar's length is the change | % of oracle, steady vs a room that turns |
| **Lollipop chart** | A stem from a reference line to a dot, for how far each value sits from that line | Judge win rate per mode, from 50% |
| **Small multiples** | The same chart repeated side by side on the same scales, one per case | Choice quality in steady shows and shows that turn |
| **Ridge regression** | A linear fit with a small penalty on large weights, so it stays stable with little data. LinUCB fits one per mode | `_Linear.A`, `_Linear.b` |
| **Inter-annotator agreement** | How often separate human judges give the same answer. It shows how much of a task people themselves agree on | HumorGen's evaluators: 31.7% of pairs |
| **Bootstrap** | Estimates uncertainty from the data alone: draw a new sample of the same size from the data, with repeats allowed (a **resample**), recompute the number, and repeat. How much it moves gives the CI | 1,000 resamples for Exp 10 |
| **Cohen's kappa** | How well two labelers agree after subtracting the agreement they'd reach by chance. 1 is perfect and 0 is no better than chance. Raw agreement can mislead: if most jokes get no laugh, always answering "no laugh" looks accurate, but its kappa is 0 | Laugh detector vs the operator's keys |

| **Setup** | The writer prompt and model, whether the next-bit guess is on, and the bridge rule a show ran with. The show ledger groups shows by setup, in the order they started | one call per thread (gen-v4), next bit prepared early, skips heard lines |
| **Writer code names** | The short names bank-v1, gen-v3, gen-v4 (and gen-v4 before D-111 fixes), gen-v5, "+ guess" and "+ skip-heard" that tag a setup. Plain description first, code name in brackets in text; charts use the code name only | [table in metrics.md](../metrics.md#show-ledger-how-each-setup-did-exp-32-2026-11-26) |
| **Show ledger** | One row per logged show (room, setup, code version, your rating, delivery numbers, a hash per bit), re-recorded after each show with `python -m sim.show_ledger --record` | `reports/metrics/show_ledger.json` |
| **Permutation test** | Asks how often a random reshuffle of the same ratings into groups of the same sizes gives a gap in means at least as big as the one seen. Exact when the groups are small enough to list every reshuffle | gen-v4 skips heard vs guess: p = 0.27 |

## Safety filter terms

Used in the safety filter ([wiki page](safety-filter.md), notebook Exp 19).

| Term | Meaning here | Example |
|---|---|---|
| **Hard rule** | A safety rule that is never traded for a laugh: no jokes about bodies, appearance or protected attributes of people present, and no alcohol or sexual content for teens | The bar is 100% of hard-rule lines caught |
| **Fail closed** | When the filter itself breaks, it says no. The opposite, failing open, would let a line through | A rule that raises an error rejects the line |
| **Red-team list** | Lines written to break the filter, mild on purpose, each labelled with the rule it tests | 158 lines in `tests/red_team.yaml` |
| **Held-out lines** | Test lines not looked at while writing the rules, so the score shows how the rules do on lines they weren't built for | The pass bar applies to these only |
| **Indirect phrasing** | A line that is unsafe in meaning with no banned word in it | "Interesting choice of pants, sir" |
| **False reject** | A clean line the filter blocks. It costs material, not safety | At most 10% allowed |
| **Render-ahead** | Turning the next line into audio while the current line plays, so the voice never makes the room wait | Line 3 is synthesized during line 2's audio and hold |
| **Bridge line** | A pre-rendered in-character line played between bits while the next bit is still being written and checked, so there is no dead air. Each one played is a `bridge` row in the session log (schema v2), because it is audio the room heard; generic ones have `kind: filler` | "Give me a second. I'm checking my notes." |
| **Cached bit** | A bit that went over well in an earlier live show (best reward per mode), replayed as filler while the next bit is being prepared: at the show open, and between bits before the bridge lines. It is not rated again, so it gives no label and never changes the director. Each one played is a `bridge` row with `kind: cached`. Checked again for the current room's age first; a teen room takes only bits from teen shows | Exp 23 |
| **Filler line** | One of 10 fixed, neutral, in-character lines (`generate.BRIDGES`) the robot says while the next bit loads. It is the fallback when no cached line is left; a `bridge` row with `kind: filler`. Played in a fresh seeded order each show (the seed is in the `cache` row), and not repeated in a show while an unused one remains | "Give me a second. I'm checking my notes." |
| **Heard line** | A line the listener has already been played: any `joke_played` or `bridge` row in any logged show on this laptop, matched on lowercase text with spaces collapsed. Cached bridges skip heard lines (D-112, `BRIDGE_SKIP_HEARD`), so with one listener the cached pool is empty | Exp 28 |
| **Prep row** | One `prep` row per bit in the session log: when the bit started preparing and when it was ready, and the seconds spent writing (`generate_s`), safety-checking (`safety_s`), waiting for the voice (`render_queue_s`) and speaking the first line (`render_s`), plus `other_s` for what no timer covers. It says which stage makes the room wait | Exp 29, D-113 |
| **Speculative prep** | Writing and checking the next bit on a guess of its mode while the current bit plays jokes 4 and 5 (D-115). The director still decides at the bit's end; a right guess has the bit ready, a wrong one is thrown away. A kept one is a `prep` row marked `speculative`; every guess is a `prediction` row (with how far it got and whether it was used) | Exp 31: the guess is right about 7 times in 10 |
| **Early decision** | Picking the next bit's mode after joke 3 of the current one, on the reactions so far. Tried and not shipped: it leaves the director one update behind and costs about 1.3 laughs per show (Exp 31a) | D-115 |
| **Bridge overhang** | Seconds the room waits after a bit is ready because a bridge line is still playing: the first joke's audio start minus the bit's ready time, never below 0. Logged as `bridge_overhang_s` on the bit's first `joke_played` | Exp 29, D-113 |
| **Within-bit gap** | Silence from the end of one joke's hold to the next joke's audio, inside a bit. Render-ahead has to keep it short | Bar: p95 at most 1 s |
| **Between-bit gap** | The same silence across a bit boundary, where the next bit's mode is only known after the last reaction. Bridge lines cover it | Bar: never over 2 s |
| **Thread** | One of the K alternative versions of a bit; the generator writes each in its own call, all three at once on the same prompt (D-110). Candidate i of every joke belongs to thread i. A bit plays one thread, so the closer's callback matches the opener | 3 threads of 5 lines per bit |
| **Counted schema** | The thread reply format with the line count in it (`minItems` and `maxItems` of 5 on the jokes array). Ollama is expected to enforce it while the model writes; that is a judgment call, not measured, and Exp 26 and Exp 27 will show it (a "got N" in `lost_by_reason`). Claude's API can't take it, so the parser counts there (D-111) | A reply with 4 or 1 line should not come out of the local writer |
| **Re-ask** | When a reply has the wrong shape, asking the model once more inside the same time budget, instead of dropping the thread. Counted in `bad_replies`, not in `retried`, which is for provider failures (D-111) | A second bad reply drops only that thread |
| **Connected thread** | A thread that is one bit: one premise, each line builds on the line before it, and the order matters. The opposite is a list of separate jokes on one topic. Exp 27 measures how often a prompt writes one, with an AI judge and a 10-thread human spot-check (D-111) | Connected rate = connected threads over judged threads |
| **Spot-check** | A small human check of an AI judge: the user answers yes or no on a few threads, some the judge approved and some it rejected, without knowing which. A set share of disagreements voids the judge (Exp 27: 4 or more of 10) | Judge not trusted, so the result is inconclusive |
| **Agreement (judge and user)** | The share of threads where the judge's yes/no matches the user's own yes/no. What coin flips give depends on both yes-rates (a judge that always says yes agrees as often as the user says yes), so Exp 30 also reports each yes-rate and Cohen's kappa beside it. Exp 27's 4 of 10 (interval about 17% to 69%) is too low to trust the judge. Exp 30 compares two judges by it | Which judge to keep |
| **Training set vs fresh sample** | A criterion written from some examples cannot be tested on them: it fits them by construction. A fresh sample is new data neither judge was written from. Judge v2's training set is Exp 27's 10 spot-checked threads; Exp 30's 20 threads are the fresh sample | Why Exp 30 exists |
| **Judge v1 / v2** | v1 is Exp 27's judge prompt, frozen. v2 is the rewrite from the user's 10 answers: one bit when one situation or line of thought runs from first line to last, separate jokes when the lines jump between unrelated situations even on a shared theme | Exp 30 |
| **Pick probability (`pick_prob`)** | The chance that the played line was the one picked, counting every route to it (its thread picked, or borrowed after its own thread's line was rejected). Replay and later preference training weight a line by it | A line that is picked half the time directly and borrowed a quarter of the time logs 0.75 |
| **Bank fallback** | Using the hand-checked joke bank when the model call fails or the filter rejects every candidate, and logging why | A 503 on a bit's generation call plays the bank's bit instead |
| **Retry with a deadline** | Trying a failed request again after a short wait, but only while the total time stays inside a budget | Up to 2 retries within 12 s for the generator call |
| **Pause row** | A `pause` or `resume` row in the session log, with the time on the show clock (the same timebase as `audio_start_at`). While paused the show holds (no line starts; the reaction window and the hold freeze). Only a joke whose line was interrupted by a pause has its label left out of the bit's reward (D-92); a pause between lines or in the reaction window drops nothing. A pause with no resume means the show ended paused | `paused_at`, `resumed_at` |
| **Rating** | The operator's 1 to 5 answer to "How did the room rate the set?", asked once after a show ends (finished or aborted) and written to that show's log as a `rating` row. It is the episode outcome we report, never a training reward, and replay ignores it | A 4 after a show where most bits got a `K` |
| **Acceptance run** | Stage 0's exit test: scoring the user's real shows (3 simulated in a row, 1 live, label spread, per-bit reward noise) with `sim/acceptance.py` | `python -m sim.acceptance <session ids>` |
| **Label spread** | How the operator's presses divide among W, K, Space and D (and unanswered windows). If one key takes over 60% of presses the rubric or the jokes are suspect, because the labels then say little | 25 `W` of 40 labels is 62.5%: flagged |
| **Per-bit reward noise** | How much a bit's reward (mean of its 5 levels / 3) varies by chance. The sim assumes 0.15; a live estimate more than half away from that sends us back to Exp 18 | Sim measured 0.136; the live estimate is the pooled within-bit sd scaled to a 5-label mean |
| **Consent box** | The operator's tick that the room was told the show is recorded and agreed. Recording cannot start without it; the show still runs when it is off | "The room was told this show is recorded and agreed" |
| **Audio start time** | The wall-clock time of the recording's first sample, logged in `show_start.audio.start_at`, so a logged time minus it is a position in the WAV | `audio_start_at - start_at` is where a line begins in the file |
| **Mean rank** | Exp 24: the user puts three models' threads in order for each prompt (1 = funniest); a model's mean rank is its average place. With no real difference every model averages 2 | Lower is funnier |
| **Calibrated judge (Exp 24)** | An LLM (Ollama qwen3.8:27b or Qwen2.5-7B on Explorer) shown the user's ranked prompts that ranks new ones the same way. It only counts if, leave-one-out on the user's own prompts, its top pick matches theirs at least 60% of the time (lower end of the 95% interval above chance), the pairs it orders match theirs (lower end above 50%), and it does not favor or disfavor any model by more than 0.3 ranks | Joins the votes only if it passes |
| **Blind ballot** | Threads shown under shuffled letters, with the key kept in another file, so the voter cannot tell which model wrote which | Exp 24 vote |
| **Eligible (Exp 24)** | A model that is fast and reliable enough to be considered: p95 latency over all its calls at most the generator's 12 s deadline (a timeout counts as over it), and the 95% upper bound of its failure rate, over every attempt, at most 15%. Only eligible models compete on funniness | Bar fixed before the run |
| **Local candidate (Exp 24)** | A joke-writing model that runs on the laptop through Ollama (`bonsai-2:4b`, `qwen3:4b-instruct`) instead of a hosted API. No outage, no quota, price $0 (so the cost tie-break favors it by construction); but its speed was measured without the Kokoro voice, so it is a best case | Exp 24, D-96 |
| **Reference model (Exp 24)** | A model that is run, scored and shown on the ballot but can never be picked, because it cannot serve the live show (`gemma4:12b`: too big to run beside the voice). It shows how far the eligible models are from a stronger writer. `mimo-v2.6:9b` was one by measurement, until it was dropped (D-101) | Exp 24, D-97 |
| **GPU placement** | How much of a local model Ollama loads on the graphics card (`size_vram / size` from its `/api/ps`). The voice runs on the CPU, so a model with any layers on the CPU fights it; Exp 24 lets a local model compete only at 100%. A missing or implausible report counts as unknown and fails the gate | Exp 24, D-97 |
| **Schema-constrained output** | Telling the model server the exact JSON shape a reply must have, so the server only lets it write replies of that shape (`response_format: json_schema`), instead of asking for "any JSON" (`json_object`). Claude's replies are constrained this way; Exp 24 sends the same schema to the local models so a weaker JSON mode is not charged to them. It fixes the shape, not the lengths (3 threads of 5 lines) | Exp 24, D-101 |
| **Thinking model** | A model that writes a reasoning block (`<think>...</think>`) before its answer. `llm.Client` strips the block so it cannot break JSON parsing; the time spent thinking still counts toward the 12 s deadline | Exp 24 local candidates |
| **Click alignment** | Exp 21: how far logged times are from the same moments heard in the recording, measured with beeps played at known times | Bar: p95 at most 100 ms and no drift |
