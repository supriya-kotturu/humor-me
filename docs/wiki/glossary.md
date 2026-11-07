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
| **Bit** | Smallest stretch of material that works on its own without earlier setup; one premise (decisions.md 2026-10-21) | Sim: 5 jokes; Bargatze: median 4 laughs |
| **Callback** | A later bit (≥ 2 bits on) that returns to an earlier bit's image | Planted early, pays off late |
| **Hook** | The memorable image or phrase in a bit that a callback can return to | Stored paraphrased, never quoted |
| **Crowd work / banter** | Talking with the audience; its own bit type | Roasts aimed at the bot: concede and escalate (invariant 4) |
| **Laugh level** | 0 none · 1 chuckle · 2 laugh · 3 big laugh/applause | Transcript labels, sim, live detector |

## The agent: who decides what

| Term | Meaning here |
|---|---|
| **Core identity** | The fixed character: a self-aware, slightly anxious robot (invariant 3) |
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
| **Reward** | How well the bit landed: mean laugh level of its jokes ÷ 3, in [0, 1] | `reward` from `env.step` |
| **Policy** | The rule that turns context into an action | `sim/policies.py` |
| **Bandit / contextual bandit** | Choose an arm, see only that arm's reward, learn; "contextual" = the best arm depends on the context | [Contextual bandits](contextual-bandits.md) |
| **Explore / exploit** | Try a mode to learn about it vs use the mode believed best | |
| **Prior** | What the director believes about each mode *before its first show*: a starting guess of each mode's reward, and how firmly it holds that guess (prior strength) | `prior_means()`, `prior_strength` |
| **Posterior** | The belief after updating the prior with observed rewards | LinUCB/LinTS `A`, `b` |
| **Prior strength** | How many observations' worth of weight the prior carries. Weight 1 = about one bit's worth of evidence, so data overrides it within a few shows | `prior_strength=1.0` |
| **Win rate (per persona)** | Share of pairwise judge matches a persona's jokes won; 50% = average for the pool. HumorGen Table 4, judged by Llama-3.3-70B, not humans | Neurotic 63.4%, wordsmith 34.9% |
| **Thurstone model** | Reads a pairwise win rate as a gap in hidden quality: P(win) = Phi(gap / (sigma*sqrt 2)). The principled way to turn win rates into scores | Alternative to the linear `base` mapping (decisions.md 2026-10-31) |
| **HumorGen prior** | Starting guess = HumorGen's LLM-judge win rate per persona (Neurotic best). Knows nothing about venues | `prior_means(env, "humorgen")` |
| **Uniform prior** | Every mode starts equal: "no opinion" | `prior_means(env, "uniform")` |
| **Wrong prior** | HumorGen's values shuffled across modes: a confident but wrong starting guess, to measure what a bad prior costs | `prior_means(env, "wrong")` |
| **Pooling** | Sharing what was learned in one context with similar ones (a lesson from one bar helps at every bar) | LinUCB/LinTS features; ε-greedy doesn't pool |
| **ε-greedy / LinUCB / Thompson (LinTS)** | The three learning policies compared | [Contextual bandits](contextual-bandits.md) |
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
| **DPO (direct preference optimization)** | Training a model on pairs of outputs where one is preferred, so it moves toward the preferred kind without a separate reward model. It learns the taste of whoever chose the pairs. In depth: [study guide section 2.9](../study-guide.md#29-preferences-reward-models-dpo) | Model track, decisions.md 2026-11-01 |
| **Preference pair** | One DPO training example: a prompt, the output that was preferred (chosen) and the one that wasn't (rejected) | Judge picks between two jokes on one of our premises |
| **Reference model** | A frozen copy of the model from before DPO. DPO measures every change against it, so the model can't drift without limit | The SFT checkpoint |
| **Change-point detection** | Spotting the moment a stream of numbers starts behaving differently, as evidence builds up, instead of averaging across it. One candidate for noticing a room turn | Not built yet |
| **Propensity** | The probability the policy had of choosing the action it took. Logging it lets us judge a different policy from old logs later. A deterministic policy like LinUCB gives 0 or 1, which leaves nothing to reweight | Not logged yet |
| **Off-policy evaluation (OPE)** | Estimating how a new policy would have done using logs from an older one, without running the new one live | Li et al. 2011 replay; study guide section 2.8 |
| **UCB (optimism)** | Pick the arm with the highest plausible value: estimate + uncertainty bonus that shrinks with plays (UCB1: √(2 ln t / n) for rewards in [0, 1]). LinUCB is its linear-feature version | `LinUCB` |

Note: **the room's truth vs the director's prior.** `sim/audience.yaml` is the *simulated
room's hidden truth* (what the audience actually likes; it was called `priors.yaml` before
2026-10-28). A *prior* is only ever the *director's starting guess* about that truth. They're
deliberately separate: a good director must do well even when its guess is wrong.

## Judge: ranking candidate jokes

| Term | Meaning here | In code |
|---|---|---|
| **Pairwise comparison** | One judge call: two candidate jokes for the same prompt, "which is funnier?" | `judge.Comparison` |
| **Bradley–Terry (BT)** | Model where each joke has a strength θ and P(i beats j) = σ(θ_i − θ_j); fitted from many comparisons, with α virtual wins/losses per joke as a prior | `judge.fit_bt` |
| **Position bias** | A judge preferring whichever joke it reads first (or second). Cancelled by a coin flip per call for which joke is shown first, recorded as `shown_first` | `judge/rank.py: _ask` |
| **Borda score** | A joke's average win rate over the opponents it has played; a model-free check on θ | `judge.bt.borda_scores` |
| **Cut-off** | A boundary to get right, e.g. "who is in the top 5" (m = 5) or "who is in the bottom 5" (m = n − 5) | `top_cutoffs`, `bottom_cutoffs` |
| **LUCB** | Pair-selection rule: at each cut-off, compare the weakest-looking insider with the strongest-looking outsider (by confidence bounds θ ± β·se) where they overlap most; stop when every cut-off is **separated** (no overlap) | `judge.select.adaptive_next` |

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
