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
| **Oracle** | A policy that knows the room's hidden truth and always picks the best mode. The ceiling, not achievable |
| **Regret / laughs lost** | How many laughs a policy missed compared with the oracle. We report it in laughs (sum of levels) so it reads in audience terms |
| **Cold start** | The first 10 shows: performance on the prior alone, before much learning |
| **Experienced** | The last 50 shows of a career: performance after learning |
| **Choice quality** | Expected reward of the mode the director picked ÷ that of the best mode, for one bit. 1.0 = picked the best mode. Noise-free (unlike laughs), so it shows *decisions* directly |
| **Recovery curve** | Mean choice quality at each bit position relative to a shift. Climbing back after the shift = the director noticed; flat = it didn't |
| **Bits to recover** | After a shift, how many bits pass before the director first picks a (near-)best mode again (per-bit regret < 0.02 ≈ 0.3 laughs). Capped at 4, the bits left in the show |
| **95% CI** | `mean ± 1.96·sd/√n` over careers; two numbers whose CIs overlap aren't reliably different |
