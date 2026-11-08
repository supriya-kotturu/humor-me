# humor-me: a stand-up agent that reads the room (report)

Status: **outline, partly backed** (results from #1 and #4 are in). Each section lists the claim
it must support and where the evidence comes from: a row in [metrics.md](metrics.md), a decision,
or a wiki page. Terms (bit, mode, prior, regret, …) are used as defined in the
[glossary](wiki/glossary.md); the background behind each section is in the
[project wiki](wiki/README.md), whose pages follow the same order as this story. A section is "done" when
every claim cites a number from a merged PR. Sections marked *(user)* are written by the
author in their own voice; the rest are drafted by the AI (Claude) and edited by the author.

## 1. Overview *(user)*

LLM can do almost anything. Models are intelligent enough to reason, implement, plan a trip, write code and many more. There's something that's still relevant and where we humans prefer a human touch. Writing, Illustration, Design anything that has a "taste". You can argue that LLM does generate joke, what's different? Give it a list of jokes, or add a RAG and it will fetch a similar joke. But that restricts humor to known set of stale one liner jokes. This is also one of the reasons that though LLM can write stories, but readers notice. In 2026 two novels were pulled by their publishers after readers flagged the prose as machine-like ([1](https://www.nytimes.com/2026/03/19/books/shy-girl-book-ai.html), [2](https://www.chroniclejournal.com/news/national/quebec/publisher-cuts-ties-with-quebec-author-after-saying-it-found-evidence-of-ai-use/article_41f60777-6d02-5670-8ce7-aee3546f7bee.html)). Both authors deny it and AI detectors are unreliable, but people sensed something was off before anyone proved anything. We can easily make out the monotonous language and the lack of depth. I think humor is something very close to this.

LLM lacks taste, Humor needs it. If we observe, the way LLMs generate jokes is sometimes just retrieval. Same list of dad jokes, that fall flat. But, how does a comedian acquire this taste? What sets them apart from a generic one-liner punchlines? How can they come up with "new bits/jokes" spontaneously? That's the million dollar question this project is trying to find.

This repo tries to relate comic generation with RL topics and see if we can mimic a stand-up comedian on a set. The goal is ambitious and this is just an exploratory interest. I believe LLMs can't do stand up. At least, not any good ones. I'm just trying to find evidence how bad it can get.

A new stand-up comedian is not a writer telling a lengthy story. They write [bits](wiki/glossary.md) - short relevant scenes that are funny. They collect these, perform at couple of venues, measure which bits did well, and which ones fell flat. It's an iterative process.

That loop is reinforcement learning. A comedian tries a bit (an action), the room laughs or
doesn't (a reward), and next time they choose differently. They don't learn it in one night
either: it takes many shows, in many rooms, to know that the cynical stuff kills at a bar and
dies at a company party. Taste, at least part of it, is what that loop leaves behind.

So that's what this project builds: a voice-only stand-up agent, a slightly anxious robot,
that performs a set one bit at a time. Before each bit a **director** picks one of six comedic
**modes** (Neurotic, Cynic, Observer, Wordsmith, Optimist, Absurdist, from
[HumorGen](wiki/llm-humor-vs-standup.md)) based on the room (venue, energy, age) and how the
last bit landed, and it carries what it learned into the next show
([glossary](wiki/glossary.md)). The question: **does adapting to the room beat performing one
fixed persona?**

In [simulation](wiki/simulated-audience.md), it does. After a few hundred shows the adaptive
director ([LinUCB](wiki/contextual-bandits.md)) gets 97% of the laughs of a director that knows
exactly what the room likes, and 18% more than always performing as Neurotic, the persona an
LLM judge rates funniest. What made the difference wasn't the starting guess; it was carrying
lessons from one room to similar rooms ([results](metrics.md)).

That's not proof an LLM can do stand-up. The simulated room's tastes are made up, and the
director only picks the attitude; it doesn't make a joke any funnier. What it shows is that the
comedian's loop, the part that builds taste, can be learned. Whether the jokes underneath are
any good, and how bad it gets in front of real people, is what the live sessions are for.

## 2. Why LLM jokes differ from a stand-up comedian

- Claim: one-shot LLM humor is generic, deaf to the room, untimed and unstructured.
- Evidence: HumorGen mode collapse and persona win rates; rJokes noise ceiling (repost spread).
- Wiki: [LLM humor vs stand-up](wiki/llm-humor-vs-standup.md) (the five gaps, HumorGen, LoL);
  [rJokes dataset](wiki/rjokes-dataset.md) (why upvotes cap how well any joke scorer can do).

## 3. How a comedian writes, and how that maps onto an agent

- Premise → setup/punch → tags → bit → set → callback.
- Mapping: bit ↔ mode choice; tag-or-move-on ↔ director; callback ↔ delayed reward;
  show-to-show rewriting ↔ learning across episodes.
- Evidence: transcript pipeline (bits F1 0.80 vs 0.21 baseline on the tuning set; held-out
  Birbiglia score if run).
- Wiki: [Stand-up structure](wiki/standup-structure.md) (comedian's decisions ↔ agent's decisions);
  [Glossary: comedy units](wiki/glossary.md#comedy-units-of-material-smallest-to-largest) (line → joke → bit → set,
  with Bargatze bit sizes); pipeline background in [beats, bits, sets](../pipeline/wiki/beats-bits-sets.md).

## 4. From bits to a contextual bandit (and where it stops being one)

- The director picks a **mode once per bit** (not per joke): a comedian commits to an attitude
  for a whole bit and varies between bits. Per joke, the sim rewarded rotation over fit
  (decisions.md 2026-10-27; metrics.md #1 v1 vs v2).
- Arms = 6 modes; context = venue/energy/age + last bit's mode and reward; reward = mean laugh
  level of the bit's jokes ÷ 3.
- Why a bandit first; why fatigue and callbacks push toward an MDP.
- Wiki: [Contextual bandits](wiki/contextual-bandits.md) ("where the bandit view breaks" is this
  section's argument); [Glossary: learning](wiki/glossary.md#learning-how-the-director-chooses).

## 5. Method

5.1 Simulated audience: the room's hidden truth (`sim/audience.yaml`, made up except `base`,
    stated plainly), 8 bits × 5 jokes, fatigue, novelty, exact oracle.
    Wiki: [Simulated audience](wiki/simulated-audience.md).
5.2 Policies: fixed, random, ε-greedy, LinUCB, Thompson (LinTS). Director's prior: HumorGen,
    uniform, or wrong (shuffled). No switching penalty: per-bit decisions keep the voice coherent.
    Wiki: [Contextual bandits](wiki/contextual-bandits.md); [Glossary: prior](wiki/glossary.md#learning-how-the-director-chooses).
5.3 Long-form generation: set planner → bit writer → director.
    Wiki: [Stand-up structure](wiki/standup-structure.md); design: [long-form set generation](design.md#long-form-set-generation).
5.4 Data: rJokes prep and mode labels; what transcripts were (and weren't) used for.
    Wiki: [rJokes dataset](wiki/rjokes-dataset.md); [Explorer HPC](wiki/explorer-hpc.md) (where labeling runs);
    pipeline: [pipeline wiki](../pipeline/wiki/README.md).
5.5 Live system: voice loop, laugh detection, operator fallback, safety filter.
    Wiki: not yet (pages added when built, weeks 3–5).

## 6. Key decisions and trade-offs

One row per decision that changed the result (to become D-xx entries: options, trade-off, choice).

| Decision | Status | Evidence | Background |
|---|---|---|---|
| Pause transcripts; rJokes + simulator first | Decided 2026-10-26 | decisions.md | [rJokes](wiki/rjokes-dataset.md) |
| Long-form sets planned hierarchically (planner → bit writer → director) | Decided 2026-10-26 | decisions.md | [Stand-up structure](wiki/standup-structure.md) |
| Reward = laugh level 0–3, scaled to [0, 1] | Decided 2026-10-27 | sim/README.md | [Glossary: laugh level](wiki/glossary.md#comedy-units-of-material-smallest-to-largest) |
| Director decides per bit, not per joke | Decided 2026-10-27 | metrics.md #1 v1 vs v2 | [Simulated audience](wiki/simulated-audience.md) |
| Director policy: LinUCB + uniform prior | Decided 2026-10-27 | metrics.md #4 | [Contextual bandits](wiki/contextual-bandits.md) |
| Learn across shows (one show is too short) | Decided 2026-10-27 | metrics.md #1: 14–177 shows to separate top modes | [Contextual bandits](wiki/contextual-bandits.md) |
| Explorer for GPU jobs only | Decided 2026-10-26 | decisions.md | [Explorer HPC](wiki/explorer-hpc.md) |
| rJokes for mode statistics only, never joke material (judge safety flags unreliable) | Decided 2026-10-31 | notebook Exp 9; metrics.md "Hand check of R2 labels" | [rJokes](wiki/rjokes-dataset.md) |
| Human mode preference: compare same-year, similar-length jokes; keep both tastes for the sim | Decided 2026-11-01 | notebook Exp 10; metrics.md "Human vs judge mode ranking" | [rJokes](wiki/rjokes-dataset.md) |

## 7. Results

How to read the numbers (laughs per show, laughs lost vs oracle, cold start, CI):
[Glossary: evaluation](wiki/glossary.md#evaluation-how-we-measure). Career figure:
[laughs lost across shows](../reports/figures/laughs_lost_across_shows.png).

| Experiment | Metric | Status | Headline so far |
|---|---|---|---|
| Is there anything to learn? | Oracle vs best fixed vs random, laughs/show | Done (#1) | Oracle 47–50, best fixed 43–47, random 38–41 |
| Policies × priors across a career | Laughs lost vs oracle over 300 shows; cold start; experienced | Done (#4) | LinUCB: 97% of oracle, +18% vs always-Neurotic; ε-greedy loses 3.4× more; prior makes no significant difference |
| Mid-set preference shift | Laughs after the shift (% of oracle); bits to recover | Done (shift PR) | Learners fall from 95% to 80–83% of oracle and recover no faster than random; a bandit reads rooms, not turns → needs show state (MDP) |
| Set length vs shift | Choice quality after the shift, 8–52-bit sets | Done (notebook Exp 6) | Longer sets don't help: quality stays ~0.15 below steady even 26 bits after the turn |
| Show state A: tonight's record as features | Choice quality after shift / in steady shows | Done (notebook Exp 7) | Fails pre-set criteria (+0.01–0.05 after a turn, −0.01–0.03 when steady): hedges, doesn't read the turn |
| Show state B: career + tonight's correction (κ sweep) | Choice quality after shift / steady; repeat and hurt-pick rates | Done (notebook Exp 8) | No κ passes: win-stay meets fatigue (repeats 29% → 54%) and untried modes are never tested → C needs in-show exploration + fatigue planning |
| Callbacks: greedy vs sequential | Late-set laugh rate, laughs lost | Planned (explore-next #5) | |
| Prior strength sweep | Laughs lost vs prior weight | Planned (explore-next #7) | |
| Hand check of R2 judge labels | Mode precision vs 105 hand labels; safety-flag misses | Done (notebook Exp 9) | Modes 92% precise at the labeled mode mix, misses lean to puns; safety flags caught 0 of 16 hand-flagged jokes. Examples: [metrics.md](metrics.md#hand-check-of-r2-labels-2026-10-31) |
| Human vs LLM-judge mode ranking | Spearman ρ, side of the field, wordsmith last; rJokes upvotes vs HumorGen | Done (notebook Exp 10) | Agree only in part: neurotic up, wordsmith down, optimist clearly disagrees (ρ +0.20 to +0.37). Mode makes a small difference in upvotes (win rates 0.480–0.527), about half of length's effect; upvote noise means this doesn't show people care less about mode |
| rJokes correlation map | Spearman ρ of each feature with `pct`, raw and within year and length group | Done (notebook Exp 12) | No flag or safety label needs an extra adjustment, so the ranking stands. Length is a weak link (ρ +0.13) but halves wordsmith's raw link; a little length remains within groups |
| Room's `base` from human data | Change in policy ranking | Done (notebook Exp 13) | The director choice holds under human taste: 96% of oracle, clearly above the best fixed mode, not beaten by the other learners. Venue offsets are still made up, so this doesn't show how much learning helps in a real room |
| Oracle vs planning optimum | Expected laughs per room, exact; share of each | Done (notebook Exp 16) | The oracle isn't the ceiling: planning scores 50.03 vs 49.20. LinUCB is 97% of the oracle, about 95% of the true best |
| Long-form sets | Word budget, closer present, planned callbacks used | Planned | |
| Demo voice | Time to render a line vs audio length | First slice done | 1.9–3.7 s per line for 5–8 s of audio on the CPU: faster than real time, but lines must be prepared ahead to meet the 1 s target |
| Live pilots | Detector vs operator agreement; episode rating | Weeks 5–7 | |

Dropped: switching-penalty ablation (per-bit decisions made it unnecessary).

## 8. Limitations

Made-up audience preferences; Reddit taste ≠ a room; laugh level as a noisy proxy; tiny live
sample; LLM judge ≠ humans (HumorGen: 58.3% judge–human agreement).
Background: [Simulated audience](wiki/simulated-audience.md) (what the sim can't answer);
[rJokes](wiki/rjokes-dataset.md) ("what it can and can't tell us").

## 9. Future work

Learned reward model from logged sessions; DPO on the generator; judge-as-audience (idea B);
H1 live experiment with enough sessions for confidence intervals.

## 10. Acknowledgements and AI use

Stand-up transcripts came from Scraps from the Loft (scrapsfromtheloft.com), used with the site admin's permission as a local research corpus for studying comedic structure. They were not redistributed or published.


Overview: paragraphs 1–4 written by the author; paragraphs 5–8 (RL loop → what was built →
result → limits) drafted by the AI (Claude) at the author's request, then edited by the author.
State exactly what the AI (Claude) wrote: code (by module), docs, wiki pages, report sections,
and what the author wrote or decided. Cite datasets (rJokes), papers (HumorGen, LoL,
Christiano et al. 2017, LinUCB, Thompson tutorial) and the Explorer cluster (NURC).

## References

Numbered in order of first use. Notes say what each source does and doesn't support.

1. Alter, A. (2026, March 19). Horror novel "Shy Girl" canceled over suspected A.I. use.
   *The New York Times*. https://www.nytimes.com/2026/03/19/books/shy-girl-book-ai.html
   Note: readers flagged the prose first (Reddit, Jan 2026; a YouTube essay with 1.2M views);
   Hachette cancelled the US edition and discontinued the UK one. The author denies using AI
   (blames a freelance editor); the AI-detector evidence was later disputed. Article not read
   directly (paywall); details cross-checked with Wikipedia ("Shy Girl") and Publishers
   Marketplace (2026-03).
2. Glorieux-Stryckman, M. (2026, October 21). Publisher cuts ties with Quebec author after
   saying it found evidence of AI use. *The Canadian Press* (via The Chronicle-Journal).
   https://www.chroniclejournal.com/news/national/quebec/publisher-cuts-ties-with-quebec-author-after-saying-it-found-evidence-of-ai-use/article_41f60777-6d02-5670-8ce7-aee3546f7bee.html
   Note: suspicion began on social media; Boréal cited document metadata and an external
   manuscript analysis, won't reprint. The author denies it; the publisher says it "will likely
   never have a definitive answer."

