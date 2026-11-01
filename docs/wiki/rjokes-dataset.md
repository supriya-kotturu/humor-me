# rJokes dataset

**Where it fits:** the working joke dataset for M1, replacing transcripts as the main data
source (decisions.md 2026-10-26).
**Used in:** `pipeline/rjokes.py` (prep, R1), `explorer/label_modes.py` (mode + safety labels,
R2), sim priors (PR 3), a learned joke scorer (stretch).

## What it is

~570k posts from Reddit's r/Jokes (2008–2019) with upvote-based scores. The official splits
hold 432k jokes with an integer score 0–11 (≈ ln upvotes). One-liners and setup/punch jokes,
Reddit taste, much offensive material. Terms: Reddit's, so use locally and never redistribute the dataset; quoting a few rows in docs is fine (decisions.md 2026-10-31).

## What prep had to fix

| Problem | Fix | Why it matters |
|---|---|---|
| Reposts (6,971 texts) spread across train and test | First split in train→dev→test owns the text | Otherwise a scorer is tested on jokes it trained on |
| Same text, scores ~7× apart in upvotes (median spread 2) | Keep max score, `n_posts`, `score_spread` | That spread is the **noise ceiling**: no scorer can beat it |
| Subreddit grew ~1000× over 11 years | `pct` = percentile within the year | A score of 5 in 2010 ≠ 5 in 2019 |
| Offensive content | Coarse topic flags now; LLM safety labels in R2 | Invariant 1 and the teens filter |

Result: 402,981 jokes (train 323,573 / dev 39,764 / test 39,644).

## What it can and can't tell us

- **Can:** how good each *mode* is in general (mode-labeled score distributions → sim `base`),
  and a rough prior for ranking candidate jokes.
- **Can't:** venue preferences (no venue), timing, or stand-up structure. Upvotes measure
  Reddit popularity, not a room's laughter.

## Further reading

- [Weller & Seppi, The rJokes Dataset (LREC 2020)](https://aclanthology.org/2020.lrec-1.753/):
  dataset construction and humor-prediction baselines.
- [orionw/rJokesData on GitHub](https://github.com/orionw/rJokesData): files and terms.
