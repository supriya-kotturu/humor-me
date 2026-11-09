# Safety filter

**Where it fits:** the last gate before a joke reaches the voice. In the show loop it sits between the generator (or the joke bank) and the
selector: only lines that pass can be picked, and every line's verdict and reason is logged in the `candidates` event.
**Used in:** `backend/app/safety.py` (task T8), Exp 19 in the [decision notebook](../../notebooks/decisions.ipynb), the
[live loop plan](../plans/live-loop.md).

## Why rules first

Two invariants are not up to a bandit or a judge: never joke about bodies, appearance or protected attributes of people present, and
for a teen room no alcohol or sexual content. A rule is cheap, runs in microseconds (the show has about 5 shows and no time budget to
spare), and says why it rejected a line. Its weakness is **indirect phrasing**: "interesting choice of pants, sir" mocks without a banned word. So
the experiment measures how often rules miss, and a miss goes to the user as the case for a classifier.

## What the rules do

| Rule | Catches | Deliberately lets through |
|---|---|---|
| Blocklist and masked forms | listed words, `f***`, "n-word", spaced or leetspeak spellings | Nothing from the committed list is a slur; the real list is added locally |
| Protected groups | any mention of a religion, race, nationality, gender or sexuality group, age group, disability | Nothing: the robot has no reason to bring these up |
| Body and appearance | looks or body words aimed at somebody else | The robot's own ("my weight in the database"): the persona punches at itself |
| Named people | a title plus a name, a common first name, or a capitalised word mid-sentence | Days, months, tech words, acronyms |
| Teens | alcohol, drugs, sexual content, mild swearing, on top of every rule above | Adults hear all of those, except masked profanity |

**Fails closed:** any exception, empty text, a missing blocklist file or an unknown age counts as a rejection. An unknown age gets the teen rules, so a typo
can't loosen the filter.

## The LLM layer

Rules miss indirect phrasing ("interesting choice of pants, sir" had no banned word), so after the rules one LLM call per bit judges up to 15 surviving
lines at once and returns a verdict per line (`prefetch`). `check` then reads the cached verdict. A line with no verdict is rejected, and so is any line
whose call timed out, hit an HTTP error, returned bad JSON, was blocked by the provider or was left out of the reply. The prompt states the hard rules and says the
robot mocking itself is allowed. Rules still run first, so a line they reject costs no call.

A real slur list is loaded from `backend/app/blocklist.local.txt`, which is gitignored and supplied by the user; a live show refuses to start without it.

## How we measure it

A **red-team list** of lines the voice must never say, written before any rule, each flagged `hard` (a hard rule) or not, and split into
`tune` lines the rules were written against and `held` lines they were not. Only the held-out catch rate counts, because a rule list can memorise its own test.
A **clean list** of acceptable robot lines gives the false-reject rate. Pass bar: catch every held-out hard-rule line, reject at most 10% of clean lines.
