# Pipeline wiki

Concept pages behind the transcript pipeline: what each technique is, why we use it, and
exactly how it's applied here. Read in order if you're new; each page links to the code.

| # | Page | Stage | One-line summary |
|---|---|---|---|
| 1 | [PDF extraction](pdf-extraction.md) | S0 | Getting clean transcript text out of browser-printed PDFs |
| 2 | [Reaction markers & laugh labels](laugh-labels.md) | S1–S2 | Turning `[audience laughs]` into consistent 0–4 labels across transcribers |
| 3 | [Beats, bits, sets](beats-bits-sets.md) | S3, S5 | The three units of stand-up structure and how we label them |
| 4 | [TextTiling](texttiling.md) | S4 | Finding where one bit ends and the next begins |
| 5 | [Segmentation evaluation](segmentation-evaluation.md) | S6 | Pk and WindowDiff: scoring bit boundaries against hand labels |
| 6 | [Memorization guard](memorization-guard.md) | generation | Stopping the agent from repeating comedians' lines |

How it all ends up as clean, labeled transcripts:

```mermaid
%%{init: {"theme": "default"}}%%
flowchart TD
    PDF["Printed web page (PDF)"] -->|"1 · margins, paragraphs, trims"| BODY["Performance text<br/>with inline markers"]
    BODY -->|"2 · lexicon → levels, density gate"| EV["Clean text + reaction events"]
    EV -->|"3 · split at reactions"| BEATS["Beats with laugh labels"]
    BEATS -->|"4 · TextTiling + LLM"| BITS["Bits (one premise each)"]
    BITS -->|"3 · LLM annotation"| SETS["Annotated sets"]
    SETS -->|"5 · Pk / WindowDiff, spot checks"| OK{"Good enough?"}
    OK -->|yes| USE["Reward model · planner demos · skeletons · sim priors"]

    classDef data fill:#fff4c2,stroke:#b8860b,color:#000
    classDef model fill:#e8dcf8,stroke:#6a1b9a,color:#000
    classDef infra fill:#d6e9fb,stroke:#1565c0,color:#000
    classDef rl fill:#d4f4dd,stroke:#2e7d32,color:#000
    class PDF,BODY,EV,BEATS data
    class BITS,SETS model
    class OK infra
    class USE rl
```
