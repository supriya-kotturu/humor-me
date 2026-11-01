# Transcript Pipeline

Turns locally stored stand-up transcripts into **structure and reward data**: never into
text the model imitates (see decisions.md, 2026-10-20). All data lives in `data/` (gitignored).

## Overview

```mermaid
%%{init: {"theme": "default"}}%%
flowchart TD
    RAW[/"data/raw/*.pdf<br/>19 specials"/] --> S0["S0 Ingest<br/>PDF → text, strip site boilerplate"]
    S0 --> S1["S1 Extract markers<br/>[...] / (...) → events with offsets"]
    LEX[("markers.yaml<br/>versioned lexicon")] --> S2
    S1 --> S2{"S2 Classify marker<br/>unknown string?"}
    S2 -- yes --> FAIL(["STOP: add mapping to lexicon"])
    S2 -- no --> S2b["audience · performer · crowd-work person<br/>level 0–4 · groan/gasp flags<br/>drop entrance/exit cheers"]
    S2b --> S3["S3 Beats<br/>text span → following reaction"]
    S3 --> S4["S4 Bits<br/>TextTiling candidates → LLM refine"]
    S4 --> S5["S5 Annotate bits (LLM)<br/>premise · mode · techniques · type<br/>callback_to · segue · position"]
    S5 --> S6{"S6 Validate<br/>Pk/WindowDiff · marker spot-check"}
    S6 --> OUT[/"data/processed/<br/>beats.jsonl · bits.jsonl · sets.jsonl"/]
    S0 --> NG[("n-gram index<br/>memorization guard")]

    OUT --> C1["Laugh predictor<br/>(reward model)"]
    OUT --> C2["Planner demos<br/>(bit sequences)"]
    OUT --> C3["Bit skeletons →<br/>synthetic robot-voice sets → SFT"]
    OUT --> C4["Sim audience priors<br/>(laugh density, timing)"]
```

## How the outputs feed the agent

```mermaid
%%{init: {"theme": "default"}}%%
flowchart LR
    subgraph Offline
        BEATS[beats.jsonl] --> LP[Laugh predictor]
        BITS[bits.jsonl] --> SK[Bit skeletons]
        SETS[sets.jsonl] --> BC[Planner prior<br/>behavior cloning]
        SK --> SYN[Synthetic sets<br/>robot voice] --> RANK{Rank by LP} --> SFT[SFT generator]
        LP --> SIM[Sim audience reward]
    end
    subgraph Live
        ROOM[Room state<br/>context · bit history · laugh trend<br/>open callbacks · elapsed] --> PL[RL planner<br/>next bit spec]
        PL --> GEN[Generator<br/>writes next bit] --> GUARD{n-gram guard<br/>+ safety filter} --> TTS[TTS → waveform]
        MIC[Mic laugh level] --> ROOM
    end
    BC --> PL
    SIM -. trains .-> PL
    SFT --> GEN
```

## Stages

### S0: Ingest
- `pymupdf` text extraction → `data/interim/<slug>.txt`. Slug from filename: `gulman_great-depresh_2019`.
- Strip print chrome (outer page margins), then take the post region between the
  "Stand-Up Transcripts" breadcrumb and the comment section.
- Trim each set by **explicit counts in `data/body_overrides.yaml`** (`skip_leading`,
  `skip_trailing`). Every set must have an entry; a missing one stops the run.
  `--suggest` proposes entries for new sets via heuristics ("Transcript" heading →
  reaction-free `* * *` divider → blurb/title match) for human review.
- Build the **n-gram index** (e.g. 8-grams) of transcript text for the generation-time
  memorization guard. Any generated line overlapping a transcript 8-gram is rejected.

### S1: Extract markers
- Regex `[\[\(]([^\]\)]{3,40})[\]\)]` → event `{offset, raw}`. Remove from the text stream,
  keep offsets so beats can be aligned.

### S2: Classify markers
Lookup in `data/markers.yaml` (lexicon is committed; it holds no transcript text).
Only **reaction-like** strings (laugh, chuckle, applause, cheer, groan, gasp…) need an entry;
an **unknown reaction string stops the pipeline** and lists what's unmapped. A laugh level is
never guessed. `python -m pipeline.markers --draft` proposes entries for human review.
The other ~300 stage directions are bucketed by keyword: `act_out` · `music` · `direction`.

Review notes: performer first names ("jeff laughs", "tom chuckles") are the **performer**,
not crowd-work people; "… continues" extends the preceding **audience** reaction.

| Field | Values |
|---|---|
| `source` | `audience` · `performer` (unattributed chuckles/sighs: subtitle convention) · `person:<name>` (crowd-work turn) |
| `level` | 0 none · 1 chuckle · 2 laugh · 3 big laugh · 4 applause break |
| `flags` | `groan`, `gasp` (separate channel, not ordinal) |

Level rules:
- **1**: audience-attributed chuckle, "scattered".
- **2**: plain laugh ("laughter", "audience laughs").
- **3**: laugh + intensity modifier ("roars", "wildly", "loudly", "continues"), or ≥ 2
  consecutive audience laugh markers.
- **4**: laugh + applause/cheer **inside the set**.
- Cheers/applause in the first/last ~2% of the set = entrance/exit → excluded.

### S3: Beats
- Beat = text span between consecutive audience reaction events (or sentence groups if a
  span exceeds N words).
- Label = level of the reaction that follows (0 if none), plus flags and words-to-reaction.
- Mid-sentence markers flagged `interrupt=true` (the laugh landed before the line finished).

### S4: Bits
- Candidate boundaries: TextTiling-style cosine drop between sentence-window embeddings.
- LLM refines: confirms/moves boundaries.
- Rules: **a bit = one premise**; tags stay inside the bit; a callback is a new bit with
  `callback_to`; crowd-work exchanges are their own bit (`type=crowd_work`).

### S5: Annotate bits (LLM)
Callback detection (two passes): (1) candidates without an LLM: rare n-grams/entities
(low corpus frequency) reappearing in a bit ≥ 2 bits later; (2) LLM confirms → `callback_to`
on the later bit, `hook` (paraphrased image) on the earlier one. Report stats: callback vs
first-time laugh level, reach distance, closer payoffs, whether the hook had landed.
These fit the sim audience's callback parameters (see design.md, Callbacks).

Per bit: `hook`, `premise`, `mode` (HumorGen six), `techniques`, `type`
(`prepared | crowd_work | callback | segue`), `callback_to`, `segue_from_prev`.
`position` (`opener | body | closer`) is assigned by index in the set, not by the LLM.
Annotations are **abstract**: premises are paraphrased descriptions, not quoted lines.

### S6: Validate
- Bit boundaries: hand-label 2 sets → **Pk / WindowDiff**.
- Markers: 50 hand-labeled markers stratified by level → agreement with lexicon mapping.
- Annotations: spot-check 30 bits for mode/premise.

## Schemas

`beats.jsonl`
```json
{"set_id": "...", "beat_idx": 12, "bit_idx": 3, "span": [4120, 4388], "words": 47,
 "level": 2, "flags": [], "interrupt": false, "source": "audience"}
```

`lines.jsonl` (laugh predictor input)
```json
{"set_id": "...", "line_idx": 40, "beat_idx": 12, "span": [4301, 4388], "text": "...",
 "words": 9, "label": 2, "flags": [], "interrupt": false, "context": "<150 words before>"}
```

`bits.jsonl`
```json
{"set_id": "...", "bit_idx": 3, "beats": [10, 14], "position": "body",
 "type": "prepared", "mode": "Neurotic", "premise": "paraphrased premise",
 "techniques": ["escalation", "act_out"], "tags": 3, "callback_to": [],
 "segue_from_prev": "topic_shift", "max_level": 3, "laugh_rate": 0.6}
```

`sets.jsonl`
```json
{"set_id": "...", "comedian": "...", "year": 2023, "has_markers": true,
 "bits": [0, 1, 2], "split": "train"}
```

## Labels and splits

- **Binary laugh** (level ≥ 2): primary target; robust across transcribers.
- **Ordinal level**: trusted **within a set only**; used as a ranking target.
- **Groan**: separate flag; mode-dependent reward (positive for Wordsmith puns).
- **Density gate**: a set is `laugh_labeled` only if words per audience laugh ≤ 150. Sparse
  transcribers leave most laughs untagged, so missing markers would read as "no laugh".
  Observed: dense sets 26–80 words/laugh; sparse 480+.
  - Labeled (11): Regan, Slay, ISMO, Arcuri, Gaffigan, Morgan, Birbiglia, Bargatze, Chieng,
    Morril, Maniscalco.
  - Structure-only (9): Martin, Bamford, Hamilton, Notaro, Papa (sparse) + Gulman, Seinfeld,
    McIntyre, Meyers (unmarked).
- Levels 1 and 3 are rare (most transcribers tag neither), so the ordinal signal is thin;
  binary stays primary.
- Laugh predictor split: **hold out whole comedians**: Mike Birbiglia (Neurotic, replaces
  Bamford, who is sparse), Jim Gaffigan (Observer), Nate Bargatze (Optimist). That leaves
  **8 training sets**; learning curve at 4/6/8 decides whether more dense transcripts are needed.

## Layout

```text
data/
  raw/          PDFs (gitignored)
  interim/      extracted text, marker events (gitignored)
  processed/    (gitignored)
    datasets.json        membership: laugh_labeled vs structure_only
    laugh_labeled/       beats/bits/sets jsonl with laugh labels
    structure_only/      bits/sets jsonl, no laugh labels
    ngrams.*             memorization guard index
  markers.yaml  lexicon (committed)
pipeline/       code (committed)
```
