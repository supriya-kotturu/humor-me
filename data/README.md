# data/

Transcript corpus and the config that controls how it's processed. **Transcript text never
leaves this machine**: `raw/`, `interim/` and `processed/` are gitignored. Only the two YAML
files are committed, and they hold counts and marker strings, no transcript text.

| Path | Committed | Written by | Contents |
|---|---|---|---|
| `raw/` | no | you (manual download) | Transcript PDFs, browser "Save as PDF" of the site pages |
| `body_overrides.yaml` | **yes** | `ingest --suggest` + human review | Per-set `skip_leading` / `skip_trailing` paragraph counts |
| `markers.yaml` | **yes** | `markers --draft` + human review | Reaction string → `kind`, `source`, `intense` |
| `interim/` | no | `pipeline.ingest`, `pipeline.markers` | `<set>.body.txt`, `.clean.txt`, `.events.jsonl`, `.meta.json` |
| `processed/datasets.json` | no | `pipeline.markers` | Which sets belong to which dataset (see below) |
| `processed/laugh_labeled/` | no | S3 (`beats`, `lines`); S4+ planned (`bits`, `sets`) | jsonl with laugh labels |
| `processed/structure_only/` | no | S3 (`beats`, `lines`); S4+ planned (`bits`, `sets`) | jsonl, labels `null` |

## Hand labels (`labels/`)

| Path | Committed | Contents |
|---|---|---|
| `labels/work/<set>.txt` | no | Numbered lines with laugh tags; you type `>>>` above each bit start. Never overwritten. |
| `labels/<set>.yaml` | **yes** | Bit-start line indices + your notes, no transcript text |
| `labels/rjokes_hand_check.csv` | **yes** | 105 rJokes ids: judge mode/flags vs your verdict, mode and safety flags, no joke text |

Marks: `>>>` new bit · `>>> continues: note` same topic as the previous bit (thread) · `>>> continues callback NNNN: note` both · `>>> callback NNNN: note` callback bit (starts at its setup; NNNN = earlier bit's start) · `@callback NNNN: note` one line inside a bit referring back (NNNN = the earlier line). Test: delete the reference; still a bit? yes → `@callback`, no → `>>> callback`.

`python -m pipeline.labels template <set>` → edit → `python -m pipeline.labels import <set>`.

## Examples for the report (`examples/`)

Small, committed samples that show *how* the hand labeling was done, since the real work files
can't be published (transcripts: copyright; rJokes: Reddit terms, decisions.md 2026-10-31).
See [examples/README.md](examples/README.md).

## Two datasets that grow independently

All PDFs go through one pipeline; membership is decided at the end of `pipeline.markers`:

| Dataset | Rule | Used for |
|---|---|---|
| `laugh_labeled` | ≤ 150 words per tagged audience laugh | laugh predictor (reward model), sim timing priors, plus everything below |
| `structure_only` | sparse or no reaction tags | bit structure, set arcs, skeletons only, **never** laugh labels |

A missing marker in a sparse transcript doesn't mean "no laugh", so the laugh predictor reads
only `processed/laugh_labeled/`. To override the gate for one set, add `dataset:` to its entry
in `body_overrides.yaml`. Each set's `meta.json` records `dataset` and `dataset_source`
(`density_gate` or `override`).

## Adding a transcript

1. Save the page as PDF into `raw/`. Keep the browser's default filename: `set_id` is parsed
   from `Comedian_ Title (Year) …pdf`.
2. `.venv/Scripts/python -m pipeline.ingest --suggest`, then open `body_overrides.yaml`.
   The comment shows which rule proposed the counts; `hit cap: verify` means check by hand.
3. `.venv/Scripts/python -m pipeline.ingest`, then open `interim/<set>.body.txt`. The first
   line should be the performance, the last the closer. Adjust the counts if not.
4. `.venv/Scripts/python -m pipeline.markers`. If it stops, run `--draft`, review the proposed
   entries (watch for the performer's own name, e.g. "jeff laughs" = performer), add them.
5. Check `interim/<set>.meta.json` → `laugh_labeled`. Only dense transcripts help the reward model.

## File formats in `interim/`

| File | Format |
|---|---|
| `.body.txt` | One paragraph per line, markers still inline |
| `.clean.txt` | Same text with markers removed; event offsets index into this |
| `.events.jsonl` | One marker per line: `offset`, `raw`, `category`, `kind`, `source`, `level`, `flags`, `interrupt`, `rel_pos` |
| `.meta.json` | `set_id`, comedian, title, year, trims, word counts, `audience_laughs`, `words_per_laugh`, `laugh_labeled` |

Everything in `interim/` and `processed/` is regenerable: delete and rerun the pipeline.
