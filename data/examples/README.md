# data/examples/

Samples of the two hand-labeling tasks, for the report and slides. The real work files stay
local: transcripts are copyrighted specials, and rJokes is under Reddit's terms (quote a few rows,
never the dataset; decisions.md 2026-10-31).

| File | Task | What it is |
|---|---|---|
| `transcript_bit_marking.txt` | Bit boundaries in stand-up transcripts | A work file as you edit it: numbered lines, laugh tags `⟨L2⟩`, your `>>>` / `@callback` marks |
| `transcript_bit_labels.yaml` | ″ | What `python -m pipeline.labels import` turns those marks into (the format of the committed `data/labels/<set>.yaml`) |
| `rjokes_hand_check_examples.csv` | Checking the R2 judge's mode labels | 8 of the 105 hand-checked jokes, one or more per finding |

## Transcript bit marking

**The lines are invented** (a 19-line set in the robot's voice), because real transcript text can't
be committed. The marks, header and YAML are the real format: the YAML was produced by running the
real importer on this file. The real labels for Nate Bargatze's *Hello World* (86 bit starts over
951 lines, no text) are in [`../labels/nate-bargatze_hello-world_2023.yaml`](../labels/nate-bargatze_hello-world_2023.yaml).

The example covers each mark once:

| Line | Mark | Meaning |
|---|---|---|
| 0 | (none) | The first line always starts a bit |
| 3 | `>>> online dating` | New bit, with a note |
| 6 | `>>> continues: online dating` | New bit on the same topic: a thread |
| 8 | `>>> airports` | New topic |
| 12 | `@callback 0002: the toaster therapist` | One line inside a bit that refers back; not a new bit |
| 13 | `>>> continues: airports, on the plane` | Thread continues |
| 15 | `>>> crowd work: accountant` | Crowd work is always its own bit |
| 18 | `>>> callback 0003: online dating` | Callback bit: returns to an earlier bit after other material |

Used for: the reference that S4's predicted bit boundaries are scored against (Pk / WindowDiff, S6).

## rJokes hand check

Columns: `judge_mode` (Qwen2.5-7B's label), `verdict` (y = agree, n = disagree, ? = joke didn't
land, no mode assignable), `human_mode` (the mode you'd give it), `what_it_shows`, `upvote_pct`
(percentile within its year). Text is verbatim from r/Jokes via
[rJokes](https://github.com/orionw/rJokesData). Full results: notebook Exp 9 and
[metrics.md](../../docs/metrics.md#hand-check-of-r2-labels-2026-10-31).
