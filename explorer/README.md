# Explorer (Northeastern HPC) jobs

GPU batch work only. Nothing on the laptop may depend on these jobs having run: the sim and
bandits start on placeholder priors.

Access is through the CS5180 course (Fall 2026). Course-only accounts and the student dir
are **deleted after final grades**: copy results back to the laptop after every run.

## Limits that shape how we use it (NURC course guide)

| Partition | Time default / max | Running jobs | RAM |
|---|---|---|---|
| `courses` (CPU) | 4 h / 24 h | 50 | 256 GB |
| `courses-gpu` | 4 h / 8 h | **1** | **12 GB** |

- A `courses-gpu` session idle for ~1 h gets an email, then is killed. Long runs go through
  `sbatch`, not an open notebook.
- 12 GB system RAM: fine for vLLM on a 7B model (weights stream to the GPU), not for loading
  all 400k jokes into pandas next to it. The script reads only the sample.
- One GPU job at a time: don't keep a JupyterLab GPU session open while a GPU batch job runs.
- GPU type varies by node: `sinfo -p courses-gpu --Format=nodes,cpus,gres,statecompact`.
  On a 16 GB card (T4/P100/V100-16) use `MODEL=Qwen/Qwen2.5-3B-Instruct`.

## One-time setup (CPU is enough)

OOD → Courses → JupyterLab, partition `courses`, then in its terminal:
```bash
ls -d /courses/CS5180.*/students/$USER       # find your dir; the folder suffix is the term code
COURSE_DIR=/courses/CS5180.<term>/students/$USER
cd $COURSE_DIR && git clone <repo url> humor-me
module avail python                          # pick a 3.10–3.12 if the default is older
python3 -m venv $COURSE_DIR/venv && source $COURSE_DIR/venv/bin/activate
pip install vllm                             # pulls a matching torch; ~10 min
```
Copy `data/processed/rjokes/train.jsonl` (gitignored; joke text never goes into git) to the
same path inside the clone: OOD Files → Upload, or `scp`.

## Runs

| Step | How | Output |
|---|---|---|
| Smoke test (interactive) | OOD JupyterLab on `courses-gpu`, terminal: `source explorer/env.sh && python explorer/label_modes.py --input data/processed/rjokes/train.jsonl --n 50 --out /tmp/smoke.jsonl`. **Then stop the session.** | Throughput, parse failures |
| Full labeling (batch) | `mkdir -p logs && sbatch explorer/label_modes.sbatch` (override: `N=5000 MODEL=... sbatch ...`) | `data/processed/rjokes/modes.jsonl` (ids + labels, no text) |
| Monitor / cancel | `squeue -u $USER`, `tail -f logs/humor-label-<id>.out`, `scancel <id>` | |
| **Interrupted?** (time limit, node failure, `scancel`) | Submit the **same** command again. It skips ids already in `modes.jsonl` (checkpointed every 2,000 jokes) and continues, on whatever node it gets | Same file, grows |
| **Is the sample big enough?** | Printed at the end of the job (`explorer/mode_precision.py`): per-mode 95% CI of the mean upvote percentile | Table; "extend" hint if any mode is wider than ±0.02 |
| **Extend the sample** | `N=60000 sbatch explorer/label_modes.sbatch` (same seed → labels the *next* 30k of the same shuffle) | Same file, grows |

**Precision rule (fixed before the first run):** label 30k; if any mode's 95% CI half-width on
mean percentile is wider than ±0.02, extend by 30k and recheck. Precision depends on jokes per
mode (±0.29/√n), not on the share of the 400k labeled.

## Storage: what survives where

Each job lands on **whatever node is free**: never assume the same machine twice. Everything that
must survive lives on the shared filesystems, which every node mounts:

| Location | Survives | Notes |
|---|---|---|
| `/courses/CS5180.<term>/students/$USER` | Across jobs, **until final grades** | Ours: repo, venv, `HF_HOME` model cache, `modes.jsonl` |
| `/home/$USER` | Permanent | 75 GB: too small for model weights |
| `/scratch/$USER` | **Purged monthly**, no backup | Not used |
| `/projects` | Permanent | Needs a research-group allocation; not available to course accounts |
| node-local `/tmp` | **Gone when the job ends** | Smoke-test output only |

No S3-style object storage. The permanent copy is the laptop + git: copy `modes.jsonl` back after
every run (ids and labels only, so it can be committed).

After the run, copy `modes.jsonl` back to the laptop, then spot-check 100 labels by hand
before the sim uses them (record agreement in dev-log.md).
