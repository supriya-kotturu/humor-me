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
- 12 GB system RAM: enough for a 7B model (weights go to the GPU), not for loading all 400k
  jokes next to it. The script reads only the sample.
- One GPU job at a time: free an interactive GPU session before `sbatch`.
- **GPUs:** `courses-gpu` has **V100-SXM2 (32 GB)** and **P100 (12 GB)** nodes
  (`sinfo -p courses-gpu --Format=nodes,gres:40,statecompact`). Neither supports bfloat16, so
  models load in float16 (`--dtype half`). The job requests a V100.
- **Driver supports CUDA ≤ 12.3** (`found version 12030`), so PyTorch must be a **cu121** build.
  Current vLLM needs newer drivers and drops Volta, so the job uses plain `transformers`
  (`--backend transformers`, the default).
- **Login node (`explorer-0X`) is for light work only** (git, editing, `sbatch`): it killed a 1-minute
  data-prep step. Anything heavy runs where the prompt shows a compute node (`cNNNN`/`dNNNN`).

## The environment that works (2026-10-31)

| Piece | Version | Why |
|---|---|---|
| Python | `module load python/3.13.5` | Default `python3` is 3.9: too old for current ML libraries. uv couldn't download its own 3.12 (GitHub transfers timed out) |
| Installer | `uv` (installed with the 3.13 module's pip) | Much faster than pip on network storage, with visible progress |
| PyTorch | `torch==2.5.1+cu121` | Matches the CUDA 12.3 driver |
| transformers | `>=4.56,<5` (4.57.6 tested) | 5.x needs a newer torch than 2.5 |
| **Not installed** | vLLM, torchvision, torchaudio | vLLM can't run here; torchvision leftovers break `transformers` imports |

Exact list: [requirements-cluster.txt](requirements-cluster.txt). `explorer/env.sh` loads the module,
activates the venv and sets `HF_HOME`/`UV_CACHE_DIR` in the course dir.

## One-time setup

Use **OOD → Clusters → Explorer Shell Access** (the "JupyterLab ML Courses" app fails with "does not
supply a sub app form file"). Run installs as **batch jobs**: they survive a closed browser, and
interactive sessions were killed mid-install by their time limit. **Paste each command as one
line**: a blank line after a trailing backslash makes `sbatch` wait for a script on stdin.

```bash
# login node: clone and get uv (light work)
export COURSE_DIR=/courses/CS5180.202710/students/$USER          # term 202710 = Fall 2026
cd $COURSE_DIR && git clone https://github.com/supriya-kotturu/humor-me.git && cd humor-me
module load python/3.13.5 && python3 -m pip install --user uv && module unload python/3.13.5
export PATH=$HOME/.local/bin:$PATH && mkdir -p logs

# venv + packages, as a batch job (~5 min); then read logs/setup-<id>.out
sbatch -p courses --cpus-per-task=4 --mem=32G --time=01:00:00 -o logs/setup-%j.out --wrap 'module load python/3.13.5; export UV_CACHE_DIR=$COURSE_DIR/uv-cache; uv venv --clear --python "$(which python3)" $COURSE_DIR/venv && source $COURSE_DIR/venv/bin/activate && uv pip install -r explorer/requirements-cluster.txt --index-strategy unsafe-best-match && python -c "import torch, transformers; from transformers import Qwen2ForCausalLM; print(torch.__version__, transformers.__version__, torch.cuda.is_available())"'
```
The log's last line should be `2.5.1+cu121 4.57.x False` (`False` is right: no GPU on a CPU node).

**Data:** `train.jsonl` is gitignored. Rebuild it on a compute node instead of uploading. It's
deterministic, so the joke ids match the laptop's (check `after_length: 402981`):
```bash
srun -p courses --cpus-per-task=2 --mem=16G --time=00:30:00 --pty bash
cd $COURSE_DIR/humor-me && git clone --depth 1 https://github.com/orionw/rJokesData data/external/rJokesData
module load python/3.13.5 && python3 -m pipeline.rjokes && exit
```

**GPU smoke test** (50 jokes; the first run also downloads the 7B model, ~15 GB, in < 1 min):
```bash
srun -p courses-gpu --gres=gpu:v100-sxm2:1 --cpus-per-task=4 --mem=12G --time=00:30:00 --pty bash
cd /courses/CS5180.202710/students/$USER/humor-me && source explorer/env.sh
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python explorer/label_modes.py --input data/processed/rjokes/train.jsonl --n 50 --out /tmp/smoke.jsonl --dtype half
head -3 /tmp/smoke.jsonl && exit
```
Measured 2026-10-31: `True Tesla V100-SXM2-32GB`; **50/50 labeled, 3.4 jokes/s, 0 parse failures**
(→ ~2.5 h for 30k). The "generation flags are not valid" warning is expected: decoding is greedy.
The batch run was faster: **5.2 jokes/s, 30k in ~1.6 h, 11 parse failures** (job 11534420;
[log screenshot](../assets/labeling/batch-labeling.png)).

## Runs

| Step | How | Output |
|---|---|---|
| Smoke test (interactive) | See "GPU smoke test" above. **Then `exit` the session.** | Throughput, parse failures |
| Full labeling (batch) | `mkdir -p logs && sbatch explorer/label_modes.sbatch` (override: `N=5000 MODEL=... sbatch ...`) | `data/processed/rjokes/modes.jsonl` (ids + labels, no text) |
| Monitor / cancel | `squeue -u $USER`, `tail -f logs/humor-label-<id>.out`, `scancel <id>` | |
| **Interrupted?** (time limit, node failure, `scancel`) | Submit the **same** command again. It skips ids already in `modes.jsonl` (checkpointed every 2,000 jokes) and continues, on whatever node it gets | Same file, grows |
| **Is the sample big enough?** | Printed at the end of the job (`explorer/mode_precision.py`): per-mode 95% CI of the mean upvote percentile | Table; "extend" hint if any mode is wider than ±0.02 |
| **Room labels (D-89)** | `mkdir -p logs && sbatch explorer/label_state.sbatch` (smoke test: `python -m explorer.label_state --n 50 --out /tmp/state_smoke.jsonl --dtype half`). Resumable like the mode labels | `data/processed/rjokes/state.jsonl` (ids + venue/energy/age, no text), then `dpo_state.jsonl` (text + mode + room, **cluster only**) |
| **Extend the sample** | `N=60000 sbatch explorer/label_modes.sbatch` (same seed → labels the *next* 30k of the same shuffle) | Same file, grows |

**Precision rule (fixed before the first run):** label 30k; if any mode's 95% CI half-width on
mean percentile is wider than ±0.02, extend by 30k and recheck. Precision depends on jokes per
mode (±0.29/√n), not on the share of the 400k labeled.

## Exp 24 judge (Qwen ranks generated bits from your votes)

Runs `explorer/judge_generators.py` on a bundle the laptop writes: the generated bits of Exp 24 and your rankings, as prompts
(our own jokes only, no rJokes, no model names, no key file). Output is a ranking of letters per call. Everything lives under
`data/model_comparison/judge/` (gitignored). Why Qwen: it is neither candidate's family, and it is already cached here
(D-93).

| Step | Where | How |
|---|---|---|
| 1. Bundle | laptop | `python -m sim.compare_generators judge --export-bundle data/model_comparison/judge` (smoke: add `--smoke 3`). Writes `calls.jsonl`. Needs your 30 votes and `run` finished |
| 2. Upload | OOD **Files** app (Clusters > Explorer > Files), or `scp` | Put `calls.jsonl` in `$COURSE_DIR/humor-me/data/model_comparison/judge/` (make the folder if it is missing). Upload that one file only |
| 3. Smoke test | cluster | `mkdir -p logs && LIMIT=3 sbatch explorer/judge_generators.sbatch`, then `cat data/model_comparison/judge/replies.jsonl`: three lines with a `ranking` each, and `logs/humor-judge-<id>.out` showing the time per call. Delete `replies.jsonl` before the full run (or upload the full bundle: finished ids are skipped) |
| 4. Full run | cluster | `sbatch explorer/judge_generators.sbatch` (about 180 prompts of 12 to 20k tokens; V100, float16, SDPA attention, greedy; `courses-gpu` allows 8 h). Interrupted? Submit the same command again: it skips ids already in `replies.jsonl`. It exits with an error if some calls were not rankings: submit again to retry them (`replies.jsonl.failed.jsonl` shows what the model said) |
| 5. Download | OOD Files app | Copy `data/model_comparison/judge/replies.jsonl` to the laptop |
| 6. Score | laptop | `python -m sim.compare_generators judge --import-replies <downloaded file>`, then `score` |

A long few-shot prompt is the risk here: if a call runs out of GPU memory the job splits it, then skips it, and the next run tries it again.
The `--max-len 30000` default is Qwen2.5's 32k context less room for the reply.

## Exp 24 judge with Ollama (first model to try: qwen3.8:27b)

The same bundle and the same steps as the Qwen2.5-7B judge above, with an Ollama server inside the job. Why: the user picked
`qwen3.8:27b` from Ollama's local list; it is neither Gemini nor Claude, and a 27B model at 4 bit (about 17 GB) fits a V100 (D-93).
Qwen2.5-7B stays the fallback.

**Before you rely on it (not verified from here):**
- The Ollama Linux build must support the **V100 (compute 7.0)** and the cluster driver (**CUDA <= 12.3**). If it cannot start on
  the GPU the job fails loudly: it checks `ollama ps` shows GPU use (and no CPU offload) before any call, instead of running a 27B
  model on the CPU for hours.
- The compute node must reach Ollama's registry for the one-time ~17 GB `ollama pull`. If it cannot, run the pull once from a
  `courses` CPU job with the same `OLLAMA_MODELS`, then submit the GPU job.
- **Context:** Ollama truncates a prompt silently to its context window. The job starts the server with `OLLAMA_CONTEXT_LENGTH=32768`,
  the judge asks for that context in each request, refuses a prompt estimated above it, and treats a reply whose `prompt_eval_count`
  is under half the estimate as truncated (listed in `replies.jsonl.failed.jsonl`, retried on the next run, never an answer).
  Each call starts with a unique line so Ollama cannot reuse a cached prefix (otherwise that count would be misleading), which means
  the examples are re-read every call: expect on the order of a minute per prompt, so a few hours for the 180 calls (an estimate).

| Step | Where | How |
|---|---|---|
| 1. Bundle | laptop | `python -m sim.compare_generators judge --judge-provider ollama --export-bundle data/model_comparison/judge-ollama` (smoke: add `--smoke 3`). Needs your 30 votes and `run` finished |
| 2. Upload | OOD **Files** app | Put `calls.jsonl` in `$COURSE_DIR/humor-me/data/model_comparison/judge-ollama/` (that one file only) |
| 3. Smoke test | cluster | `mkdir -p logs && LIMIT=3 sbatch explorer/judge_ollama.sbatch`. The first job installs Ollama under `$COURSE_DIR/ollama` (no root) and downloads the model to `$COURSE_DIR/ollama/models`, so it is slow; read `logs/humor-judge-ollama-<id>.out` (it prints `nvidia-smi` and `ollama ps`: the PROCESSOR column must say `100% GPU`) and `data/model_comparison/judge-ollama/replies.jsonl` (three lines, each with a `ranking`) |
| 4. Full run | cluster | `sbatch explorer/judge_ollama.sbatch` (V100, 8 h max, 12 GB system RAM: the model lives in GPU memory). Interrupted? Submit the same command again: it skips ids already in `replies.jsonl` and the model is already on disk. It exits with an error if calls were not rankings: submit again |
| 5. Download | OOD Files app | Copy `replies.jsonl` to the laptop |
| 6. Score | laptop | `python -m sim.compare_generators judge --judge-provider ollama --import-replies <file>`, then `score` |

Other model: `MODEL=<name> sbatch explorer/judge_ollama.sbatch` (and `--judge-model <name>` on the laptop); fallbacks in order are
muse-glimmer 30b, then the transformers Qwen2.5-7B job above.

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
before the sim uses them: hand labels in `data/labels/rjokes_hand_check.csv`, agreement from
`python explorer/hand_check.py` (notebook Exp 9).
