#!/bin/bash
# Exp 24 judge on Explorer: ranks the bundle's prompts with Qwen2.5-7B-Instruct. Submit from the repo root after
# uploading data/model_comparison/judge/calls.jsonl (explorer/README.md, "Exp 24 judge"):
#   mkdir -p logs && sbatch explorer/judge_generators.sbatch          # LIMIT=3 sbatch ... for the smoke test
# Interrupted (time limit, node failure)? Submit the same command again: it skips calls already in replies.jsonl.
# Calls that did not parse as a ranking exit with an error and are retried by the next submission.
#SBATCH --job-name=humor-judge
#SBATCH --partition=courses-gpu
# V100: float16 and SDPA attention; no flash-attn (Volta) and no vLLM (driver), as for label_modes.
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
# About 180 prompts of 12 to 20k tokens, a few seconds to a minute each.
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source explorer/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv
python explorer/judge_generators.py --input "${INPUT:-data/model_comparison/judge/calls.jsonl}" \
    --out "${OUT:-data/model_comparison/judge/replies.jsonl}" --dtype "${DTYPE:-half}" \
    ${LIMIT:+--limit "$LIMIT"} ${MODEL:+--model "$MODEL"}
