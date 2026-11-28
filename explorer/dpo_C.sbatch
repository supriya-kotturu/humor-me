#!/bin/bash
# Writer C (D-123): DPO of the base model on the user's own 120 decided pairs. After smoke.sbatch (independent of writer B):
#   sbatch explorer/dpo_C.sbatch
# The pairs are bonsai-2:4b's threads, so for this qwen3 base they are off-policy (D-123). 120 pairs are a few steps at an
# effective batch of 8: three epochs by default (EPOCHS=... to change). Interrupted? Submit the same command again.
#SBATCH --job-name=humor-dpo-C
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source explorer/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
RUNS=data/dpo/train/runs
python -m explorer.train_dpo --data data/dpo/train/dpo_C.jsonl --out $RUNS/dpo_C --epochs "${EPOCHS:-3}" ${LR:+--lr "$LR"}
