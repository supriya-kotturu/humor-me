#!/bin/bash
# Writer B, step 1 (D-123): SFT of Qwen3-4B-Instruct-2507 on the Haiku-chosen threads. After smoke.sbatch, from the repo root:
#   mkdir -p logs && sbatch explorer/sft_B.sbatch
# Interrupted (time limit, node failure)? Submit the same command again: it resumes from the last checkpoint in the run folder.
# --requeue lets Slurm do that itself after a node failure. The script saves and stops itself at 7.5 h, before the 8 h limit.
#SBATCH --job-name=humor-sft-B
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source explorer/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
RUNS=data/dpo/train/runs
python -m explorer.train_sft --data data/dpo/train/sft_B.jsonl --out $RUNS/sft_B ${EPOCHS:+--epochs "$EPOCHS"} ${LR:+--lr "$LR"} ${DTYPE:+--dtype "$DTYPE"}
