#!/bin/bash
# Batch version of label_state.py. Submit from the repo root: sbatch explorer/label_state.sbatch
# 90k at label_modes' 5.2 jokes/s is ~4.8 h, inside the 8 h limit; resubmit the same command if cut off.
#SBATCH --job-name=humor-state
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
nvidia-smi --query-gpu=name,memory.total --format=csv
python -m explorer.label_state --dtype "${DTYPE:-half}" --backend "${BACKEND:-transformers}" ${N:+--n "$N"}
python -m explorer.label_state --export data/processed/rjokes/dpo_state.jsonl
