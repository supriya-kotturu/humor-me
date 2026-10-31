#!/bin/bash
# Batch version of label_modes.py: survives closing the browser, and isn't killed as an idle
# OOD session. Submit from the repo root: sbatch explorer/label_modes.sbatch
# Interrupted (time limit, node failure)? Submit the same command again: the job skips jokes
# already in modes.jsonl, on whichever node it lands. --requeue lets Slurm do that itself
# after a node failure.
#SBATCH --job-name=humor-label
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source explorer/env.sh
nvidia-smi --query-gpu=name,memory.total --format=csv
python explorer/label_modes.py --input data/processed/rjokes/train.jsonl --n "${N:-30000}" \
    --seed "${SEED:-0}" ${MODEL:+--model "$MODEL"}
# Exit code 1 here means "sample too small for the precision rule", not a failed labeling run.
python explorer/mode_precision.py || echo "precision rule not met: extend with N=$(( ${N:-30000} + 30000 ))"
