#!/bin/bash
# Batch version of label_modes.py: survives closing the browser, and isn't killed as an idle
# OOD session. Submit from the repo root: sbatch explorer/label_modes.sbatch
# Interrupted (time limit, node failure)? Submit the same command again: the job skips jokes
# already in modes.jsonl, on whichever node it lands. --requeue lets Slurm do that itself
# after a node failure.
#SBATCH --job-name=humor-label
#SBATCH --partition=courses-gpu
# courses-gpu has V100-SXM2 and P100 nodes; vLLM doesn't support the P100, so ask for a V100.
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
# courses-gpu's maximum; a 60k extension takes ~3.3 h, too close to 4.
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source explorer/env.sh
# The OOM at 72k/90k had 4.5 GB reserved but unusable (fragmentation); this lets PyTorch reuse it.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv
python explorer/label_modes.py --input data/processed/rjokes/train.jsonl --n "${N:-30000}" \
    --seed "${SEED:-0}" --dtype "${DTYPE:-half}" --backend "${BACKEND:-transformers}" \
    ${MODEL:+--model "$MODEL"}
# Exit code 1 here means "sample too small for the precision rule", not a failed labeling run.
python explorer/mode_precision.py || echo "precision rule not met: extend with N=$(( ${N:-30000} + 30000 ))"
