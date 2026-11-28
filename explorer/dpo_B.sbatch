#!/bin/bash
# Writer B, step 2 (D-123): DPO on Haiku's labelled pairs, starting from the SFT model. After sft_B.sbatch has finished:
#   sbatch explorer/dpo_B.sbatch
# First it merges the SFT adapter into the base (fp16, on the GPU, about a minute) so the DPO adapter sits on the SFT model and
# its reference (adapter off) is the SFT model. Interrupted? Submit the same command again: the merge is skipped when done and
# DPO resumes from its last checkpoint.
#SBATCH --job-name=humor-dpo-B
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
[ -f $RUNS/merged_sft_B/config.json ] || python -m explorer.merge_lora --adapter $RUNS/sft_B --out $RUNS/merged_sft_B --device cuda
python -m explorer.train_dpo --data data/dpo/train/dpo_B.jsonl --base $RUNS/merged_sft_B --out $RUNS/dpo_B ${EPOCHS:+--epochs "$EPOCHS"} ${LR:+--lr "$LR"} ${DTYPE:+--dtype "$DTYPE"}
