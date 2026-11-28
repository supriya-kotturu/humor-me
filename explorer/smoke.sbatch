#!/bin/bash
# 10-step smoke test of the D-123 training stack. Run it BEFORE sft_B / dpo_B / dpo_C (explorer/README.md, "Training the writer"):
#   mkdir -p logs && sbatch explorer/smoke.sbatch
# Prints the library versions, checks the tokenizer's chat template against explorer/chatml.py, then runs 10 SFT steps and 10 DPO
# steps (on dpo_C, the small file) and prints seconds per step, peak GPU memory and the tail of each tokenized example. Read
# logs/humor-smoke-<id>.out before submitting anything long: the full runs' times are this times the step counts.
#SBATCH --job-name=humor-smoke
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source explorer/env.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name,memory.total --format=csv
RUNS=data/dpo/train/runs
python - <<'PYEOF'
from transformers import AutoTokenizer
from explorer import train_common as tcm
print(tcm.versions())
tok = AutoTokenizer.from_pretrained(tcm.DEFAULT_BASE)
tcm.template_check(tok, [{"role": "system", "content": "SYS"}, {"role": "user", "content": "{\"venue\": \"bar\"}"}])
PYEOF
python -m explorer.train_sft --data data/dpo/train/sft_B.jsonl --out $RUNS/smoke-sft --max-steps 10 --save-steps 1000
python -m explorer.train_dpo --data data/dpo/train/dpo_C.jsonl --out $RUNS/smoke-dpo --max-steps 10 --save-steps 1000
rm -rf $RUNS/smoke-sft $RUNS/smoke-dpo
