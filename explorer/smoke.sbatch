#!/bin/bash
# 10-step smoke test of the D-123 training stack. Run it BEFORE sft_B / dpo_B / dpo_C (explorer/README.md, "Training the writer"):
#   mkdir -p logs && sbatch explorer/smoke.sbatch
# Prints the library versions, checks the tokenizer's chat template against explorer/chatml.py, then runs every path a long job
# takes: 10 SFT steps, the merge, DPO on the merged model (stopped at a checkpoint and resumed), and 10 DPO steps of writer C. It
# prints seconds per step, peak GPU memory, the longest row and the tail of each tokenized example. Read
# logs/humor-smoke-<id>.out before submitting anything long: the full runs' times are this times the step counts.
#SBATCH --job-name=humor-smoke
#SBATCH --partition=courses-gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=01:30:00
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
rm -rf $RUNS/smoke-sft $RUNS/smoke-merged $RUNS/smoke-dpo $RUNS/smoke-dpo-C $RUNS/smoke-dpo_B.jsonl
python -m explorer.train_sft --data data/dpo/train/sft_B.jsonl --out $RUNS/smoke-sft --max-steps 10 --save-steps 1000
# Writer B's path: merge the SFT adapter on the GPU, then DPO on the merged model. A 48-pair slice keeps the reference pass short.
python -m explorer.merge_lora --adapter $RUNS/smoke-sft --out $RUNS/smoke-merged --device cuda
head -n 48 data/dpo/train/dpo_B.jsonl > $RUNS/smoke-dpo_B.jsonl
# 6 steps with a checkpoint every 3, then the same command to step 10: the second run must say it resumes from checkpoint-6.
python -m explorer.train_dpo --data $RUNS/smoke-dpo_B.jsonl --base $RUNS/smoke-merged --out $RUNS/smoke-dpo --max-steps 6 --save-steps 3
python -m explorer.train_dpo --data $RUNS/smoke-dpo_B.jsonl --base $RUNS/smoke-merged --out $RUNS/smoke-dpo --max-steps 10 --save-steps 3
# Writer C's path: the base model on the user's pairs (canonical JSON replies).
python -m explorer.train_dpo --data data/dpo/train/dpo_C.jsonl --out $RUNS/smoke-dpo-C --max-steps 10 --save-steps 1000
rm -rf $RUNS/smoke-sft $RUNS/smoke-merged $RUNS/smoke-dpo $RUNS/smoke-dpo-C $RUNS/smoke-dpo_B.jsonl
