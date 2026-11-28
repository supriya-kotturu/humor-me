#!/bin/bash
# 10-step smoke test of the D-123 training stack. Run it BEFORE sft_B / dpo_B / dpo_C (explorer/README.md, "Training the writer"):
#   mkdir -p logs && sbatch explorer/smoke.sbatch
# Prints the library versions, checks the tokenizer's chat template against explorer/chatml.py, then runs every path a long job
# takes: 10 SFT steps, the merge, a tokenizer check, DPO on the merged model (stopped at a checkpoint and resumed), and 10 DPO
# steps of writer C. It logs every step, prints seconds per step, peak GPU memory, the longest row and the tail of each tokenized
# example, and ends with one [summary] line per stage (ok / failed / skipped): paste that block. It stops at the first failed
# stage. Read logs/humor-smoke-<id>.out before submitting anything long: the full runs' times are this times the step counts.
# DTYPE=fp32 sbatch explorer/smoke.sbatch runs every stage in plain fp32 (the fallback when fp16 is unstable).
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
DT=${DTYPE:+--dtype $DTYPE}

STAGES=(template sft merge tokenizer dpo-B-to-6 dpo-B-resume dpo-C)
declare -A STATUS
for s in "${STAGES[@]}"; do STATUS[$s]=skipped; done
summary() {
    echo "==== smoke summary (DTYPE=${DTYPE:-fp16})"
    for s in "${STAGES[@]}"; do printf '[summary] %-13s %s\n' "$s" "${STATUS[$s]}"; done
}
trap summary EXIT
stage() {  # stage <name> <command...>: runs it, records ok or failed, and stops the job on a failure
    local name=$1 rc=0
    shift
    echo "==== stage $name"
    "$@" || rc=$?
    if [ "$rc" -eq 0 ]; then
        STATUS[$name]=ok
    else
        STATUS[$name]="failed (exit $rc)"
        if [ "$rc" -eq 3 ]; then STATUS[$name]="failed (exit 3: NaN stop, see the [stop] lines)"; fi
        exit "$rc"
    fi
}

template() {
    python - <<'PYEOF'
from transformers import AutoTokenizer
from explorer import train_common as tcm
print(tcm.versions())
tok = AutoTokenizer.from_pretrained(tcm.DEFAULT_BASE)
tcm.end_token_check(tok)
tcm.template_check(tok, [{"role": "system", "content": "SYS"}, {"role": "user", "content": "{\"venue\": \"bar\"}"}])
PYEOF
}

# The merged folder's tokenizer (what the GGUF converter reads) and the one training uses must give the base's ids. Loading the
# merged folder may print transformers 4.57's "incorrect regex pattern" warning: a false alarm for Qwen (train_common.tokenizer_source).
tokenizer() {
    python - "$RUNS/smoke-merged" <<'PYEOF'
import sys
from transformers import AutoTokenizer
from explorer import train_common as tcm
merged = sys.argv[1]
base = AutoTokenizer.from_pretrained(tcm.DEFAULT_BASE)
tcm.same_tokens(AutoTokenizer.from_pretrained(merged), base, "merged folder")
tcm.same_tokens(AutoTokenizer.from_pretrained(tcm.tokenizer_source(merged)), base, f"training ({tcm.tokenizer_source(merged)})")
PYEOF
}

rm -rf $RUNS/smoke-sft $RUNS/smoke-merged $RUNS/smoke-dpo $RUNS/smoke-dpo-C $RUNS/smoke-dpo_B.jsonl
stage template template
stage sft python -m explorer.train_sft --data data/dpo/train/sft_B.jsonl --out $RUNS/smoke-sft --max-steps 10 --save-steps 1000 $DT
# Writer B's path: merge the SFT adapter on the GPU, then DPO on the merged model. A 48-pair slice keeps the reference pass short.
stage merge python -m explorer.merge_lora --adapter $RUNS/smoke-sft --out $RUNS/smoke-merged --device cuda
stage tokenizer tokenizer
head -n 48 data/dpo/train/dpo_B.jsonl > $RUNS/smoke-dpo_B.jsonl
# 6 steps with a checkpoint every 3, then the same command to step 10: the second run must say it resumes from checkpoint-6.
stage dpo-B-to-6 python -m explorer.train_dpo --data $RUNS/smoke-dpo_B.jsonl --base $RUNS/smoke-merged --out $RUNS/smoke-dpo --max-steps 6 --save-steps 3 $DT
stage dpo-B-resume python -m explorer.train_dpo --data $RUNS/smoke-dpo_B.jsonl --base $RUNS/smoke-merged --out $RUNS/smoke-dpo --max-steps 10 --save-steps 3 $DT
# Writer C's path: the base model on the user's pairs (canonical JSON replies).
stage dpo-C python -m explorer.train_dpo --data data/dpo/train/dpo_C.jsonl --out $RUNS/smoke-dpo-C --max-steps 10 --save-steps 1000 $DT
rm -rf $RUNS/smoke-sft $RUNS/smoke-merged $RUNS/smoke-dpo $RUNS/smoke-dpo-C $RUNS/smoke-dpo_B.jsonl
