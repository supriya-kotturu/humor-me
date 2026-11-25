#!/bin/bash
# Exp 24 judge with an Ollama model on Explorer (default qwen3.8:27b, 4-bit): ranks the bundle's prompts. Submit from the repo
# root after uploading data/model_comparison/judge-ollama/calls.jsonl (explorer/README.md, "Exp 24 judge with Ollama"):
#   mkdir -p logs && LIMIT=3 sbatch explorer/judge_ollama.sbatch     # smoke test: the first 3 calls (the first job also downloads)
#   mkdir -p logs && sbatch explorer/judge_ollama.sbatch             # the full run
# Interrupted? Submit the same command again: calls already in replies.jsonl are skipped, and the model is already on disk.
#
# Ollama runs without root, from the course dir: the official Linux tarball is unpacked under $COURSE_DIR/ollama and the
# model lives in $COURSE_DIR/ollama/models (about 17 GB; the 75 GB home quota is not touched). The 12 GB here is system
# RAM: the model sits in GPU memory. The V100 is compute 7.0 and the driver supports CUDA up to 12.3: the Ollama build must
# support both, and if it falls back to the CPU this job stops at once instead of crawling for hours.
#SBATCH --job-name=humor-judge-ollama
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
MODEL="${MODEL:-qwen3.8:27b}"
CONTEXT="${CONTEXT:-32768}"
PORT="${PORT:-11434}"
OLLAMA_DIR="$COURSE_DIR/ollama"
mkdir -p "$OLLAMA_DIR/models" logs
export OLLAMA_MODELS="$OLLAMA_DIR/models"
export OLLAMA_HOST="127.0.0.1:$PORT"
# Ollama's default context is far smaller than our 12 to 20k-token prompts and it truncates silently: set it here, and the
# judge asks for it again per request and refuses prompts that do not fit.
export OLLAMA_CONTEXT_LENGTH="$CONTEXT"
export OLLAMA_KEEP_ALIVE=-1
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv

if [ ! -x "$OLLAMA_DIR/bin/ollama" ]; then
    echo "installing Ollama under $OLLAMA_DIR"
    if curl -fL --retry 3 -o "$OLLAMA_DIR/ollama.tgz" https://ollama.com/download/ollama-linux-amd64.tgz; then
        tar -C "$OLLAMA_DIR" -xzf "$OLLAMA_DIR/ollama.tgz"
    else  # newer releases ship a zstd archive
        curl -fL --retry 3 -o "$OLLAMA_DIR/ollama.tar.zst" https://ollama.com/download/ollama-linux-amd64.tar.zst
        tar -C "$OLLAMA_DIR" --zstd -xf "$OLLAMA_DIR/ollama.tar.zst"
    fi
fi
export PATH="$OLLAMA_DIR/bin:$PATH"
ollama --version

ollama serve > "logs/ollama-serve-$SLURM_JOB_ID.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for i in $(seq 1 60); do
    curl -sf "http://127.0.0.1:$PORT/api/version" > /dev/null && break
    sleep 2
done
curl -sf "http://127.0.0.1:$PORT/api/version" || { echo "ollama serve did not come up: see logs/ollama-serve-$SLURM_JOB_ID.log"; exit 1; }

# First job only: the download (about 17 GB). A node without a route to the registry fails here; then download once from a
# `courses` CPU job with the same OLLAMA_MODELS and rerun this.
ollama list | grep -q "^$MODEL" || ollama pull "$MODEL"

# Load the model and require the GPU. A 27B model on the CPU would run for days: fail now.
curl -sf "http://127.0.0.1:$PORT/api/generate" -d "{\"model\": \"$MODEL\", \"prompt\": \"ok\", \"stream\": false, \"options\": {\"num_ctx\": $CONTEXT, \"num_predict\": 1}}" > /dev/null
ollama ps
if ! ollama ps | tail -n +2 | grep -q "GPU"; then
    echo "ollama is not running $MODEL on the GPU (V100 compute 7.0, driver CUDA <= 12.3 needs a build that supports them): see logs/ollama-serve-$SLURM_JOB_ID.log" >&2
    exit 1
fi
if ollama ps | tail -n +2 | grep -q "CPU"; then
    echo "$MODEL is partly on the CPU (offloaded): too slow for 180 long prompts; use a smaller model or a larger GPU" >&2
    exit 1
fi

python explorer/judge_generators.py --backend ollama --model "$MODEL" --context "$CONTEXT" --url "http://127.0.0.1:$PORT" \
    --input "${INPUT:-data/model_comparison/judge-ollama/calls.jsonl}" --out "${OUT:-data/model_comparison/judge-ollama/replies.jsonl}" \
    ${LIMIT:+--limit "$LIMIT"}
