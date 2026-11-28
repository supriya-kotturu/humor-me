#!/bin/bash
# Merge a trained adapter and convert the result to a Q4_K_M GGUF for Ollama (D-124), on the CPU partition (no GPU needed, 256 GB RAM):
#   WRITER=B ADAPTER=data/dpo/train/runs/dpo_B BASE=data/dpo/train/runs/merged_sft_B sbatch explorer/merge_gguf.sbatch
#   WRITER=C ADAPTER=data/dpo/train/runs/dpo_C sbatch explorer/merge_gguf.sbatch      # BASE defaults to Qwen/Qwen3-4B-Instruct-2507
# Output: data/dpo/train/runs/gguf/humor-qwen3-<WRITER>-Q4_K_M.gguf (about 2.5 GB), to copy back to the laptop.
# Steps: merge_and_unload to fp16 safetensors, llama.cpp convert_hf_to_gguf.py --outtype f16, build llama-quantize once, quantize.
# First run builds llama.cpp under $COURSE_DIR/llama.cpp: it needs a C++ compiler and cmake. CPU nodes have no cmake on PATH, so the
# job installs the PyPI build into the venv when it is missing; a missing compiler stops the job before any work (`module avail gcc`).
#SBATCH --job-name=humor-gguf
#SBATCH --partition=courses
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --open-mode=append
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source explorer/env.sh
: "${WRITER:?set WRITER=B or C}"
: "${ADAPTER:?set ADAPTER=<adapter folder>}"
BASE="${BASE:-Qwen/Qwen3-4B-Instruct-2507}"
RUNS=data/dpo/train/runs
MERGED=$RUNS/merged_$WRITER
GGUF=$RUNS/gguf
NAME=humor-qwen3-$WRITER
LLAMA=$COURSE_DIR/llama.cpp
if [ ! -x "$LLAMA/build/bin/llama-quantize" ]; then
    command -v cmake >/dev/null || uv pip install cmake
    command -v c++ >/dev/null || { echo "[stop] no C++ compiler on PATH: load a gcc module (module avail gcc)" >&2; exit 2; }
fi
mkdir -p "$GGUF"
[ -f "$MERGED/config.json" ] || python -m explorer.merge_lora --adapter "$ADAPTER" --base "$BASE" --out "$MERGED" --device cpu
[ -d "$LLAMA" ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA"
python "$LLAMA/convert_hf_to_gguf.py" "$MERGED" --outtype f16 --outfile "$GGUF/$NAME-f16.gguf"
if [ ! -x "$LLAMA/build/bin/llama-quantize" ]; then
    cmake -S "$LLAMA" -B "$LLAMA/build" -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
    cmake --build "$LLAMA/build" --target llama-quantize -j 8
fi
"$LLAMA/build/bin/llama-quantize" "$GGUF/$NAME-f16.gguf" "$GGUF/$NAME-Q4_K_M.gguf" Q4_K_M
ls -l "$GGUF"
rm -f "$GGUF/$NAME-f16.gguf"
