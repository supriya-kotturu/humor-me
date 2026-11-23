#!/bin/bash
#SBATCH --partition=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=eval_judge
#SBATCH --output=logs/eval_judge.%j.out
#SBATCH --error=logs/eval_judge.%j.err

source ~/humor-me/explorer/env.sh
export HF_HUB_OFFLINE=1
cd ~/humor-me
python explorer/eval_judge.py "$@"