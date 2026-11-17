#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --job-name=dpo_judge
#SBATCH --output=logs/dpo_judge.%j.out
#SBATCH --error=logs/dpo_judge.%j.err

source ~/humor-me/explorer/env.sh
export HF_HUB_OFFLINE=1
cd ~/humor-me
python explorer/train_dpo.py "$@"