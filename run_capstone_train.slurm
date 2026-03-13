#!/bin/bash
#SBATCH --job-name=capstone-train
#SBATCH --output=capstone_agent/benchmarks/slurm-%j.out
#SBATCH --error=capstone_agent/benchmarks/slurm-%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
# If your cluster requires GPU, uncomment next line:
# #SBATCH --gres=gpu:1

set -euo pipefail

cd ~/catan_ai
eval "$(micromamba shell hook --shell bash)"
micromamba activate catan311

mkdir -p capstone_agent/benchmarks

# Continue from current weights (auto-resume), train 5000 more games in chunks
for i in $(seq 1 50); do
  python capstone_agent/run_simulation.py \
    --games 100 \
    --train \
    --save capstone_agent/capstone_model.pt \
    --run-name iter_full
done
