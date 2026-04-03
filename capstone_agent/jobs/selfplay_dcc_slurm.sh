#!/bin/bash
# Duke DCC Slurm: self-play ladder training (edit #SBATCH and paths, then: sbatch this file)
#
# Docs: https://oit-rc.pages.oit.duke.edu/rcsupportdocs/dcc/slurm/
#
#SBATCH --job-name=catan_selfplay
#SBATCH --partition=common
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
## SBATCH --account=YOUR_ACCOUNT     # uncomment if required by your allocation

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/catan_ai}"
cd "$REPO_ROOT"
mkdir -p logs

# Unbuffered Python + immediate shell line so Slurm .out files show progress while
# imports / model load run (can be minutes before first Python print otherwise).
export PYTHONUNBUFFERED=1
echo "=== $(date -Is) job start cwd=$(pwd) ==="

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
# shellcheck source=/dev/null
source ~/.micromamba.sh
micromamba activate catan311

python -u capstone_agent/run_simulation.py \
  --train \
  --self-play-ladder \
  --champion-history-dir capstone_agent/models/champion_history/selfplay_run1 \
  --load capstone_agent/models/capstone_model.pt \
  --placement-model capstone_agent/models/placement_model.pt \
  --save capstone_agent/models/challenger_main_play.pt \
  --save-placement-model capstone_agent/models/challenger_placement.pt \
  --champion-main-model capstone_agent/models/champion_main_play.pt \
  --champion-placement-model capstone_agent/models/champion_placement.pt \
  --placement-strategy model \
  --games 1000000 \
  --save-every-games 10000 \
  --train-update-trigger steps \
  --train-every-steps 4096 \
  --self-play-eval-every-games 5000 \
  --self-play-eval-games 500 \
  --self-play-promotion-threshold 0.55 \
  --map-template TOURNAMENT \
  --map-mode fixed \
  --fixed-map-seed 0 \
  --run-name selfplay_10m_run1 \
  --benchmark-csv capstone_agent/benchmarks/selfplay_10m_run1.csv
