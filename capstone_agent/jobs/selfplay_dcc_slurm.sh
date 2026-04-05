#!/bin/bash
# Duke DCC Slurm: self-play ladder training (GPU)
#
# Submit from repo root (or anywhere): sbatch capstone_agent/jobs/selfplay_dcc_slurm.sh
# Docs: https://oit-rc.pages.oit.duke.edu/rcsupportdocs/dcc/slurm/
#
# Wall time: 30 days. If sbatch rejects with MaxTime, your GPU partition may cap lower
# (e.g. gpu-common is often 2d). Then use a lab/courses GPU partition, or chain jobs with
# --dependency=afterok:<jobid> and rely on checkpoint resume (--save paths + existing weights).
#
#SBATCH --job-name=catan_selfplay_gpu
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --time=30-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
## SBATCH --account=YOUR_ACCOUNT     # uncomment if required by your allocation
## SBATCH --partition=courses-gpu    # alternative for ECE / course GPU pool
## SBATCH --gres=gpu:2080:1          # optional: pin GPU type (see: gpuavail)

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

echo "=== GPU check ==="
nvidia-smi -L || true
python -u -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu-only')"

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
  --progress-env-steps 250 \
  --self-play-eval-every-games 5000 \
  --self-play-eval-games 500 \
  --self-play-promotion-threshold 0.55 \
  --map-template TOURNAMENT \
  --map-mode fixed \
  --fixed-map-seed 0 \
  --run-name selfplay_10m_run1 \
  --benchmark-csv capstone_agent/benchmarks/selfplay_10m_run1.csv
