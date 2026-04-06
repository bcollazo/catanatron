#!/bin/bash
# Duke DCC Slurm: self-play ladder training (GPU)
#
# Submit from repo root (or anywhere): sbatch capstone_agent/jobs/selfplay_dcc_slurm.sh
# Docs: https://oit-rc.pages.oit.duke.edu/rcsupportdocs/dcc/slurm/
#
# Wall time must be <= partition MaxTime. gpu-common / courses-gpu are typically 2 days max
# (30 days will be rejected). For longer training: re-submit with
#   sbatch --dependency=afterok:<JOBID> capstone_agent/jobs/selfplay_dcc_slurm.sh
# Check limits:  scontrol show partition gpu-common | grep -i max
# scavenger-gpu often allows ~7d (lower priority): uncomment partition line below if you use it.
#
#SBATCH --job-name=catan_selfplay_gpu
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
## SBATCH --time=7-00:00:00
## SBATCH --partition=scavenger-gpu
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

# Challenger initial weights (NOT the champion — see below):
#   Resume last job:  leave LOAD_MAIN unset (default) so run_simulation auto-loads
#     --save / --save-placement-model if those files exist; else falls back to *_FALLBACK.
#   Start from champion:  export LOAD_MAIN LOAD_PLACEMENT to champion paths (or *_FALLBACK).
# Champion opponent files: always --champion-main-model / --champion-placement-model.
CHAMP_MAIN="$REPO_ROOT/capstone_agent/models/champion_main_play.pt"
CHAMP_PLACE="$REPO_ROOT/capstone_agent/models/champion_placement.pt"
CHAL_MAIN="$REPO_ROOT/capstone_agent/models/challenger_main_play.pt"
CHAL_PLACE="$REPO_ROOT/capstone_agent/models/challenger_placement.pt"
LOAD_MAIN_FALLBACK="${LOAD_MAIN_FALLBACK:-$REPO_ROOT/capstone_agent/models/capstone_model.pt}"
LOAD_PLACE_FALLBACK="${LOAD_PLACE_FALLBACK:-$REPO_ROOT/capstone_agent/models/placement_model.pt}"
if [[ -n "${LOAD_MAIN:-}" ]]; then
  MAIN_SRC="$LOAD_MAIN"
elif [[ -f "$CHAL_MAIN" ]]; then
  MAIN_SRC="" # omit --load → run_simulation resumes from --save (challenger_main_play.pt)
else
  MAIN_SRC="$LOAD_MAIN_FALLBACK"
fi
if [[ -n "${LOAD_PLACEMENT:-}" ]]; then
  PLACE_SRC="$LOAD_PLACEMENT"
elif [[ -f "$CHAL_PLACE" ]]; then
  PLACE_SRC="" # omit → resume from --save-placement-model
else
  PLACE_SRC="$LOAD_PLACE_FALLBACK"
fi
LOAD_ARGS=()
[[ -n "$MAIN_SRC" ]] && LOAD_ARGS+=(--load "$MAIN_SRC")
[[ -n "$PLACE_SRC" ]] && LOAD_ARGS+=(--placement-model "$PLACE_SRC")

python -u capstone_agent/run_simulation.py \
  --train \
  --self-play-ladder \
  --champion-history-dir capstone_agent/models/champion_history/selfplay_run1 \
  "${LOAD_ARGS[@]}" \
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
