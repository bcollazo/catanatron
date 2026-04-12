#!/bin/bash
# Simple reward + fixed opponent curriculum (preset ab-ramp):
#   weighted -> value -> alphabeta@1 -> alphabeta@2 (per --difficulty segment lengths).
# Placement defaults to AlphaBeta (preset); override with PLACEMENT_STRATEGY=model if needed.
#
# Submit from repo root:
#   sbatch capstone_agent/jobs/neuner_simple_ab_ramp_gpu.slurm
# Optional: DIFFICULTY=hard GAMES=500000 sbatch ...
#
#SBATCH --job-name=neuner_simple_ramp
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
## SBATCH --account=YOUR_ACCOUNT

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/catan_ai}"
cd "$REPO_ROOT"
mkdir -p logs

export PYTHONUNBUFFERED=1
echo "=== $(date -Is) neuner_simple_ab_ramp start cwd=$(pwd) ==="

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
# shellcheck source=/dev/null
source ~/.micromamba.sh
micromamba activate catan311

echo "=== GPU check ==="
nvidia-smi -L || true
python -u -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu-only')"

MODEL_DIR="$REPO_ROOT/capstone_agent/models/neuner_simple_ab_ramp"
mkdir -p "$MODEL_DIR/replays"

MAIN_PT="$MODEL_DIR/main_play.pt"
SEED_PT="${SEED_PT:-$REPO_ROOT/capstone_agent/models/capstone_model.pt}"
LOAD_ARGS=()
if [[ "${TRAIN_FROM_SCRATCH:-0}" == "1" ]]; then
  echo "TRAIN_FROM_SCRATCH=1: no --load (random init)."
elif [[ -n "${LOAD_MAIN:-}" ]]; then
  LOAD_ARGS+=(--load "$LOAD_MAIN")
elif [[ -f "$MAIN_PT" ]]; then
  LOAD_ARGS+=(--load "$MAIN_PT")
elif [[ -f "$SEED_PT" ]]; then
  LOAD_ARGS+=(--load "$SEED_PT")
fi

DIFF="${DIFFICULTY:-medium}"
PLACEMENT="${PLACEMENT_STRATEGY:-}"

EXTRA=()
if [[ -n "$PLACEMENT" ]]; then
  EXTRA+=(--placement-strategy "$PLACEMENT")
fi

python -u capstone_agent/run_simulation.py \
  --train \
  --preset ab-ramp \
  --difficulty "$DIFF" \
  --reward-function simple \
  --games "${GAMES:-1000000}" \
  --save "$MAIN_PT" \
  --save-every-games 2000 \
  --save-numbered-checkpoints \
  --benchmark-csv "$MODEL_DIR/benchmark_games.csv" \
  --training-metrics-csv "$MODEL_DIR/ppo_training.csv" \
  --save-games-json-dir "$MODEL_DIR/replays" \
  --save-games-json-every 100 \
  --train-update-trigger steps \
  --train-every-steps 4096 \
  --map-template TOURNAMENT \
  --map-mode fixed \
  --fixed-map-seed 0 \
  --run-name neuner_simple_ab_ramp \
  --progress-env-steps 250 \
  "${EXTRA[@]}" \
  "${LOAD_ARGS[@]}"
