#!/bin/bash
# Simple reward (terminal ±1), vs RandomPlayer, random placement for our agent.
# Good baseline: critic aligns with winning; avoids dense VP/pip shaping.
#
# From scratch (no --load): export TRAIN_FROM_SCRATCH=1 before sbatch, or rm main_play.pt.
# Resume like other neuner jobs: omit TRAIN_FROM_SCRATCH; existing main_play.pt loads.
#
# Submit from repo root:
#   TRAIN_FROM_SCRATCH=1 sbatch capstone_agent/jobs/neuner_simple_rand_vs_rand_gpu.slurm
#   sbatch capstone_agent/jobs/neuner_simple_rand_vs_rand_gpu.slurm
#
#SBATCH --job-name=neuner_simple_rr
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
echo "=== $(date -Is) neuner_simple_rand_vs_rand start cwd=$(pwd) ==="

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
# shellcheck source=/dev/null
source ~/.micromamba.sh
micromamba activate catan311

echo "=== GPU check ==="
nvidia-smi -L || true
python -u -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu-only')"

MODEL_DIR="$REPO_ROOT/capstone_agent/models/neuner_simple_rand_vs_rand"
mkdir -p "$MODEL_DIR/replays"

MAIN_PT="$MODEL_DIR/main_play.pt"
SEED_PT="${SEED_PT:-$REPO_ROOT/capstone_agent/models/capstone_model.pt}"
LOAD_ARGS=()
if [[ "${TRAIN_FROM_SCRATCH:-0}" == "1" ]]; then
  echo "TRAIN_FROM_SCRATCH=1: no --load (random init main + placement)."
elif [[ -n "${LOAD_MAIN:-}" ]]; then
  LOAD_ARGS+=(--load "$LOAD_MAIN")
elif [[ -f "$MAIN_PT" ]]; then
  LOAD_ARGS+=(--load "$MAIN_PT")
elif [[ -f "$SEED_PT" ]]; then
  LOAD_ARGS+=(--load "$SEED_PT")
fi

python -u capstone_agent/run_simulation.py \
  --train \
  --preset none \
  --reward-function simple \
  --enemy random \
  --placement-strategy random \
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
  --run-name neuner_simple_rr \
  --progress-env-steps 250 \
  "${LOAD_ARGS[@]}"
