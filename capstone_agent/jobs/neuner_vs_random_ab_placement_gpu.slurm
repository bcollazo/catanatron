#!/bin/bash
# Train main-play policy vs RandomPlayer with AlphaBeta placement for our agent only.
# (Learned main play + AB setup; opponent remains full random.)
# GPU job for Duke DCC — submit from repo root:
#   sbatch capstone_agent/jobs/neuner_vs_random_ab_placement_gpu.slurm
#
# Outputs under capstone_agent/models/neuner_vs_random_ab_placement/:
#   main_play.pt              — latest weights (also final save)
#   main_play_game_*.pt       — copy every --save-every-games (--save-numbered-checkpoints)
#   benchmark_games.csv       — per-game stats
#   ppo_training.csv          — one row per PPO update
#   replays/*.json            — every 100th completed game
#
# Resume: existing main_play.pt in MODEL_DIR; else capstone_agent/models/capstone_model.pt
#         (override with LOAD_MAIN or SEED_PT).
# Tune setup strength: PLACEMENT_AB_DEPTH (default 2), optional --no-placement-ab-prunning via editing script.
#
# Wall time: raise only up to partition MaxTime.

#SBATCH --job-name=neuner_rand_abp
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
echo "=== $(date -Is) neuner_vs_random_ab_placement start cwd=$(pwd) ==="

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
# shellcheck source=/dev/null
source ~/.micromamba.sh
micromamba activate catan311

echo "=== GPU check ==="
nvidia-smi -L || true
python -u -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu-only')"

MODEL_DIR="$REPO_ROOT/capstone_agent/models/neuner_vs_random_ab_placement"
mkdir -p "$MODEL_DIR/replays"

MAIN_PT="$MODEL_DIR/main_play.pt"
SEED_PT="${SEED_PT:-$REPO_ROOT/capstone_agent/models/capstone_model.pt}"
LOAD_ARGS=()
if [[ -n "${LOAD_MAIN:-}" ]]; then
  LOAD_ARGS+=(--load "$LOAD_MAIN")
elif [[ -f "$MAIN_PT" ]]; then
  LOAD_ARGS+=(--load "$MAIN_PT")
elif [[ -f "$SEED_PT" ]]; then
  LOAD_ARGS+=(--load "$SEED_PT")
fi

python -u capstone_agent/run_simulation.py \
  --train \
  --enemy random \
  --placement-strategy alphabeta \
  --placement-ab-depth "${PLACEMENT_AB_DEPTH:-2}" \
  --placement-ab-prunning \
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
  --run-name neuner_vs_random_ab_placement \
  --progress-env-steps 250 \
  "${LOAD_ARGS[@]}"
