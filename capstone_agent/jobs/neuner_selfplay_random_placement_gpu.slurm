#!/bin/bash
# Self-play ladder: challenger (Blue) trains against a frozen champion (Red), both using
# random initial settlement/road placement. Blue: --placement-strategy random.
# Red: RL champion with --enemy-random-initial-build (random legal actions in initial build only).
#
# Starts from the latest main weights in neuner_phase_threshold_curriculum by default
# (--load PHASE_DIR/main_play.pt). Boots champion copies if missing (same run_simulation logic).
#
# Promotion: built-in eval is seat-balanced (half games challenger as Blue, half as Red), which
# removes first-player bias. Random openings still add variance: a 52% result over 500 games is
# only a modest edge (noise can flip outcomes). If promotions feel random, raise SELF_PLAY_EVAL_GAMES
# (e.g. 800–1200) before raising the threshold.
#
# Defaults: 10k training games per job; eval every 10k; promotion if challenger wins >= 52% over
# 500 eval games (override with env vars below).
#
# Export before sbatch (optional):
#   PHASE_DIR=.../neuner_phase_threshold_curriculum   # source main_play.pt
#   MODEL_DIR=.../neuner_selfplay_random_ladder       # challenger + champion saves
#   LOAD_MAIN=...                                     # override initial weights
#   GAMES=10000
#   SELF_PLAY_EVAL_EVERY=10000
#   SELF_PLAY_EVAL_GAMES=500
#   SELF_PLAY_PROMOTION_THRESHOLD=0.52
#
# Submit from repo root:
#   sbatch capstone_agent/jobs/neuner_selfplay_random_placement_gpu.slurm
#
#SBATCH --job-name=neuner_selfplay_r
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
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
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$REPO_ROOT/catanatron"
echo "=== $(date -Is) neuner_selfplay_random_placement cwd=$(pwd) ==="

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
# shellcheck source=/dev/null
source ~/.micromamba.sh
micromamba activate catan311

echo "=== GPU check ==="
nvidia-smi -L || true
python -u -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

PHASE_DIR="${PHASE_DIR:-$REPO_ROOT/capstone_agent/models/neuner_phase_threshold_curriculum}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/capstone_agent/models/neuner_selfplay_random_ladder}"
mkdir -p "$MODEL_DIR/replays"

SEED_PT="${SEED_PT:-$REPO_ROOT/capstone_model.pt}"
CHALLENGER_PT="$MODEL_DIR/challenger_main.pt"
CHAMPION_MAIN_PT="$MODEL_DIR/champion_main.pt"
CHAMPION_PLACE_PT="$MODEL_DIR/champion_placement_model.pt"

LOAD_ARGS=()
FRESH_ARGS=()
if [[ "${FRESH_START:-0}" == "1" ]]; then
  FRESH_ARGS+=(--fresh-start)
fi
if [[ "${TRAIN_FROM_SCRATCH:-0}" == "1" ]]; then
  echo "TRAIN_FROM_SCRATCH=1: no --load."
elif [[ -n "${LOAD_MAIN:-}" ]]; then
  echo "LOAD_MAIN: $LOAD_MAIN"
  LOAD_ARGS+=(--load "$LOAD_MAIN")
elif [[ -f "$CHALLENGER_PT" ]]; then
  echo "Resuming from existing challenger: $CHALLENGER_PT"
  LOAD_ARGS+=(--load "$CHALLENGER_PT")
elif [[ -f "$PHASE_DIR/main_play.pt" ]]; then
  echo "Bootstrapping from phase-threshold run: $PHASE_DIR/main_play.pt"
  LOAD_ARGS+=(--load "$PHASE_DIR/main_play.pt")
elif [[ -f "$SEED_PT" ]]; then
  LOAD_ARGS+=(--load "$SEED_PT")
fi

GAMES="${GAMES:-10000}"
SELF_PLAY_EVAL_EVERY="${SELF_PLAY_EVAL_EVERY:-10000}"
SELF_PLAY_EVAL_GAMES="${SELF_PLAY_EVAL_GAMES:-500}"
SELF_PLAY_PROMO="${SELF_PLAY_PROMOTION_THRESHOLD:-0.52}"

echo "GAMES=$GAMES  SELF_PLAY_EVAL_EVERY=$SELF_PLAY_EVAL_EVERY  SELF_PLAY_EVAL_GAMES=$SELF_PLAY_EVAL_GAMES  PROMO=$SELF_PLAY_PROMO"

python -u capstone_agent/run_simulation.py \
  --train \
  --preset none \
  --games "$GAMES" \
  --placement-strategy random \
  --enemy-random-initial-build \
  --self-play-ladder \
  --champion-main-model "$CHAMPION_MAIN_PT" \
  --champion-placement-model "$CHAMPION_PLACE_PT" \
  --self-play-eval-every-games "$SELF_PLAY_EVAL_EVERY" \
  --self-play-eval-games "$SELF_PLAY_EVAL_GAMES" \
  --self-play-promotion-threshold "$SELF_PLAY_PROMO" \
  --map-template TOURNAMENT \
  --map-mode fixed \
  --fixed-map-seed 0 \
  --rollout-collection-workers 1 \
  --reward-function "${REWARD_FUNCTION:-full}" \
  --save "$CHALLENGER_PT" \
  --save-every-games "${SAVE_EVERY_GAMES:-10000}" \
  --save-numbered-checkpoints \
  --benchmark-csv "$MODEL_DIR/benchmark_selfplay.csv" \
  --training-metrics-csv "$MODEL_DIR/ppo_training_selfplay.csv" \
  --run-name neuner_selfplay_random_ladder \
  --train-update-trigger steps \
  --train-every-steps 4096 \
  --progress-env-steps 250 \
  "${FRESH_ARGS[@]}" \
  "${LOAD_ARGS[@]}"
