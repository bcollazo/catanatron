#!/bin/bash
# Train from a checkpoint with fixed curriculum:
#   value 100k -> alphabeta@1 100k -> alphabeta@2 100k -> alphabeta@3 100k (400k total).
# Saves every 10k games; optional numbered checkpoints.
# Uses parallel CPU rollout workers (--rollout-collection-workers); PPO stays on the GPU process.
#
# Submit (simple reward, default):
#   sbatch capstone_agent/jobs/neuner_value_ab123_curriculum_from_ckpt_gpu.slurm
#
# Submit (full / dense reward):
#   REWARD_FUNCTION=full sbatch capstone_agent/jobs/neuner_value_ab123_curriculum_from_ckpt_gpu.slurm
#
# Optional environment overrides:
#   MODEL_DIR=...              output dir (default includes reward name)
#   START_CKPT=/path/main.pt   explicit weights (highest priority)
#   START_MODEL_DIR=...        if START_CKPT unset: newest main_play_game_*.pt in dir, else main_play.pt
#   GAMES=400000
#   ROLLOUT_WORKERS=6          parallel game collection (match cpus-per-task roughly)
#   SAVE_EVERY_GAMES=10000
#   PLACEMENT_STRATEGY=model|random|alphabeta
#
#SBATCH --job-name=neuner_vab123
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
## SBATCH --account=YOUR_ACCOUNT

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/catan_ai}"
cd "$REPO_ROOT"
mkdir -p logs

export PYTHONUNBUFFERED=1
REWARD_FUNCTION="${REWARD_FUNCTION:-simple}"
if [[ "$REWARD_FUNCTION" != "simple" && "$REWARD_FUNCTION" != "full" ]]; then
  echo "REWARD_FUNCTION must be 'simple' or 'full' (got: $REWARD_FUNCTION)" >&2
  exit 1
fi

MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/capstone_agent/models/neuner_value_ab123_curriculum_${REWARD_FUNCTION}}"
mkdir -p "$MODEL_DIR/replays"

GAMES="${GAMES:-400000}"
SAVE_EVERY_GAMES="${SAVE_EVERY_GAMES:-10000}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-6}"
PLACEMENT="${PLACEMENT_STRATEGY:-}"

SCHEDULE="value:100000,alphabeta@1:100000,alphabeta@2:100000,alphabeta@3:100000"

LOAD_ARGS=()
if [[ -n "${START_CKPT:-}" ]]; then
  if [[ ! -f "$START_CKPT" ]]; then
    echo "START_CKPT is not a file: $START_CKPT" >&2
    exit 1
  fi
  LOAD_ARGS+=(--load "$START_CKPT")
elif [[ -n "${START_MODEL_DIR:-}" ]]; then
  if [[ ! -d "$START_MODEL_DIR" ]]; then
    echo "START_MODEL_DIR is not a directory: $START_MODEL_DIR" >&2
    exit 1
  fi
  PICK=""
  if compgen -G "$START_MODEL_DIR"/main_play_game_*.pt > /dev/null; then
    PICK="$(ls -t "$START_MODEL_DIR"/main_play_game_*.pt 2>/dev/null | head -1)"
  fi
  if [[ -z "$PICK" && -f "$START_MODEL_DIR/main_play.pt" ]]; then
    PICK="$START_MODEL_DIR/main_play.pt"
  fi
  if [[ -z "$PICK" ]]; then
    echo "No main_play.pt or main_play_game_*.pt under START_MODEL_DIR=$START_MODEL_DIR" >&2
    exit 1
  fi
  LOAD_ARGS+=(--load "$PICK")
else
  DEFAULT_PT="$REPO_ROOT/capstone_agent/models/neuner_simple_ab_ramp/main_play.pt"
  if [[ -f "$DEFAULT_PT" ]]; then
    LOAD_ARGS+=(--load "$DEFAULT_PT")
  else
    echo "Default checkpoint missing: $DEFAULT_PT (set START_CKPT or START_MODEL_DIR)" >&2
    exit 1
  fi
fi

echo "=== $(date -Is) neuner_value_ab123 curriculum start ==="
echo "REWARD_FUNCTION=$REWARD_FUNCTION MODEL_DIR=$MODEL_DIR GAMES=$GAMES"
echo "LOAD_ARGS=${LOAD_ARGS[*]:-none}"

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
# shellcheck source=/dev/null
source ~/.micromamba.sh
micromamba activate catan311

echo "=== GPU check ==="
nvidia-smi -L || true
python3 -u -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

MAIN_PT="$MODEL_DIR/main_play.pt"
EXTRA=()
if [[ -n "$PLACEMENT" ]]; then
  EXTRA+=(--placement-strategy "$PLACEMENT")
fi

python3 -u capstone_agent/run_simulation.py \
  --train \
  --preset none \
  --enemy-fixed-schedule \
  --enemy-schedule "$SCHEDULE" \
  --reward-function "$REWARD_FUNCTION" \
  --games "$GAMES" \
  --save "$MAIN_PT" \
  --save-every-games "$SAVE_EVERY_GAMES" \
  --save-numbered-checkpoints \
  --rollout-collection-workers "$ROLLOUT_WORKERS" \
  --benchmark-csv "$MODEL_DIR/benchmark_games.csv" \
  --training-metrics-csv "$MODEL_DIR/ppo_training.csv" \
  --save-games-json-dir "$MODEL_DIR/replays" \
  --save-games-json-every 100 \
  --train-update-trigger steps \
  --train-every-steps 4096 \
  --map-template TOURNAMENT \
  --map-mode fixed \
  --fixed-map-seed 0 \
  --run-name "neuner_value_ab123_${REWARD_FUNCTION}" \
  --progress-env-steps 250 \
  "${EXTRA[@]}" \
  "${LOAD_ARGS[@]}"
