#!/bin/bash
# Fixed-game-count ladder (NO win-rate gating), random placement for both sides.
#
# Default ladder (no mcts):
#   random:100000,
#   weighted:100000,
#   value:200000,
#   alphabeta@1:200000,
#   alphabeta@2:200000,
#   alphabeta@3:200000
#
# Starts from no_mcts curriculum weights by default:
#   ~/catan_ai/capstone_agent/models/neuner_tournament_map_wr_curriculum_no_mcts/main_play.pt
#
# START_PHASE is 0-based index into the ladder above:
#   0=random, 1=weighted, 2=value, 3=alphabeta@1, 4=alphabeta@2, 5=alphabeta@3
#
# Submit from repo root:
#   sbatch capstone_agent/jobs/neuner_random_placement_fixed_ladder_v2_gpu.slurm
#
# Optional overrides before sbatch:
#   START_PHASE=2
#   GAMES=300000
#   LOAD_MAIN=/path/to/model.pt
#   MODEL_DIR=.../neuner_random_placement_fixed_ladder_v2
#   SOURCE_DIR=.../neuner_tournament_map_wr_curriculum_no_mcts
#
#SBATCH --job-name=neuner_fixed_rp_v2
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
echo "=== $(date -Is) neuner_random_placement_fixed_ladder_v2 cwd=$(pwd) ==="

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
# shellcheck source=/dev/null
source ~/.micromamba.sh
micromamba activate catan311

echo "=== GPU check ==="
nvidia-smi -L || true
python -u -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/capstone_agent/models/neuner_random_placement_fixed_ladder_v2}"
SOURCE_DIR="${SOURCE_DIR:-$REPO_ROOT/capstone_agent/models/neuner_tournament_map_wr_curriculum_no_mcts}"
mkdir -p "$MODEL_DIR/replays"

MAIN_PT="$MODEL_DIR/main_play.pt"
SEED_PT="${SEED_PT:-$REPO_ROOT/capstone_model.pt}"
SOURCE_MAIN_PT="$SOURCE_DIR/main_play.pt"

LOAD_ARGS=()
if [[ "${TRAIN_FROM_SCRATCH:-0}" == "1" ]]; then
  echo "TRAIN_FROM_SCRATCH=1: no --load."
elif [[ -n "${LOAD_MAIN:-}" ]]; then
  echo "LOAD_MAIN: loading $LOAD_MAIN"
  LOAD_ARGS+=(--load "$LOAD_MAIN")
elif [[ -f "$MAIN_PT" ]]; then
  echo "Resuming from existing MAIN_PT: $MAIN_PT"
  LOAD_ARGS+=(--load "$MAIN_PT")
elif [[ -f "$SOURCE_MAIN_PT" ]]; then
  echo "Bootstrapping from SOURCE_MAIN_PT: $SOURCE_MAIN_PT"
  LOAD_ARGS+=(--load "$SOURCE_MAIN_PT")
elif [[ -f "$SEED_PT" ]]; then
  LOAD_ARGS+=(--load "$SEED_PT")
fi

PHASE_NAMES=("random" "weighted" "value" "alphabeta@1" "alphabeta@2" "alphabeta@3")
PHASE_GAMES=(100000 100000 200000 200000 200000 200000)

START_PHASE="${START_PHASE:-0}"
if ! [[ "$START_PHASE" =~ ^[0-9]+$ ]]; then
  echo "ERROR: START_PHASE must be an integer >= 0" >&2
  exit 1
fi
if (( START_PHASE < 0 || START_PHASE >= ${#PHASE_NAMES[@]} )); then
  echo "ERROR: START_PHASE=$START_PHASE out of range (0..$(( ${#PHASE_NAMES[@]} - 1 )))" >&2
  exit 1
fi

SCHEDULE=""
remaining_games=0
for ((i=START_PHASE; i<${#PHASE_NAMES[@]}; i++)); do
  token="${PHASE_NAMES[$i]}:${PHASE_GAMES[$i]}"
  if [[ -z "$SCHEDULE" ]]; then
    SCHEDULE="$token"
  else
    SCHEDULE="$SCHEDULE,$token"
  fi
  remaining_games=$((remaining_games + PHASE_GAMES[$i]))
done

GAMES="${GAMES:-$remaining_games}"

echo "START_PHASE=$START_PHASE"
echo "SCHEDULE=$SCHEDULE"
echo "GAMES=$GAMES"

python -u capstone_agent/run_simulation.py \
  --train \
  --preset none \
  --games "$GAMES" \
  --placement-strategy random \
  --enemy-random-initial-build \
  --enemy-fixed-schedule \
  --enemy-schedule "$SCHEDULE" \
  --map-template TOURNAMENT \
  --map-mode fixed \
  --fixed-map-seed 0 \
  --rollout-collection-workers 1 \
  --reward-function "${REWARD_FUNCTION:-full}" \
  --save "$MAIN_PT" \
  --save-every-games "${SAVE_EVERY_GAMES:-10000}" \
  --save-numbered-checkpoints \
  --benchmark-csv "$MODEL_DIR/benchmark_games.csv" \
  --training-metrics-csv "$MODEL_DIR/ppo_training.csv" \
  --save-games-json-dir "$MODEL_DIR/replays" \
  --save-games-json-every 0 \
  --train-update-trigger steps \
  --train-every-steps 4096 \
  --run-name neuner_random_placement_fixed_ladder_v2 \
  --progress-env-steps 250 \
  "${LOAD_ARGS[@]}"
