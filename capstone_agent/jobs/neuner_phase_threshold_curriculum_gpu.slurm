#!/bin/bash
# Fixed TOURNAMENT board with random placement for both sides.
# Opponent curriculum phases:
#   random -> weighted -> mcts -> value -> alphabeta@1 -> alphabeta@2 -> alphabeta@3
# Advancement is win-rate gated PER PHASE (via --schedule-gate-spec), matching:
#   random:    90% over 10000 games
#   weighted:  90% over 10000 games
#   mcts:      80% over 1000 games
#   value:     70% over 10000 games
#   ab@1:      70% over 1000 games
#   ab@2:      60% over 1000 games
#   ab@3:      60% over 1000 games (terminal phase; gate won't advance further)
#
# Resume / override (export before sbatch):
#   LOAD_MAIN=/path/to/main_play.pt
#   TRAIN_FROM_SCRATCH=1
#   FRESH_START=1
#   SCHEDULE_INITIAL_PHASE=N              # 0-based index into schedule below
#   MODEL_DIR=...                         # where main_play.pt and logs live
#   ENEMY_MCTS_N=16                       # smaller MCTS opponent (default here: 16)
#   GATE_SPEC='0.9/10000,0.9/10000,0.8/1000,0.7/10000,0.7/1000,0.6/1000,0.6/1000'
#
# Submit from repo root:
#   sbatch capstone_agent/jobs/neuner_phase_threshold_curriculum_gpu.slurm
#
#SBATCH --job-name=neuner_phasegate
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
echo "=== $(date -Is) neuner_phase_threshold_curriculum start cwd=$(pwd) ==="

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
# shellcheck source=/dev/null
source ~/.micromamba.sh
micromamba activate catan311

echo "=== GPU check ==="
nvidia-smi -L || true
python -u -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/capstone_agent/models/neuner_phase_threshold_curriculum}"
mkdir -p "$MODEL_DIR/replays"

MAIN_PT="$MODEL_DIR/main_play.pt"
SEED_PT="${SEED_PT:-$REPO_ROOT/capstone_model.pt}"
LOAD_ARGS=()
FRESH_ARGS=()
if [[ "${FRESH_START:-0}" == "1" ]]; then
  FRESH_ARGS+=(--fresh-start)
  echo "FRESH_START=1: --fresh-start (will not auto-load from --save path if missing explicit --load)"
fi
if [[ "${TRAIN_FROM_SCRATCH:-0}" == "1" ]]; then
  echo "TRAIN_FROM_SCRATCH=1: no --load (random init)."
elif [[ -n "${LOAD_MAIN:-}" ]]; then
  echo "LOAD_MAIN: loading $LOAD_MAIN"
  LOAD_ARGS+=(--load "$LOAD_MAIN")
elif [[ -f "$MAIN_PT" ]]; then
  LOAD_ARGS+=(--load "$MAIN_PT")
elif [[ -f "$SEED_PT" ]]; then
  LOAD_ARGS+=(--load "$SEED_PT")
fi

# Placeholders for game-count schedule; win-rate gate decides advancement.
SCHEDULE="${SCHEDULE:-random:99999999,weighted:99999999,mcts:99999999,value:99999999,alphabeta@1:99999999,alphabeta@2:99999999,alphabeta@3:99999999}"
GATE_SPEC="${GATE_SPEC:-0.9/10000,0.9/10000,0.8/1000,0.7/10000,0.7/1000,0.6/1000,0.6/1000}"

GAMES="${GAMES:-20000000}"
ENEMY_MCTS_N="${ENEMY_MCTS_N:-16}"
SCHEDULE_INITIAL_PHASE="${SCHEDULE_INITIAL_PHASE:-0}"

echo "ENEMY_MCTS_N=$ENEMY_MCTS_N"
echo "GATE_SPEC=$GATE_SPEC"
if [[ "$SCHEDULE_INITIAL_PHASE" != "0" ]]; then
  echo "SCHEDULE_INITIAL_PHASE=$SCHEDULE_INITIAL_PHASE (0-based index into --enemy-schedule)"
fi

python -u capstone_agent/run_simulation.py \
  --train \
  --preset none \
  --difficulty medium \
  --games "$GAMES" \
  --placement-strategy random \
  --enemy-random-initial-build \
  --map-template TOURNAMENT \
  --map-mode fixed \
  --fixed-map-seed 0 \
  --enemy-fixed-schedule \
  --enemy-schedule "$SCHEDULE" \
  --schedule-advance-by-winrate \
  --schedule-gate-spec "$GATE_SPEC" \
  --schedule-initial-phase "$SCHEDULE_INITIAL_PHASE" \
  --schedule-gate-window 1000 \
  --schedule-gate-min-win-rate 0.6 \
  --enemy-mcts-n "$ENEMY_MCTS_N" \
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
  --run-name neuner_phase_threshold_curriculum \
  --progress-env-steps 250 \
  "${FRESH_ARGS[@]}" \
  "${LOAD_ARGS[@]}"
