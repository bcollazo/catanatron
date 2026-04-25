#!/bin/bash
# Same as neuner_tournament_map_winrate_curriculum_gpu.slurm, but the curriculum skips the
# mcts segment (MCTS is slow). Order: random -> weighted -> value -> alphabeta@1/2/3.
# Rolling win-rate gate default: 70% over the last 1000 games to advance.
#
# Resume / override (export before sbatch):
#   LOAD_MAIN=...  TRAIN_FROM_SCRATCH=1  FRESH_START=1
#   SCHEDULE_INITIAL_PHASE=N   # 0-based: no-mcts schedule has 6 phases (0..5)
#   MODEL_DIR=...  SCHEDULE_GATE_MIN_WIN_RATE=0.6  # override 70% default if needed
#
# Submit from repo root:
#   sbatch capstone_agent/jobs/neuner_tournament_map_winrate_curriculum_no_mcts_gpu.slurm
#
#SBATCH --job-name=neuner_wr_nomc
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
echo "=== $(date -Is) neuner_tournament_map_winrate_curriculum_no_mcts cwd=$(pwd) ==="

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
# shellcheck source=/dev/null
source ~/.micromamba.sh
micromamba activate catan311

echo "=== GPU check ==="
nvidia-smi -L || true
python -u -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/capstone_agent/models/neuner_tournament_map_wr_curriculum_no_mcts}"
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

# No mcts segment. Placeholders; phase length is win-rate gated.
SCHEDULE="${SCHEDULE:-random:99999999,weighted:99999999,value:99999999,alphabeta@1:99999999,alphabeta@2:99999999,alphabeta@3:99999999}"

GAMES="${GAMES:-20000000}"
SCHEDULE_INITIAL_PHASE="${SCHEDULE_INITIAL_PHASE:-0}"
if [[ "$SCHEDULE_INITIAL_PHASE" != "0" ]]; then
  echo "SCHEDULE_INITIAL_PHASE=$SCHEDULE_INITIAL_PHASE (0-based; this schedule has 6 phases, indices 0..5)"
fi

# Default 70% rolling win rate in last 1000 games to advance; override e.g. SCHEDULE_GATE_MIN_WIN_RATE=0.6
GATE_WR="${SCHEDULE_GATE_MIN_WIN_RATE:-0.7}"
echo "SCHEDULE_GATE_MIN_WIN_RATE=$GATE_WR"

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
  --schedule-initial-phase "$SCHEDULE_INITIAL_PHASE" \
  --schedule-gate-window "${SCHEDULE_GATE_WINDOW:-1000}" \
  --schedule-gate-min-win-rate "$GATE_WR" \
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
  --run-name neuner_tournament_map_wr_curriculum_no_mcts \
  --progress-env-steps 250 \
  "${FRESH_ARGS[@]}" \
  "${LOAD_ARGS[@]}"
