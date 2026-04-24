#!/bin/bash
# Fixed TOURNAMENT board; random *placement* (settlements/roads) for us (--placement-strategy random)
# and for scripted opponents (--enemy-random-initial-build). Main-play curriculum advances only on
# rolling win rate (default: 60% over the last 1000 games) before the next opponent.
# Phases: random -> weighted -> mcts -> value -> alphabeta@1 -> @2 -> @3 (placeholder :games ignored).
# Checkpoints every 10,000 games.
#
# Resume / override (export before sbatch):
#   LOAD_MAIN=/path/to/main_play.pt     # explicit main weights (overrides auto pick below)
#   TRAIN_FROM_SCRATCH=1                # random init, no --load
#   FRESH_START=1                      # ignore existing --save weights for load (still set --save)
#   SCHEDULE_INITIAL_PHASE=4            # 0-based; start curriculum at 5th enemy (e.g. skip to value/AB)
#   MODEL_DIR=...                       # where main_play.pt and logs live
#
# Submit from repo root:
#   sbatch capstone_agent/jobs/neuner_tournament_map_winrate_curriculum_gpu.slurm
# Shorter walltime if needed (overrides #SBATCH below), e.g. 24h:
#   sbatch --time=24:00:00 capstone_agent/jobs/neuner_tournament_map_winrate_curriculum_gpu.slurm
#
#SBATCH --job-name=neuner_wr_gate
#SBATCH --partition=gpu-common
#SBATCH --gres=gpu:1
# 48 hours — max allowed on DCC for this queue (do not exceed).
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
echo "=== $(date -Is) neuner_tournament_map_winrate_curriculum start cwd=$(pwd) ==="

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/micromamba}"
# shellcheck source=/dev/null
source ~/.micromamba.sh
micromamba activate catan311

echo "=== GPU check ==="
nvidia-smi -L || true
python -u -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/capstone_agent/models/neuner_tournament_map_wr_curriculum}"
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

# Large placeholders; actual phase length is win-rate gated.
SCHEDULE="${SCHEDULE:-random:99999999,weighted:99999999,mcts:99999999,value:99999999,alphabeta@1:99999999,alphabeta@2:99999999,alphabeta@3:99999999}"

GAMES="${GAMES:-20000000}"
SCHEDULE_INITIAL_PHASE="${SCHEDULE_INITIAL_PHASE:-0}"
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
  --schedule-initial-phase "$SCHEDULE_INITIAL_PHASE" \
  --schedule-gate-window "${SCHEDULE_GATE_WINDOW:-1000}" \
  --schedule-gate-min-win-rate "${SCHEDULE_GATE_MIN_WIN_RATE:-0.6}" \
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
  --run-name neuner_tournament_map_wr_curriculum \
  --progress-env-steps 250 \
  "${FRESH_ARGS[@]}" \
  "${LOAD_ARGS[@]}"
