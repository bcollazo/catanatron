#!/bin/bash
# Run on your LOCAL machine (laptop). Copies the largest-numbered
# main_play_game_*.pt from DCC for each neuner experiment into this repo.
#
# Uses a single SSH connection and streams one tar archive (stdout = data only;
# progress messages go to stderr on the remote), so you are not prompted once
# per experiment for ls + scp.
#
# Usage:
#   ./capstone_agent/jobs/pull_latest_game_checkpoints_from_dcc.sh
#
# Optional env:
#   DCC_SSH         default wjn7@dcc-login.oit.duke.edu
#   DCC_CATAN_PATH  absolute path to repo on DCC if not ~/catan_ai (no $ in value)
#   LOCAL_REPO      default parent of capstone_agent (this repo root)

set -euo pipefail

SSH_TARGET="${DCC_SSH:-wjn7@dcc-login.oit.duke.edu}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_REPO="${LOCAL_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
LOCAL_MODELS="$LOCAL_REPO/capstone_agent/models"

EXPS=(neuner_vs_alphabeta neuner_vs_random neuner_vs_random_ab_placement)

echo "SSH: $SSH_TARGET"
if [[ -n "${DCC_CATAN_PATH:-}" ]]; then
  echo "Remote: ${DCC_CATAN_PATH}/capstone_agent/models/<exp>/ (on DCC)"
else
  echo "Remote: \$HOME/catan_ai/capstone_agent/models/<exp>/ (on DCC)"
fi
echo "Local models: $LOCAL_MODELS"
echo "---"

mkdir -p "$LOCAL_MODELS"

# Safe for spaces/special chars in DCC_CATAN_PATH when set.
quoted_path="$(printf '%q' "${DCC_CATAN_PATH:-}")"

set -o pipefail
# Remote: only tar bytes on stdout; all logging on stderr.
if ! ssh "$SSH_TARGET" "DCC_CATAN_PATH=$quoted_path bash -s" <<'EOS' | tar xf - -C "$LOCAL_MODELS"
set -euo pipefail
if [[ -n "${DCC_CATAN_PATH:-}" ]]; then
  BASE="$DCC_CATAN_PATH"
else
  BASE="$HOME/catan_ai"
fi
MODELS="$BASE/capstone_agent/models"
cd "$MODELS" || { echo "Cannot cd to $MODELS" >&2; exit 1; }

files=()
for exp in neuner_vs_alphabeta neuner_vs_random neuner_vs_random_ab_placement; do
  f="$(ls -1v "$exp"/main_play_game_*.pt 2>/dev/null | tail -1)"
  if [[ -n "$f" && -f "$f" ]]; then
    files+=("$f")
    echo "[$exp] including $(basename "$f")" >&2
  else
    echo "[$exp] no main_play_game_*.pt (skip)" >&2
  fi
done

if ((${#files[@]} == 0)); then
  echo "No numbered checkpoints to pack." >&2
  exit 1
fi

tar cf - "${files[@]}"
EOS
then
  echo "Pull failed (SSH or tar extract). See messages above." >&2
  exit 1
fi

echo "Done."
