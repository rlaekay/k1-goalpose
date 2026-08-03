#!/bin/bash
# Re-evaluate only the invalid E-batch arms on GPU 1, isolated from G-batch
# outputs. This is eval-only: no training, no checkpoint mutation.
#
# Usage on the training server:
#   bash tools/reeval_e_batch_gpu1.sh
#
# Optional overrides:
#   SESSION=e_reval_gpu1_b bash tools/reeval_e_batch_gpu1.sh
#   ARMS="E1_path V7_full" bash tools/reeval_e_batch_gpu1.sh
#   VIDEO_S=30 bash tools/reeval_e_batch_gpu1.sh
#   CONDA_ENV=k1goalpose bash tools/reeval_e_batch_gpu1.sh
#   ALLOW_BUSY_GPU=1 bash tools/reeval_e_batch_gpu1.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SESSION="${SESSION:-e_reval_gpu1}"
ARMS="${ARMS:-E1_path E2_robust V7_full}"
GPUS="${GPUS:-1}"
VIDEO_S="${VIDEO_S:-60}"
RUN_ROOT="${RUN_ROOT:-logs/K1/K1/Goal_Pose_V7}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/shared_eval_videos/reeval_e_batch_gpu1_$(date +%Y%m%d-%H%M%S)}"
ALLOW_BUSY_GPU="${ALLOW_BUSY_GPU:-0}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required on the training server." >&2
  exit 1
fi

if [ ! -d "$RUN_ROOT" ]; then
  echo "Run root not found: $RUN_ROOT" >&2
  echo "Run this on the training server, or sync logs first." >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1 && [ "$ALLOW_BUSY_GPU" != "1" ]; then
  BUSY="$(nvidia-smi -i "$GPUS" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' || true)"
  if [ -n "$BUSY" ]; then
    echo "GPU $GPUS already has compute processes; not starting re-eval." >&2
    echo "$BUSY" >&2
    echo "Re-run with ALLOW_BUSY_GPU=1 only if sharing GPU $GPUS is intentional." >&2
    exit 1
  fi
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION" >&2
  echo "Attach with: tmux attach -t $SESSION" >&2
  exit 1
fi

# Rebuild the conda env inside tmux. New tmux windows inherit the tmux server's
# original environment, not necessarily this shell's active conda env. This also
# supports non-interactive SSH launches where CONDA_PREFIX is empty.
ENV_PRELUDE=""
CONDA_ENV="${CONDA_ENV:-${CONDA_DEFAULT_ENV:-k1goalpose}}"
if [ -n "${CONDA_PREFIX:-}" ]; then
  CONDA_BASE="$(conda info --base 2>/dev/null || echo "${CONDA_PREFIX%/envs/*}")"
  ENV_NAME="${CONDA_DEFAULT_ENV:-$(basename "$CONDA_PREFIX")}"
  ENV_PRELUDE="source '$CONDA_BASE/etc/profile.d/conda.sh' && conda activate '$ENV_NAME' && export LD_LIBRARY_PATH=\"\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}\" &&"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  ENV_PRELUDE="source '$CONDA_BASE/etc/profile.d/conda.sh' && conda activate '$CONDA_ENV' && export LD_LIBRARY_PATH=\"\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-}\" &&"
else
  echo "No active conda env and conda is not on PATH." >&2
  echo "Activate the Isaac/K1 env first, or run with CONDA_ENV=<env-name> from a shell where conda is available." >&2
  exit 1
fi

mkdir -p "$OUT_ROOT"
# Persistent start marker for the read-only live monitor.  It is not inside a
# run/checkpoint directory, so it cannot affect selection or evaluation.
date '+%F %T' > "$OUT_ROOT/STARTED_AT"

CMD="$ENV_PRELUDE cd '$REPO_ROOT' && ARMS='$ARMS' GPUS='$GPUS' VIDEO_S='$VIDEO_S' RUN_ROOT='$RUN_ROOT' OUT_ROOT='$OUT_ROOT' bash tools/reeval_v7.sh; exec bash"
tmux new-session -d -s "$SESSION" -n reeval_e "$CMD"

echo "Started isolated E-batch re-eval."
echo "  session: $SESSION"
echo "  arms:    $ARMS"
echo "  gpu:     $GPUS"
echo "  output:  $OUT_ROOT"
echo "  attach:  tmux attach -t $SESSION"
echo "  log:     tail -f $OUT_ROOT/gpu${GPUS}.log"
