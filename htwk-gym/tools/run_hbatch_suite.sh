#!/bin/bash
# Generate -> independent static/dynamic/video smoke -> launch only passing arms.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SESSION="${SESSION:-hbatch}"
ITERS="${ITERS:-12000}"
ENVS="${ENVS:-4096}"
VIDEO_S="${VIDEO_S:-8}"
CKPT="${CKPT:-logs/K1/K1/Goal_Pose_V7/2026-07-28-17-02-35_G1_speed/nn/model_10700.pth}"
FAIL_DIR="$REPO_ROOT/logs/hbatch/smoke_failures"
mkdir -p "$FAIL_DIR"

if [ ! -f "$CKPT" ]; then
  echo "!!! G1@10700 warm start missing: $CKPT" >&2
  exit 1
fi
python tools/make_hbatch_configs.py >/dev/null

PASS=()
for arm in H0 H1 H2 H3; do
  cfg="sweeps/hbatch/${arm}-codex.yaml"
  log=$(mktemp)
  echo "=== smoke $arm ==="
  if python tools/smoke_hbatch.py --config "$cfg" --checkpoint "$CKPT" \
       --sim_device cuda:0 --rl_device cuda:0 --steps 300 >"$log" 2>&1; then
    PASS+=("$arm")
    rm -f "$FAIL_DIR/${arm}-codex.log" "$log"
    echo "PASS $arm"
  else
    mv "$log" "$FAIL_DIR/${arm}-codex.log"
    echo "FAIL $arm -> $FAIL_DIR/${arm}-codex.log" >&2
  fi
done

if [ ${#PASS[@]} -eq 0 ]; then
  echo "!!! no HBatch arm passed smoke; nothing launched" >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "!!! tmux session already exists: $SESSION" >&2
  exit 1
fi

ENV_PRELUDE=""
if [ -n "${CONDA_PREFIX:-}" ]; then
  CONDA_BASE="$(conda info --base 2>/dev/null || echo "${CONDA_PREFIX%/envs/*}")"
  ENV_NAME="${CONDA_DEFAULT_ENV:-$(basename "$CONDA_PREFIX")}"
  ENV_PRELUDE="source '$CONDA_BASE/etc/profile.d/conda.sh' && conda activate '$ENV_NAME' && export LD_LIBRARY_PATH='$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}' &&"
fi

made=0
for i in "${!PASS[@]}"; do
  arm=${PASS[$i]}
  gpu=$((i % 2))
  cfg="sweeps/hbatch/${arm}-codex.yaml"
  cmd="$ENV_PRELUDE cd '$REPO_ROOT' && ITERS='$ITERS' ENVS='$ENVS' VIDEO_S='$VIDEO_S' bash tools/train_and_eval_hbatch.sh '$arm' '$cfg' '$CKPT' 'cuda:$gpu' 'cuda:$gpu'"
  if [ $made -eq 0 ]; then
    tmux new-session -d -s "$SESSION" -n "$arm" "$cmd; exec bash"
    made=1
  else
    tmux new-window -t "$SESSION" -n "$arm" "$cmd; exec bash"
  fi
  echo "launched $arm on GPU $gpu"
done

echo "passing arms: ${PASS[*]}"
echo "smoke failures only: $FAIL_DIR"
echo "attach: tmux attach -t $SESSION"
