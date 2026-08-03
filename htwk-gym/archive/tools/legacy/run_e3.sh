#!/bin/bash
# Launch E3_wide_nosched alone, into the existing v7 tmux session.
#
#   bash tools/run_e3.sh          # GPU 0 (default)
#   GPU=1 bash tools/run_e3.sh
#
# E3 removes the speed SCHEDULER from E1 and replaces it with one wide fixed
# commanded-speed distribution, U(0.2, 2.0) m/s per env. Compare against E1
# (same everything, curriculum on) to answer: is a scheduler worth its moving
# parts at all?
#
# The payoff either way is the "명령속도 vs 실제속도" table in E3's report: the
# knee of that curve is K1's physical speed ceiling, measured rather than
# assumed. Nothing we have run so far can produce that number.
#
# Run this when a GPU slot frees up -- i.e. after one of the four v7 arms
# finishes. Check with:
#   nvidia-smi --query-compute-apps=pid,used_memory --format=csv
#   tmux list-windows -t v7

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:-0}"
ITERS="${ITERS:-12000}"
ENVS="${ENVS:-4096}"
VIDEO_S="${VIDEO_S:-60}"
SESSION="${SESSION:-v7}"
NAME="E3_wide_nosched"
CKPT="${CKPT:-logs/K1/K1/Goal_Pose/2026-07-24-17-22-03_armB_goal_reached/nn/model_11500.pth}"

[ -f "$CKPT" ] || { echo "!!! warm-start 체크포인트 없음: $CKPT" >&2; exit 1; }

if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "!!! conda 환경이 활성화돼 있지 않습니다. 'conda activate k1goalpose' 후 실행하십시오." >&2
  exit 1
fi
# tmux windows inherit the tmux SERVER's environment, not this shell's -- see
# run_v7_suite.sh for the full story (isaacgym dies on libpython3.8.so.1.0).
CONDA_BASE="$(conda info --base 2>/dev/null || echo "${CONDA_PREFIX%/envs/*}")"
ENV_NAME="${CONDA_DEFAULT_ENV:-$(basename "$CONDA_PREFIX")}"
PRELUDE="source '$CONDA_BASE/etc/profile.d/conda.sh' && conda activate '$ENV_NAME' && export LD_LIBRARY_PATH='$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}' &&"

echo "=== E3 config 생성 ==="
python tools/make_v7_arms.py --only "$NAME" --checkpoint "$CKPT" \
  --num_envs "$ENVS" --max_iterations "$ITERS"

CMD="$PRELUDE cd $REPO_ROOT && TRAIN=train_v7.py STRESS=1 VIDEO_S=$VIDEO_S \
bash tools/train_and_eval.sh cuda:$GPU cuda:$GPU -- \
--task=K1/Goal_Pose_V7 --config sweeps/$NAME.yaml --headless True \
--checkpoint $CKPT --num_envs $ENVS --max_iterations $ITERS \
--sim_device cuda:$GPU --rl_device cuda:$GPU"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux new-window -t "$SESSION" -n "$NAME" "$CMD; exec bash"
else
  tmux new-session -d -s "$SESSION" -n "$NAME" "$CMD; exec bash"
fi
echo "=== $NAME 를 GPU $GPU 에 띄웠습니다 (tmux $SESSION:$NAME) ==="

echo "기동 확인 중 (90s)..."
sleep 90
pane_pid=$(tmux list-panes -t "$SESSION:$NAME" -F '#{pane_pid}' | head -1)
if pstree -p "$pane_pid" 2>/dev/null | grep -q python; then
  echo "  ✅ $NAME 살아 있음"
  echo "  진행 확인: tmux capture-pane -p -t $SESSION:$NAME | tail -20"
else
  echo "  ❌ $NAME 죽었음 — tmux capture-pane -p -t $SESSION:$NAME | tail -30" >&2
  exit 1
fi
