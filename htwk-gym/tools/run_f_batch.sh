#!/bin/bash
# Launch the F1-F3 batch (masterplan2.md §8) into a new tmux session.
#
#   bash tools/run_f_batch.sh
#   ITERS=8000 bash tools/run_f_batch.sh
#   CKPT=<path> bash tools/run_f_batch.sh   # override the warm-start for all three
#
# F1_timed  GPU 0   Rudin timed-window gate, never measured anywhere before
# F2_grid   GPU 0   fixed lookahead (pace/floor/leash) + grid-adaptive (speed x
#                   curvature) curriculum -- THE arm masterplan2 #2/#3 exists for
# F3_stress GPU 1   BT flicker 0.004 -> 0.01, alone on its GPU (heavier: full
#                   disturbance + higher flicker)
#
# All three warm-start from E0's OWN final checkpoint (already adapted to the
# arms-down URDF for 12000 iters), not from armB directly -- see
# tools/make_v7_arms.py's ARMSDOWN_CKPT comment for why, and for the caveat
# that the URDF was rebuilt 80->90 deg + shoulder tuck AFTER E0 finished, so
# there is one more small dynamics step on top of what E0 already absorbed.
#
# Same smoke gate as run_v7_suite.sh -- the grid curriculum and the fixed
# lookahead are new code, exercised for the first time here.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SESSION="${SESSION:-f}"
ITERS="${ITERS:-12000}"
ENVS="${ENVS:-4096}"
VIDEO_S="${VIDEO_S:-60}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
CKPT="${CKPT:-logs/K1/K1/Goal_Pose_V7/2026-07-26-19-36-15_E0_armB_armsdown/nn/model_12000.pth}"

if [ ! -f "$CKPT" ]; then
  echo "!!! warm-start 체크포인트가 없습니다: $CKPT" >&2
  echo "    E0의 실제 최종 체크포인트 경로를 확인하십시오:" >&2
  echo "    ls logs/K1/K1/Goal_Pose_V7/*E0*/nn/model_12000.pth" >&2
  exit 1
fi

if [ "$SKIP_SMOKE" != "1" ]; then
  echo "=== [1/3] 스모크 테스트 (grid 커리큘럼 + 새 lookahead는 이번이 첫 실행) ==="
  if ! python tools/smoke_v7.py --checkpoint "$CKPT" --sim_device cuda:0 --rl_device cuda:0 --steps 300; then
    echo "!!! 스모크 테스트 실패 — 학습을 시작하지 않습니다." >&2
    exit 1
  fi
else
  echo "=== [1/3] 스모크 테스트 건너뜀 (SKIP_SMOKE=1) ==="
fi

echo ""
echo "=== [2/3] F1-F3 config 생성 ==="
for arm in F1_timed F2_grid F3_stress; do
  python tools/make_v7_arms.py --only "$arm" --checkpoint "$CKPT" --num_envs "$ENVS" --max_iterations "$ITERS"
done

echo ""
echo "=== [3/3] tmux 세션 '$SESSION' 실행 ==="
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "!!! tmux 세션 '$SESSION'이 이미 있습니다. 지우려면: tmux kill-session -t $SESSION" >&2
  exit 1
fi

if [ -z "${CONDA_PREFIX:-}" ]; then
  echo "!!! conda 환경이 활성화돼 있지 않습니다. 'conda activate k1goalpose' 후 실행하십시오." >&2
  exit 1
fi
CONDA_BASE="$(conda info --base 2>/dev/null || echo "${CONDA_PREFIX%/envs/*}")"
ENV_NAME="${CONDA_DEFAULT_ENV:-$(basename "$CONDA_PREFIX")}"
PRELUDE="source '$CONDA_BASE/etc/profile.d/conda.sh' && conda activate '$ENV_NAME' && export LD_LIBRARY_PATH='$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}' &&"

launch() {  # name gpu
  local name=$1 gpu=$2
  local cmd="$PRELUDE cd $REPO_ROOT && TRAIN=train_v7.py STRESS=1 VIDEO_S=$VIDEO_S \
bash tools/train_and_eval.sh cuda:$gpu cuda:$gpu -- \
--task=K1/Goal_Pose_V7 --config sweeps/$name.yaml --headless True \
--checkpoint $CKPT --num_envs $ENVS --max_iterations $ITERS \
--sim_device cuda:$gpu --rl_device cuda:$gpu"
  if [ -z "${_SESSION_MADE:-}" ]; then
    tmux new-session -d -s "$SESSION" -n "$name" "$cmd; exec bash"
    _SESSION_MADE=1
  else
    tmux new-window -t "$SESSION" -n "$name" "$cmd; exec bash"
  fi
  echo "  launched $name on GPU $gpu"
}

launch F1_timed  0
launch F2_grid   0
launch F3_stress 1

echo ""
echo "기동 확인 중 (90s)..."
sleep 90
DEAD=""
for name in F1_timed F2_grid F3_stress; do
  pane_pid=$(tmux list-panes -t "$SESSION:$name" -F '#{pane_pid}' 2>/dev/null | head -1)
  if [ -n "$pane_pid" ] && pstree -p "$pane_pid" 2>/dev/null | grep -q python; then
    echo "  ✅ $name 살아 있음"
  else
    echo "  ❌ $name 죽었음"
    DEAD="$DEAD $name"
  fi
done
if [ -n "$DEAD" ]; then
  echo "!!! 죽은 arm:$DEAD — tmux capture-pane -p -t $SESSION:<이름> | tail -30" >&2
  exit 1
fi

echo ""
echo "완료. GPU 0: F1_timed + F2_grid   |   GPU 1: F3_stress"
echo "  진행:  python tools/progress.py"
echo "  확인:  tmux attach -t $SESSION"
