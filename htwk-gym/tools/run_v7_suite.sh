#!/bin/bash
# Launch the full v7 ablation ladder across both A6000s, gated on a smoke test.
#
#   bash tools/run_v7_suite.sh
#   ITERS=8000 bash tools/run_v7_suite.sh          # shorter runs
#   CKPT=logs/.../model_11500.pth bash tools/run_v7_suite.sh
#   SKIP_SMOKE=1 bash tools/run_v7_suite.sh        # only if smoke already passed
#
# The smoke test is a HARD gate: v7 has never been executed, and a silent bug in
# the path/disturbance machinery would burn GPU-days producing numbers that mean
# nothing. It costs about a minute.
#
# Layout (2 processes per GPU; the earlier sweep showed the bottleneck is a
# single CPU core per process, not VRAM -- 3 procs fit in 14/49 GB):
#   GPU 0: E0_armB_armsdown   E1_path
#   GPU 1: E2_robust          V7_full
#
# Each window trains, then searches for its best checkpoint, evaluates it,
# records a video, runs the goal-jitter stress eval, and drops everything in
# htwk-gym/shared_eval_videos/.
#
# Watch:    tmux attach -t v7
# Windows:  tmux list-windows -t v7
# Kill one: tmux kill-window -t v7:E2_robust
# Kill all: tmux kill-session -t v7

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SESSION="${SESSION:-v7}"
ITERS="${ITERS:-12000}"
ENVS="${ENVS:-4096}"
VIDEO_S="${VIDEO_S:-60}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
CKPT="${CKPT:-logs/K1/K1/Goal_Pose/2026-07-24-17-22-03_armB_goal_reached/nn/model_11500.pth}"

if [ ! -f "$CKPT" ]; then
  echo "!!! warm-start 체크포인트가 없습니다: $CKPT" >&2
  echo "    CKPT=<경로> 로 지정하거나, 처음부터 학습하려면 CKPT=null 로 두십시오." >&2
  exit 1
fi

# ---- gate 1: does the v7 machinery actually work? --------------------------
if [ "$SKIP_SMOKE" != "1" ]; then
  echo "=== [1/3] 스모크 테스트 (v7은 아직 한 번도 실행된 적이 없음) ==="
  if ! python tools/smoke_v7.py --sim_device cuda:0 --rl_device cuda:0 --steps 300; then
    echo "" >&2
    echo "!!! 스모크 테스트 실패 — 학습을 시작하지 않습니다." >&2
    echo "    위의 FAIL 항목을 고친 뒤 다시 실행하십시오." >&2
    exit 1
  fi
else
  echo "=== [1/3] 스모크 테스트 건너뜀 (SKIP_SMOKE=1) ==="
fi

# ---- gate 2: generate the ladder -------------------------------------------
echo ""
echo "=== [2/3] 실험 config 생성 ==="
python tools/make_v7_arms.py --checkpoint "$CKPT" --num_envs "$ENVS" --max_iterations "$ITERS"

# ---- launch ----------------------------------------------------------------
echo ""
echo "=== [3/3] tmux 세션 '$SESSION' 실행 ==="
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "!!! tmux 세션 '$SESSION'이 이미 있습니다. 지우려면: tmux kill-session -t $SESSION" >&2
  exit 1
fi

launch() {  # name gpu
  local name=$1 gpu=$2
  local cmd="cd $REPO_ROOT && TRAIN=train_v7.py STRESS=1 VIDEO_S=$VIDEO_S \
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

launch E0_armB_armsdown 0
launch E1_path          0
launch E2_robust        1
launch V7_full          1

echo ""
echo "완료. 4개 실행 중 (GPU당 2개)."
echo "  확인:  tmux attach -t $SESSION      (분리: Ctrl-b d)"
echo "  진행:  watch -n30 nvidia-smi"
echo "  결과:  $REPO_ROOT/shared_eval_videos/"
echo ""
echo "각 arm이 답하는 질문:"
echo "  E0_armB_armsdown  팔 내린 URDF가 warm start를 견디나? 대칭손실이 도움이 되나?"
echo "  E1_path           path mode가 실제로 몸통 속도를 올리나?  <- 핵심 질문"
echo "  E2_robust         강건성 강화가 게이트에 얼마나 비용을 물리나?"
echo "  V7_full           통합 후보"
