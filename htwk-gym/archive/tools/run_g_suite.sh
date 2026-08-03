#!/bin/bash
# Launch the G batch (4 arms) across both A6000s, gated on a smoke test.
#
#   bash tools/run_g_suite.sh
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
#   GPU 0: G1_speed   G2_robust
#   GPU 1: G3_full    G4_smoothturn
#
# G4 is the SmoothTurn arm and runs task K1/Goal_Pose_V8 from its own base
# config; the other three are K1/Goal_Pose_V7. Everything warm-starts from
# E0@6200, the corrected best (2.7 cm / 89% strict / 2 falls).
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

# Undefined names cost a 100-second GPU run to discover on 2026-07-27
# (CATEGORY_PATH in summarize()): ast.parse only checks syntax, so a NameError
# waits until that branch executes. This is instant -- run it before anything.
if ! python tools/check_names.py >/tmp/check_names.$$ 2>&1; then
  grep -v "import \*" /tmp/check_names.$$ | grep UNDEFINED && {
    echo "!!! 정의되지 않은 이름이 있습니다 — 실행을 중단합니다." >&2
    rm -f /tmp/check_names.$$
    exit 1
  }
fi
rm -f /tmp/check_names.$$

SESSION="${SESSION:-g}"
ITERS="${ITERS:-12000}"
ENVS="${ENVS:-4096}"
VIDEO_S="${VIDEO_S:-60}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
CKPT="${CKPT:-logs/K1/K1/Goal_Pose_V7/2026-07-26-19-36-15_E0_armB_armsdown/nn/model_6200.pth}"

if [ ! -f "$CKPT" ]; then
  echo "!!! warm-start 체크포인트가 없습니다: $CKPT" >&2
  echo "    CKPT=<경로> 로 지정하거나, 처음부터 학습하려면 CKPT=null 로 두십시오." >&2
  exit 1
fi

# ---- rebuild this shell's environment inside each tmux window ---------------
# A tmux window does NOT inherit the launching shell's environment: if a tmux
# server is already running, new windows get the environment that server was
# started with. That server predates the conda activation, so isaacgym dies with
# "libpython3.8.so.1.0: cannot open shared object file" -- while the smoke test,
# run directly in this shell, passes. Reconstruct the env explicitly.
ENV_PRELUDE=""
if [ -n "${CONDA_PREFIX:-}" ]; then
  CONDA_BASE="$(conda info --base 2>/dev/null || echo "${CONDA_PREFIX%/envs/*}")"
  ENV_NAME="${CONDA_DEFAULT_ENV:-$(basename "$CONDA_PREFIX")}"
  ENV_PRELUDE="source '$CONDA_BASE/etc/profile.d/conda.sh' && conda activate '$ENV_NAME' && export LD_LIBRARY_PATH='$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}' &&"
  echo "conda 환경 '$ENV_NAME'을 각 tmux 창에서 재활성화합니다."
else
  echo "!!! conda 환경이 활성화돼 있지 않습니다. 'conda activate k1goalpose' 후 다시 실행하십시오." >&2
  exit 1
fi

# ---- gate 1: does the v7 machinery actually work? --------------------------
if [ "$SKIP_SMOKE" != "1" ]; then
  echo "=== [1/3] 스모크 테스트 (v7은 아직 한 번도 실행된 적이 없음) ==="
  # Each arm gets its own smoke run: G1 carries the lookahead floor + dwell +
  # grid and G4 carries the whole sequential-navigation stack, none of which has
  # ever executed. A single smoke on the base config would not touch them.
  for arm in G1_speed G2_robust G3_full G4_smoothturn; do
    python tools/make_v7_arms.py --only "$arm" --checkpoint "$CKPT" >/dev/null
    task="K1/Goal_Pose_V7"
    [ "$arm" = "G4_smoothturn" ] && task="K1/Goal_Pose_V8"
    echo "--- smoke: $arm ($task) ---"
    if ! python tools/smoke_v7.py --config "sweeps/$arm.yaml" --task "$task" \
         --checkpoint "$CKPT" --sim_device cuda:0 --rl_device cuda:0 --steps 300; then
      echo "" >&2
      echo "!!! $arm 스모크 실패 — 학습을 시작하지 않습니다." >&2
      exit 1
    fi
  done
else
  echo "=== [1/3] 스모크 테스트 건너뜀 (SKIP_SMOKE=1) ==="
fi

# ---- gate 2: generate the ladder -------------------------------------------
echo ""
echo "=== [2/3] 실험 config 생성 ==="
for arm in G1_speed G2_robust G3_full G4_smoothturn; do
  python tools/make_v7_arms.py --only "$arm" --checkpoint "$CKPT" \
    --num_envs "$ENVS" --max_iterations "$ITERS"
done

# ---- launch ----------------------------------------------------------------
echo ""
echo "=== [3/3] tmux 세션 '$SESSION' 실행 ==="
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "!!! tmux 세션 '$SESSION'이 이미 있습니다. 지우려면: tmux kill-session -t $SESSION" >&2
  exit 1
fi

launch() {  # name gpu
  local name=$1 gpu=$2 task="K1/Goal_Pose_V7"
  [ "$name" = "G4_smoothturn" ] && task="K1/Goal_Pose_V8"
  local cmd="$ENV_PRELUDE cd $REPO_ROOT && TRAIN=train_v7.py STRESS=1 VIDEO_S=$VIDEO_S \
bash tools/train_and_eval.sh cuda:$gpu cuda:$gpu -- \
--task=$task --config sweeps/$name.yaml --headless True \
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

launch G1_speed      0
launch G2_robust     0
launch G3_full       1
launch G4_smoothturn 1

# ---- verify they are actually alive ----------------------------------------
# Previously this script announced "4개 실행 중" the instant tmux accepted the
# windows, which it does even when every process dies on the first import.
WAIT="${HEALTHCHECK_S:-90}"
echo ""
echo "기동 확인 중 (${WAIT}s)..."
sleep "$WAIT"

DEAD=""
for name in G1_speed G2_robust G3_full G4_smoothturn; do
  pane_pid=$(tmux list-panes -t "$SESSION:$name" -F '#{pane_pid}' 2>/dev/null | head -1)
  if [ -n "$pane_pid" ] && pgrep -P "$pane_pid" -f python >/dev/null 2>&1 \
     || ([ -n "$pane_pid" ] && pstree -p "$pane_pid" 2>/dev/null | grep -q python); then
    echo "  ✅ $name  살아 있음"
  else
    echo "  ❌ $name  죽었음"
    DEAD="$DEAD $name"
  fi
done

if [ -n "$DEAD" ]; then
  echo "" >&2
  echo "!!! 죽은 arm:$DEAD" >&2
  echo "    마지막 출력 확인:  tmux capture-pane -p -t $SESSION:<이름> | tail -30" >&2
  exit 1
fi

echo ""
echo "완료. 4개 실행 중 (GPU당 2개)."
echo "  확인:  tmux attach -t $SESSION      (분리: Ctrl-b d)"
echo "  진행:  watch -n30 nvidia-smi"
echo "  결과:  $REPO_ROOT/shared_eval_videos/"
echo ""
echo "각 arm이 답하는 질문:"
echo "  G1_speed       path가 속도를 올리면서 E0의 2.7cm를 지키나?  <- 핵심"
echo "  G2_robust      강건성이 게이트에 무엇을 물리나? (E2는 재평가 미완이라 미측정)"
echo "  G3_full        통합 후보"
echo "  G4_smoothturn  연속 turn을 감속 없이 하나? (SmoothTurn, 관측 54 유지)"
