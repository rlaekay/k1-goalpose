#!/bin/bash
# Overnight: wait for any running re-eval, then smoke-gate and launch the G batch.
#
#   nohup bash tools/tonight.sh > tonight.log 2>&1 &
#   tail -f tonight.log
#
# Written to be started and walked away from. It waits rather than killing the
# re-eval, because E2's own-task number is the baseline G2 gets compared against
# and it is 15 minutes from being finished; throwing that away to save 15 minutes
# of an overnight run is a bad trade.
#
# Nothing here trains until every arm has passed its own smoke test. G1 carries
# the lookahead floor + dwell + grid, G3 carries the scripted elbows, and G4
# carries the whole sequential-navigation stack -- none of which has ever
# executed. Burning a night on unproven code is exactly what the gate is for.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SESSION="${SESSION:-g}"
ITERS="${ITERS:-12000}"
ENVS="${ENVS:-4096}"
CKPT="${CKPT:-logs/K1/K1/Goal_Pose_V7/2026-07-26-19-36-15_E0_armB_armsdown/nn/model_6200.pth}"
MAX_WAIT_MIN="${MAX_WAIT_MIN:-90}"

say() { echo "[$(date +%H:%M:%S)] $*"; }

say "=== 오늘 밤 계획: 재평가 대기 -> 스모크 4종 -> G 배치 4종 ==="

# ---- 0) warm-start 체크포인트가 실제로 있는지 ------------------------------
if [ ! -f "$CKPT" ]; then
  say "!!! warm-start 체크포인트 없음: $CKPT"
  say "    확인:  ls logs/K1/K1/Goal_Pose_V7/*E0*/nn/model_*.pth | tail"
  exit 1
fi
say "warm start: $CKPT"

# ---- 1) 재평가가 끝나기를 기다린다 ------------------------------------------
waited=0
while pgrep -f "reeval_v7.sh|select_best_checkpoint.py" >/dev/null 2>&1; do
  if [ "$waited" -ge "$MAX_WAIT_MIN" ]; then
    say "재평가가 ${MAX_WAIT_MIN}분째 안 끝남 — 더 기다리지 않고 진행한다."
    say "  (재평가는 학습이 아니라 언제든 다시 돌릴 수 있다)"
    break
  fi
  [ $((waited % 10)) -eq 0 ] && say "재평가 진행 중... ${waited}분 경과 (최대 ${MAX_WAIT_MIN}분 대기)"
  sleep 60
  waited=$((waited + 1))
done
say "재평가 대기 종료 (${waited}분)"

# GPU가 실제로 비었는지 — 공유 서버라 남의 작업이 올라와 있을 수 있다
say "현재 GPU 점유:"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null | sed 's/^/    /'

# ---- 2) 정적 검사 (즉시, GPU 불필요) ---------------------------------------
# ---- 모니터: 서버가 headless라 이게 없으면 진행 상황을 볼 방법이 없다 --------
# 이미 떠 있으면 다시 띄우지 않는다. 폴링 전용이라 죽어도 학습에는 영향이 없다.
MON_PORT="${MON_PORT:-8420}"
if ! pgrep -f "tools/monitor.py --serve" >/dev/null 2>&1; then
  mkdir -p logs
  nohup python -u tools/monitor.py --serve --port "$MON_PORT" > logs/monitor.log 2>&1 &
  sleep 1
  say "모니터 기동: http://<서버IP>:$MON_PORT/"
  say "  터널:  ssh -L $MON_PORT:localhost:$MON_PORT <user>@<host> -p <port>  ->  http://localhost:$MON_PORT/"
else
  say "모니터 이미 실행 중 (포트 $MON_PORT)"
fi
say "터미널만 쓸 경우:  python tools/monitor.py --tui"

say "=== 이름 해석 검사 ==="
if ! python tools/check_names.py > /tmp/tonight_names.txt 2>&1; then
  if grep -v "import \*" /tmp/tonight_names.txt | grep -q UNDEFINED; then
    say "!!! 정의되지 않은 이름 발견 — 중단"
    grep UNDEFINED /tmp/tonight_names.txt
    exit 1
  fi
fi
say "이름 검사 통과"

# ---- 3) arm별 config 생성 + 스모크 -----------------------------------------
# 통과한 arm만 먼저 띄우고 싶을 때:  ARMS="G1_speed G2_robust" bash tools/tonight.sh
ARMS="${ARMS:-G1_speed G2_robust G3_full G4_smoothturn}"
NARMS=$(echo $ARMS | wc -w)
say "=== config 생성 + 스모크 (${NARMS}종: $ARMS) ==="
for arm in $ARMS; do
  task="K1/Goal_Pose_V7"
  [ "$arm" = "G4_smoothturn" ] && task="K1/Goal_Pose_V8"

  if ! python tools/make_v7_arms.py --only "$arm" --checkpoint "$CKPT" \
        --num_envs "$ENVS" --max_iterations "$ITERS" > /dev/null 2>&1; then
    say "!!! $arm config 생성 실패 — 중단"
    python tools/make_v7_arms.py --only "$arm" --checkpoint "$CKPT" 2>&1 | tail -20
    exit 1
  fi

  say "--- 스모크: $arm ($task) ---"
  if ! python tools/smoke_v7.py --config "sweeps/$arm.yaml" --task "$task" \
        --checkpoint "$CKPT" --sim_device cuda:0 --rl_device cuda:0 --steps 300; then
    say "!!! $arm 스모크 실패 — 아무것도 학습시키지 않고 중단한다."
    say "    위 FAIL 항목을 고친 뒤 다시 실행하십시오."
    exit 1
  fi
done
say "스모크 ${NARMS}종 전부 통과"

# ---- 4) tmux 세션 실행 ------------------------------------------------------
# launch()는 세션이 이미 있으면 창을 추가하도록 되어 있다 (아래) -- G1/G2를
# 먼저 세션 'g'로 띄워둔 뒤 G3만 따로 통과시켜 재실행하는 것이 정상 경로이므로,
# 세션 존재 자체가 아니라 "이번에 띄우려는 arm의 창이 이미 있는지"만 막는다.
for arm in $ARMS; do
  if tmux has-session -t "$SESSION" 2>/dev/null && tmux list-windows -t "$SESSION" -F '#{window_name}' 2>/dev/null | grep -qx "$arm"; then
    say "!!! tmux 창 '$SESSION:$arm'이 이미 있다. 지우려면: tmux kill-window -t $SESSION:$arm"
    exit 1
  fi
done

# tmux 창은 실행 셸이 아니라 tmux SERVER의 환경을 물려받는다. 그 서버는 conda
# 활성화보다 먼저 떠 있어서, 그냥 띄우면 isaacgym이 libpython3.8.so.1.0을 못 찾는다.
if [ -z "${CONDA_PREFIX:-}" ]; then
  say "!!! conda 환경이 없다. 'conda activate k1goalpose' 후 다시 실행."
  exit 1
fi
CONDA_BASE="$(conda info --base 2>/dev/null || echo "${CONDA_PREFIX%/envs/*}")"
ENV_NAME="${CONDA_DEFAULT_ENV:-$(basename "$CONDA_PREFIX")}"
PRELUDE="source '$CONDA_BASE/etc/profile.d/conda.sh' && conda activate '$ENV_NAME' && export LD_LIBRARY_PATH='$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}' &&"

launch() {  # name gpu
  local name=$1 gpu=$2 task="K1/Goal_Pose_V7"
  [ "$name" = "G4_smoothturn" ] && task="K1/Goal_Pose_V8"
  local cmd="$PRELUDE cd $REPO_ROOT && TRAIN=train_v7.py STRESS=1 VIDEO_S=60 \
bash tools/train_and_eval.sh cuda:$gpu cuda:$gpu -- \
--task=$task --config sweeps/$name.yaml --headless True \
--checkpoint $CKPT --num_envs $ENVS --max_iterations $ITERS \
--sim_device cuda:$gpu --rl_device cuda:$gpu"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-window -t "$SESSION" -n "$name" "$cmd; exec bash"
  else
    tmux new-session -d -s "$SESSION" -n "$name" "$cmd; exec bash"
  fi
  say "  launched $name on GPU $gpu"
}

# GPU 라운드로빈. 4종이면 2/2로 갈리고, 2종이면 카드 하나씩 잡는다 --
# 통과한 둘만 돌릴 때 굳이 한 장에 몰아 넣어 서로 느려질 이유가 없다.
say "=== 배치 실행 (${NARMS}종) ==="
i=0
for arm in $ARMS; do
  launch "$arm" $((i % 2))
  i=$((i + 1))
done

# ---- 5) 기동 확인 ----------------------------------------------------------
say "기동 확인 중 (120s)..."
sleep 120
DEAD=""
for name in $ARMS; do
  pid=$(tmux list-panes -t "$SESSION:$name" -F '#{pane_pid}' 2>/dev/null | head -1)
  if [ -n "$pid" ] && pstree -p "$pid" 2>/dev/null | grep -q python; then
    say "  ✅ $name 살아 있음"
  else
    say "  ❌ $name 죽었음"
    DEAD="$DEAD $name"
  fi
done

if [ -n "$DEAD" ]; then
  say "!!! 죽은 arm:$DEAD"
  for n in $DEAD; do
    say "--- $n 마지막 출력 ---"
    tmux capture-pane -p -t "$SESSION:$n" 2>/dev/null | tail -25
  done
  exit 1
fi

say ""
say "=== 4개 전부 실행 중. 자러 가셔도 됩니다. ==="
say "아침에:"
say "  python tools/progress.py --task Goal_Pose_V7"
say "  python tools/progress.py --task Goal_Pose_V8      # G4"
say "  bash htwk-gym/tools/fetch_results.sh              # (Mac에서)"
say ""
say "예상: 4개가 GPU당 2개씩 동시에 돈다(직렬 아님). v7 배치 실측이 3.5~3.9 s/iter"
say "      였으므로 12000 iter면 12~13시간. 아침에 거의 끝나 있거나 평가 단계일 것."
say "      학습 후 최적 체크포인트 탐색 + 영상 + stress가 30~60분 더 붙는다."
