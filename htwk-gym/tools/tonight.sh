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
MAX_WAIT_MIN="${MAX_WAIT_MIN:-20}"   # GPU 여유 대기 상한. 0 = 즉시 진행

say() { echo "[$(date +%H:%M:%S)] $*"; }
# 통과한 arm만 먼저 띄우고 싶을 때:  ARMS="G1_speed G2_robust" bash tools/tonight.sh
ARMS="${ARMS:-G1_speed G2_robust G3_full G4_smoothturn}"
NARMS=$(echo $ARMS | wc -w | tr -d " ")


say "=== ${NARMS}종: 스모크 -> 학습 (${ITERS} iter) ==="

# ---- 0) warm-start 체크포인트가 실제로 있는지 ------------------------------
DRYRUN="${DRYRUN:-0}"    # 1 = GPU 없이 변수/문구/순서만 검사하고 빠져나온다.
                         # `bash -n`은 문법만 보고 `set -u`의 미정의 변수는 못 잡는다.
                         # NARMS가 정의보다 먼저 쓰여 launch가 죽은 사고에서 나왔다.
if [ "$DRYRUN" != "1" ] && [ ! -f "$CKPT" ]; then
  say "!!! warm-start 체크포인트 없음: $CKPT"
  say "    확인:  ls logs/K1/K1/Goal_Pose_V7/*E0*/nn/model_*.pth | tail"
  exit 1
fi
say "warm start: $CKPT"

# ---- 1) GPU가 빌 때까지 기다린다 --------------------------------------------
# 예전에는 `pgrep reeval_v7.sh|select_best_checkpoint.py`로 대기했다. 그 조건은
# "평가가 도는 중"일 뿐 "GPU가 없다"가 아니다. R1 재실행이 정확히 여기 걸렸다 —
# 완주한 두 arm의 후처리 평가가 카드 하나만 쓰고 있는데 90분을 기다리려 했다.
# 이제 실제 여유 VRAM을 본다. 학습 프로세스 하나가 약 2.6 GB를 쓴다.
NEED_MB="${NEED_MB:-6000}"      # arm 하나를 안전하게 띄우는 데 필요한 여유
waited=0
free_mb() {
  nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
    | sort -rn | head -1
}
while :; do
  f=$(free_mb); f="${f:-999999}"
  [ "$f" -ge "$NEED_MB" ] && break
  if [ "$waited" -ge "$MAX_WAIT_MIN" ]; then
    say "GPU가 ${MAX_WAIT_MIN}분째 안 비었다 (여유 ${f} MiB) — 그래도 진행한다."
    break
  fi
  [ $((waited % 5)) -eq 0 ] && say "GPU 대기... 여유 ${f} MiB < ${NEED_MB} (${waited}/${MAX_WAIT_MIN}분)"
  sleep 60
  waited=$((waited + 1))
done
say "GPU 대기 종료 (${waited}분, 여유 $(free_mb) MiB)"

# GPU가 실제로 비었는지 — 공유 서버라 남의 작업이 올라와 있을 수 있다
say "현재 GPU 점유:"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null | sed 's/^/    /'

# ---- 2) 정적 검사 (즉시, GPU 불필요) ---------------------------------------
# ---- 모니터: 서버가 headless라 이게 없으면 진행 상황을 볼 방법이 없다 --------
# 이미 떠 있으면 다시 띄우지 않는다. 폴링 전용이라 죽어도 학습에는 영향이 없다.
PY="${PY:-python}"        # conda 환경엔 `python`이 있고, 맥 로컬 검증엔 PY=python3
MON_PORT="${MON_PORT:-8420}"
SRV_IP=$(hostname -I 2>/dev/null | awk '{print $1}'); SRV_IP="${SRV_IP:-localhost}"
if [ "$DRYRUN" = "1" ]; then
  say "DRYRUN: 모니터 기동 생략 (실제 실행 시 http://$SRV_IP:$MON_PORT/)"
else
  # ALWAYS restart. Skipping when one is already up meant a collector started
  # before a `git pull` kept serving the old code forever -- start times and new
  # fields simply never appeared, and it looked like a dashboard bug.
  pkill -f "tools/monitor.py --serve" >/dev/null 2>&1 && sleep 1
  mkdir -p logs
  nohup python -u tools/monitor.py --serve --port "$MON_PORT" > logs/monitor.log 2>&1 &
  sleep 1
  say "모니터 (재)기동: http://$SRV_IP:$MON_PORT/"
  say "  안 열리면 맥에서 터널:  ssh -L $MON_PORT:localhost:$MON_PORT <user>@$SRV_IP -p <sshport>"
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
if [ "$DRYRUN" = "1" ]; then
  # Force make_v7_arms to import and build every arm dict. The I1 arms died at
  # import time on a duplicate key (dict(**a, **b) with a shared lever), and
  # nothing before the GPU had touched that module -- so the failure surfaced
  # 28 minutes into a launch instead of in one second here.
  for a in $ARMS; do
    if ! "$PY" tools/make_v7_arms.py --gpu-of "$a" >/dev/null; then
      say "!!! DRYRUN: $a — make_v7_arms 임포트/구성 실패"
      "$PY" tools/make_v7_arms.py --gpu-of "$a" 2>&1 | tail -5
      exit 1
    fi
  done
  say "DRYRUN: 변수/문구 + arm 구성 검사 통과. GPU 작업은 건너뛴다."
  exit 0
fi

say "=== config 생성 + 스모크 (${NARMS}종: $ARMS) ==="
for arm in $ARMS; do
  task="K1/Goal_Pose_V7"

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
# ---- checkpoint가 나오는 즉시 채점하고, 사전 확정한 판정에 걸리면 중단 --------
# H는 12000까지 다 돌린 뒤에야 "4개 전부 나빠짐"을 알았다. 그 신호는 iteration
# 100에 이미 있었다(7.25 -> 10.4~13.2 cm). 판정은 학습 전에 고정하고 코드가 집행한다.
# REF/REF_FALLS는 비교 대상 정책의 실측값이다: E0@6200 = 2.72 cm / 낙상 2.
LAUNCH_T=$(date +%s)
# Reference = the best VERIFIED policy under the current task, not E0's raw
# number. R1 ran with REF_FALLS=2 (E0) and killed three of four arms at
# iteration 50 on the warm-start transient; I0c_h055 measured 2.71 cm / 4 falls
# under the corrected foot, and that is the honest bar.
REF="${REF:-0.0271}"
REF_FALLS="${REF_FALLS:-4}"
STOP_RATIO="${STOP_RATIO:-1.3}"
WARMUP_ITERS="${WARMUP_ITERS:-100}"   # no stop before this: re-adaptation dips first
STRIKES="${STRIKES:-2}"               # consecutive violations required
if [ "${WATCH_EVAL:-1}" = "1" ]; then
  for name in $ARMS; do
    gpu=$(python tools/make_v7_arms.py --gpu-of "$name" 2>/dev/null || echo 0)
    mkdir -p logs/watch_eval
    nohup python -u tools/watch_eval.py --run-glob "logs/*/*/*/*${name}" \
        --newer-than "$LAUNCH_T" --device "cuda:$gpu" \
        --ref "$REF" --ref-falls "$REF_FALLS" --stop-ratio "$STOP_RATIO" \
        --warmup-iters "$WARMUP_ITERS" --strikes "$STRIKES" \
        > "logs/watch_eval/${name}.log" 2>&1 &
    say "  watch_eval: $name (ref ${REF} m / 낙상 ${REF_FALLS}, ${STOP_RATIO}x·${STRIKES}회 연속·iter ${WARMUP_ITERS} 이후)"
  done
fi

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
say "=== ${NARMS}종 실행 중 ==="
say "진행 상황:"
say "  http://$SRV_IP:$MON_PORT/          웹 (배치별 분류 + reward/정확도 그래프)"
say "  python tools/monitor.py --tui      터미널"
say "  tmux attach -t $SESSION"
say ""
say "예상: 실측 ~2.0 s/iter (카드당 2 프로세스) -> ${ITERS} iter 기준"
say "      $(awk -v i=$ITERS 'BEGIN{printf "%.0f분", i*2.0/60}') 내외."
say "      학습 후 체크포인트 탐색 + 영상 + stress가 30~60분 더 붙는다."
