#!/usr/bin/env bash
# GPU 유휴 감시자. 서버에서 계속 돌면서 queue/idle_state.json 을 갱신한다.
#
# 왜 필요한가: 2026-08-06/07 밤에 GPU 두 장이 13.8시간을 완전히 놀았다. 큐에 작업을
# 넣는 명령이 한 번 실패했고, 아무도 그걸 눈치채지 못했다. 사람이든 에이전트든
# "다음에 확인하겠다"는 신뢰할 수 없는 감시자다. 이 스크립트가 그 자리를 대신한다.
#
# 왜 한 번의 nvidia-smi 로는 안 되는가:
#   * 학습 중에도 epoch 경계에서 util 이 0% 로 찍힌다. 한 샘플은 유휴의 증거가 아니다.
#   * 반대로 죽은 프로세스가 메모리를 붙잡고 있으면 memory.used 는 0 이 아니다.
#   * 그래서 세 가지 독립 신호를 모두 본다: util 샘플 이력, 살아 있는 학습 프로세스,
#     큐 깊이. 셋 다 유휴여야 유휴다.
#
# `set -u` 는 쓰지 않는다 -- conda.sh 가 미설정 변수를 참조해서 활성화 전에 죽는다.
# (gpu_queue.sh 에 같은 주석이 있고, 그걸 잊어서 sig_sweep.sh 를 한 번 더 죽였다.)

cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"
STATE="$ROOT/queue/idle_state.json"
INTERVAL="${INTERVAL:-30}"        # 샘플 주기(초)
# 유휴로 세려면 두 장 모두 이 값 미만이어야 한다. 0 이 아니라 5 인 이유: 유휴 카드도
# ECC/팬 폴링으로 1-3% 가 찍힌다.
IDLE_UTIL="${IDLE_UTIL:-5}"

mkdir -p "$ROOT/queue"

# 연속 유휴 시작 시각. 0 이면 "지금 유휴가 아님".
idle_since=0

json_escape() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }

while true; do
    now=$(date +%s)

    # --- 신호 1: GPU util (두 장) ---------------------------------------
    utils=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
    u0=$(echo "$utils" | sed -n 1p); u1=$(echo "$utils" | sed -n 2p)
    [ -z "$u0" ] && u0=999          # nvidia-smi 실패는 유휴가 아니다(보수적으로)
    [ -z "$u1" ] && u1=999

    # --- 신호 2: 살아 있는 GPU 작업 프로세스 ----------------------------
    # util 이 0 이어도 이게 있으면 유휴가 아니다.
    nproc_gpu=$(pgrep -fc 'train_v7\.py|train\.py|eval_goal_pose\.py|select_best_checkpoint\.py|play_mujoco' 2>/dev/null)
    [ -z "$nproc_gpu" ] && nproc_gpu=0

    # --- 신호 3: 큐 깊이 + 실행 표식 ------------------------------------
    q0=$(find "$ROOT/queue/gpu0" -maxdepth 1 -name '*.sh' 2>/dev/null | wc -l | tr -d ' ')
    q1=$(find "$ROOT/queue/gpu1" -maxdepth 1 -name '*.sh' 2>/dev/null | wc -l | tr -d ' ')
    running=$(find "$ROOT/queue/done" -maxdepth 1 -name '*.running' 2>/dev/null | wc -l | tr -d ' ')
    workers=$(pgrep -fc 'gpu_queue\.sh [01]' 2>/dev/null)
    [ -z "$workers" ] && workers=0

    # --- 판정 -----------------------------------------------------------
    if [ "$u0" -lt "$IDLE_UTIL" ] && [ "$u1" -lt "$IDLE_UTIL" ] \
       && [ "$nproc_gpu" -eq 0 ] && [ "$q0" -eq 0 ] && [ "$q1" -eq 0 ] \
       && [ "$running" -eq 0 ]; then
        [ "$idle_since" -eq 0 ] && idle_since=$now
    else
        idle_since=0
    fi

    if [ "$idle_since" -eq 0 ]; then idle_s=0; else idle_s=$((now - idle_since)); fi

    # 마지막으로 끝난 작업 세 개 -- 깨어난 에이전트가 무엇을 채점해야 하는지 알려면 필요하다.
    recent=$(ls -1t "$ROOT/queue/done"/*.rc* 2>/dev/null | head -3 \
             | while read -r f; do printf '"%s",' "$(json_escape "$(basename "$f")")"; done | sed 's/,$//')

    tmp="$STATE.tmp"
    cat > "$tmp" <<EOF
{
  "ts": $now,
  "ts_kst": "$(TZ=Asia/Seoul date +'%F %T')",
  "gpu_util": [$u0, $u1],
  "gpu_jobs_alive": $nproc_gpu,
  "queue_depth": {"gpu0": $q0, "gpu1": $q1},
  "running_markers": $running,
  "queue_workers_alive": $workers,
  "idle_seconds": $idle_s,
  "idle_30min": $([ "$idle_s" -ge 1800 ] && echo true || echo false),
  "recent_done": [${recent}]
}
EOF
    mv "$tmp" "$STATE"      # 원자적 교체. 읽는 쪽이 절반 쓰인 파일을 보지 않는다.

    sleep "$INTERVAL"
done
