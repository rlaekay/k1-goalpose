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
stall_since=0
idle_since_g0=0
idle_since_g1=0
idle_g0=0
idle_g1=0

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
    #
    # ⛔ 2026-08-07: 여기가 `pgrep -fc 'train_v7\.py|...'` 였는데 **명령줄에 그
    # 문자열이 들어 있기만 한 셸까지 셌다.** 13.5시간 전 내 ssh 셸 하나가
    # heredoc 안에 train_v7.py 를 담은 채 살아 있어서 계수가 항상 2 이상이었다.
    # 그러면 학습이 전부 끝나도 nproc_gpu 가 0 이 되지 않고, 유휴 조건이
    # **원리적으로 절대 성립하지 않는다** -- autopilot 승격이 한 번도 안 일어난
    # 이유이고, GPU 13.8시간 유휴 사고가 그대로 재현될 수 있는 구멍이었다.
    #
    # 그래서 **실행 파일이 python 인 것만** 센다. 셸은 comm 이 bash 라 걸러진다.
    nproc_gpu=$(ps -eo comm=,args= | awk '$1 ~ /^python/ && /train_v7\.py|train\.py|eval_goal_pose\.py|select_best_checkpoint\.py|play_mujoco/ {n++} END{print n+0}')
    [ -z "$nproc_gpu" ] && nproc_gpu=0

    # --- 신호 2b: **카드별** GPU 작업 --------------------------------------
    # ⛔ 2026-08-07 감사(S1-1): 유휴 판정이 두 장 AND 두 큐 전역 AND 였다. 그래서
    # "GPU1 이 6시간짜리 학습을 도는 동안 GPU0 큐가 말라 6시간을 논다"가 유휴로도
    # 정체로도 안 잡혔다. plan/ 승격 기구가 존재하는 바로 그 상황에서 발동하지 않는다.
    # nvidia-smi 가 PID->카드 매핑을 주므로 카드별로 센다.
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
    n0=0; n1=0
    for pid in $apps; do
        # 카드 인덱스는 uuid 로 오므로 PID 의 CUDA_VISIBLE_DEVICES 대신
        # nvidia-smi 의 카드별 조회 두 번으로 가른다(간단하고 확실하다).
        :
    done
    n0=$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . )
    n1=$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . )

    # --- 신호 3: 큐 깊이 + 실행 표식 ------------------------------------
    # 백필(small) 레인도 센다. 안 세면 짧은 작업이 대기 중인데 "카드 유휴"로
    # 잡혀서 autopilot 이 그 위에 3.5시간짜리 학습을 얹는다.
    q0=$(find "$ROOT/queue/gpu0" "$ROOT/queue/small/gpu0" -maxdepth 1 -name '*.sh' 2>/dev/null | wc -l | tr -d ' ')
    q1=$(find "$ROOT/queue/gpu1" "$ROOT/queue/small/gpu1" -maxdepth 1 -name '*.sh' 2>/dev/null | wc -l | tr -d ' ')
    running=$(find "$ROOT/queue/done" -maxdepth 1 -name '*.running' 2>/dev/null | wc -l | tr -d ' ')
    # 워커도 같은 이유로 실행 파일 기준으로 센다.
    workers=$(ps -eo comm=,args= | awk '$1 ~ /^bash/ && $3 == "tools/gpu_queue.sh" {n++} END{print n+0}')
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

    # --- 판정 2: **정체**(stall) -- 할 일이 있는데 아무것도 안 돈다 ---------
    #
    # 유휴 판정은 큐가 비어 있을 것을 요구한다. 그래서 "큐에 대기 작업이 있는데
    # 워커가 죽어서 아무도 안 집는" 상태는 유휴로도 잡히지 않고 정상으로도 안 잡힌다.
    # 그건 GPU 가 노는 것과 결과가 똑같은데 **아무 신호도 안 나가는** 최악의 모드다.
    # 고아 `.running` 표식(워커가 작업 도중 죽으면 남는다)도 같은 부류다.
    if [ "$u0" -lt "$IDLE_UTIL" ] && [ "$u1" -lt "$IDLE_UTIL" ] \
       && [ "$nproc_gpu" -eq 0 ] \
       && { [ "$q0" -gt 0 ] || [ "$q1" -gt 0 ] || [ "$running" -gt 0 ]; }; then
        [ "$stall_since" -eq 0 ] && stall_since=$now
    else
        stall_since=0
    fi
    if [ "$stall_since" -eq 0 ]; then stall_s=0; else stall_s=$((now - stall_since)); fi

    # --- 판정 3: **카드별** 유휴 -------------------------------------------
    # 한 장만 노는 상태. 전역 유휴와 달리 다른 카드가 무엇을 하든 무관하다.
    for g in 0 1; do
        eval "u=\$u$g; nq=\$q$g; np=\$n$g; since=\${idle_since_g$g:-0}"
        if [ "$u" -lt "$IDLE_UTIL" ] && [ "$np" -eq 0 ] && [ "$nq" -eq 0 ]; then
            [ "$since" -eq 0 ] && since=$now
        else
            since=0
        fi
        eval "idle_since_g$g=$since"
        if [ "$since" -eq 0 ]; then eval "idle_g$g=0"; else eval "idle_g$g=$((now - since))"; fi
    done

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
  "stall_seconds": $stall_s,
  "stalled": $([ "$stall_s" -ge 300 ] && echo true || echo false),
  "gpu_apps": [$n0, $n1],
  "idle_seconds_gpu0": $idle_g0,
  "idle_seconds_gpu1": $idle_g1,
  "card_idle_30min": $([ "$idle_g0" -ge 1800 ] || [ "$idle_g1" -ge 1800 ] && echo true || echo false),
  "recent_done": [${recent}]
}
EOF
    mv "$tmp" "$STATE"      # 원자적 교체. 읽는 쪽이 절반 쓰인 파일을 보지 않는다.

    sleep "$INTERVAL"
done
