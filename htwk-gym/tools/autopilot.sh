#!/usr/bin/env bash
# 서버 자체 자율 운전. 에이전트 세션이 죽어도 GPU 가 놀지 않게 하는 마지막 방어선.
#
# 왜: 2026-08-06/07 밤에 GPU 두 장이 13.8시간을 놀았다. 큐 투입이 한 번 실패했고,
# "다음에 확인하겠다"고 한 쪽(에이전트)이 확인하지 않았다. 감시자가 사람이든
# 에이전트든, **어딘가에 사람이 개입해야 다음 작업이 시작되는 구조 자체가 결함**이다.
#
# 동작:
#   * idle_watch.sh 가 쓴 queue/idle_state.json 을 읽는다.
#   * 유휴가 PROMOTE_AFTER 초를 넘으면 queue/plan/gpu<N>/ 의 사전순 첫 작업
#     하나를 live 큐(queue/gpu<N>/)로 옮긴다. 한 번에 하나씩만 옮긴다.
#   * plan 이 비면 아무것도 하지 않고 계속 감시한다.
#
# 왜 30분을 기다리는가: 그 30분이 에이전트가 **결과를 보고** 다음 학습을 정할
# 기회다. 사용자가 원한 것은 결과 반영이지 무한 자동 학습이 아니다. plan/ 은
# 에이전트가 못 왔을 때만 쓰이는 대체재이고, 그래서 plan 에는 결과와 무관하게
# 가치가 있는 작업만 둔다.
#
# `set -u` 는 쓰지 않는다 -- conda.sh 가 미설정 변수를 참조해 활성화 전에 죽는다.

cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"
STATE="$ROOT/queue/idle_state.json"
PLAN="$ROOT/queue/plan"
LOG="$ROOT/queue/autopilot.log"
PROMOTE_AFTER="${PROMOTE_AFTER:-1800}"   # 30분
INTERVAL="${INTERVAL:-120}"

mkdir -p "$PLAN/gpu0" "$PLAN/gpu1"

say() { printf '[%s] %s\n' "$(TZ=Asia/Seoul date +'%F %T')" "$*" | tee -a "$LOG"; }

say "autopilot 시작 (유휴 ${PROMOTE_AFTER}초 후 승격, ${INTERVAL}초마다 확인)"

while true; do
    idle=0
    if [ -f "$STATE" ]; then
        age=$(( $(date +%s) - $(stat -c %Y "$STATE" 2>/dev/null || echo 0) ))
        if [ "$age" -le 180 ]; then
            idle=$(sed -n 's/.*"idle_seconds": *\([0-9]*\).*/\1/p' "$STATE")
            [ -z "$idle" ] && idle=0
        else
            say "⚠️ idle_state.json 이 ${age}초 낡았다 -- idle_watch.sh 가 죽었나?"
            # 감시자가 죽었으면 승격하지 않는다. 유휴인지 아닌지 모르는 상태에서
            # 4096-env 학습을 띄우면 돌고 있는 학습과 카드를 나눠 쓰게 된다.
            idle=0
        fi
    fi

    if [ "$idle" -ge "$PROMOTE_AFTER" ]; then
        promoted=0
        for g in 0 1; do
            live_n=$(find "$ROOT/queue/gpu$g" -maxdepth 1 -name '*.sh' 2>/dev/null | wc -l | tr -d ' ')
            [ "$live_n" -gt 0 ] && continue          # 이미 대기 작업이 있으면 건드리지 않는다
            job=$(find "$PLAN/gpu$g" -maxdepth 1 -type f -name '*.sh' | sort | head -1)
            [ -z "$job" ] && continue
            if mv "$job" "$ROOT/queue/gpu$g/"; then
                say "승격: gpu$g <- $(basename "$job")  (유휴 $((idle / 60))분)"
                promoted=$((promoted + 1))
            fi
        done
        if [ "$promoted" -eq 0 ]; then
            say "유휴 $((idle / 60))분인데 plan/ 이 비었다. 사람 또는 에이전트가 필요하다."
            sleep 900     # 같은 메시지로 로그를 채우지 않는다
        fi
    fi

    sleep "$INTERVAL"
done
