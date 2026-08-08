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

# 카드 $1 에 **큰 레인** 워커가 살아 있는가.
#
# ⛔ 2026-08-08. 예전에는 `$3 == "tools/gpu_queue.sh" && $4 == g` 로만 봤다. 지금
# 작은 레인은 같은 카드 번호로 `gpu_worker.sh <g>` (LANE=small) 로 돌고, 큰 레인도
# gpu_worker.sh 로 갈아탈 예정이다. 그러면 **작은 레인 워커가 큰 레인 워커로 오인돼서
# 큰 레인이 죽어도 복구가 발동하지 않는다** -- 감시자가 있는데 아무 신호가 안 나가는,
# 이 파일이 막으라고 있는 바로 그 실패 모드다. 레인은 이름이 아니라 환경변수에 있다.
big_worker_alive() {
    local g="$1" p lane
    for p in $(ps -eo pid=,args= | awk -v g="$g" '$3 ~ /tools\/gpu_(queue|worker)\.sh$/ && $4 == g {print $1}'); do
        lane=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | sed -n 's/^LANE=//p' | head -1)
        [ -z "$lane" ] && lane=big
        [ "$lane" = "big" ] && return 0
    done
    return 1
}

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

    # --- 정체 복구: 할 일이 있는데 아무것도 안 돈다 -----------------------
    #
    # 유휴 승격보다 **먼저** 본다. 큐에 대기 작업이 있는데 워커가 죽어 있으면
    # plan 에서 하나 더 승격해봐야 아무도 안 집는다. 원인부터 고쳐야 한다.
    #
    # 두 가지를 복구한다:
    #   1. 워커가 2개 미만이면 다시 띄운다.
    #   2. 고아 `.running` 표식을 원래 큐로 되돌린다. 워커가 작업 도중 죽으면
    #      그 표식이 남고, 그 작업은 **영원히 사라진다** -- 큐에도 없고 done 에도
    #      완료로 안 남는다. 아무 신호도 안 나가는 최악의 모드다.
    stall=0
    if [ -f "$STATE" ]; then
        age=$(( $(date +%s) - $(stat -c %Y "$STATE" 2>/dev/null || echo 0) ))
        if [ "$age" -le 180 ]; then
            stall=$(sed -n 's/.*"stall_seconds": *\([0-9]*\).*/\1/p' "$STATE")
            [ -z "$stall" ] && stall=0
        fi
    fi
    if [ "$stall" -ge 300 ]; then
        say "⚠️ 정체 $((stall / 60))분 -- 할 일이 있는데 아무것도 안 돈다. 복구 시도."
        # 카드마다 큰 레인 워커가 살아 있는지 **레인 기준**으로 본다(위 함수 주석).
        for g in 0 1; do
            big_worker_alive "$g" && continue
            say "  큰 레인 워커 $g 가 없다. 재기동."
            ( cd "$ROOT" && setsid nohup bash tools/gpu_queue.sh "$g" \
                < /dev/null >> "queue/worker$g.log" 2>&1 & )
        done
        for f in "$ROOT/queue/done"/*.running; do
            [ -e "$f" ] || continue
            n=$(basename "$f" .running)
            # 되돌릴 곳은 `.owner`(pid gpu lane)에서 읽는다. gpu_worker.sh 가 작업을
            # 집을 때 쓴다. 없으면(구 gpu_queue.sh 가 집은 것) gpu0 큰 레인으로 간다.
            # ⛔ 예전에는 무조건 gpu0 큰 레인이었다. 그러면 2시간짜리 채점 작업이
            # 큰 레인에 얹혀서 그 카드의 다음 학습을 그만큼 미룬다.
            own="$ROOT/queue/done/$n.owner"
            dest="$ROOT/queue/gpu0"
            if [ -f "$own" ]; then
                read -r _opid ogpu olane < "$own"
                [ -z "$ogpu" ] && ogpu=0
                if [ "$olane" = "small" ]; then dest="$ROOT/queue/small/gpu$ogpu"
                else                            dest="$ROOT/queue/gpu$ogpu"; fi
                mkdir -p "$dest"
            fi
            mv "$f" "$dest/$n.sh" 2>/dev/null \
                && { rm -f "$own"; say "  고아 표식 복구: $n -> ${dest#$ROOT/}/"; }
        done
        sleep "$INTERVAL"
        continue
    fi

    # ⛔ 승격을 **카드별 유휴**로 판단한다(2026-08-07 감사 S1-1). 전역 유휴를 쓰면
    # 다른 카드가 6시간짜리 학습을 도는 동안 이쪽 카드가 놀아도 발동하지 않는다 --
    # plan/ 기구가 존재하는 바로 그 상황이다.
    promote_any=0
    for g in 0 1; do
        ci=$(sed -n "s/.*\"idle_seconds_gpu$g\": *\([0-9]*\).*/\1/p" "$STATE" 2>/dev/null)
        [ -z "$ci" ] && ci=0
        [ "$ci" -ge "$PROMOTE_AFTER" ] && promote_any=1
    done
    if [ "$idle" -ge "$PROMOTE_AFTER" ] || [ "$promote_any" -eq 1 ]; then
        promoted=0
        for g in 0 1; do
            ci=$(sed -n "s/.*\"idle_seconds_gpu$g\": *\([0-9]*\).*/\1/p" "$STATE" 2>/dev/null)
            [ -z "$ci" ] && ci=0
            [ "$ci" -lt "$PROMOTE_AFTER" ] && continue   # 이 카드는 아직 안 논다
            live_n=$(find "$ROOT/queue/gpu$g" -maxdepth 1 -name '*.sh' 2>/dev/null | wc -l | tr -d ' ')
            [ "$live_n" -gt 0 ] && continue          # 이미 대기 작업이 있으면 건드리지 않는다
            job=$(find "$PLAN/gpu$g" -maxdepth 1 -type f -name '*.sh' | sort | head -1)
            [ -z "$job" ] && continue
            if mv "$job" "$ROOT/queue/gpu$g/"; then
                say "승격: gpu$g <- $(basename "$job")  (카드 유휴 $((ci / 60))분)"
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
