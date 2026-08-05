#!/usr/bin/env bash
# GPU를 놀리지 않기 위한 최소 작업 큐. GPU 한 장당 워커 하나.
#
# 왜 필요한가: 어제 하루에서 GPU가 실제로 돈 시간보다 "내가 결과를 보고 다음 것을
# 띄우기까지 비어 있던 시간"이 길었다. 사람이 스케줄러였기 때문이다. 이 스크립트는
# 그 자리를 대신한다 -- 작업이 끝나면 즉시 다음 것을 집는다.
#
# 설계 원칙 두 가지:
#   1. 큐는 파일이다. DB도 데몬도 없다. `queue/gpu0/010-M1.sh` 같은 실행 파일을
#      사전순으로 집는다. 우선순위를 바꾸려면 파일 이름을 바꾸면 된다.
#   2. 워커는 conda를 자기가 활성화한다. setsid 서브셸이 PATH를 잃어
#      `python: command not found`가 났던 전례가 있어서, 모든 작업이
#      `bash -lc "source .../conda.sh && conda activate ..."`를 거친다.
#      `set -u`는 쓰지 않는다 -- conda.sh가 미설정 변수를 참조해서 활성화 전에
#      죽고, 그러면 로그 리다이렉트에 닿지 못해 로그 파일조차 안 생긴다.
#
# 사용:
#   tools/gpu_queue.sh 0 &        # GPU 0 워커
#   tools/gpu_queue.sh 1 &        # GPU 1 워커
#   tools/gpu_queue.sh 0 --once   # 큐가 비면 종료(야간 배치용)
#
# 작업 추가: queue/gpu<N>/ 에 실행 가능한 스크립트를 놓는다. 끝나면
# queue/done/ 으로 옮겨지고 로그는 queue/logs/<이름>.log 에 남는다.

GPU="${1:?사용법: gpu_queue.sh <gpu index> [--once]}"
ONCE="${2:-}"

cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"
Q="$ROOT/queue/gpu$GPU"
DONE="$ROOT/queue/done"
LOGS="$ROOT/queue/logs"
mkdir -p "$Q" "$DONE" "$LOGS"

echo "[worker $GPU] 큐 감시 시작: $Q"

while true; do
    # 사전순 첫 번째 작업. ls는 파일이 없으면 빈 문자열을 준다.
    JOB="$(find "$Q" -maxdepth 1 -type f -name '*.sh' | sort | head -1)"

    if [ -z "$JOB" ]; then
        if [ "$ONCE" = "--once" ]; then
            echo "[worker $GPU] 큐가 비었다. 종료."
            exit 0
        fi
        sleep 30
        continue
    fi

    NAME="$(basename "$JOB" .sh)"
    LOG="$LOGS/$NAME.log"

    # 집는 즉시 옮긴다. 두 워커가 같은 작업을 집는 경합을 mv의 원자성으로 막는다.
    # mv가 실패하면 다른 워커가 먼저 집은 것이므로 조용히 다음으로 넘어간다.
    RUNNING="$DONE/$NAME.running"
    mv "$JOB" "$RUNNING" 2>/dev/null || continue

    echo "[worker $GPU] 시작: $NAME  -> $LOG"
    START=$(date +%s)
    chmod +x "$RUNNING"
    GPU_INDEX="$GPU" bash -lc "source /mnt/DATA/workspace/ws_eungkyu/miniconda3/etc/profile.d/conda.sh && conda activate k1goalpose && cd '$ROOT' && GPU_INDEX=$GPU '$RUNNING'" > "$LOG" 2>&1
    RC=$?
    ELAPSED=$(( $(date +%s) - START ))

    mv "$RUNNING" "$DONE/$NAME.rc$RC"
    printf '[worker %s] 완료: %-28s rc=%s  %d분\n' \
        "$GPU" "$NAME" "$RC" "$((ELAPSED / 60))" | tee -a "$LOGS/_history.txt"
done
