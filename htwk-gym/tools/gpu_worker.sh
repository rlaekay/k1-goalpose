#!/usr/bin/env bash
# GPU 워커 v2. `gpu_queue.sh` 를 대체한다. 실무 관행 조사(2026-08-07) 결과 우리
# 큐에 없던 원시기능 네 개를 넣었다.
#
#   tools/gpu_worker.sh 0            # gpu0 의 큰 작업 레인
#   LANE=small tools/gpu_worker.sh 0 # gpu0 의 짧은 작업 레인(백필)
#
# ---- 왜 자체 큐를 유지하는가 (조사 결론) ------------------------------------
#
# 단일 노드 GPU 랩의 표준 후보는 Slurm 과 task-spooler 두 개다.
#   * Slurm 은 이 서버에 root 권한이 없어 설치 자체가 불가능하다(CLAUDE.md: 내
#     디렉터리에서만 작업). 그리고 단일 노드에서는 과하다.
#   * task-spooler(GPU fork)가 우리 큐와 사실상 같은 물건인데, 문서를 읽어 보면
#     **timeout / 실패 재투입 / 재부팅 후 복구를 지원하지 않는다.** 우리가 실제로
#     아쉬운 셋이 정확히 그 셋이다. 즉 갈아타도 얻는 게 없다.
# 그래서 **바꾸지 않고 빠진 원시기능만 넣는다.** 아래 넷이다.
#
# ---- 1. 타임아웃 -------------------------------------------------------------
# Slurm 의 `--time=` 에 해당한다. 지금까지 없었고, 그래서 작업이 멈추면(IsaacGym
# 락, NCCL 대기, 디스크 stall) 카드가 **영구히** 잠긴다. 살아 있는 python 은
# 유휴 판정도 정체 판정도 무효화하므로 아무 신호도 안 나간다.
# 작업 스크립트가 `# MAX_HOURS=N` 주석으로 자기 상한을 선언한다. 없으면 기본값.
#
# ---- 2. 실패 재투입 ----------------------------------------------------------
# Slurm 의 `--requeue`. 지금까지 실패한 작업은 그냥 사라졌다 -- 큐에도 없고
# 완료로도 안 남는다. 재시도 횟수를 파일 이름에 박아 상한을 건다.
# ⛔ 단, **설정 오류로 죽는 작업을 무한 재시도하면 안 된다.** 그래서 상한 1회이고,
# 1분 미만에 죽은 작업은 재시도하지 않는다(빠른 실패 = 대개 설정 문제다).
#
# ---- 3. 로그 보존 ------------------------------------------------------------
# 기존 워커는 `> "$LOG"` 로 덮어썼다. 같은 이름을 재투입하면 이전 실행의 로그가
# 통째로 사라진다 -- 실제로 `830-NZ_zeroiid` 의 rc1 로그가 그렇게 없어져서
# 빠른 실패의 원인을 사후에 못 봤다. 타임스탬프를 붙이고 latest 심볼릭 링크를 둔다.
#
# ---- 4. 싱글턴 락 ------------------------------------------------------------
# flock. 워커가 같은 카드에 둘 뜨면 한 카드에서 서로 다른 작업 둘이 동시에 돈다
# (`mv` 원자성은 그걸 막지 못한다 -- 서로 다른 파일을 집으면 그만이다).
#
# `set -u` 는 쓰지 않는다 -- conda.sh 가 미설정 변수를 참조해 활성화 전에 죽는다.

GPU="${1:?사용법: gpu_worker.sh <gpu index>}"
LANE="${LANE:-big}"
DEFAULT_MAX_HOURS="${DEFAULT_MAX_HOURS:-8}"

cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"
if [ "$LANE" = "small" ]; then
    Q="$ROOT/queue/small/gpu$GPU"
    POLL=20
    DEFAULT_MAX_HOURS="${SMALL_MAX_HOURS:-2}"
else
    Q="$ROOT/queue/gpu$GPU"
    POLL=30
fi
DONE="$ROOT/queue/done"
LOGS="$ROOT/queue/logs"
mkdir -p "$Q" "$DONE" "$LOGS"

# 싱글턴: 같은 (카드, 레인) 조합에 워커가 둘 뜨지 못하게 한다.
exec 9> "$ROOT/queue/.lock.worker$GPU.$LANE"
if ! flock -n 9; then
    echo "[worker $GPU/$LANE] 이미 돌고 있다. 종료."
    exit 0
fi

echo "[worker $GPU/$LANE] 큐 감시 시작: $Q  (기본 상한 ${DEFAULT_MAX_HOURS}h)"

while true; do
    JOB="$(find "$Q" -maxdepth 1 -type f -name '*.sh' | sort | head -1)"
    if [ -z "$JOB" ]; then sleep "$POLL"; continue; fi

    NAME="$(basename "$JOB" .sh)"
    STAMP="$(date +%Y%m%d-%H%M%S)"
    LOG="$LOGS/$NAME.$STAMP.log"

    RUNNING="$DONE/$NAME.running"
    mv "$JOB" "$RUNNING" 2>/dev/null || { sleep 1; continue; }
    # 소유자를 박는다. autopilot 이 고아 표식을 복구할 때 `kill -0` 로 확인하고,
    # 되돌릴 카드도 여기서 읽는다. 정체 신호의 정확도에 복구가 기대지 않게 된다.
    printf '%s %s %s\n' "$$" "$GPU" "$LANE" > "$DONE/$NAME.owner"

    # 작업이 선언한 상한. 없으면 기본값.
    MAXH=$(grep -m1 -oE '^# *MAX_HOURS *= *[0-9.]+' "$RUNNING" | grep -oE '[0-9.]+')
    [ -z "$MAXH" ] && MAXH="$DEFAULT_MAX_HOURS"

    echo "[worker $GPU/$LANE] 시작: $NAME  상한 ${MAXH}h  -> $LOG"
    ln -sfn "$(basename "$LOG")" "$LOGS/$NAME.log"      # 기존 경로 호환
    START=$(date +%s)
    chmod +x "$RUNNING"
    timeout --signal=TERM --kill-after=120 "${MAXH}h" \
        bash -lc "source /mnt/DATA/workspace/ws_eungkyu/miniconda3/etc/profile.d/conda.sh && conda activate k1goalpose && cd '$ROOT' && GPU_INDEX=$GPU LANE=$LANE '$RUNNING'" \
        > "$LOG" 2>&1
    RC=$?
    ELAPSED=$(( $(date +%s) - START ))
    rm -f "$DONE/$NAME.owner"

    # 재투입 판정. 이름에 이미 .retryN 이 있으면 상한에 걸린 것이다.
    case "$NAME" in
        *.retry*) RETRIED=1 ;;
        *)        RETRIED=0 ;;
    esac
    if [ "$RC" -ne 0 ] && [ "$RETRIED" -eq 0 ] && [ "$ELAPSED" -ge 60 ] && [ "$RC" -ne 130 ]; then
        # 1분 이상 돌다 죽은 것만 한 번 다시 넣는다. 빠른 실패는 대개 설정
        # 오류라 재시도가 GPU 를 태울 뿐이다. rc130(SIGINT)은 사람이 끊은 것이다.
        cp "$RUNNING" "$Q/$NAME.retry1.sh"
        chmod +x "$Q/$NAME.retry1.sh"
        echo "[worker $GPU/$LANE] 재투입: $NAME -> $NAME.retry1 (rc=$RC, ${ELAPSED}s)"
    fi

    mv "$RUNNING" "$DONE/$NAME.rc$RC"
    touch "$DONE/$NAME.rc$RC"     # recent_done 이 완료 시각 순이 되도록
    printf '[worker %s/%s] 완료: %-30s rc=%-4s %d분\n' \
        "$GPU" "$LANE" "$NAME" "$RC" "$((ELAPSED / 60))" | tee -a "$LOGS/_history.txt"
done
