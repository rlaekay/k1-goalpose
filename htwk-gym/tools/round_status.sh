#!/usr/bin/env bash
# 깨어난 에이전트가 ssh 한 번으로 읽는 서버 상태 요약.
#
#   ssh a6000 'bash /mnt/DATA/.../htwk-gym/tools/round_status.sh'
#
# idle_watch.sh 가 만든 idle_state.json 을 우선 읽되, 그 파일이 없거나 낡았으면
# (감시자가 죽었으면) 직접 샘플링해서 대체한다 -- 감시자를 믿는 것이 감시자가
# 죽었을 때 조용히 "유휴 아님"으로 읽히는 실패 모드를 만들면 안 된다.

cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"
STATE="$ROOT/queue/idle_state.json"
STALE_S=180        # 이보다 오래된 상태 파일은 믿지 않는다(샘플 주기 30초의 6배)

echo "=== 시각 ==="
TZ=Asia/Seoul date +'KST %F %T'

echo
echo "=== 유휴 감시자 ==="
if [ -f "$STATE" ]; then
    age=$(( $(date +%s) - $(stat -c %Y "$STATE" 2>/dev/null || echo 0) ))
    if [ "$age" -le "$STALE_S" ]; then
        echo "상태: 살아있음 (${age}초 전 갱신)"
        cat "$STATE"
    else
        echo "⚠️ 상태 파일이 ${age}초 낡았다 -- 감시자가 죽었을 수 있다. 직접 샘플한다."
        echo "STALE"
    fi
else
    echo "⚠️ idle_state.json 없음 -- 감시자가 돈 적이 없다. 직접 샘플한다."
    echo "MISSING"
fi

echo
echo "=== 직접 샘플 (감시자와 무관한 교차확인) ==="
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
# ⛔ pgrep -f 를 쓰지 않는다. 명령줄에 그 문자열이 들어 있기만 한 셸까지 세서
# 2026-08-07 에 유휴 감지가 원리적으로 발동하지 못하게 만든 버그의 원인이었다.
# 이 블록은 감시자와 **무관한 교차확인**이 목적이므로, 감시자와 같은 방식으로
# 세야 두 숫자가 어긋났을 때 그것이 실제 불일치임을 알 수 있다.
echo "GPU 작업 프로세스: $(ps -eo comm=,args= | awk '$1 ~ /^python/ && /train_v7\.py|eval_goal_pose\.py|select_best_checkpoint\.py/ {n++} END{print n+0}')개"
echo "큐 워커: $(ps -eo comm=,args= | awk '$1 ~ /^bash/ && $3 == "tools/gpu_queue.sh" {n++} END{print n+0}')개"

echo
echo "=== 큐 ==="
echo "gpu0 대기: $(ls -1 "$ROOT/queue/gpu0" 2>/dev/null | tr '\n' ' ')"
echo "gpu1 대기: $(ls -1 "$ROOT/queue/gpu1" 2>/dev/null | tr '\n' ' ')"
echo "실행 중  : $(ls -1 "$ROOT/queue/done"/*.running 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')"

echo
echo "=== 최근 완료 8건 (rc0 = 정상) ==="
ls -1t "$ROOT/queue/done"/*.rc* 2>/dev/null | head -8 | xargs -n1 basename 2>/dev/null

echo
echo "=== 진행 중인 학습의 마지막 줄 ==="
for f in $(ls -1t "$ROOT/queue/logs"/*.log 2>/dev/null | head -4); do
    printf '%-34s %s\n' "$(basename "$f")" "$(grep -E 'epoch:|Traceback|Error|nan' "$f" 2>/dev/null | tail -1)"
done

echo
echo "=== 채점 상태 ==="
# ⛔ 2026-08-08 수정. 예전에는 `$d/eval` 하나만 봤다. 그런데 **현행 라운드 파이프라인은
# 결과를 `logs/eval_rounds/<라운드>/<run>.accuracy` 에 쓴다** -- run 디렉터리 안이 아니다.
# 그래서 NC_actfilter·NZ_zeroiid·N0_ctrl·NA_histzero 처럼 **이미 채점이 끝난 run 이
# 전부 "미채점" 으로 찍혔다**(실측 확인). 그 표를 믿고 재채점하면 GPU 를 그냥 버린다.
# 두 경로를 모두 본다.
for d in $(ls -1td "$ROOT"/logs/K1/K1/Goal_Pose_V7/*/ 2>/dev/null | head -14); do
    name=$(basename "$d")
    n=$(ls "$d/nn"/model_*.pth 2>/dev/null | wc -l | tr -d ' ')
    rounds=$(ls -d "$ROOT"/logs/eval_rounds/*/"$name".accuracy 2>/dev/null | wc -l | tr -d ' ')
    if [ -d "$d/eval" ]; then
        mark="채점됨 (run내 eval/)"
    elif [ "$rounds" -gt 0 ]; then
        mark="채점됨 (eval_rounds ${rounds}건)"
    else
        mark="미채점"
    fi
    printf '%-46s ckpt=%-4s %s\n' "$name" "$n" "$mark"
done
