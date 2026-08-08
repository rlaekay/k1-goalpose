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
# ⛔ 워커를 `gpu_queue.sh` 하나로만 세면 **작은 레인이 통째로 안 보인다**. 지금 실제
# 구성은 big=`gpu_queue.sh`(v1) 둘 + small=`gpu_worker.sh LANE=small` 둘인데, 예전
# 코드는 "큐 워커: 2개"만 찍어서 백필 워커가 죽어도 아무 신호가 안 났다. 그리고
# big 을 gpu_worker.sh 로 갈아탈 예정이라 스크립트 이름으로 레인을 판정할 수도 없다.
# 그래서 **프로세스 환경변수 LANE 을 직접 읽는다**(없으면 big).
_worker_lanes() {
    local p lane big=0 small=0
    for p in $(ps -eo pid=,args= | awk '$3 ~ /tools\/gpu_(queue|worker)\.sh$/ {print $1}'); do
        lane=$(tr '\0' '\n' < "/proc/$p/environ" 2>/dev/null | sed -n 's/^LANE=//p' | head -1)
        [ -z "$lane" ] && lane=big
        if [ "$lane" = "small" ]; then small=$((small + 1)); else big=$((big + 1)); fi
    done
    echo "big ${big}개 / small ${small}개"
}
echo "큐 워커: $(_worker_lanes)"

echo
echo "=== 큐 ==="
echo "gpu0 대기: $(ls -1 "$ROOT/queue/gpu0" 2>/dev/null | tr '\n' ' ')"
echo "gpu1 대기: $(ls -1 "$ROOT/queue/gpu1" 2>/dev/null | tr '\n' ' ')"
# 작은 레인과 plan 레인도 찍는다. 예전에는 큰 레인 둘만 보여서, 큰 레인이 비었을 때
# **다음에 승격될 것이 있는지**를 이 요약만으로는 알 수 없었다(2026-08-08 실측:
# gpu0 이 5분 뒤 비는데 그 사실이 여기 안 보여서 따로 ssh 를 한 번 더 했다).
echo "small0    : $(ls -1 "$ROOT/queue/small/gpu0" 2>/dev/null | tr '\n' ' ')"
echo "small1    : $(ls -1 "$ROOT/queue/small/gpu1" 2>/dev/null | tr '\n' ' ')"
echo "plan(승격대기): gpu0[$(ls -1 "$ROOT/queue/plan/gpu0" 2>/dev/null | tr '\n' ' ')] gpu1[$(ls -1 "$ROOT/queue/plan/gpu1" 2>/dev/null | tr '\n' ' ')]"
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
# ⛔ 2026-08-08 **두 번째** 수정. 첫 수정(같은 날 오전)은 `logs/eval_rounds/<라운드>/
# <run>.accuracy` 라는 **파일 이름 규칙**에 기댔다. 그런데 현행 작업 스크립트가 쓰는
# 이름은 `<run>_final.accuracy` 이거나 아예 `<arm>_final.accuracy`(타임스탬프 없음)라
# 글로브가 하나도 안 맞았다 -- N9_zerostruct·NB_zerocritic·NA_histzero 가 채점이
# 끝났는데도 전부 "미채점"으로 찍혔다(2026-08-08 14:3x 실측). 같은 거짓음성을 두 번
# 낸 것이고, 그 표를 믿고 재채점하면 GPU 를 그냥 버린다.
#
# **그래서 이름에 기대는 것을 그만둔다.** 리포트는 자기 `checkpoint` 경로를 안에
# 적는다(`report.json`). 그 경로에 run 이름이 들어 있는지로 판정하면 파일명을 어떻게
# 붙이든 맞는다 -- 판정 근거를 규칙이 아니라 **데이터**에 둔다.
#
# 덤으로 **몇 번째 체크포인트로 채점됐는지**를 같이 찍는다. best.pth 가 arm 마다 다른
# iteration 을 가리켜서 "레버 차이"와 "학습량 차이"가 교락된 사고(RETRACTIONS C3)가
# 이 열에서 눈으로 보인다.
for d in $(ls -1td "$ROOT"/logs/K1/K1/Goal_Pose_V7/*/ 2>/dev/null | head -14); do
    name=$(basename "$d")
    n=$(ls "$d/nn"/model_*.pth 2>/dev/null | wc -l | tr -d ' ')
    # segments.csv 는 크다(수백 KB). --include 로 report.json 만 읽는다.
    hits=$(grep -rlF --include=report.json "/$name/nn/" "$ROOT/logs/eval_rounds" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$hits" -gt 0 ]; then
        its=$(grep -rhoE --include=report.json "$name/nn/[A-Za-z0-9_.]+\.pth" "$ROOT/logs/eval_rounds" 2>/dev/null \
              | sed 's|.*/||; s|\.pth$||' | sort -u | tr '\n' ',' | sed 's/,$//')
        mark="채점됨 ${hits}건 [${its}]"
    elif [ -d "$d/eval" ]; then
        mark="채점됨 (run내 eval/)"
    else
        mark="미채점"
    fi
    printf '%-46s ckpt=%-4s %s\n' "$name" "$n" "$mark"
done
