#!/usr/bin/env bash
# 스모크 테스트가 남긴 빈 run 디렉터리만 지운다.
#
#   bash tools/clean_smoke_runs.sh          # 무엇을 지울지 보여주기만 한다(기본)
#   bash tools/clean_smoke_runs.sh --yes    # 실제로 지운다
#
# ⛔ 왜 스크립트인가 -- 2026-08-07 에 내가 이걸 임시 `rm -rf` 루프로 세 번 했고
# 세 번째에 **막 시작한 학습 두 개를 지웠다**(NZ_zeroiid, N4_hist). 둘 다
# FileNotFoundError 로 죽었다. 판정 기준이 "체크포인트 0개 = 스모크"였는데,
# **방금 시작한 진짜 학습도 체크포인트가 0개다**(save_interval 100 이면 첫 저장까지
# 4분쯤 걸린다).
#
# 그래서 세 조건을 **전부** 만족해야 지운다:
#   1. 체크포인트 0개
#   2. 디렉터리가 MIN_AGE_MIN 분보다 오래됐다 (기본 20분 -- 첫 저장 시간의 5배)
#   3. 살아 있는 train_v7.py 가 그 디렉터리를 참조하지 않는다
#
# 3번은 2번을 통과한 뒤에도 필요하다: 중단됐다 재시작한 run 은 오래된 디렉터리를
# 계속 쓸 수 있다.

cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"
LOGS="$ROOT/logs/K1/K1/Goal_Pose_V7"
MIN_AGE_MIN="${MIN_AGE_MIN:-20}"
APPLY=""
[ "${1:-}" = "--yes" ] && APPLY=1

# 살아 있는 학습이 쓰는 run 디렉터리 목록. /proc 의 cwd 와 명령줄 양쪽을 본다.
live=$(ps -eo args | grep "train_v7\.py" | grep -v grep)

del=0; keep=0
for d in "$LOGS"/*/; do
    name=$(basename "$d")
    n=$(ls "$d"/nn/model_*.pth 2>/dev/null | wc -l | tr -d ' ')
    [ "$n" -gt 0 ] && { keep=$((keep+1)); continue; }

    age_min=$(( ( $(date +%s) - $(stat -c %Y "$d") ) / 60 ))
    if [ "$age_min" -lt "$MIN_AGE_MIN" ]; then
        echo "  보존 $name  (체크포인트 0 이지만 ${age_min}분밖에 안 됐다 -- 방금 시작한 학습일 수 있다)"
        keep=$((keep+1)); continue
    fi
    # 살아 있는 프로세스가 이 arm 을 돌고 있으면 건드리지 않는다.
    arm="${name#*_}"
    if printf '%s' "$live" | grep -q "sweeps/${arm}\.yaml"; then
        echo "  보존 $name  (같은 arm 이 지금 돌고 있다)"
        keep=$((keep+1)); continue
    fi

    if [ -n "$APPLY" ]; then
        rm -rf "$d"; echo "  삭제 $name  (체크포인트 0, ${age_min}분)"
    else
        echo "  삭제예정 $name  (체크포인트 0, ${age_min}분)"
    fi
    del=$((del+1))
done

echo
if [ -n "$APPLY" ]; then
    echo "삭제 $del 개, 보존 $keep 개."
else
    echo "삭제예정 $del 개, 보존 $keep 개.  실제로 지우려면 --yes 를 붙여라."
fi
