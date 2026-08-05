#!/usr/bin/env bash
# 3인자 격자 탐색: 실기 signature를 재현하는 조건을 찾는다.
#
# 단일 레버로는 아무것도 다 못 맞췄다(2026-08-05 밤):
#   필터 50-70 Hz  -> 추종률 0.64-0.67 (실기 0.61) + 낙상 다수.  발 들기는 과억제
#   obs 지연 20 ms -> 다리 교차 9.2% (실기 9.9%).                추종률은 정상
#   다리 이득 0.35 -> 낙상 44/분.                                발목 속도 재현 못함
#   어느 것도 dq_ankR(실기 7.0/8.9)을 재현 못함 -- 모든 sim이 1.4-3.5
#
# 세 축이 서로 다른 지표를 맞추므로 조합을 본다. 축을 셋으로 제한한 것은,
# armD가 12개 레버를 동시에 바꿔 아무것도 배우지 못한 실패를 피하기 위해서다 --
# 셋이면 어느 축이 무엇을 옮기는지 사후에 분해할 수 있다.
#
# set -u 를 쓰지 않는다: conda.sh가 미설정 변수를 참조해 활성화 전에 죽는다.
cd "$(dirname "$0")/.."
source /mnt/DATA/workspace/ws_eungkyu/miniconda3/etc/profile.d/conda.sh
conda activate k1goalpose
PT=logs/K1/K1/Goal_Pose_V7/2026-08-04-09-48-36_I3b_stance10/nn/model_200.pt
D=logs/mujoco/grid
mkdir -p "$D"
DUR="${DUR:-25}"

run() {
    local n="$1"; shift
    [ -f "$D/$n.csv" ] && return 0            # 재시작 안전
    python play_mujoco_goalpose.py --duration "$DUR" --policy "$PT" --goal-hold \
        --dump-csv "$D/$n.csv" --out "$D/$n.json" "$@" > "$D/$n.log" 2>&1
    echo "  done $n"
}

echo "### 격자: 필터Hz x obs지연 x 다리이득  ($(date '+%H:%M'))"
for FH in 0 100 70 50; do
  for LG in 20 0; do
    for GN in 1.0 0.7 0.5; do
      A=""
      [ "$FH" != "0" ] && A="$A --deploy-filter --filter-hz $FH"
      [ "$LG" != "0" ] && A="$A --sense-lag-ms $LG"
      [ "$GN" != "1.0" ] && A="$A --leg-gain $GN"
      run "f${FH}_l${LG}_g${GN}" $A
    done
  done
done

echo "### 점수 ($(date '+%H:%M'))"
python3 tools/signature_score.py "$D"/*.csv 2>&1 | tail -30
