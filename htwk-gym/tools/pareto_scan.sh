#!/usr/bin/env bash
# 한 run 의 여러 체크포인트를 **두 축 모두**로 재서 파레토 프론트를 그린다.
#
#   bash tools/pareto_scan.sh <run_dir> [체크포인트 개수]
#
# ⛔ 왜 필요한가 -- 이 라운드에서 드러난 가장 큰 구멍이다.
#
# run 하나에 체크포인트가 60개인데 우리가 본 것은 **항상 두 개뿐**이었다:
# `best.pth`(선택기가 정확도로 뽑은 것)와 마지막 것. 그런데 N1_path 에서 그 둘의
# 보행 낙상률이 0.038 대 0.0007 로 **50배** 갈렸고, 정확도는 7.93 대 47.70 cm 로
# 반대 방향으로 갈렸다. 즉 학습이 진행되면서 두 축이 서로 반대로 움직이는데,
# 우리는 그 곡선의 **양 끝점만** 보고 "path 는 정확도를 버린다"고 말하고 있었다.
#
# 가운데를 본 적이 없다. 무릎이 있을 수 있고, 있으면 그게 배포 후보다.
# 새 학습이 필요 없다 -- 이미 디스크에 있는 체크포인트를 읽기만 하면 된다.
#
# 각 단계는 60 s 다. 선택기의 --fast 근거(20 s 1위 == 120 s 1위)보다 후하게 잡았다:
# 여기서는 순위가 아니라 **두 축의 값 자체**가 필요하고, 낙상률처럼 드문 사건은
# 짧은 창에서 분산이 크다.

cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"
GPU="${GPU_INDEX:-0}"
DEV="cuda:$GPU"
COMMON_CFG="${COMMON_CFG:-sweeps/N0_ctrl.yaml}"
DUR="${DUR:-60}"

RUN="${1:?사용법: pareto_scan.sh <run_dir> [개수]}"
N="${2:-8}"
NAME=$(basename "$RUN")
OUT="${PARETO_OUT:-$ROOT/logs/pareto/$NAME}"
mkdir -p "$OUT"

CFG="$OUT/eval.cfg.yaml"
python -u tools/make_eval_cfg.py --common "$COMMON_CFG" --run "$RUN" --out "$CFG" || exit 1

# 체크포인트를 iteration 순으로 정렬해 균등하게 N개 뽑는다. 마지막 것은 반드시 포함한다
# -- 지금까지 "잘 걷는 쪽"이 거기였다.
mapfile -t ALL < <(ls -1v "$RUN"/nn/model_*.pth 2>/dev/null)
TOTAL=${#ALL[@]}
if [ "$TOTAL" -lt 2 ]; then echo "체크포인트가 부족하다: $TOTAL"; exit 1; fi
PICK=()
for i in $(seq 0 $((N - 1))); do
    idx=$(( i * (TOTAL - 1) / (N - 1) ))
    PICK+=("${ALL[$idx]}")
done

echo "파레토 스캔: $NAME  ($TOTAL 개 중 $N 개, 각 ${DUR}s x 2축)"
for ck in "${PICK[@]}"; do echo "  $(basename "$ck")"; done
echo

for ck in "${PICK[@]}"; do
    it=$(basename "$ck" .pth | sed 's/model_//')
    echo "---- iteration $it ----"
    python -u eval_goal_pose.py --task K1/Goal_Pose_V7 --config "$CFG" \
        --checkpoint "$ck" --terrain plane --duration_s "$DUR" \
        --sim_device "$DEV" --rl_device "$DEV" \
        --out "$OUT/it${it}.accuracy" 2>&1 | tail -3
    python -u eval_goal_pose.py --task K1/Goal_Pose_V7 --config "$CFG" \
        --checkpoint "$ck" --terrain plane --duration_s "$DUR" \
        --goal_pattern forward_hold \
        --sim_device "$DEV" --rl_device "$DEV" \
        --out "$OUT/it${it}.walk" 2>&1 | tail -3
done

echo
echo "==== 파레토 표: $NAME ===="
python -u tools/pareto_table.py "$OUT"
