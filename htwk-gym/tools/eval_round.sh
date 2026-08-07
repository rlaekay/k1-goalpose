#!/usr/bin/env bash
# 한 라운드의 채점. 학습이 끝난 run 들을 받아서 **두 개의 공통 프로토콜**로 잰다.
#
#   bash tools/eval_round.sh <run_dir> [<run_dir> ...]
#
# ⛔ 왜 두 개인가 -- 이 저장소가 반복해서 낸 오류 두 가지를 동시에 막기 위해서다.
#
# (1) "각 arm 을 자기 config 로 채점" 은 arm 끼리 비교가 안 된다. v7 배치가 정확히
#     그렇게 해서 E1(path) 을 path 과제로, E0 을 waypoint 과제로 재고 순위를 매겼다.
#     그래서 정확도 순위는 **모든 arm 을 N0_ctrl.yaml 하나로** 잰다.
#
# (2) 더 중요한 것: 지금까지의 지표가 **보행을 잰 적이 없다.** 학습·평가 과제 모두
#     waypoint 이고, 그 과제의 요구 속도는 median 0.126 m/s / 42% 가 0.5 m 미만이라
#     사실상 "서 있기" 다. 3.86 cm 라는 좋아 보이는 숫자가 거기서 나왔고, 그 정책은
#     실기에서 세 걸음을 못 갔다. 그래서 --goal_pattern forward_hold 를 **판정에
#     정식으로 넣는다**: 목표를 항상 2 m 앞에 다시 두어 도착이 없게 만들면 body_speed
#     가 과도구간이 아니라 지속 속도가 된다. forward_hold 는 자기가 알아서
#     goal_mode_mixture 를 waypoint 1.0 으로 덮으므로 그 자체가 공통 프로토콜이다.
#
# 두 수치는 트레이드오프 관계일 수 있고, 그걸 보는 것이 이 스크립트의 목적이다.
# 하나만 보고 배포 후보를 고르지 않는다.

cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"
GPU="${GPU_INDEX:-0}"
DEV="cuda:$GPU"
# 모든 arm 이 같은 자로 재도록 고정한 공통 정확도 프로토콜. 배포가 실제로 요구하는
# 과제(2 m 안쪽 목표로 가서 선다)가 이것이고, 게이트도 여기서 정의돼 있다.
COMMON_CFG="${COMMON_CFG:-sweeps/N0_ctrl.yaml}"
# EVAL_OUT 을 주면 그 디렉터리에 쌓는다. 서로 다른 시각에 끝나는 arm 들을 **한 표**로
# 보려면 필요하다 -- round_table.py 는 디렉터리 하나를 읽으므로, 각 채점이 자기
# 타임스탬프 디렉터리를 만들면 arm 끼리 비교가 안 된다.
OUTROOT="${EVAL_OUT:-$ROOT/logs/eval_rounds/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUTROOT"

if [ $# -eq 0 ]; then
    echo "사용법: bash tools/eval_round.sh <run_dir> [<run_dir> ...]" >&2
    exit 1
fi

echo "라운드 채점 시작  device=$DEV  공통 config=$COMMON_CFG"
echo "출력: $OUTROOT"
echo

for RUN in "$@"; do
    NAME=$(basename "$RUN")
    if [ ! -d "$RUN/nn" ]; then
        echo "!! $NAME: nn/ 이 없다. 건너뛴다."; continue
    fi
    # ⛔ nn/ 이 있는 것만으로는 부족하다. 2026-08-07 에 체크포인트 0개짜리 스모크
    # 디렉터리가 여기를 통과했고, --checkpoint 가 빈 문자열이 되어 eval_goal_pose 의
    # 기본값("-1" = logs 아래 아무 최신 모델)로 떨어져서 **엉뚱한 정책을 이 arm 의
    # 이름으로 채점했다.** 죽지 않고 틀린 숫자를 내는 쪽이 훨씬 나쁘다.
    NCK=$(ls "$RUN"/nn/model_*.pth 2>/dev/null | wc -l)
    if [ "$NCK" -lt 2 ]; then
        echo "!! $NAME: 체크포인트가 ${NCK}개뿐이다(스모크/중단 run). 건너뛴다."; continue
    fi
    echo "===================================================================="
    echo "== $NAME"
    echo "===================================================================="

    # 공통 프로토콜에 이 arm 의 **관측 인터페이스만** 이식한다. 프로토콜(목표 분포,
    # 노이즈, 외란, 자산)은 그대로 공통이고, 체크포인트의 성질인 num_observations 와
    # observation 블록만 정책을 따라간다. 이게 없으면 관측을 넓힌 arm 은 채점 자체가
    # 안 된다 -- NA_histzero(obs 270)가 공통 config(54)로 안 올라가서 죽었다.
    CFG="$OUTROOT/$NAME.cfg.yaml"
    if ! python -u tools/make_eval_cfg.py --common "$COMMON_CFG" \
            --run "$RUN" --out "$CFG"; then
        echo "!! $NAME: 평가 config 생성 실패. 건너뛴다."; continue
    fi

    # --- 1단계: 최고 체크포인트 고르기 (공통 정확도 프로토콜) -------------
    # tail_frac 1.0: 봉우리가 run 후반에 있다는 보장이 없다. L 배치에서 12000
    # iteration run 의 최고점이 1250 이었다 -- 기본값 0.6 이면 그 지점을 아예
    # 후보에서 뺀다. max_candidates 12 는 그대로라 비용은 같다.
    SEL="$OUTROOT/$NAME.select"
    echo "-- 1/3 선택기 (--fast, tail_frac 1.0)"
    python -u tools/select_best_checkpoint.py \
        --run_dir "$RUN" --task K1/Goal_Pose_V7 --config "$CFG" \
        --fast --tail_frac 1.0 --terrain plane \
        --sim_device "$DEV" --rl_device "$DEV" \
        --out "$SEL" --link_best 2>&1 | tail -40
    rc=$?
    if [ $rc -ne 0 ]; then echo "!! $NAME 선택기 실패 rc=$rc"; continue; fi

    BEST="$RUN/nn/best.pth"
    if [ ! -e "$BEST" ]; then
        BEST=$(ls -1v "$RUN"/nn/model_*.pth 2>/dev/null | tail -1)
        echo "   (best.pth 없음 -> 마지막 체크포인트 $BEST 로 진행)"
    fi

    # --- 2단계: 공통 정확도 프로토콜 정식 채점 ---------------------------
    echo "-- 2/3 정확도 (공통 waypoint 프로토콜, 120 s)"
    python -u eval_goal_pose.py --task K1/Goal_Pose_V7 --config "$CFG" \
        --checkpoint "$BEST" --terrain plane --duration_s 120 \
        --sim_device "$DEV" --rl_device "$DEV" \
        --out "$OUTROOT/$NAME.accuracy" 2>&1 | tail -25

    # --- 3단계: 지속 보행 (그동안 아무도 재지 않은 축) --------------------
    echo "-- 3/4 지속 보행 (forward_hold, 120 s)"
    python -u eval_goal_pose.py --task K1/Goal_Pose_V7 --config "$CFG" \
        --checkpoint "$BEST" --terrain plane --duration_s 120 \
        --goal_pattern forward_hold \
        --sim_device "$DEV" --rl_device "$DEV" \
        --out "$OUTROOT/$NAME.walk" 2>&1 | tail -25

    # --- 4단계: 마지막 체크포인트도 보행으로 재본다 -----------------------
    # 왜: 위 선택기는 **정확도**(waypoint) 로 승자를 뽑는다. path arm 에서는 그
    # 승자가 "잘 서는 체크포인트" 이고 정작 잘 걷는 체크포인트가 따로일 수 있다.
    # 한 축으로 뽑아 다른 축으로 채점하는 것이 이 저장소가 반복한 오류라, 두 번째
    # 후보를 싸게(60 s) 같이 재서 그 가능성을 열어둔다.
    FINAL=$(ls -1v "$RUN"/nn/model_*.pth 2>/dev/null | tail -1)
    if [ -n "$FINAL" ] && [ "$(readlink -f "$FINAL")" != "$(readlink -f "$BEST")" ]; then
        echo "-- 4/4 지속 보행: 마지막 체크포인트 $(basename "$FINAL") (60 s)"
        python -u eval_goal_pose.py --task K1/Goal_Pose_V7 --config "$CFG" \
            --checkpoint "$FINAL" --terrain plane --duration_s 60 \
            --goal_pattern forward_hold \
            --sim_device "$DEV" --rl_device "$DEV" \
            --out "$OUTROOT/${NAME}_final.walk" 2>&1 | tail -20
    else
        echo "-- 4/4 건너뜀: 선택기 승자가 곧 마지막 체크포인트다"
    fi

    echo
done

echo "===================================================================="
echo "== 라운드 요약"
echo "===================================================================="
python -u tools/round_table.py "$OUTROOT" 2>&1 || echo "(요약 표 생성 실패 -- report.json 을 직접 읽어라)"
echo
echo "결과 위치: $OUTROOT"
