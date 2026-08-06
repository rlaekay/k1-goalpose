#!/usr/bin/env bash
# 이름 패턴에 맞는 run 디렉터리 중 **실제로 학습된** 가장 최근 것을 고른다.
#
#   tools/pick_run.sh N0_ctrl          -> logs/K1/K1/Goal_Pose_V7/2026-...-N0_ctrl
#   MIN_CKPT=10 tools/pick_run.sh N1_path
#
# 왜 필요한가 (2026-08-07 실제 사고): 큐 작업들이 `ls -1td .../*_N0_ctrl | head -1`
# 로 run 을 찾았는데, 그 사이 내가 돌린 **3-iteration 스모크**가 더 최신이라
# eval_round 가 체크포인트 0개짜리 디렉터리를 집었다. nn/ 은 존재하므로 "nn 이 없으면
# 건너뛴다" 가드를 통과했고, 체크포인트가 없으니 --checkpoint 가 빈 문자열이 되어
# eval_goal_pose 의 기본값("-1" = logs 아래 아무 최신 모델)로 떨어졌다. 즉 죽지 않고
# **엉뚱한 정책을 N0_ctrl 이라는 이름으로 채점했다.**
#
# 시간순 최신은 "내가 원하는 run" 의 안전한 대용이 아니다. 최소 체크포인트 수를
# 요구하면 스모크와 중단된 run 이 자동으로 걸러진다.

MIN_CKPT="${MIN_CKPT:-5}"
ROOT_DIR="${ROOT_DIR:-logs/K1/K1/Goal_Pose_V7}"

pat="${1:?사용법: pick_run.sh <arm 이름>}"

best=""
for d in $(ls -1td "$ROOT_DIR"/*_"$pat" 2>/dev/null); do
    n=$(ls "$d"/nn/model_*.pth 2>/dev/null | wc -l)
    if [ "$n" -ge "$MIN_CKPT" ]; then best="$d"; break; fi
done

if [ -z "$best" ]; then
    echo "pick_run: '$pat' 에 맞으면서 체크포인트 ${MIN_CKPT}개 이상인 run 이 없다" >&2
    exit 1
fi
echo "$best"
