#!/bin/bash
# 학습이 끝나면 자동으로 eval(2분 영상)을 돌리고 결과를 shared_eval_videos/ 에 모아둔다.
# 영상은 git으로 옮기지 않는다 (.gitignore) -- 서버에서 scp/rsync로 pull할 것.
#
# 사용법:
#   bash tools/train_and_eval.sh <sim_device> <rl_device> -- <train.py 인자들...>
#
# 예:
#   bash tools/train_and_eval.sh cuda:1 cuda:1 -- \
#     --task=K1/Goal_Pose --config sweeps/armA_continue.yaml --headless True \
#     --checkpoint logs/K1/K1/Goal_Pose/2026-07-23-21-54-01/nn/model_20000.pth \
#     --num_envs 4096 --max_iterations 20000

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SHARED_DIR="$REPO_ROOT/shared_eval_videos"
mkdir -p "$SHARED_DIR"

if [ $# -lt 3 ]; then
  echo "사용법: bash tools/train_and_eval.sh <sim_device> <rl_device> -- <train.py 인자들...>"
  exit 1
fi

SIM_DEV=$1
RL_DEV=$2
shift 2
if [ "$1" == "--" ]; then shift; fi

LOGFILE=$(mktemp)

echo "=== 학습 시작: python train.py $* ==="
cd "$REPO_ROOT"
python train.py "$@" 2>&1 | tee "$LOGFILE"

# 학습 로그에서 run 디렉토리 추출 ("Saving model to logs/.../nn/model_X.pth" 마지막 줄 기준)
RUN_DIR=$(grep "Saving model to" "$LOGFILE" | tail -1 | sed -E 's|Saving model to (.*)/nn/model_[0-9]+\.pth|\1|')

if [ -z "$RUN_DIR" ]; then
  echo "!!! run 디렉토리를 찾지 못했습니다. 로그: $LOGFILE"
  exit 1
fi

CKPT=$(ls -t "$RUN_DIR"/nn/model_*.pth | head -1)
echo "=== 학습 종료. checkpoint: $CKPT ==="

echo "=== eval 시작 (2분 영상) ==="
python eval_goal_pose.py \
  --task K1/Goal_Pose \
  --checkpoint "$CKPT" \
  --sim_device "$SIM_DEV" \
  --rl_device "$RL_DEV" \
  --record_video \
  --record_video_s 120

EVAL_DIR=$(ls -td "$RUN_DIR"/eval/*/ | head -1)
VIDEO_SRC="${EVAL_DIR}rollout_env0.mp4"

RUN_NAME=$(basename "$RUN_DIR")
TS=$(date +%Y%m%d-%H%M%S)
VIDEO_DST="$SHARED_DIR/${RUN_NAME}_${TS}.mp4"

if [ -f "$VIDEO_SRC" ]; then
  cp "$VIDEO_SRC" "$VIDEO_DST"
  echo "=== 영상 복사 완료: $VIDEO_DST ==="
else
  echo "!!! 영상 파일을 찾지 못했습니다: $VIDEO_SRC"
fi

rm -f "$LOGFILE"
