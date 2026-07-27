#!/bin/bash
# Train -> pick the best checkpoint -> record a video -> drop it in a shared folder.
#
# The videos and reports land in htwk-gym/shared_eval_videos/, which is gitignored:
# pull them with scp/rsync, they are never carried by git push/pull.
#
# Usage:
#   bash tools/train_and_eval.sh <sim_device> <rl_device> -- <train.py args...>
#
# Example:
#   bash tools/train_and_eval.sh cuda:1 cuda:1 -- \
#     --task=K1/Goal_Pose --config sweeps/armA_continue.yaml --headless True \
#     --checkpoint logs/K1/K1/Goal_Pose/2026-07-23-21-54-01/nn/model_20000.pth \
#     --num_envs 4096 --max_iterations 20000
#
# Environment overrides:
#   SELECT_BEST=0   evaluate only the final checkpoint instead of searching for the best
#   VIDEO_S=60      video length in seconds (default 120)
#   SHARED_DIR=...  where to drop results (default htwk-gym/shared_eval_videos)
#   TRAIN=train_v7.py  trainer entrypoint (default train.py; v7 needs the v3 runner)
#   STRESS=1        also run the goal-jitter robustness eval on the winner

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SHARED_DIR="${SHARED_DIR:-$REPO_ROOT/shared_eval_videos}"
SELECT_BEST="${SELECT_BEST:-1}"
VIDEO_S="${VIDEO_S:-120}"
TRAIN="${TRAIN:-train.py}"
STRESS="${STRESS:-0}"
mkdir -p "$SHARED_DIR"

if [ $# -lt 3 ]; then
  echo "usage: bash tools/train_and_eval.sh <sim_device> <rl_device> -- <train.py args...>" >&2
  exit 1
fi

SIM_DEV=$1
RL_DEV=$2
shift 2
[ "${1:-}" = "--" ] && shift

cd "$REPO_ROOT"
LOGFILE=$(mktemp)
trap 'rm -f "$LOGFILE"' EXIT

echo "=== 학습 시작: python $TRAIN $* ==="
# -u: Python fully buffers stdout (~8 KB) whenever it isn't a tty, which a pipe
# into `tee` always triggers. Short per-iteration prints then sit unflushed for
# hundreds of iterations, so a live tmux pane looks stalled while GPU-Util sits
# at 95% -- training is fine, only the console feedback is delayed. -u forces
# line buffering so `tmux capture-pane` reflects reality within one print.
python -u "$TRAIN" "$@" 2>&1 | tee "$LOGFILE"

# The run dir is whatever the trainer last saved into; parsing its own log avoids
# guessing at timestamped directory names.
RUN_DIR=$(grep "Saving model to" "$LOGFILE" | tail -1 \
          | sed -E 's|.*Saving model to (.*)/nn/model_[0-9]+\.pth.*|\1|')
if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
  echo "!!! run 디렉토리를 찾지 못했습니다. 학습 로그를 확인하세요." >&2
  exit 1
fi
RUN_NAME=$(basename "$RUN_DIR")
TASK=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['basic']['task'])" "$RUN_DIR/config.yaml")
echo "=== 학습 종료: $RUN_DIR (task $TASK) ==="

TS=$(date +%Y%m%d-%H%M%S)
DEST="$SHARED_DIR/${RUN_NAME}_${TS}"
mkdir -p "$DEST"

if [ "$SELECT_BEST" = "1" ]; then
  echo "=== 최적 체크포인트 탐색 + 평가 + 영상 ==="
  python tools/select_best_checkpoint.py \
    --run_dir "$RUN_DIR" --task "$TASK" \
    --sim_device "$SIM_DEV" --rl_device "$RL_DEV" \
    --record_video --record_video_s "$VIDEO_S" --link_best
  EVAL_DIR=$(ls -td "$RUN_DIR"/eval/select_*/ 2>/dev/null | head -1)
  VIDEO_SRC="${EVAL_DIR}winner_video/rollout_env0.mp4"
else
  echo "=== 마지막 체크포인트 평가 + 영상 ==="
  CKPT=$(ls -t "$RUN_DIR"/nn/model_*.pth | head -1)
  python eval_goal_pose.py \
    --task "$TASK" --checkpoint "$CKPT" \
    --sim_device "$SIM_DEV" --rl_device "$RL_DEV" \
    --record_video --record_video_s "$VIDEO_S"
  EVAL_DIR=$(ls -td "$RUN_DIR"/eval/*/ | head -1)
  VIDEO_SRC="${EVAL_DIR}rollout_env0.mp4"
fi

# Ship the video together with the reports that explain it, so the mp4 is never
# looked at without the numbers next to it.
for f in "$VIDEO_SRC" "${EVAL_DIR}report.md" "${EVAL_DIR}report.json" "${EVAL_DIR}selection.md" "${EVAL_DIR}BEST_CHECKPOINT"; do
  [ -f "$f" ] && cp "$f" "$DEST/"
done
[ -f "$DEST/rollout_env0.mp4" ] || echo "!!! 영상이 생성되지 않았습니다 (VIDEO_SRC=$VIDEO_SRC 없음) -- eval 로그에서 record_video 단계 실패 원인 확인 필요" >&2

if [ "$STRESS" = "1" ]; then
  echo "=== 강건성 스트레스 평가 (goal jitter 50Hz) ==="
  BEST_CKPT=$(cat "${EVAL_DIR}BEST_CHECKPOINT" 2>/dev/null || ls -t "$RUN_DIR"/nn/model_*.pth | head -1)
  # Non-fatal: a failed stress run must not discard the training result above.
  if python eval_goal_pose.py --task "$TASK" --checkpoint "$BEST_CKPT" \
      --sim_device "$SIM_DEV" --rl_device "$RL_DEV" --stress jitter --duration_s 60; then
    STRESS_DIR=$(ls -td "$RUN_DIR"/eval/*_stress_jitter/ 2>/dev/null | head -1)
    [ -n "$STRESS_DIR" ] && cp "${STRESS_DIR}report.md" "$DEST/report_stress_jitter.md" 2>/dev/null || true
  else
    echo "!!! 스트레스 평가 실패 (학습/기본 평가 결과는 유효)" >&2
  fi
fi

if [ -f "$DEST/rollout_env0.mp4" ]; then
  echo "=== 완료: $DEST ==="
  ls -lh "$DEST"
else
  echo "!!! 영상을 찾지 못했습니다 (평가 결과는 $EVAL_DIR 확인)" >&2
fi
