#!/bin/bash
# One H arm: train -> select including warm-start/early checkpoints -> full eval suite.
set -euo pipefail

if [ $# -lt 5 ]; then
  echo "usage: $0 <H0..H3> <config> <checkpoint> <sim_device> <rl_device>" >&2
  exit 2
fi

ARM=$1
CONFIG=$2
WARM_START=$3
SIM_DEV=$4
RL_DEV=$5
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ITERS="${ITERS:-12000}"
ENVS="${ENVS:-4096}"
VIDEO_S="${VIDEO_S:-8}"
SHARED_DIR="${SHARED_DIR:-$REPO_ROOT/shared_eval_videos/hbatch}"
mkdir -p "$SHARED_DIR"
cd "$REPO_ROOT"

TRAIN_LOG=$(mktemp)
trap 'rm -f "$TRAIN_LOG"' EXIT
python -u train_hbatch.py --task=K1/Goal_Pose_HBatch --config "$CONFIG" \
  --headless True --checkpoint "$WARM_START" --num_envs "$ENVS" \
  --max_iterations "$ITERS" --sim_device "$SIM_DEV" --rl_device "$RL_DEV" \
  2>&1 | tee "$TRAIN_LOG"

RUN_DIR=$(grep "Saving model to" "$TRAIN_LOG" | tail -1 | \
  sed -E 's|.*Saving model to (.*)/nn/model_[0-9]+\.pth.*|\1|')
if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
  echo "!!! $ARM run directory not found" >&2
  exit 1
fi

# This is a fine-tune, so step 0 is a legitimate candidate.  The old tail-only
# selector excluded precisely the useful early robust checkpoints.
cp "$WARM_START" "$RUN_DIR/nn/model_0.pth"
SELECT_DIR="$RUN_DIR/eval/select_hbatch"
python tools/select_best_checkpoint.py --run_dir "$RUN_DIR" \
  --task K1/Goal_Pose_HBatch --config "$RUN_DIR/config.yaml" \
  --tail_frac 1.0 --max_candidates 24 \
  --include 0,100,200,300,400,500,700,1000,1500,2000,3000 \
  --sim_device "$SIM_DEV" --rl_device "$RL_DEV" --out "$SELECT_DIR"
BEST=$(sed -n '1p' "$SELECT_DIR/BEST_CHECKPOINT")

eval_one() {
  local name=$1
  shift
  python eval_goal_pose.py --task K1/Goal_Pose_HBatch \
    --config "$RUN_DIR/config.yaml" --checkpoint "$BEST" \
    --sim_device "$SIM_DEV" --rl_device "$RL_DEV" \
    --out "$RUN_DIR/eval/$name" "$@"
}

eval_one clean
eval_one force --keep_perturbations
eval_one jitter --stress jitter --duration_s 60
eval_one combined --stress jitter --keep_perturbations --duration_s 60
eval_one lateral --goal_pattern lateral --duration_s 60
eval_one reverse --goal_pattern reverse --duration_s 60
eval_one video_force --keep_perturbations --record_video --record_video_s "$VIDEO_S" \
  --duration_s 10 --num_envs 16 --force_visualization_probe

VIDEO_REPORT="$RUN_DIR/eval/video_force/report.json"
VIDEO_MP4="$RUN_DIR/eval/video_force/rollout_env0.mp4"
if [ ! -s "$VIDEO_MP4" ] || [ ! -s "$VIDEO_REPORT" ]; then
  echo "!!! $ARM video_force is missing a nonempty mp4 or report" >&2
  exit 1
fi
python -c 'import json,sys; d=json.load(open(sys.argv[1])); e=d.get("disturbance_eval",{}); assert e.get("video_recorded_frames",0)>0 and e.get("video_force_arrow_drawn_frames",0)>0 and e.get("video_path_carrot_drawn_frames",0)>0 and e.get("video_path_trace_drawn_frames",0)>0, e' "$VIDEO_REPORT"

DEST="$SHARED_DIR/${ARM}_$(date +%Y%m%d-%H%M%S)"
PARTIAL="${DEST}-partial-$$"
mkdir -p "$PARTIAL"
cp "$SELECT_DIR/BEST_CHECKPOINT" "$SELECT_DIR/selection.json" "$SELECT_DIR/selection.md" "$PARTIAL/"
for name in clean force jitter combined lateral reverse video_force; do
  mkdir -p "$PARTIAL/$name"
  for f in report.json report.md segments.csv rollout_env0.mp4; do
    [ ! -f "$RUN_DIR/eval/$name/$f" ] || cp "$RUN_DIR/eval/$name/$f" "$PARTIAL/$name/"
  done
done
touch "$PARTIAL/COMPLETE"
mv "$PARTIAL" "$DEST"
python tools/compare_hbatch_results.py --root "$SHARED_DIR" \
  --out "$SHARED_DIR/hbatch-comparison-codex.md"
echo "HBatch $ARM complete: $DEST"
