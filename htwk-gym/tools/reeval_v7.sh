#!/bin/bash
# Re-evaluate the four v7 arms with the FIXED harness. No training.
#
#   bash tools/reeval_v7.sh
#   ARMS="E1_path V7_full" bash tools/reeval_v7.sh
#
# Why: the 2026-07-27 batch was scored against envs/K1/Goal_Pose_V7.yaml instead
# of each arm's own sweeps/*.yaml, because train_and_eval.sh called
# select_best_checkpoint.py without --config. E0 and E2 trained with path mode
# OFF and were then graded on a task that was 44-46% path segments; 90 of E0's
# 93 falls and 123 of E2's 125 falls were in exactly those segments. On top of
# that, path segments cannot satisfy a position gate by construction (the goal
# is deliberately lookahead_min ahead and keeps moving), and they were half the
# sample. Both are fixed now, so every number from that batch needs re-deriving
# before it can be used as the F-batch baseline.
#
# Costs ~30 min total, no GPU-days. Each arm is scored on its OWN training task
# AND on the common v7 task, because the two answer different questions:
#   own    -- how good is this policy at the job it was trained for?
#   common -- how well does it generalise to the integrated v7 task?
# The first batch only ever produced the second, and reported it as the first.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RUN_ROOT="${RUN_ROOT:-logs/K1/K1/Goal_Pose_V7}"
ARMS="${ARMS:-E0_armB_armsdown E1_path E2_robust V7_full}"
GPU="${GPU:-0}"
DEV="cuda:$GPU"
VIDEO_S="${VIDEO_S:-60}"
COMMON="${COMMON:-envs/K1/Goal_Pose_V7.yaml}"
OUT_ROOT="$REPO_ROOT/shared_eval_videos/reeval_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT_ROOT"

for arm in $ARMS; do
  RUN_DIR=$(ls -d "$RUN_ROOT"/*"$arm" 2>/dev/null | head -1)
  if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
    echo "!!! run 디렉토리 없음: $RUN_ROOT/*$arm — 건너뜀" >&2
    continue
  fi
  echo ""
  echo "================ $arm ================"
  echo "run: $RUN_DIR"

  # 1) best checkpoint, re-selected on the arm's OWN task
  echo "--- [1/3] 자기 과제로 최적 체크포인트 재선택 + 평가 + 영상 ---"
  python tools/select_best_checkpoint.py \
    --run_dir "$RUN_DIR" --task K1/Goal_Pose_V7 \
    --sim_device "$DEV" --rl_device "$DEV" \
    --record_video --record_video_s "$VIDEO_S" --link_best
  SEL_DIR=$(ls -td "$RUN_DIR"/eval/select_*/ 2>/dev/null | head -1)
  BEST=$(cat "${SEL_DIR}BEST_CHECKPOINT" 2>/dev/null || ls -t "$RUN_DIR"/nn/model_*.pth | head -1)

  D="$OUT_ROOT/${arm}"
  mkdir -p "$D/own_task"
  for f in report.md report.json segments.csv selection.md BEST_CHECKPOINT; do
    [ -f "${SEL_DIR}$f" ] && cp "${SEL_DIR}$f" "$D/own_task/"
  done
  [ -f "${SEL_DIR}winner_video/rollout_env0.mp4" ] && cp "${SEL_DIR}winner_video/rollout_env0.mp4" "$D/own_task/"

  # 2) same winner, scored on the shared v7 task -> cross-arm comparison
  echo "--- [2/3] 공통 v7 과제로 평가 (arm 간 비교용) ---"
  python eval_goal_pose.py --task K1/Goal_Pose_V7 --config "$COMMON" \
    --checkpoint "$BEST" --sim_device "$DEV" --rl_device "$DEV" \
    --out "$D/common_task"

  # 3) jitter stress
  echo "--- [3/3] stress jitter ---"
  python eval_goal_pose.py --task K1/Goal_Pose_V7 --config "$COMMON" \
    --checkpoint "$BEST" --sim_device "$DEV" --rl_device "$DEV" \
    --stress jitter --duration_s 60 --out "$D/stress_jitter" || \
    echo "!!! $arm stress 실패 (나머지 결과는 유효)" >&2
done

echo ""
echo "================ 요약 ================"
python - "$OUT_ROOT" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
hdr = "  {:<20} {:>9} {:>9} {:>7} {:>7} {:>8} {:>9}"


def dig(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


for scope in ("own_task", "common_task"):
    print("\n[{}]".format(scope))
    print(hdr.format("arm", "pos_med", "pos_p90", "head", "falls", "성공률", "속도p99"))
    print("  " + "-" * 74)
    for p in sorted(glob.glob(os.path.join(root, "*", scope, "report.json"))):
        try:
            r = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        arm = p.split(os.sep)[-3][:20]
        g = lambda *k: dig(r, *k)
        pm, pp = g("pos_err_m", "median"), g("pos_err_m", "p90")
        print(hdr.format(
            arm,
            "{:.1f}cm".format(pm * 100) if pm is not None else "—",
            "{:.1f}cm".format(pp * 100) if pp is not None else "—",
            "{:.1f}".format(g("heading_err_deg", "median") or 0),
            str(r.get("falls", "—")),
            "{:.0f}%".format((r.get("success_rate_strict") or 0) * 100),
            "{:.2f}".format(g("body_speed", "p99") or 0)))
        n_g, n_p = r.get("segments_scored_by_gates"), r.get("segments_path_excluded_from_gates")
        if n_p:
            print("  {:<20} (게이트 {}구간, path {}구간 제외)".format("", n_g, n_p))
PY
echo ""
echo "결과: $OUT_ROOT"
echo "Mac으로: bash htwk-gym/tools/fetch_results.sh reeval"
