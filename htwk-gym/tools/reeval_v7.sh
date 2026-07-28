#!/bin/bash
# Re-evaluate the four v7 arms with the FIXED harness. No training.
#
#   bash tools/reeval_v7.sh
#   ARMS="E1_path V7_full" bash tools/reeval_v7.sh
#   GPUS="0" bash tools/reeval_v7.sh          # force everything onto one GPU
#
# Runs arms in parallel, one queue per GPU in $GPUS (default "0 1"): arms are
# dealt round-robin into as many queues as there are GPUs, and each queue's
# arms run one after another while the queues themselves run concurrently.
# 256-env eval is light (~2.6 GB, observed) so two queues fit easily even
# alongside another user's job; this was serial on a single GPU before, which
# left the second GPU idle for the whole ~30 min run for no reason.
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

# Undefined names cost a 100-second GPU run to discover on 2026-07-27
# (CATEGORY_PATH in summarize()): ast.parse only checks syntax, so a NameError
# waits until that branch executes. This is instant -- run it before anything.
if ! python tools/check_names.py >/tmp/check_names.$$ 2>&1; then
  grep -v "import \*" /tmp/check_names.$$ | grep UNDEFINED && {
    echo "!!! 정의되지 않은 이름이 있습니다 — 실행을 중단합니다." >&2
    rm -f /tmp/check_names.$$
    exit 1
  }
fi
rm -f /tmp/check_names.$$

RUN_ROOT="${RUN_ROOT:-logs/K1/K1/Goal_Pose_V7}"
ARMS="${ARMS:-E0_armB_armsdown E1_path E2_robust V7_full}"
GPUS="${GPUS:-0 1}"
VIDEO_S="${VIDEO_S:-60}"
COMMON="${COMMON:-envs/K1/Goal_Pose_V7.yaml}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/shared_eval_videos/reeval_$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$OUT_ROOT"

run_arm() {  # arm gpu
  local arm=$1 gpu=$2 dev="cuda:$2"
  local RUN_DIR
  RUN_DIR=$(ls -d "$RUN_ROOT"/*"$arm" 2>/dev/null | head -1)
  if [ -z "$RUN_DIR" ] || [ ! -d "$RUN_DIR" ]; then
    echo "!!! run 디렉토리 없음: $RUN_ROOT/*$arm — 건너뜀" >&2
    return 1
  fi
  echo "================ $arm (GPU $gpu) ================"
  echo "run: $RUN_DIR"

  echo "--- [1/3] 자기 과제로 최적 체크포인트 재선택 + 평가 + 영상 ---"
  python tools/select_best_checkpoint.py \
    --run_dir "$RUN_DIR" --task K1/Goal_Pose_V7 \
    --sim_device "$dev" --rl_device "$dev" \
    --record_video --record_video_s "$VIDEO_S" --link_best
  local SEL_DIR BEST
  SEL_DIR=$(ls -td "$RUN_DIR"/eval/select_*/ 2>/dev/null | head -1)
  BEST=$(cat "${SEL_DIR}BEST_CHECKPOINT" 2>/dev/null || ls -t "$RUN_DIR"/nn/model_*.pth | head -1)

  local D="$OUT_ROOT/${arm}"
  mkdir -p "$D/own_task"
  for f in report.md report.json segments.csv selection.md BEST_CHECKPOINT; do
    [ -f "${SEL_DIR}$f" ] && cp "${SEL_DIR}$f" "$D/own_task/"
  done
  [ -f "${SEL_DIR}winner_video/rollout_env0.mp4" ] && cp "${SEL_DIR}winner_video/rollout_env0.mp4" "$D/own_task/"

  echo "--- [2/3] 공통 v7 과제로 평가 (arm 간 비교용) ---"
  python eval_goal_pose.py --task K1/Goal_Pose_V7 --config "$COMMON" \
    --checkpoint "$BEST" --sim_device "$dev" --rl_device "$dev" \
    --out "$D/common_task"

  echo "--- [3/3] stress jitter ---"
  python eval_goal_pose.py --task K1/Goal_Pose_V7 --config "$COMMON" \
    --checkpoint "$BEST" --sim_device "$dev" --rl_device "$dev" \
    --stress jitter --duration_s 60 --out "$D/stress_jitter" || \
    echo "!!! $arm stress 실패 (나머지 결과는 유효)" >&2
  echo "================ $arm 완료 ================"
}

# Deal arms round-robin across the GPU list; each GPU's arms run one after
# another inside that GPU's own background subshell, and the subshells
# themselves run concurrently. set -e inside a `(...) &` only kills that
# subshell, not this script, so one GPU's failure doesn't take down the other.
# Plain indexed arrays only (no associative arrays / declare -A): those need
# bash 4+, and this needs to work the same whether tested on macOS's bash 3.2
# or run on the Linux server.
read -ra GPU_LIST <<< "$GPUS"
read -ra ARM_LIST <<< "$ARMS"
NGPU=${#GPU_LIST[@]}
QUEUES=()
for ((k = 0; k < NGPU; k++)); do QUEUES[k]=""; done
i=0
for arm in "${ARM_LIST[@]}"; do
  idx=$((i % NGPU))
  QUEUES[idx]="${QUEUES[idx]} $arm"
  i=$((i + 1))
done

echo "GPU 배정:"
for ((k = 0; k < NGPU; k++)); do
  [ -n "${QUEUES[k]}" ] && echo "  GPU ${GPU_LIST[k]}:${QUEUES[k]}"
done
echo ""

pids=()
for ((k = 0; k < NGPU; k++)); do
  [ -n "${QUEUES[k]}" ] || continue
  gpu="${GPU_LIST[k]}"
  LOG="$OUT_ROOT/gpu${gpu}.log"
  (
    for arm in ${QUEUES[k]}; do
      run_arm "$arm" "$gpu"
    done
  ) > "$LOG" 2>&1 &
  pids[k]=$!
  echo "GPU $gpu 큐 시작 (백그라운드, pid $!) — 로그: $LOG"
done

echo ""
echo "대기 중... (진행 확인: tail -f $OUT_ROOT/gpu*.log)"
fail=0
for ((k = 0; k < NGPU; k++)); do
  [ -n "${pids[k]:-}" ] || continue
  if ! wait "${pids[k]}"; then
    echo "!!! GPU ${GPU_LIST[k]} 큐에서 오류 발생 — $OUT_ROOT/gpu${GPU_LIST[k]}.log 확인" >&2
    fail=1
  fi
done
for ((k = 0; k < NGPU; k++)); do
  [ -n "${QUEUES[k]}" ] || continue
  echo ""
  echo "----- GPU ${GPU_LIST[k]} 로그 (마지막 20줄) -----"
  tail -20 "$OUT_ROOT/gpu${GPU_LIST[k]}.log"
done
[ "$fail" = "1" ] && echo "" && echo "!!! 일부 arm이 실패했습니다. 위 로그를 확인하십시오." >&2

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
