#!/bin/bash
# End-to-end short screen: generate -> smoke -> train -> paired eval -> report.
# Intended invocation on the two-A6000 compute server:
#   nohup bash tools/run_mcells.sh > logs/mcells/launcher-codex.log 2>&1 &

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

BASE_CONFIG="${BASE_CONFIG:-logs/K1/K1/Goal_Pose_V7/2026-07-28-17-02-35_G1_speed/config.yaml}"
CKPT="${CKPT:-logs/K1/K1/Goal_Pose_V7/2026-07-28-17-02-35_G1_speed/nn/model_10700.pth}"
ENVS="${ENVS:-4096}"
ITERS="${ITERS:-200}"
SMOKE_ENVS="${SMOKE_ENVS:-256}"
SMOKE_STEPS="${SMOKE_STEPS:-300}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-240}"
CELLS="${CELLS:-M0_control-codex M1_force-codex M2_jointdr-codex M3_mirror_off-codex}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
STATE="logs/mcells/state-$RUN_ID-codex"
FAIL="logs/mcells/failures-$RUN_ID-codex"
mkdir -p "$STATE" "$FAIL"

say() { echo "[$(date +%H:%M:%S)] $*"; }
gpu_for() {
  case "$1" in
    M0_control-codex|M2_jointdr-codex) echo 0 ;;
    *) echo 1 ;;
  esac
}

say "M-cell causal screen $RUN_ID"
say "base: $BASE_CONFIG"
say "warm start: $CKPT"

if [ ! -f "$BASE_CONFIG" ] || [ ! -f "$CKPT" ]; then
  say "FAIL: frozen G1 config or checkpoint is missing"
  exit 1
fi
if [ "$ITERS" -ne 200 ]; then
  say "FAIL: protocol is frozen at 200 iterations (got ITERS=$ITERS)"
  exit 1
fi

say "[1/5] generate and verify causal configs"
python tools/make_mcell_configs.py \
  --base_config "$BASE_CONFIG" --checkpoint "$CKPT" || exit 1
python tools/make_mcell_configs.py \
  --base_config "$BASE_CONFIG" --checkpoint "$CKPT" --check || exit 1
python -m py_compile \
  envs/K1/goal_pose_hbatch.py eval_goal_pose.py utils/runner_v3.py \
  tools/make_mcell_configs.py tools/compare_mcells.py || exit 1
if ! python tools/check_names.py > "$STATE/names-codex.log" 2>&1; then
  if grep -v "import \*" "$STATE/names-codex.log" | grep -qE "UNDEFINED|IMPORT|FORMAT"; then
    say "FAIL: name/import check (see $STATE/names-codex.log)"
    exit 1
  fi
fi

say "[2/5] independent inference/mechanics smoke (all cells in one GPU wave)"
declare -A SMOKE_PID
for cell in $CELLS; do
  gpu="$(gpu_for "$cell")"
  probe=""
  [ "$cell" = "M1_force-codex" ] && probe="--disturbance_probe"
  python -u tools/smoke_v7.py \
    --task K1/Goal_Pose_HBatch \
    --config "sweeps/mcells/$cell.yaml" \
    --checkpoint "$CKPT" --num_envs "$SMOKE_ENVS" --steps "$SMOKE_STEPS" \
    --sim_device "cuda:$gpu" --rl_device "cuda:$gpu" $probe \
    > "$STATE/smoke-$cell.log" 2>&1 &
  SMOKE_PID[$cell]=$!
  say "  smoke $cell -> cuda:$gpu pid ${SMOKE_PID[$cell]}"
done

PASSED=""
for cell in $CELLS; do
  if wait "${SMOKE_PID[$cell]}"; then
    PASSED="$PASSED $cell"
    say "  PASS $cell"
  else
    cp "$STATE/smoke-$cell.log" "$FAIL/smoke-$cell-codex.log"
    say "  FAIL $cell -> $FAIL/smoke-$cell-codex.log"
  fi
done
PASSED="${PASSED# }"
if [ -z "$PASSED" ]; then
  say "no cell passed smoke; nothing launched"
  exit 1
fi

train_one() { # cell gpu token health_path
  local cell="$1" gpu="$2" token="$3" health="$4"
  local log="$STATE/train-$cell.log"
  HBATCH_HEALTH_MARKER="$health" \
  HBATCH_HEALTH_TOKEN="$token" \
  HBATCH_HEALTH_ITERATIONS=2 \
  python -u train_hbatch.py \
    --task K1/Goal_Pose_HBatch --config "sweeps/mcells/$cell.yaml" \
    --headless True --checkpoint "$CKPT" --num_envs "$ENVS" \
    --max_iterations "$ITERS" --sim_device "cuda:$gpu" --rl_device "cuda:$gpu" \
    > "$log" 2>&1
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    cp "$log" "$FAIL/train-$cell-codex.log"
    return "$rc"
  fi
  local run_dir
  run_dir="$(sed -n -E 's|.*Saving model to (.*)/nn/model_[0-9]+\.pth.*|\1|p' "$log" | tail -1)"
  if [ -z "$run_dir" ] || [ ! -d "$run_dir/nn" ]; then
    cp "$log" "$FAIL/train-$cell-codex.log"
    return 90
  fi
  cp "$CKPT" "$run_dir/nn/model_0.pth"
  printf '%s\n' "$run_dir" > "$STATE/run-$cell.txt"
  return 0
}

say "[3/5] train 4 cells concurrently (2 processes per A6000)"
declare -A TRAIN_PID HEALTH TOKEN
for cell in $PASSED; do
  gpu="$(gpu_for "$cell")"
  TOKEN[$cell]="$RUN_ID-$cell"
  HEALTH[$cell]="$STATE/health-$cell.json"
  train_one "$cell" "$gpu" "${TOKEN[$cell]}" "${HEALTH[$cell]}" &
  TRAIN_PID[$cell]=$!
  say "  train $cell -> cuda:$gpu pid ${TRAIN_PID[$cell]}"
done

# Do not wait 25-40 minutes to discover a broken backward pass.  RunnerV3
# atomically writes a signed marker after two full PPO iterations.
say "  waiting up to ${HEALTH_TIMEOUT_S}s for two-iteration health markers"
deadline=$((SECONDS + HEALTH_TIMEOUT_S))
while [ "$SECONDS" -lt "$deadline" ]; do
  pending=0
  for cell in $PASSED; do
    [ -f "${HEALTH[$cell]}" ] || pending=$((pending + 1))
  done
  [ "$pending" -eq 0 ] && break
  sleep 2
done

HEALTHY=""
for cell in $PASSED; do
  if [ -f "${HEALTH[$cell]}" ] && python tools/verify_hbatch_health.py \
      --marker "${HEALTH[$cell]}" --health_token "${TOKEN[$cell]}" \
      --num_envs "$ENVS" --min_iterations 2 \
      --expected_configured_lr 2e-6 --min_lr 5e-7 --max_lr 2e-6 \
      > "$STATE/health-$cell.log" 2>&1; then
    HEALTHY="$HEALTHY $cell"
    say "  HEALTHY $cell"
  else
    say "  UNHEALTHY $cell -> $FAIL/health-$cell-codex.log"
    cp "$STATE/health-$cell.log" "$FAIL/health-$cell-codex.log" 2>/dev/null || \
      cp "$STATE/train-$cell.log" "$FAIL/health-$cell-codex.log"
    kill "${TRAIN_PID[$cell]}" 2>/dev/null || true
  fi
done
HEALTHY="${HEALTHY# }"
if [ -z "$HEALTHY" ]; then
  say "no healthy cell; evaluation cancelled"
  exit 1
fi

TRAINED=""
for cell in $HEALTHY; do
  if wait "${TRAIN_PID[$cell]}"; then
    TRAINED="$TRAINED $cell"
    say "  COMPLETE $cell"
  else
    say "  FAIL during training $cell -> $FAIL/train-$cell-codex.log"
  fi
done
TRAINED="${TRAINED# }"
if [ -z "$TRAINED" ]; then
  say "no completed cell; evaluation cancelled"
  exit 1
fi

say "[4/5] paired targeted evaluation (two persistent GPU queues)"
python -u tools/compare_mcells.py \
  --state_dir "$STATE" --cells $TRAINED \
  > "$STATE/eval-and-report-codex.log" 2>&1
EVAL_RC=$?
if [ "$EVAL_RC" -ne 0 ]; then
  cp "$STATE/eval-and-report-codex.log" "$FAIL/eval-codex.log"
  say "evaluation/report failed -> $FAIL/eval-codex.log"
  exit "$EVAL_RC"
fi

say "[5/5] COMPLETE"
say "report: logs/mcells/compare/$(basename "$STATE")/mcell-report-codex.md"
say "state:  $STATE"
say "failures (only failed cells/stages): $FAIL"
