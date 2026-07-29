#!/bin/bash
# Verify frozen configs -> independent mechanics/train/disturbance/video smoke
# -> launch only passing arms -> require a production-shape finite health gate.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SESSION="${SESSION:-hbatch}"
ITERS="${ITERS:-12000}"
ENVS="${ENVS:-4096}"
VIDEO_S="${VIDEO_S:-8}"
HEALTH_ITERATIONS="${HEALTH_ITERATIONS:-2}"
HEALTH_TIMEOUT_S="${HEALTH_TIMEOUT_S:-300}"
HEALTH_GRACE_S="${HEALTH_GRACE_S:-10}"
CKPT="${CKPT:-logs/K1/K1/Goal_Pose_V7/2026-07-28-17-02-35_G1_speed/nn/model_10700.pth}"
FAIL_DIR="$REPO_ROOT/logs/hbatch/smoke_failures"
STATUS_DIR="$REPO_ROOT/logs/hbatch/launch_status"
TRAIN_FAIL_DIR="$REPO_ROOT/logs/hbatch/training_failures"
mkdir -p "$FAIL_DIR" "$STATUS_DIR" "$TRAIN_FAIL_DIR"

write_status() {
  local arm=$1
  local value=$2
  local status="$STATUS_DIR/${arm}-codex.status"
  local temporary="${status}.tmp-$$"
  printf '%s\n' "$value" >"$temporary"
  mv "$temporary" "$status"
}

preserve_launch_failure() {
  local arm=$1
  local reason=$2
  local pending="$TRAIN_FAIL_DIR/.${arm}-codex.pending.log"
  local failure="$TRAIN_FAIL_DIR/${arm}-codex.log"
  if [ -f "$pending" ]; then
    cp "$pending" "$failure"
  elif [ ! -f "$failure" ]; then
    printf 'HBatch %s launch failure: %s\n' "$arm" "$reason" >"$failure"
  fi
  write_status "$arm" "FAILED $reason"
}

pane_alive() {
  local pane=$1
  local pane_dead
  [ -n "$pane" ] || return 1
  pane_dead=$(tmux display-message -p -t "$pane" '#{pane_dead}' \
    2>/dev/null) || return 1
  [ "$pane_dead" = "0" ]
}

terminate_pane() {
  local pane=$1
  if pane_alive "$pane"; then
    tmux kill-pane -t "$pane" >/dev/null 2>&1 || true
  fi
  # kill-pane normally tears down the foreground process group immediately.
  # Poll the exact pane so remain-on-exit and delayed teardown cannot be
  # mistaken for termination.
  local attempts=0
  while pane_alive "$pane" && [ "$attempts" -lt 20 ]; do
    sleep 0.1
    attempts=$((attempts + 1))
  done
  ! pane_alive "$pane"
}

supervisor_ok() {
  local status_value=$1
  local pane=$2
  if [[ "$status_value" == COMPLETE* ]]; then
    return 0
  fi
  [[ "$status_value" == RUNNING* ]] && pane_alive "$pane"
}

health_ok() {
  local i=$1
  python tools/verify_hbatch_health.py \
    --marker "${HEALTH_MARKERS[$i]}" \
    --health_token "${HEALTH_TOKENS[$i]}" \
    --num_envs "$ENVS" --min_iterations "$HEALTH_ITERATIONS" \
    >/dev/null 2>&1
}

if [ ! -f "$CKPT" ]; then
  echo "!!! G1@10700 warm start missing: $CKPT" >&2
  exit 1
fi
# The committed configs are frozen experiment inputs. Verify generator
# agreement without rewriting tracked YAMLs (which previously dirtied the
# server worktree and made the next git pull abort).
python tools/make_hbatch_configs.py --check >/dev/null

PASS=()
for arm in H0 H1 H2 H3; do
  cfg="sweeps/hbatch/${arm}-codex.yaml"
  log=$(mktemp)
  echo "=== smoke $arm ==="
  if python tools/smoke_hbatch.py --config "$cfg" --checkpoint "$CKPT" \
       --sim_device cuda:0 --rl_device cuda:0 --steps 300 >"$log" 2>&1; then
    PASS+=("$arm")
    rm -f "$FAIL_DIR/${arm}-codex.log" "$log"
    echo "PASS $arm"
  else
    mv "$log" "$FAIL_DIR/${arm}-codex.log"
    echo "FAIL $arm -> $FAIL_DIR/${arm}-codex.log" >&2
    echo "--- $arm failure tail ---" >&2
    tail -n 80 "$FAIL_DIR/${arm}-codex.log" >&2
  fi
done

if [ ${#PASS[@]} -eq 0 ]; then
  echo "!!! no HBatch arm passed smoke; nothing launched" >&2
  exit 1
fi
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "!!! tmux session already exists: $SESSION" >&2
  exit 1
fi

CONDA_BASE=""
CONDA_ENV=""
ACTIVE_CONDA_PREFIX=""
if [ -n "${CONDA_PREFIX:-}" ]; then
  CONDA_BASE="$(conda info --base 2>/dev/null || echo "${CONDA_PREFIX%/envs/*}")"
  CONDA_ENV="${CONDA_DEFAULT_ENV:-$(basename "$CONDA_PREFIX")}"
  ACTIVE_CONDA_PREFIX="$CONDA_PREFIX"
fi

HEALTH_TOKENS=()
HEALTH_MARKERS=()
PANE_IDS=()
CANDIDATE_AT=()
RESULTS=()
resolved=0
failed=0

# Keep the tmux server/session alive until every real arm pane has been
# created. A first arm that fails instantly must not make later new-window
# calls fail merely because the session disappeared between launches.
set +e
anchor_output=$(tmux new-session -d -P -F '#{pane_id}' \
  -s "$SESSION" -n _hbatch_launcher_anchor 'sleep 86400' 2>&1)
anchor_rc=$?
set -e
if [ "$anchor_rc" -ne 0 ]; then
  for arm in "${PASS[@]}"; do
    preserve_launch_failure "$arm" "tmux-anchor-bootstrap-rc=$anchor_rc"
  done
  echo "!!! tmux anchor bootstrap failed: $anchor_output" >&2
  exit 1
fi
ANCHOR_PANE="$anchor_output"

for i in "${!PASS[@]}"; do
  arm=${PASS[$i]}
  gpu=$((i % 2))
  cfg="sweeps/hbatch/${arm}-codex.yaml"
  health_token=$(python -c 'import secrets; print(secrets.token_hex(16))')
  health_marker="$STATUS_DIR/${arm}-health-codex.json"
  HEALTH_TOKENS[$i]="$health_token"
  HEALTH_MARKERS[$i]="$health_marker"
  PANE_IDS[$i]=""
  rm -f "$STATUS_DIR/${arm}-codex.status" "$health_marker" \
    "$TRAIN_FAIL_DIR/${arm}-codex.log" \
    "$TRAIN_FAIL_DIR/.${arm}-codex.pending.log"

  launch=(
    env
    "HBATCH_CONDA_BASE=$CONDA_BASE"
    "HBATCH_CONDA_ENV=$CONDA_ENV"
    "HBATCH_CONDA_PREFIX=$ACTIVE_CONDA_PREFIX"
    "HBATCH_HEALTH_ITERATIONS=$HEALTH_ITERATIONS"
    bash "$SCRIPT_DIR/run_hbatch_arm.sh"
    "$arm" "$cfg" "$CKPT" "cuda:$gpu" "cuda:$gpu"
    "$ITERS" "$ENVS" "$VIDEO_S" "$STATUS_DIR" "$TRAIN_FAIL_DIR"
    "$health_marker" "$health_token"
  )
  printf -v cmd '%q ' "${launch[@]}"
  cmd=${cmd% }
  set +e
  pane_output=$(tmux new-window -d -P -F '#{pane_id}' \
    -t "$SESSION" -n "$arm" "$cmd" 2>&1)
  launch_rc=$?
  set -e
  if [ "$launch_rc" -ne 0 ]; then
    preserve_launch_failure "$arm" "tmux-bootstrap rc=$launch_rc: $pane_output"
    RESULTS[$i]="failed"
    resolved=$((resolved + 1))
    failed=$((failed + 1))
    echo "FAIL $arm tmux bootstrap -> $TRAIN_FAIL_DIR/${arm}-codex.log" >&2
    continue
  fi
  PANE_IDS[$i]="$pane_output"
  echo "started $arm on GPU $gpu; awaiting $HEALTH_ITERATIONS finite iterations + ${HEALTH_GRACE_S}s grace"
done
terminate_pane "$ANCHOR_PANE" || {
  echo "!!! tmux anchor pane did not terminate cleanly" >&2
  exit 1
}

deadline=$((SECONDS + HEALTH_TIMEOUT_S))
while [ "$resolved" -lt "${#PASS[@]}" ] && [ "$SECONDS" -lt "$deadline" ]; do
  for i in "${!PASS[@]}"; do
    if [ "${RESULTS[$i]:-}" != "" ]; then
      continue
    fi
    arm=${PASS[$i]}
    status="$STATUS_DIR/${arm}-codex.status"
    status_value=$(head -n 1 "$status" 2>/dev/null || true)
    if [[ "$status_value" == FAILED* ]]; then
      RESULTS[$i]="failed"
      resolved=$((resolved + 1))
      failed=$((failed + 1))
      echo "FAIL $arm production launch -> $TRAIN_FAIL_DIR/${arm}-codex.log" >&2
      continue
    fi
    if [[ "$status_value" == RUNNING* ]] && ! pane_alive "${PANE_IDS[$i]}"; then
      preserve_launch_failure "$arm" "supervisor-pane-disappeared"
      RESULTS[$i]="failed"
      resolved=$((resolved + 1))
      failed=$((failed + 1))
      echo "FAIL $arm supervisor pane disappeared -> $TRAIN_FAIL_DIR/${arm}-codex.log" >&2
      continue
    fi
    if [ -z "$status_value" ] && ! pane_alive "${PANE_IDS[$i]}"; then
      preserve_launch_failure "$arm" "supervisor-exited-before-status"
      RESULTS[$i]="failed"
      resolved=$((resolved + 1))
      failed=$((failed + 1))
      echo "FAIL $arm supervisor exited before status -> $TRAIN_FAIL_DIR/${arm}-codex.log" >&2
      continue
    fi
    if [[ "$status_value" == COMPLETE* ]] && ! health_ok "$i"; then
      preserve_launch_failure "$arm" "completed-without-valid-health-marker"
      RESULTS[$i]="failed"
      resolved=$((resolved + 1))
      failed=$((failed + 1))
      echo "FAIL $arm completed without valid health attestation" >&2
      continue
    fi
    if health_ok "$i"; then
      if [ -z "${CANDIDATE_AT[$i]:-}" ]; then
        CANDIDATE_AT[$i]=$SECONDS
        echo "candidate $arm: marker valid; checking ${HEALTH_GRACE_S}s startup grace"
      elif [ $((SECONDS - CANDIDATE_AT[$i])) -ge "$HEALTH_GRACE_S" ]; then
        status_value=$(head -n 1 "$status" 2>/dev/null || true)
        if ! supervisor_ok "$status_value" "${PANE_IDS[$i]}"; then
          preserve_launch_failure "$arm" "failed-during-health-grace"
          RESULTS[$i]="failed"
          failed=$((failed + 1))
          echo "FAIL $arm during post-marker grace -> $TRAIN_FAIL_DIR/${arm}-codex.log" >&2
        else
          RESULTS[$i]="healthy"
          echo "HEALTHY $arm: production-shape finite gate and startup grace passed"
        fi
        resolved=$((resolved + 1))
      fi
    fi
  done
  if [ "$resolved" -lt "${#PASS[@]}" ]; then
    sleep 2
  fi
done

for i in "${!PASS[@]}"; do
  if [ "${RESULTS[$i]:-}" = "" ]; then
    arm=${PASS[$i]}
    preserve_launch_failure "$arm" "health-timeout-after-${HEALTH_TIMEOUT_S}s"
    if ! terminate_pane "${PANE_IDS[$i]}"; then
      echo "WARN $arm timeout pane did not terminate cleanly" >&2
    fi
    RESULTS[$i]="timeout"
    failed=$((failed + 1))
    echo "FAIL $arm launch health timed out after ${HEALTH_TIMEOUT_S}s -> $TRAIN_FAIL_DIR/${arm}-codex.log" >&2
  fi
done

# Close the marker/status race once more immediately before returning.  A
# policy that attested iteration 2 and died at iteration 3 must not be reported
# as a healthy launch merely because the marker check won the race.
if [ "$failed" -eq 0 ]; then
  sleep 2
  for i in "${!PASS[@]}"; do
    arm=${PASS[$i]}
    status="$STATUS_DIR/${arm}-codex.status"
    marker_valid=0
    if health_ok "$i"; then
      marker_valid=1
    fi
    # Read status only after verifier returns. This closes a RUNNING->FAILED
    # transition that occurs while the JSON marker is being checked.
    status_value=$(head -n 1 "$status" 2>/dev/null || true)
    if [ "$marker_valid" -ne 1 ] || \
       ! supervisor_ok "$status_value" "${PANE_IDS[$i]}"; then
      preserve_launch_failure "$arm" "failed-final-health-recheck"
      if ! terminate_pane "${PANE_IDS[$i]}"; then
        echo "WARN $arm failed pane did not terminate cleanly" >&2
      fi
      RESULTS[$i]="failed"
      failed=$((failed + 1))
      echo "FAIL $arm final health/status recheck -> $TRAIN_FAIL_DIR/${arm}-codex.log" >&2
    fi
  done
fi

if [ "$failed" -ne 0 ]; then
  echo "!!! $failed HBatch arm(s) failed initial production health; surviving arms were not killed" >&2
  exit 1
fi

echo "passing arms: ${PASS[*]}"
echo "smoke failures only: $FAIL_DIR"
echo "training failures only: $TRAIN_FAIL_DIR"
echo "attach: tmux attach -t $SESSION"
