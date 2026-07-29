#!/bin/bash
# One supervised HBatch arm.  Bootstrap, train and eval are all inside the
# captured process so every failure mode has an arm-specific status and log.
set -euo pipefail

if [ $# -ne 12 ]; then
  echo "usage: $0 ARM CONFIG CKPT SIM_DEV RL_DEV ITERS ENVS VIDEO_S STATUS_DIR FAILURE_DIR HEALTH_MARKER HEALTH_TOKEN" >&2
  exit 2
fi

ARM=$1
CONFIG=$2
CKPT=$3
SIM_DEV=$4
RL_DEV=$5
ITERS=$6
ENVS=$7
VIDEO_S=$8
STATUS_DIR=$9
FAILURE_DIR=${10}
HEALTH_MARKER=${11}
HEALTH_TOKEN=${12}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$STATUS_DIR" "$FAILURE_DIR"
STATUS="$STATUS_DIR/${ARM}-codex.status"
FAILURE_LOG="$FAILURE_DIR/${ARM}-codex.log"
PENDING_LOG="$FAILURE_DIR/.${ARM}-codex.pending.log"

write_status() {
  local value=$1
  local temporary="${STATUS}.tmp-$$"
  printf '%s\n' "$value" >"$temporary"
  mv "$temporary" "$STATUS"
}

write_failed_status_unless_present() {
  local value=$1
  if [ ! -f "$STATUS" ] || ! grep -q '^FAILED' "$STATUS"; then
    write_status "$value"
  fi
}

preserve_failure_log() {
  if [ -f "$PENDING_LOG" ]; then
    mv -f "$PENDING_LOG" "$FAILURE_LOG"
  elif [ ! -f "$FAILURE_LOG" ]; then
    printf 'HBatch %s failed before a captured child log was created.\n' "$ARM" >"$FAILURE_LOG"
  fi
}

on_signal() {
  local signal=$1
  trap - HUP INT TERM
  preserve_failure_log
  write_failed_status_unless_present "FAILED signal=$signal"
  exit 130
}
trap 'on_signal HUP' HUP
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM

rm -f "$FAILURE_LOG" "$PENDING_LOG" "$HEALTH_MARKER"
write_status "RUNNING pid=$$"

run_job() (
  set -euo pipefail
  if [ -n "${HBATCH_CONDA_BASE:-}" ]; then
    # Activation is deliberately inside the captured subprocess: a missing
    # conda hook/env is a real launch failure, not a silent 300 s timeout.
    source "$HBATCH_CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$HBATCH_CONDA_ENV"
  fi
  if [ -n "${HBATCH_CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH="$HBATCH_CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
  fi
  cd "$REPO_ROOT"
  HBATCH_HEALTH_MARKER="$HEALTH_MARKER" \
  HBATCH_HEALTH_TOKEN="$HEALTH_TOKEN" \
  HBATCH_HEALTH_ITERATIONS="${HBATCH_HEALTH_ITERATIONS:-2}" \
  ITERS="$ITERS" ENVS="$ENVS" VIDEO_S="$VIDEO_S" \
  bash tools/train_and_eval_hbatch.sh \
    "$ARM" "$CONFIG" "$CKPT" "$SIM_DEV" "$RL_DEV"

  # A zero exit is only valid when this exact launch produced the requested
  # production-shape post-update attestation.
  python tools/verify_hbatch_health.py \
    --marker "$HEALTH_MARKER" --health_token "$HEALTH_TOKEN" \
    --num_envs "$ENVS" \
    --min_iterations "${HBATCH_HEALTH_ITERATIONS:-2}"
)

set +e
run_job 2>&1 | tee "$PENDING_LOG"
PIPELINE_STATUS=("${PIPESTATUS[@]}")
JOB_RC=${PIPELINE_STATUS[0]}
TEE_RC=${PIPELINE_STATUS[1]}
set -e

if [ "$JOB_RC" -eq 0 ] && [ "$TEE_RC" -eq 0 ]; then
  write_status "COMPLETE"
  rm -f "$PENDING_LOG" "$FAILURE_LOG"
  exit 0
fi

RC=$JOB_RC
if [ "$RC" -eq 0 ]; then
  RC=$TEE_RC
fi
preserve_failure_log
write_failed_status_unless_present "FAILED job_rc=$JOB_RC tee_rc=$TEE_RC"
exit "$RC"
