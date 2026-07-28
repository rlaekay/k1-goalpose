#!/bin/bash
# Live status monitor for tools/reeval_v7.sh.
#
# It is deliberately read-only: it never sends a signal to the evaluation or
# touches checkpoints/results.  It works for an evaluation already started
# before this script existed by using the gpu log and result file mtimes.
#
# Examples (on the training server, from htwk-gym):
#   bash tools/watch_reeval.sh
#   bash tools/watch_reeval.sh --session e_reval_gpu1_shared --interval 10
#   bash tools/watch_reeval.sh --out /path/to/shared_eval_videos/reeval_e_batch_gpu1_... \
#     --arms "E1_path E2_robust V7_full"
#
# ETA is an operational estimate, not a result metric.  Override the defaults
# after observing a batch if this server is consistently faster/slower:
#   OWN_MIN=12 COMMON_MIN=3 STRESS_MIN=3 bash tools/watch_reeval.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SESSION="${SESSION:-e_reval_gpu1}"
OUT_ROOT="${OUT_ROOT:-}"
ARMS="${ARMS:-E1_path E2_robust V7_full}"
GPU="${GPU:-1}"
INTERVAL="${INTERVAL:-10}"
OWN_MIN="${OWN_MIN:-12}"
COMMON_MIN="${COMMON_MIN:-3}"
STRESS_MIN="${STRESS_MIN:-3}"
ONCE=0

usage() {
  sed -n '1,19p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --session) SESSION="$2"; shift 2 ;;
    --out) OUT_ROOT="$2"; shift 2 ;;
    --arms) ARMS="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --once) ONCE=1; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

mtime() {
  # Linux server uses GNU stat; the second form makes local macOS inspection
  # work too.
  stat -c %Y "$1" 2>/dev/null || stat -f %m "$1" 2>/dev/null || echo 0
}

hms() {
  local sec=$1
  [ "$sec" -lt 0 ] && sec=0
  printf '%dh %02dm %02ds' "$((sec / 3600))" "$(((sec % 3600) / 60))" "$((sec % 60))"
}

latest_out() {
  # Never use an arbitrary shared_eval_videos directory: only this wrapper's
  # isolated output prefix is eligible for autodiscovery.
  ls -td "$REPO_ROOT"/shared_eval_videos/reeval_e_batch_gpu1_* 2>/dev/null | head -1 || true
}

current_from_log() {
  local log=$1
  [ -f "$log" ] || return 0
  awk '
    /================ .* \(GPU [0-9]+\) =+$/ {
      x=$0; sub(/^.*================ /, "", x); split(x, a, " "); arm=a[1]
    }
    /--- \[[123]\/3\]/ { stage=$0 }
    END { if (arm != "") print arm "\t" stage }
  ' "$log"
}

stage_started_from_log() {
  # Newer reeval logs prefix stage markers with [YYYY-MM-DD HH:MM:SS].
  # Older jobs simply return nothing and use result-file timestamps instead.
  local log=$1
  [ -f "$log" ] || return 0
  awk '
    /--- \[[123]\/3\]/ { line=$0 }
    END {
      if (line ~ /^\[[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9] [0-9][0-9]:[0-9][0-9]:[0-9][0-9]\]/)
        print substr(line, 2, 19)
    }
  ' "$log"
}

epoch() {
  [ -n "$1" ] || { echo 0; return; }
  date -d "$1" +%s 2>/dev/null || date -j -f '%Y-%m-%d %H:%M:%S' "$1" +%s 2>/dev/null || echo 0
}

render() {
  local now log line current_arm current_stage queue_started stage_start=0 log_stage_epoch
  local own_s=$((OWN_MIN * 60)) common_s=$((COMMON_MIN * 60)) stress_s=$((STRESS_MIN * 60))
  local total_per_arm=$((own_s + common_s + stress_s)) remaining=0 started=0
  local idx=0 arm own common stress state elapsed est stage_left min_remaining
  local previous_arm_end

  now="$(date +%s)"
  if [ -z "$OUT_ROOT" ]; then
    OUT_ROOT="$(latest_out)"
  fi
  if [ -z "$OUT_ROOT" ] || [ ! -d "$OUT_ROOT" ]; then
    echo "reeval output directory not found. Pass --out <printed output path>."
    return
  fi
  log="$OUT_ROOT/gpu${GPU}.log"
  line="$(current_from_log "$log")"
  current_arm="${line%%$'\t'*}"
  current_stage="${line#*$'\t'}"
  [ "$current_stage" = "$line" ] && current_stage=""
  log_stage_epoch="$(epoch "$(stage_started_from_log "$log")")"
  # STARTED_AT is written by the wrapper for new jobs.  Older jobs do not have
  # it, so their output-directory mtime is the best available fallback.
  queue_started="$(mtime "$OUT_ROOT/STARTED_AT")"
  [ "$queue_started" = 0 ] && queue_started="$(mtime "$OUT_ROOT")"
  previous_arm_end=$queue_started

  printf '\033[2J\033[H'
  echo "E-batch re-eval monitor  $(date '+%F %T')  (refresh ${INTERVAL}s)"
  echo "output: $OUT_ROOT"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "tmux:   $SESSION  [running]"
  else
    echo "tmux:   $SESSION  [not found — job may have finished or exited]"
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -i "$GPU" --query-gpu=utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits 2>/dev/null | awk -F, '{printf "GPU %s: %s%% util, %s / %s MiB\n", "'"$GPU"'", $1, $2, $3}' || true
  fi
  echo ""
  printf '%-22s %-10s %-9s %s\n' "arm" "state" "runtime" "detail"
  printf '%s\n' '--------------------------------------------------------------------'

  for arm in $ARMS; do
    own="$OUT_ROOT/$arm/own_task/report.json"
    common="$OUT_ROOT/$arm/common_task/report.json"
    stress="$OUT_ROOT/$arm/stress_jitter/report.json"
    state="QUEUED"
    elapsed=0
    est=0
    stage_left=$total_per_arm
    min_remaining=0

    if [ -f "$stress" ] || { [ -f "$log" ] && grep -q "================ $arm 완료" "$log"; }; then
      state="DONE"
      stage_left=0
      # Use the completed arm's final report as the next arm's start estimate.
      # If stress intentionally failed, common_task is the closest fallback.
      previous_arm_end="$(mtime "$stress")"
      [ "$previous_arm_end" = 0 ] && previous_arm_end="$(mtime "$common")"
    elif [ -f "$common" ]; then
      state="STRESS 3/3"
      stage_start="$(mtime "$common")"
      [ "$arm" = "$current_arm" ] && [ "$log_stage_epoch" != 0 ] && stage_start=$log_stage_epoch
      elapsed=$((now - stage_start))
      est=$stress_s
      stage_left=$((stress_s - elapsed))
      min_remaining=0
    elif [ -f "$own" ]; then
      state="COMMON 2/3"
      stage_start="$(mtime "$own")"
      [ "$arm" = "$current_arm" ] && [ "$log_stage_epoch" != 0 ] && stage_start=$log_stage_epoch
      elapsed=$((now - stage_start))
      est=$common_s
      stage_left=$((common_s - elapsed + stress_s))
      min_remaining=$stress_s
    elif [ "$arm" = "$current_arm" ]; then
      state="OWN 1/3"
      # For an already-running legacy job there is no stage-start timestamp.
      # A prior arm's final report is a much closer estimate than the queue
      # start; for the first arm this falls back to the queue start marker.
      stage_start=$previous_arm_end
      [ "$log_stage_epoch" != 0 ] && stage_start=$log_stage_epoch
      elapsed=$((now - stage_start))
      est=$own_s
      stage_left=$((own_s - elapsed + common_s + stress_s))
      min_remaining=$((common_s + stress_s))
    fi

    [ "$stage_left" -lt "$min_remaining" ] && stage_left=$min_remaining
    remaining=$((remaining + stage_left))
    if [ "$state" != "QUEUED" ] && [ "$state" != "DONE" ]; then started=1; fi
    case "$state" in
      DONE) printf '%-22s %-10s %-9s %s\n' "$arm" "$state" "—" "reports written" ;;
      QUEUED) printf '%-22s %-10s %-9s %s\n' "$arm" "$state" "—" "~$(hms "$total_per_arm") after earlier arms" ;;
      *)
        printf '%-22s %-10s %-9s %s\n' "$arm" "$state" "$(hms "$elapsed")" "stage budget ~$(hms "$est")"
        ;;
    esac
  done

  echo ""
  if [ "$started" = 1 ]; then
    echo "Estimated queue completion: $(hms "$remaining")  →  $(date -d "+$remaining seconds" '+%F %H:%M' 2>/dev/null || true)"
  else
    echo "Estimated queue completion: ~$(hms "$remaining") after the first arm starts"
  fi
  echo "ETA model: own ${OWN_MIN}m + common ${COMMON_MIN}m + stress ${STRESS_MIN}m per arm; GPU contention can extend it."
  echo ""
  echo "Active evaluator process:"
  ps -eo pid=,etime=,pcpu=,pmem=,args= 2>/dev/null | \
    awk -v root="$OUT_ROOT" '/(eval_goal_pose\.py|select_best_checkpoint\.py)/ && (index($0, root) || /select_best_checkpoint\.py/) {print}' | \
    tail -3 || true
  echo "Latest log:"
  tail -4 "$log" 2>/dev/null || true
}

while true; do
  render
  [ "$ONCE" = 1 ] && break
  sleep "$INTERVAL"
done
