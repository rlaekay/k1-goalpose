#!/bin/bash
# Live waterfall view of this user's GPU processes.
#
# Run on the training server from htwk-gym:
#   bash tools/watch_gpu_waterfall.sh
#   bash tools/watch_gpu_waterfall.sh --interval 10
#   bash tools/watch_gpu_waterfall.sh --user user --width 50
#   bash tools/watch_gpu_waterfall.sh --once
#
# This is intentionally read-only. It combines nvidia-smi's GPU/PID view with
# ps' owner/runtime/command view, then draws a simple waterfall from the oldest
# currently running GPU process to "now".

set -euo pipefail

INTERVAL="${INTERVAL:-5}"
WIDTH="${WIDTH:-42}"
USER_FILTER="${USER_FILTER:-${USER:-$(id -un 2>/dev/null || whoami)}}"
ALL_USERS=0
ONCE=0

usage() {
  sed -n '1,13p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --interval) INTERVAL="$2"; shift 2 ;;
    --width) WIDTH="$2"; shift 2 ;;
    --user) USER_FILTER="$2"; shift 2 ;;
    --all-users) ALL_USERS=1; shift ;;
    --once) ONCE=1; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

tmp_dir="${TMPDIR:-/tmp}/gpu_waterfall.$$"
mkdir -p "$tmp_dir"
trap 'rm -rf "$tmp_dir"' EXIT

hms() {
  local sec=$1
  [ -z "$sec" ] && sec=0
  [ "$sec" -lt 0 ] && sec=0
  printf '%dh %02dm %02ds' "$((sec / 3600))" "$(((sec % 3600) / 60))" "$((sec % 60))"
}

shorten() {
  local s="$1" n="$2"
  s="${s//$'\t'/ }"
  if [ "${#s}" -gt "$n" ]; then
    printf '%s...' "${s:0:$((n - 3))}"
  else
    printf '%s' "$s"
  fi
}

classify() {
  local args="$1"
  case "$args" in
    *select_best_checkpoint.py*) echo "eval-select" ;;
    *eval_goal_pose.py*) echo "eval" ;;
    *train_and_eval.sh*) echo "train+eval" ;;
    *train_v7.py*|*train.py*) echo "train" ;;
    *) echo "gpu-job" ;;
  esac
}

bar() {
  local elapsed=$1 max_elapsed=$2 width=$3
  local start len i
  [ "$max_elapsed" -le 0 ] && max_elapsed=1
  [ "$elapsed" -lt 0 ] && elapsed=0
  start=$(((max_elapsed - elapsed) * width / max_elapsed))
  len=$((elapsed * width / max_elapsed))
  [ "$len" -lt 1 ] && len=1
  [ "$start" -lt 0 ] && start=0
  [ "$start" -ge "$width" ] && start=$((width - 1))
  [ $((start + len)) -gt "$width" ] && len=$((width - start))

  printf '['
  for ((i = 0; i < width; i++)); do
    if [ "$i" -ge "$start" ] && [ "$i" -lt $((start + len)) ]; then
      printf '#'
    else
      printf ' '
    fi
  done
  printf ']'
}

collect() {
  local gpu_file="$tmp_dir/gpus.tsv"
  local proc_file="$tmp_dir/procs.tsv"
  local apps_file="$tmp_dir/apps.csv"
  local uuid pid proc_name gpu_mem owner etimes pcpu pmem args tag

  : > "$gpu_file"
  : > "$proc_file"
  : > "$apps_file"

  command -v nvidia-smi >/dev/null 2>&1 || return 0

  nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits 2>/dev/null |
  awk -F, '
    {
      for (i = 1; i <= NF; i++) {
        gsub(/^[[:space:]]+|[[:space:]]+$/, "", $i)
      }
      print $1 "\t" $2 "\t" $3 "\t" $4 "\t" $5 "\t" $6
    }
  ' > "$gpu_file" || true

  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits 2>/dev/null > "$apps_file" || true

  while IFS=, read -r uuid pid proc_name gpu_mem; do
    uuid="$(echo "${uuid:-}" | awk '{$1=$1; print}')"
    pid="$(echo "${pid:-}" | awk '{$1=$1; print}')"
    proc_name="$(echo "${proc_name:-}" | awk '{$1=$1; print}')"
    gpu_mem="$(echo "${gpu_mem:-}" | awk '{$1=$1; print}')"
    [ -n "$pid" ] || continue
    [ "$pid" != "[N/A]" ] || continue

    owner="$(ps -o user= -p "$pid" 2>/dev/null | awk '{$1=$1; print}')"
    [ -n "$owner" ] || continue
    if [ "$ALL_USERS" != 1 ] && [ "$owner" != "$USER_FILTER" ]; then
      continue
    fi

    etimes="$(ps -o etimes= -p "$pid" 2>/dev/null | awk '{$1=$1; print}')"
    pcpu="$(ps -o pcpu= -p "$pid" 2>/dev/null | awk '{$1=$1; print}')"
    pmem="$(ps -o pmem= -p "$pid" 2>/dev/null | awk '{$1=$1; print}')"
    args="$(ps -o args= -p "$pid" 2>/dev/null | tr '\t' ' ' | awk '{$1=$1; print}')"
    [ -n "$etimes" ] || etimes=0
    [ -n "$pcpu" ] || pcpu=0
    [ -n "$pmem" ] || pmem=0
    [ -n "$args" ] || args="$proc_name"
    tag="$(classify "$args")"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$uuid" "$pid" "$owner" "$etimes" "$gpu_mem" "$pcpu" "$pmem" "$tag" "$args" >> "$proc_file"
  done < "$apps_file"
}

render() {
  local gpu_file="$tmp_dir/gpus.tsv"
  local proc_file="$tmp_dir/procs.tsv"
  local max_elapsed total_procs scope gpu_idx uuid name util used total
  local pid owner etimes gpu_mem pcpu pmem tag args line

  collect

  max_elapsed="$(awk -F'\t' 'BEGIN {m = 0} $4 > m {m = $4} END {print m + 0}' "$proc_file")"
  total_procs="$(awk 'END {print NR + 0}' "$proc_file")"
  [ "$max_elapsed" -le 0 ] && max_elapsed=1

  if [ "$ALL_USERS" = 1 ]; then
    scope="all users"
  else
    scope="user=$USER_FILTER"
  fi

  printf '\033[2J\033[H'
  echo "GPU process waterfall  $(date '+%F %T')  (refresh ${INTERVAL}s)"
  echo "scope: $scope, processes: $total_procs"
  echo "waterfall: oldest current GPU process ($(hms "$max_elapsed")) -> now"
  echo ""

  if [ ! -s "$gpu_file" ]; then
    echo "nvidia-smi GPU query returned nothing."
    return
  fi

  while IFS=$'\t' read -r gpu_idx uuid name util used total; do
    printf 'GPU %-2s  %-24s  %3s%% util  %6s / %-6s MiB\n' \
      "$gpu_idx" "$(shorten "$name" 24)" "$util" "$used" "$total"

    if ! awk -F'\t' -v u="$uuid" '$1 == u {found = 1} END {exit !found}' "$proc_file"; then
      echo "  no matching GPU process for this scope"
      echo ""
      continue
    fi

    printf '  %-8s %-10s %-8s %-6s %-5s %-11s %-12s %s\n' \
      "PID" "runtime" "gpu_mem" "CPU%" "MEM%" "kind" "waterfall" "command"
    printf '  %s\n' '----------------------------------------------------------------------------------------------------'

    while IFS=$'\t' read -r _ pid owner etimes gpu_mem pcpu pmem tag args; do
      printf '  %-8s %-10s %-8s %-6s %-5s %-11s ' \
        "$pid" "$(hms "$etimes")" "${gpu_mem}MiB" "$pcpu" "$pmem" "$tag"
      bar "$etimes" "$max_elapsed" "$WIDTH"
      printf ' %s\n' "$(shorten "$args" 96)"
    done < <(awk -F'\t' -v u="$uuid" '$1 == u {print}' "$proc_file" | sort -t $'\t' -k4,4nr)
    echo ""
  done < "$gpu_file"

  echo "Notes:"
  echo "  - This shows live GPU ownership/runtime/memory for current processes only."
  echo "  - Generic GPU PIDs do not expose true ETA. For training ETA, use checkpoint progress:"
  echo "    python tools/progress.py --task Goal_Pose_V7 --watch 60"
  echo "    python tools/progress.py --task Goal_Pose_V8 --watch 60"
}

while true; do
  render
  [ "$ONCE" = 1 ] && break
  sleep "$INTERVAL"
done
