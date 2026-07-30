#!/usr/bin/env bash

# Send locomotion-test mission commands from macOS without installing ROS 2.
# ROS 2 CLI runs on the robot through SSH; no password is stored in this file.

set -euo pipefail

ROBOT_SSH="${ROBOT_SSH:-booster@192.168.10.102}"
SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-5}"
SSH_ARGS=(-o "ConnectTimeout=${SSH_CONNECT_TIMEOUT}")

usage() {
  cat <<'EOF'
Usage:
  ./missionctl.sh 0|1|2|3|4|5
  ./missionctl.sh stop
  ./missionctl.sh check
  ./missionctl.sh once status|telemetry|goal|goal-rel|policy
  ./missionctl.sh watch status|telemetry|goal|goal-rel|policy

Environment:
  ROBOT_SSH=booster@192.168.10.102   SSH destination
  SSH_CONNECT_TIMEOUT=5              Connection timeout in seconds

Mission 0/stop only stops the BT goal stream. For an emergency or policy fault,
use the robot remote/E-stop; the deploy process separately requests DAMPING on
Ctrl-C, stale LowState, excessive roll/pitch, or non-finite policy output.
EOF
}

remote_ros() {
  ssh -t "${SSH_ARGS[@]}" "${ROBOT_SSH}" \
    "source /opt/ros/humble/setup.bash && $1"
}

publish_mission() {
  local mission_id="$1"
  if [[ ! "${mission_id}" =~ ^[0-5]$ ]]; then
    echo "mission id must be one digit from 0 through 5" >&2
    exit 2
  fi
  remote_ros "ros2 topic pub --once /locomotion_test/mission_id std_msgs/msg/Int32 '{data: ${mission_id}}'"
}

topic_for_name() {
  case "$1" in
    status) echo "/locomotion_test/status" ;;
    telemetry) echo "/locomotion_test/telemetry" ;;
    goal) echo "/locomotion_test/goal_pose" ;;
    goal-rel) echo "/locomotion_test/goal_rel" ;;
    policy) echo "/locomotion_test/policy_debug" ;;
    *)
      echo "unknown stream '$1'" >&2
      exit 2
      ;;
  esac
}

command_name="${1:-}"
case "${command_name}" in
  0|1|2|3|4|5)
    publish_mission "${command_name}"
    ;;
  stop)
    publish_mission 0
    ;;
  check)
    remote_ros "ros2 topic list | grep '^/locomotion_test/' | sort"
    ;;
  once|watch)
    stream_name="${2:-}"
    [[ -n "${stream_name}" ]] || { usage >&2; exit 2; }
    topic="$(topic_for_name "${stream_name}")"
    if [[ "${command_name}" == "once" ]]; then
      remote_ros "ros2 topic echo --once '${topic}'"
    else
      remote_ros "ros2 topic echo '${topic}'"
    fi
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
