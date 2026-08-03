#!/usr/bin/env bash
#
# Does the SDK fall detector keep publishing once we take the joints?
#
# rt/fall_down has no Python binding in this SDK build, so it can only be read
# through the ROS bridge. This watches /fall_down_recovery_state (and the other
# state topics) and reports whether each is alive, without changing any mode.
#
# Run on the robot, in the mode you want to characterise:
#   ./probe_fall_topic.sh            # 10 s sample
#   ./probe_fall_topic.sh 20         # 20 s sample
#
# The interesting comparison is the same command run once in walking/soccer mode
# and once in CUSTOM. If /fall_down_recovery_state goes silent in CUSTOM, the
# deploy has to detect falls from IMU itself.

# ROS setup scripts reference unset vars; -u would abort on sourcing them.
set -o pipefail
SECS="${1:-10}"

source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

TOPICS=(
  /fall_down_recovery_state
  /fall_down
  /low_state
  /odometer_state
  /robot_states
)

printf '%-30s %-8s %s\n' TOPIC STATUS DETAIL
printf '%-30s %-8s %s\n' "------------------------------" "--------" "------"

for t in "${TOPICS[@]}"; do
  if ! ros2 topic info "$t" >/dev/null 2>&1; then
    printf '%-30s %-8s %s\n' "$t" "ABSENT" "not advertised"
    continue
  fi
  # `topic hz` prints nothing at all when no message ever arrives, so treat an
  # empty result as DEAD rather than waiting forever.
  line="$(timeout "$((SECS + 3))" ros2 topic hz "$t" 2>/dev/null | grep -m1 'average rate' || true)"
  if [ -n "$line" ]; then
    printf '%-30s %-8s %s\n' "$t" "ALIVE" "$line"
  else
    pubs="$(ros2 topic info "$t" 2>/dev/null | grep -i 'Publisher count' || echo '?')"
    printf '%-30s %-8s %s\n' "$t" "DEAD" "no msg in ${SECS}s (${pubs})"
  fi
done

echo
echo "Current mode (if the agent exposes it):"
timeout 5 ros2 topic echo --once /robot_states 2>/dev/null | head -20 || echo "  (no /robot_states data)"
