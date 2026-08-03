#!/usr/bin/env bash
# Start the E0 GoalPose deploy. Run this from a robot terminal.
#
# Exists so the long command does not have to be retyped: the keyboard listener
# leaves the tty in raw mode when the process dies badly, and typed characters
# then get eaten -- which is how "python3" became "fpython3" and a path ended up
# as a stray argument. `stty sane` up front clears any leftover of that.
#
#   ./run_e0.sh                 # fixed goal (0,0,0), bring-up
#   ./run_e0.sh ros             # mission mode, goal from Brain
#   ./run_e0.sh fixed 0.2,0,0   # fixed goal, 0.2 m forward

set -o pipefail
cd "$(dirname "$0")"

stty sane 2>/dev/null || true

source /opt/ros/humble/setup.bash
source "$HOME/Workspace/INHA-Soccer/INHA-Player/install/setup.bash"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

SOURCE="${1:-fixed}"
GOAL="${2:-0,0,0}"

ARGS=(--config Goal_Pose_E0.yaml --net 127.0.0.1 --goal-source "$SOURCE")
[ "$SOURCE" = "fixed" ] && ARGS+=(--goal "$GOAL")

echo "starting: python3 deploy_goal_pose.py ${ARGS[*]}"
python3 deploy_goal_pose.py "${ARGS[@]}"
rc=$?

# The wrapper's own cleanup does this too, but not if it was killed outright.
stty sane 2>/dev/null || true
exit $rc
