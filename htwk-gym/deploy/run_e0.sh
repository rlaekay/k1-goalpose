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
#
# Anything after those two is passed through to deploy_goal_pose.py, so the
# diagnostic runs do not need the long command either:
#
#   ./run_e0.sh fixed 0,0,0 --hold-prepare

set -o pipefail
cd "$(dirname "$0")"

stty sane 2>/dev/null || true

source /opt/ros/humble/setup.bash
source "$HOME/Workspace/INHA-Soccer/INHA-Player/install/setup.bash"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"

SOURCE="${1:-fixed}"
GOAL="${2:-0,0,0}"
[ $# -gt 0 ] && shift
[ $# -gt 0 ] && shift

ARGS=(--config Goal_Pose_E0.yaml --net 127.0.0.1 --goal-source "$SOURCE")
[ "$SOURCE" = "fixed" ] && ARGS+=(--goal "$GOAL")
ARGS+=("$@")

# InitChannel의 GIL 데드락은 경합이라 대개 통과하고 가끔 걸린다. 걸린 실행은
# deploy_goal_pose.py의 faulthandler 워치독이 8초 만에 잘라내고 DEADLOCK_LOG에
# 스택을 남긴다(종료코드는 1로 고정이라 그것으로는 구분이 안 된다).
# 그 파일이 남아 있을 때만 재시도한다 -- 다른 실패는 그대로 올린다.
DEADLOCK_LOG=/tmp/e0_init_deadlock.log
echo "starting: python3 deploy_goal_pose.py ${ARGS[*]}"
for attempt in 1 2 3 4 5; do
    rm -f "$DEADLOCK_LOG"
    python3 deploy_goal_pose.py "${ARGS[@]}"
    rc=$?
    if [ $rc -eq 0 ] || [ ! -s "$DEADLOCK_LOG" ]; then
        break
    fi
    echo "[run_e0] 초기화 GIL 데드락 (시도 $attempt/5). 스택: $DEADLOCK_LOG -- 재시도한다."
    echo "[run_e0]   창은 InitChannel 하나가 아니다 -- LowState 콜백이 500 Hz 로 도는"
    echo "[run_e0]   동안 rclpy 노드(ModeMonitor/FallMonitor)를 세우는 구간도 포함이다."
    stty sane 2>/dev/null || true
    sleep 2
done

# The wrapper's own cleanup does this too, but not if it was killed outright.
stty sane 2>/dev/null || true
exit $rc
