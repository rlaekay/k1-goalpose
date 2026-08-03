#!/usr/bin/env bash
#
# learning -> deploy in one command.
#
# Takes a training checkpoint on the server, exports it to actor-only
# TorchScript, verifies the exported module, copies it to the robot's deploy
# staging area, and checks the hashes match on both ends.
#
# A .pt on its own is NOT a policy: the frozen run's normalization scales,
# default_qpos, action_scale, decimation, PD gains and gait_frequency must match
# what the deploy config says, or the robot moves differently for reasons that
# are invisible at runtime. So this script also diffs the checkpoint's frozen
# config against the deploy YAML and refuses to install on a mismatch unless you
# pass --force.
#
# Usage:
#   tools/install_policy.sh --checkpoint <path-on-server> [options]
#
#   --name <policy>      deploy policy name (default: goal_pose_e0)
#                        -> models/<policy>.pt on the robot
#   --config <yaml>      deploy config basename (default: Goal_Pose_E0.yaml)
#   --task <task>        export task name (default: K1/Goal_Pose_V7)
#   --export-only        stop after the server-side export + verify
#   --skip-export        reuse an existing .pt next to the checkpoint
#   --force              install even if the config cross-check fails
#   --dry-run            print what would happen, change nothing
#
# Connection settings come from the environment so no host or account detail is
# baked into the repo:
#
#   SERVER=user@host  SERVER_PORT=22  SERVER_REPO=/path/to/k1-goalpose
#   ROBOT=user@host   ROBOT_PORT=22   ROBOT_WS=/path/to/mission_ws
#   CONDA_ENV=k1goalpose
#   SSH_BIND=<local ip>   # only if the Mac needs to pick a specific interface
#
# Put them in tools/deploy_env.sh (gitignored) and this script will source it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
[ -f "${SCRIPT_DIR}/deploy_env.sh" ] && source "${SCRIPT_DIR}/deploy_env.sh"

POLICY_NAME="goal_pose_e0"
DEPLOY_CONFIG="Goal_Pose_E0.yaml"
TASK="K1/Goal_Pose_V7"
CHECKPOINT=""
EXPORT_ONLY=0
SKIP_EXPORT=0
FORCE=0
DRY_RUN=0

die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33mwarn:\033[0m %s\n' "$*" >&2; }

while [ $# -gt 0 ]; do
  case "$1" in
    --checkpoint) CHECKPOINT="${2:-}"; shift 2 ;;
    --name)       POLICY_NAME="${2:-}"; shift 2 ;;
    --config)     DEPLOY_CONFIG="${2:-}"; shift 2 ;;
    --task)       TASK="${2:-}"; shift 2 ;;
    --export-only) EXPORT_ONLY=1; shift ;;
    --skip-export) SKIP_EXPORT=1; shift ;;
    --force)      FORCE=1; shift ;;
    --dry-run)    DRY_RUN=1; shift ;;
    -h|--help)    sed -n '2,40p' "$0"; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[ -n "${CHECKPOINT}" ] || die "--checkpoint is required (path to a .pth on the server)"
[ -n "${SERVER:-}" ]      || die "SERVER is not set (see tools/deploy_env.sh.example)"
[ -n "${SERVER_REPO:-}" ] || die "SERVER_REPO is not set"
if [ "${EXPORT_ONLY}" -eq 0 ]; then
  [ -n "${ROBOT:-}" ]    || die "ROBOT is not set (or pass --export-only)"
  [ -n "${ROBOT_WS:-}" ] || die "ROBOT_WS is not set (or pass --export-only)"
fi

SERVER_PORT="${SERVER_PORT:-22}"
ROBOT_PORT="${ROBOT_PORT:-22}"
CONDA_ENV="${CONDA_ENV:-k1goalpose}"
# Resolve a python on the server without relying on conda being on PATH.
if [ -n "${CONDA_BASE:-}" ]; then
  SERVER_PY="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"
else
  SERVER_PY="python"
fi

SSH_OPTS=()
SCP_OPTS=()
if [ -n "${SSH_BIND:-}" ]; then
  SSH_OPTS+=(-b "${SSH_BIND}")
  SCP_OPTS+=(-o "BindAddress=${SSH_BIND}")
fi

run_server() {
  if [ "${DRY_RUN}" -eq 1 ]; then printf '  [dry-run] server$ %s\n' "$*"; return 0; fi
  ssh ${SSH_OPTS[@]+"${SSH_OPTS[@]}"} -p "${SERVER_PORT}" "${SERVER}" "$@"
}
run_robot() {
  if [ "${DRY_RUN}" -eq 1 ]; then printf '  [dry-run] robot$ %s\n' "$*"; return 0; fi
  ssh -p "${ROBOT_PORT}" "${ROBOT}" "$@"
}

# Resolve to absolute paths. A relative --checkpoint is convenient (it is what
# you type on the server, from htwk-gym/), but ssh and scp both start in the
# home directory, so anything relative silently misses.
case "${CHECKPOINT}" in
  /*) CKPT_ABS="${CHECKPOINT}" ;;
  *)  CKPT_ABS="${SERVER_REPO}/htwk-gym/${CHECKPOINT}" ;;
esac
PT_PATH="${CKPT_ABS%.pth}.pt"
FROZEN_CONFIG="$(dirname "$(dirname "${CKPT_ABS}")")/config.yaml"

echo
info "policy   : ${POLICY_NAME}"
info "config   : ${DEPLOY_CONFIG}"
info "ckpt     : ${CHECKPOINT}"
info "export   : ${PT_PATH}"
echo

# ---------------------------------------------------------------- 1. export --
if [ "${SKIP_EXPORT}" -eq 0 ]; then
  info "1/5 exporting on server (${SERVER})"
  run_server "set -e
    cd '${SERVER_REPO}/htwk-gym'
    '${SERVER_PY}' export_model.py --task '${TASK}' --checkpoint '${CKPT_ABS}'
  "
else
  info "1/5 skipping export (--skip-export)"
fi

# ------------------------------------------------------- 2. verify the actor --
info "2/5 verifying exported actor is (1,54) -> (1,12) and finite"
run_server "set -e
  cd '${SERVER_REPO}/htwk-gym'
  '${SERVER_PY}' - <<'PY'
import torch
m = torch.jit.load('${PT_PATH}', map_location='cpu').eval()
with torch.inference_mode():
    y = m(torch.zeros(1, 54))
assert tuple(y.shape) == (1, 12), y.shape
assert torch.isfinite(y).all(), 'non-finite actor output'
print('actor ok: (1,54) -> %s  |max|=%.4f' % (tuple(y.shape), float(y.abs().max())))
PY
"

# ------------------------------------- 3. cross-check frozen config vs deploy --
# Two comparisons, and the second one matters more. sim<->deploy catches a
# mistyped gain. deploy<->hardware catches a config written for the wrong robot,
# which is how a 23-joint config (SDK B1JointCnt says 23, with a waist) came to
# be shipped for a robot that reports 22 joints with the legs starting one index
# earlier. Both configs agreed with each other and both were wrong.
info "3/5 cross-checking frozen run config and real robot layout"

LAYOUT_LOCAL=""
if [ "${EXPORT_ONLY}" -eq 0 ] && [ "${DRY_RUN}" -eq 0 ]; then
  if run_robot "test -f '${ROBOT_WS}/dump_robot_layout.py'" 2>/dev/null; then
    :
  else
    scp -q -P "${ROBOT_PORT}" "${SCRIPT_DIR}/dump_robot_layout.py" \
        "${ROBOT}:${ROBOT_WS}/" 2>/dev/null || true
  fi
  LAYOUT_LOCAL="$(mktemp -t robot_layout).json"
  if run_robot "cd '${ROBOT_WS}' && python3 dump_robot_layout.py --out /tmp/robot_layout.json >/dev/null 2>&1" \
     && scp -q -P "${ROBOT_PORT}" "${ROBOT}:/tmp/robot_layout.json" "${LAYOUT_LOCAL}" 2>/dev/null; then
    info "      robot layout: $(python3 -c "
import json;d=json.load(open('${LAYOUT_LOCAL}'))
print('joint_cnt=%s leg_dof_start=%s parallel=%s' % (d['joint_cnt'], d['inferred_leg_dof_start'], d['parallel_mech_indexes']))" 2>/dev/null || echo '?')"
  else
    warn "could not read the robot's joint layout (is it powered and its motion stack running?)"
    warn "the deploy config will NOT be checked against hardware"
    LAYOUT_LOCAL=""
  fi
fi

SERVER_FROZEN="$( { run_server "cat '${FROZEN_CONFIG}'" 2>/dev/null || true; } )"
if [ "${DRY_RUN}" -eq 0 ]; then
  if [ -z "${SERVER_FROZEN}" ]; then
    warn "could not read frozen config at ${FROZEN_CONFIG}; comparing against hardware only"
    SERVER_FROZEN="{}"
  fi
  CHECK_RC=0
  if python3 -c "import yaml" 2>/dev/null; then
    CHECK_ARGS=(--deploy-config "${REPO_ROOT}/htwk-gym/deploy/configs/${DEPLOY_CONFIG}"
                --frozen-config -)
    [ -n "${LAYOUT_LOCAL}" ] && CHECK_ARGS+=(--robot-layout "${LAYOUT_LOCAL}")
    printf '%s' "${SERVER_FROZEN}" | python3 "${SCRIPT_DIR}/check_policy_contract.py" \
        ${CHECK_ARGS[@]+"${CHECK_ARGS[@]}"} || CHECK_RC=$?
  else
    # No PyYAML here (this Mac cannot reach PyPI). Run the check on the robot,
    # which has PyYAML and already holds the layout dump.
    info "      (no local PyYAML -- running the contract check on the robot)"
    printf '%s' "${SERVER_FROZEN}" | run_robot "cat > /tmp/frozen_config.yaml"
    scp -q -P "${ROBOT_PORT}" "${SCRIPT_DIR}/check_policy_contract.py" \
        "${ROBOT}:${ROBOT_WS}/" 2>/dev/null || true
    scp -q -P "${ROBOT_PORT}" "${REPO_ROOT}/htwk-gym/deploy/configs/${DEPLOY_CONFIG}" \
        "${ROBOT}:/tmp/deploy_config.yaml" 2>/dev/null || true
    run_robot "cd '${ROBOT_WS}' && python3 check_policy_contract.py \
        --frozen-config /tmp/frozen_config.yaml \
        --deploy-config /tmp/deploy_config.yaml \
        --robot-layout /tmp/robot_layout.json" || CHECK_RC=$?
  fi
  [ -n "${LAYOUT_LOCAL}" ] && rm -f "${LAYOUT_LOCAL}"
  if [ "${CHECK_RC}" -ne 0 ]; then
    if [ "${FORCE}" -eq 1 ]; then
      warn "contract mismatch, continuing because --force was given"
      warn "a joint-layout mismatch means every leg command lands on the wrong joint"
    else
      die "contract mismatch (see above). Fix ${DEPLOY_CONFIG}, or pass --force if you are sure."
    fi
  fi
fi

if [ "${EXPORT_ONLY}" -eq 1 ]; then
  info "done (--export-only). Exported: ${PT_PATH}"
  exit 0
fi

# --------------------------------------------------------- 4. copy to robot --
DEST_DIR="${ROBOT_WS}/deploy/models"
DEST="${DEST_DIR}/${POLICY_NAME}.pt"
info "4/5 copying to robot ${ROBOT}:${DEST}"
run_robot "mkdir -p '${DEST_DIR}'"

if [ "${DRY_RUN}" -eq 1 ]; then
  printf '  [dry-run] scp server:%s -> /tmp/%s.pt -> robot:%s\n' "${PT_PATH}" "${POLICY_NAME}" "${DEST}"
else
  TMP_LOCAL="$(mktemp -t "${POLICY_NAME}").pt"
  trap 'rm -f "${TMP_LOCAL}"' EXIT
  # Relay through this machine: the server usually has no route to the robot's
  # private network.
  scp ${SCP_OPTS[@]+"${SCP_OPTS[@]}"} -P "${SERVER_PORT}" "${SERVER}:${PT_PATH}" "${TMP_LOCAL}"
  scp -P "${ROBOT_PORT}" "${TMP_LOCAL}" "${ROBOT}:${DEST}"
fi

# Also ship the deploy config and wrapper, so the robot never runs a .pt against
# a stale YAML. These are plain Python/YAML -- no build step on the robot.
info "      shipping deploy wrapper + config (pure Python, no build needed)"
if [ "${DRY_RUN}" -eq 0 ]; then
  run_robot "mkdir -p '${ROBOT_WS}/deploy/configs' '${ROBOT_WS}/deploy/utils'"
  scp -P "${ROBOT_PORT}" \
      "${REPO_ROOT}/htwk-gym/deploy/deploy_goal_pose.py" \
      "${ROBOT}:${ROBOT_WS}/deploy/"
  scp -P "${ROBOT_PORT}" \
      "${REPO_ROOT}/htwk-gym/deploy/configs/${DEPLOY_CONFIG}" \
      "${ROBOT}:${ROBOT_WS}/deploy/configs/"
  scp -P "${ROBOT_PORT}" \
      "${REPO_ROOT}"/htwk-gym/deploy/utils/*.py \
      "${ROBOT}:${ROBOT_WS}/deploy/utils/"
fi

# -------------------------------------------------------- 5. verify integrity --
info "5/5 verifying sha256 on both ends"
if [ "${DRY_RUN}" -eq 0 ]; then
  SERVER_SHA="$(run_server "sha256sum '${PT_PATH}'" | awk '{print $1}')"
  ROBOT_SHA="$(run_robot  "sha256sum '${DEST}'"    | awk '{print $1}')"
  echo "    server: ${SERVER_SHA}"
  echo "    robot : ${ROBOT_SHA}"
  [ "${SERVER_SHA}" = "${ROBOT_SHA}" ] || die "sha256 mismatch after copy"

  info "      robot-side load smoke test"
  run_robot "cd '${ROBOT_WS}/deploy' && python3 - <<'PY'
import torch
m = torch.jit.load('models/${POLICY_NAME}.pt', map_location='cpu').eval()
with torch.inference_mode():
    y = m(torch.zeros(1, 54))
assert tuple(y.shape) == (1, 12), y.shape
assert torch.isfinite(y).all()
print('robot actor ok:', tuple(y.shape))
PY
"
fi

echo
info "installed ${POLICY_NAME}"
cat <<EOF

Next, on the robot (terminal C):

  cd ${ROBOT_WS}/deploy
  source /opt/ros/humble/setup.bash
  python3 deploy_goal_pose.py --config ${DEPLOY_CONFIG} --goal-source ros --net 127.0.0.1

No colcon build is needed for the deploy side -- it is plain Python. Only Brain
(the C++ BT node) needs building.
EOF
