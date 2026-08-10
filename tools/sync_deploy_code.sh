#!/usr/bin/env bash
#
# 배포 **코드**를 로봇으로 옮긴다.
#
# ⛔ 왜 별도 스크립트인가: `install_policy.sh` 는 **정책(.pt)과 config, 검증 도구**만
# 옮긴다. `deploy_goal_pose.py` 와 `utils/` 는 안 옮긴다. 로봇에는 git 이 없다.
# 그래서 코드 수정이 로봇에 반영되는 경로가 **문서에도 스크립트에도 없었고**,
# 2026-08-09 실행 직전에 로봇이 3주 전 빌드(`d883ed17…`)로 서 있는 것이 발견됐다.
#
# 옮기는 것 (딱 이것만 — 최소 표면):
#   deploy_goal_pose.py          메인 루프. 발목 관측 게이트 · 토크 박스 · 시계 레버
#   utils/policy_goal_pose.py    정책 래퍼. gait_clock_scale
#   configs/Goal_Pose_E0.yaml    torque_box_limits 등
#   run_e0.sh                    실행 래퍼 (ROS source)
#
# 안전:
#   * 옮기기 전 로봇 쪽 파일을 `<name>.bak.<UTC>` 로 백업한다
#   * 옮긴 뒤 **양쪽 SHA-256 을 대조**하고 하나라도 어긋나면 exit 1
#   * `--dry-run` 으로 무엇이 다른지만 볼 수 있다
#   * ⛔ 로봇에서 **아무 프로세스도 시작하지 않는다.** 파일만 놓는다
#
# 사용:
#   bash tools/sync_deploy_code.sh --dry-run
#   bash tools/sync_deploy_code.sh
set -o pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "${REPO_ROOT}/tools/deploy_env.sh" 2>/dev/null || {
  echo "⛔ tools/deploy_env.sh 가 없다. deploy_env.sh.example 을 복사해서 채워라." >&2
  exit 1
}
: "${ROBOT:?ROBOT 미설정}" "${ROBOT_PORT:=22}" "${ROBOT_WS:=/home/booster/Workspace/deploy}"

DRY=0
[ "${1:-}" = "--dry-run" ] && DRY=1

FILES=(
  deploy_goal_pose.py
  utils/policy_goal_pose.py
  configs/Goal_Pose_E0.yaml
  run_e0.sh
)
SRC="${REPO_ROOT}/htwk-gym/deploy"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
CHANGED=()

echo "로봇: ${ROBOT}:${ROBOT_WS}"
echo "저장소 HEAD: $(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
echo

for f in "${FILES[@]}"; do
  [ -f "${SRC}/${f}" ] || { echo "⛔ 로컬에 없다: ${f}"; exit 1; }
  L=$(shasum -a 256 "${SRC}/${f}" | cut -d' ' -f1)
  R=$(ssh -p "${ROBOT_PORT}" "${ROBOT}" "sha256sum '${ROBOT_WS}/${f}' 2>/dev/null | cut -d' ' -f1")
  if [ "${L}" = "${R}" ]; then
    printf '  같음   %s\n' "${f}"
  else
    printf '⛔ 다름   %s\n           local %s\n           robot %s\n' \
      "${f}" "${L:0:16}" "${R:0:16}"
    CHANGED+=("${f}")
  fi
done

if [ ${#CHANGED[@]} -eq 0 ]; then
  echo; echo "✅ 로봇이 이미 최신이다. 옮길 것 없음."
  exit 0
fi

echo
if [ "${DRY}" = "1" ]; then
  echo "[dry-run] 옮길 파일 ${#CHANGED[@]} 개: ${CHANGED[*]}"
  exit 0
fi

for f in "${CHANGED[@]}"; do
  # 백업 -> 전송. 디렉터리는 이미 있다(파일이 이미 있으므로).
  ssh -p "${ROBOT_PORT}" "${ROBOT}" \
    "[ -f '${ROBOT_WS}/${f}' ] && cp -p '${ROBOT_WS}/${f}' '${ROBOT_WS}/${f}.bak.${STAMP}' || true" || exit 1
  scp -q -P "${ROBOT_PORT}" "${SRC}/${f}" "${ROBOT}:${ROBOT_WS}/${f}" || exit 1
  echo "  전송 ${f}  (백업 ${f}.bak.${STAMP})"
done

echo
echo "=== 전송 후 대조 ==="
FAIL=0
for f in "${FILES[@]}"; do
  L=$(shasum -a 256 "${SRC}/${f}" | cut -d' ' -f1)
  R=$(ssh -p "${ROBOT_PORT}" "${ROBOT}" "sha256sum '${ROBOT_WS}/${f}' 2>/dev/null | cut -d' ' -f1")
  if [ "${L}" = "${R}" ]; then
    printf '  ✅ %-32s %s\n' "${f}" "${L:0:16}"
  else
    printf '  ⛔ %-32s local %s / robot %s\n' "${f}" "${L:0:16}" "${R:0:16}"
    FAIL=1
  fi
done

# ⛔ 파이썬 캐시가 옛 모듈을 물고 있으면 새 코드가 안 도는 수가 있다.
ssh -p "${ROBOT_PORT}" "${ROBOT}" "rm -rf '${ROBOT_WS}/__pycache__' '${ROBOT_WS}/utils/__pycache__'" || true
echo "  (파이썬 캐시 비움)"

if [ "${FAIL}" != "0" ]; then
  echo; echo "⛔ 해시 불일치. 로봇을 돌리지 마라."; exit 1
fi
echo
echo "✅ 동기화 완료. 시험 기록에 남길 해시:"
ssh -p "${ROBOT_PORT}" "${ROBOT}" "sha256sum '${ROBOT_WS}/deploy_goal_pose.py'"
