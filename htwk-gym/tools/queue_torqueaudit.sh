#!/usr/bin/env bash
# 토크 포화가 학습에서 실제로 걸리는가 -- **관절별로** 잰다.
#
#   bash tools/queue_torqueaudit.sh
#
# ---- 이미 확정된 것 (코드에서 직접 확인, 추측 아님) ------------------------
#
# ① **학습은 토크를 클립한다.** `envs/K1/goal_pose.py:866`
#      dof_torques = torch.clip(dof_torques - friction,
#                               min=-self.torque_limits, max=self.torque_limits)
#    `torque_limits[i]` 는 `goal_pose.py:68` 에서 **URDF `<limit effort=...>`**
#    로 채워진다(Isaac 의 DOF properties 경유). config 값이 아니다.
#    ⇒ "아무 데서도 안 걸린다" 는 기각. 걸린다.
#
# ② ⛔ **그런데 기준이 세 군데에서 서로 다르다.** 같은 물리 관절인데:
#
#      관절          boxfoot(N배치)  armsdown(배포계보 I3b)  deploy config
#      Hip_Pitch          30              30                   45
#      Hip_Roll         **35**          **20**               **30**
#      Hip_Yaw            20              20                   30
#      Knee_Pitch         40              40                   45
#      Ankle_Pitch        20              20                   20
#      Ankle_Roll       **20**          **15**                 20
#
#    `Hip_Roll` 은 **35 / 20 / 30 으로 셋 다 다르다**(최대 1.75배).
#    N 배치는 `K1_robot_boxfoot.urdf`, 배포된 정책 계보 `I3b_stance10` 은
#    `Goal_Pose_V7.yaml` 기본값인 `K1_locomotion_armsdown.urdf` 를 썼다.
#    ⇒ **N 배치는 배포 계보보다 Hip_Roll 토크 권한이 1.75배 크다.**
#
# ③ ⛔ **배포의 `common.torque_limit` 은 다리 대부분에 안 걸린다.**
#    `deploy_goal_pose.py:1920` 의 클립은 `if self._parallel_torque and
#    self._policy_gains_active` 안에 있고 `mech = parallel_mech_indexes
#    = [14,15,20,21]` = **양발 Ankle_Pitch/Ankle_Roll 뿐**이다.
#    Hip_Roll(11/17) 의 30 은 우리 코드가 한 번도 적용하지 않는다 -- 펌웨어가 정한다.
#
# ---- 왜 이게 sim-real 격차 후보인가 -----------------------------------------
#
# `Hip_Roll` 은 다리를 좌우로 벌리는 관절이고, **실기 증상이 "다리가 모여 발끼리
# 부딪힌다"** 이다. 정책이 35 N·m 를 쓸 수 있다는 전제로 측면 자세를 유지하도록
# 학습했는데 실기가 그보다 낮은 곳에서 포화하면, **정확히 그 증상이 나온다.**
# 그리고 `anti_mirror` 영점 모드(IMU 가 구조적으로 못 보는 모드)가 같은 관절이다.
#
# ---- 판정 기준 (숫자 보기 **전에** 고정한다) --------------------------------
#
# 재는 값: `torque_by_joint[*].mean_occ` (평균 점유율), `peak_occ`, `sat_share`.
# ⛔ `torque_occupancy` / `torque_saturated` 는 쓰지 마라 -- `_v7_extras` 가
#    호출되는 **한 프레임**의 값이고 eval 은 그걸 롤아웃 끝에 한 번 읽는다.
#    RETRACTIONS C11 이 발 간격으로 이미 당한 결함이다. 새 키는 롤아웃 전체 누산이다.
#
#   확증 : boxfoot arm 의 Hip_Roll `mean_Nm` 이 **20 N·m 를 넘거나**
#          `peak_occ` 가 20/35 = **0.571 을 넘는다**
#          => 정책이 armsdown/배포가 안 주는 토크를 상시로 쓰고 있다.
#             URDF effort 를 실기 값으로 맞추는 arm 이 즉시 정당화된다.
#
#   기각 : Hip_Roll `peak_occ` < 0.571 이고 `sat_share` ~ 0
#          => 35 라는 값이 이 정책의 행동에 안 걸린다(여유가 남는다).
#             URDF 불일치는 실재하지만 **이 실패의 원인은 아니다.** 다른 데를 봐라.
#
#   ⚠️ 어느 쪽이든 **URDF 세 값이 다르다는 사실 자체는 고쳐야 한다.** 기각은
#      "지금 이 증상의 원인이 아니다" 이지 "문제가 없다" 가 아니다.
#
# 대조: 같은 프로토콜로 **두 URDF 를 다 잰다.** N1_path(boxfoot) 와 배포 계보
#       I3b_stance10(armsdown). 한쪽만 재면 "35 를 쓴다" 가 URDF 때문인지
#       정책 때문인지 안 갈린다.

cd "$(dirname "$0")/.." || exit 1
OUT_DIR="$PWD/queue/small/gpu0"
mkdir -p "$OUT_DIR"

cat > "$OUT_DIR/015-eval_torqueaudit.sh" <<'JOB'
#!/usr/bin/env bash
# MAX_HOURS=2
# 관절별 토크 점유율. 판정 기준은 tools/queue_torqueaudit.sh 주석에 사전 고정돼 있다.
set -e
OUT=logs/eval_rounds/torqueaudit
mkdir -p "$OUT"

# boxfoot 계열(N 배치) 과 armsdown 계열(배포 계보) 을 같은 프로토콜로 잰다.
for arm in N1_path N0_ctrl NE_ctrl100 I3b_stance10; do
    D=$(MIN_CKPT=1 bash tools/pick_run.sh "$arm") || { echo "건너뜀 $arm: run 없음"; continue; }
    CK="$D/nn/model_6000.pth"
    [ -e "$CK" ] || CK=$(ls -1 "$D"/nn/model_*.pth 2>/dev/null | sort -t_ -k2 -n | tail -1)
    [ -n "$CK" ] && [ -e "$CK" ] || { echo "건너뜀 $arm: 체크포인트 없음"; continue; }
    CFG="$OUT/$(basename "$D").cfg.yaml"
    python -u tools/make_eval_cfg.py --common sweeps/N0_ctrl.yaml --run "$D" --out "$CFG" || continue

    # ⛔ 자산은 **arm 자신의 것**을 써야 한다. 공통 config 로 덮으면 URDF 가 통일돼
    # 버려서 이 실험이 재려는 바로 그 차이가 사라진다. make_eval_cfg 는 asset 을
    # 이식하지 않으므로 여기서 명시한다.
    ASSET=$(python - "$D" <<'PY'
import sys, os, yaml
d = sys.argv[1]
for c in ("config.yaml", "cfg.yaml", "params.yaml"):
    p = os.path.join(d, c)
    if os.path.exists(p):
        try:
            print((yaml.safe_load(open(p)) or {}).get("asset", {}).get("file", ""))
        except Exception:
            print("")
        break
else:
    print("")
PY
)
    echo "== $arm  $(basename "$CK")  자산=${ASSET:-(config 기본)} =="
    python -u eval_goal_pose.py --task K1/Goal_Pose_V7 --config "$CFG" \
        --checkpoint "$CK" --terrain plane --duration_s 120 \
        --goal_pattern forward_hold \
        --sim_device "cuda:$GPU_INDEX" --rl_device "cuda:$GPU_INDEX" \
        --out "$OUT/$(basename "$D").torque" 2>&1 | tail -2
done

python -u tools/torque_table.py "$OUT"
JOB
chmod +x "$OUT_DIR/015-eval_torqueaudit.sh"
bash -n "$OUT_DIR/015-eval_torqueaudit.sh" && echo "문법 OK"
echo "걸었다: queue/small/gpu0/015-eval_torqueaudit.sh"
