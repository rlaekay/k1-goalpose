#!/usr/bin/env bash
# 영점 강건성 레인을 백필 큐에 넣는다.
#
#   bash tools/queue_zerostress.sh
#
# ⛔ 왜 필요한가. `NZ_zeroiid` 는 `joint_zero.enabled: false` 로 채점됐다 --
# 공통 평가 config 에 그 블록이 없고 `make_eval_cfg.py` 는 DR 을 이식하지 않는다.
# 즉 **영점 강건화 arm 이 원리적으로 자기 이득을 보일 수 없었다.**
#
# 기존 held-out 프로브(`--joint_encoder_bias_rad` / `--joint_target_offset_rad`)로는
# 안 된다. 그 둘은 legacy 키를 쓰고 **독립 추첨**이라 실기 영점 오차
# (`encoder = +b`, `target = −b`)를 재현하지 못한다(ibatch §8-46).
# 그래서 `--joint_zero_probe_deg` 를 새로 만들었다.
#
# ---- 설계 근거 --------------------------------------------------------------
#
# **3도인 이유**: `anti_mirror 0.20` 의 근거였던 `logs/mujoco/sig/hiprbias*` 를
# 다시 읽으면 **±3도는 양쪽 다 낙상 0**, −5도만 6낙상이고 **n=1** 이다
# (`--goal-hold` 는 결정론적이라 seed 를 늘려도 표본이 안 는다). ±5도를 근거로
# 쓸 수 없다. 문헌이 실제로 거는 에피소드 상수 오프셋도 ±0.57~2.3도다.
#
# **모드 셋인 이유**: 한 모드만 재면 "그 모드가 얼마나 어려운가"는 나오지만
# "어느 모드에 얼마를 줄까"는 안 나온다. 비중을 데이터로 잡으려면 **모드 간
# 상대 난이도**가 필요하다.
# ⚠️ 다만 비중이 반영해야 하는 것은 **P(그 고장이 실기에서 일어난다)** 이지
# **P(그 모드가 어렵다)** 가 아니다. 발생률은 δ 를 실측하기 전에는 모른다 --
# 그래서 이 스캔이 끝나도 비중은 근거 있는 추측이고, `N9_zerostruct` 는
# 배포단 영점 추정기(R7)가 δ 의 상관 구조를 줄 때까지 plan 에 둔다.
#
# **`anti_mirror` 가 가장 어려운 시험인 이유**: URDF 에서 `Left_Hip_Roll` 과
# `Right_Hip_Roll` 이 **둘 다 `axis=(1,0,0)`** 이다(좌우 미러가 아니라 공통 몸통
# 좌표계). 그래서 anti_mirror δ 는 양발이 모이거나 벌어지면서 몸통 자세 변화가
# ≈0 이라 **IMU 가 구조적으로 못 본다.** 그리고 그것이 실기 증상(다리가 모여
# 발끼리 부딪힘)의 모드다.
#
# **대조군에도 같은 시험을 거는 이유**: 학습 때 이 축이 꺼져 있던 arm 도 받아야
# "영점 랜덤화가 강건성을 준다"를 말할 수 있다. 한쪽만 시험하면 비교가 아니다.

cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"
OUT_DIR="$ROOT/queue/small/gpu1"
mkdir -p "$OUT_DIR"

cat > "$OUT_DIR/014-eval_zerostress.sh" <<'JOB'
#!/usr/bin/env bash
# MAX_HOURS=4
# 영점 강건성: 모든 arm 에 **같은** 오차를 건다(대조군 포함).
# 기준선(0도) + 3도 x 모드 3종. 자세한 근거는 tools/queue_zerostress.sh 주석.
set -e
OUT=logs/eval_rounds/zerostress
mkdir -p "$OUT"
for arm in NE_ctrl100 NZ_zeroiid N0_ctrl N1_path; do
    D=$(MIN_CKPT=10 bash tools/pick_run.sh "$arm") || continue
    CK="$D/nn/model_6000.pth"
    [ -e "$CK" ] || { echo "건너뜀 $arm: model_6000 없음"; continue; }
    CFG="$OUT/$(basename "$D").cfg.yaml"
    python -u tools/make_eval_cfg.py --common sweeps/N0_ctrl.yaml --run "$D" --out "$CFG" || continue
    for spec in 0:none 3:anti_mirror 3:mirror 3:iid; do
        deg=${spec%%:*}
        mode=${spec##*:}
        probe=""
        if [ "$deg" != "0" ]; then
            probe="--joint_zero_probe_deg $deg --joint_zero_probe_modes $mode"
        fi
        echo "== $arm model_6000  영점 ${deg}도 ${mode} =="
        python -u eval_goal_pose.py --task K1/Goal_Pose_V7 --config "$CFG" \
            --checkpoint "$CK" --terrain plane --duration_s 120 $probe \
            --sim_device "cuda:$GPU_INDEX" --rl_device "cuda:$GPU_INDEX" \
            --out "$OUT/$(basename "$D")_z${deg}_${mode}.accuracy" 2>&1 | tail -2
    done
done
python -u tools/round_table.py "$OUT"
JOB
chmod +x "$OUT_DIR/014-eval_zerostress.sh"
bash -n "$OUT_DIR/014-eval_zerostress.sh" && echo "문법 OK"
echo "걸었다: queue/small/gpu1/014-eval_zerostress.sh"
echo "  arm 4 x (기준선 + 3도 x 모드 3) = 16 회 x 120 s"
