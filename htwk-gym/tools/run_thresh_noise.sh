#!/usr/bin/env bash
# 시계 팽창 문턱에 **진짜 반복**을 만든다 + 판정 셀 영상.
#
# ⛔ 왜 필요한가: 앞선 문턱 스윕의 깨끗한 열이 `10/10/10` · `22/22/22` · `45/45/45` 로
# **정수까지 동일**했다. `--goal-hold` 에 잡음이 하나도 없어 완전히 결정론적이고,
# 시드가 아무것도 안 바꾼다. ⇒ 문턱 추정 25.3 ms 는 **점추정이고 CI 가 없다.**
# (내가 HANDOFF_MUJOCO_FILTER §7-① 에 "반복 금지" 라고 적어 둔 오류를 다시 냈다.)
#
# ⚠️ IMU 잡음을 켜면 **같은 양의 CI 가 아니라 다른 조건의 문턱**이다. 그렇게 보고한다.
# 실기에는 센서 잡음이 있으므로 이쪽이 오히려 현실적인 값이다.
#
# 그리고 판정을 지는 셀 셋을 렌더한다(CLAUDE.md 영상 규칙):
#   clean 25.0 (p1 +0.020, 낙상 0)  vs  clean 25.7 (p1 -0.0007, 낙상 10)  = 기하 문턱
#   corr  25.0 (p1 +0.017, 낙상 9)                                        = 비기하 붕괴
# 셋을 나란히 보면 "발이 교차해서 넘어지는 것" 과 "안 부딪히고 넘어지는 것" 이 갈린다.
set -u

ROOT=/mnt/DATA/workspace/ws_eungkyu/k1-goalpose/htwk-gym
cd "$ROOT" || exit 1
P=${POLICY:-logs/K1/K1/Goal_Pose_V7/2026-08-04-09-48-36_I3b_stance10/nn/model_200.pt}
BASE="--policy $P --real-asset --goal-hold --duration 120 --armature-preset vendor --clock-mode counter"

O=logs/mujoco/thresh_noise
mkdir -p "$O"
for MS in 24.0 25.0 25.35 25.7 26.14; do
    for S in 0 1 2 3 4; do
        python play_mujoco_goalpose.py $BASE --imu-noise-deg 0.5 \
            --seed "$S" --period-ms "$MS" --out "$O/n_${MS}_s${S}.json" || exit 1
    done
done

V=logs/mujoco/verdict
mkdir -p "$V"
python play_mujoco_goalpose.py $BASE --seed 0 --period-ms 25.0 \
    --out "$V/clean_25.0.json" --video "$V/clean_25.0.mp4"
python play_mujoco_goalpose.py $BASE --seed 0 --period-ms 25.7 \
    --out "$V/clean_25.7.json" --video "$V/clean_25.7.mp4"
python play_mujoco_goalpose.py $BASE --seed 0 --period-ms 25.0 \
    --corrupt-lankr 0.44 --corrupt-rankr 0.21 \
    --out "$V/corr_25.0.json" --video "$V/corr_25.0.mp4"

echo "=== 잡음 켠 문턱 (5시드, 진짜 반복) ==="
python - "$O" <<'PY'
import json, glob, os, sys, statistics as st
for ms in ("24.0", "25.0", "25.35", "25.7", "26.14"):
    ps = sorted(glob.glob(os.path.join(sys.argv[1], "n_%s_s*.json" % ms)))
    if not ps:
        continue
    f = [json.load(open(p))["falls"] for p in ps]
    p1 = [json.load(open(p)).get("foot_sep_upright_m", {}).get("p1") for p in ps]
    p1 = [v for v in p1 if v is not None]
    print("%-7s 낙상 %-18s 평균 %5.1f  SD %5.2f | p1 %+.4f (SD %.4f)"
          % (ms, "/".join(map(str, f)), st.mean(f),
             st.pstdev(f), st.mean(p1), st.pstdev(p1)))
PY
