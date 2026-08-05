#!/usr/bin/env bash
# 감지 열화를 키워가며 정책이 어디서 무너지는지 찾는다.
#
# 왜: MuJoCo가 강체 동역학을 용의선상에서 지웠다 -- 엔진, 접촉 모델, 발 관성,
# 팔 자유도를 전부 바꿔도 180초 낙상 0이다. 그런데 **두 시뮬레이터 모두
# obs[0:6](projected_gravity, base_ang_vel)을 정확히 준다.** 실기에서는 그게
# IMU와 상태추정에서 온다. 균형에 가장 중요한 6채널인데 한 번도 열화시켜 본 적이 없다.
#
# 읽는 법: 무너지는 크기가 실기 IMU 품질보다 **크면** 이 가설도 죽는다.
# **작으면** 원인을 잡은 것이고, 그 크기가 곧 학습에 넣어야 할 랜덤화 폭이다.
#
# 연속보행(goal_hold)으로 잰다. 도착-정지가 없어 균형이 계속 시험되고,
# 실기에서 세 걸음 만에 무너지는 그 조건에 가장 가깝다.
set -e
cd "$(dirname "$0")/.."
PT="${1:-logs/K1/K1/Goal_Pose_V7/2026-08-04-09-48-36_I3b_stance10/nn/model_200.pt}"
DUR="${2:-60}"
OUT=logs/mujoco/sense
mkdir -p "$OUT"

run() {  # 이름, 인자...
    local name="$1"; shift
    python play_mujoco_goalpose.py --duration "$DUR" --policy "$PT" --goal-hold \
        --out "$OUT/$name.json" "$@" > "$OUT/$name.log" 2>&1
    python3 -c "
import json,sys
d=json.load(open('$OUT/$name.json'))
print('  %-22s 낙상 %3d  (%.2f/분)' % ('$name', d['falls'], d['falls_per_min']))
"
}

echo "=== 기준 (열화 없음) ==="
run base

echo "=== IMU 기울기 잡음 (매 스텝, std 도) ==="
for v in 1 2 5 10; do run "noise_${v}deg" --imu-noise-deg "$v"; done

echo "=== IMU 기울기 바이어스 (고정, 도) -- 학습이 본 적 없는 형태 ==="
for v in 1 2 5 10; do run "bias_${v}deg" --imu-bias-deg "$v"; done

echo "=== 자이로 잡음 (std rad/s) ==="
for v in 0.2 0.5 1.0; do run "gyro_${v}" --gyro-noise "$v"; done

echo "=== dof_vel 잡음 (std rad/s) -- 실기는 인코더 미분이다 ==="
for v in 0.5 1.0 3.0; do run "dofvel_${v}" --dofvel-noise "$v"; done

echo "=== 관측 지연 (ms) -- 학습은 액션 쪽 0-18 ms만 모델링한다 ==="
for v in 10 20 40; do run "lag_${v}ms" --sense-lag-ms "$v"; done

echo
echo "결과: $OUT/*.json"
