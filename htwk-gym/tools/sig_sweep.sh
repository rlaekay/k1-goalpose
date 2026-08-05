#!/usr/bin/env bash
# 실기 증상을 재현하는 조건을 찾는다. 레버를 하나씩 켜며 signature 점수를 낸다.
#
# 1단계는 **단일 레버 스캔**이다. 어느 축이 점수를 내리는지부터 본다 --
# 여러 개를 한꺼번에 켜면 armD가 12개 레버를 동시에 바꿔 아무것도 배우지 못한
# 그 실패를 반복한다.
#
# 목표 지표(실기, 만충 2.4초): Hip_Roll 폭 29.8/45.4도, roll 액션 0.910/1.313,
# 발목 pitch 토크 rms 6.92/7.26(sim보다 낮다), 몸통 roll 폭 35.3도.
# set -u 를 쓰지 않는다 -- conda.sh가 LD_LIBRARY_PATH 등 미설정 변수를 참조해
# 활성화 전에 죽는다. gpu_queue.sh 주석에 같은 전례를 적어두고 또 반복했다.
cd "$(dirname "$0")/.."
source /mnt/DATA/workspace/ws_eungkyu/miniconda3/etc/profile.d/conda.sh
conda activate k1goalpose
PT=logs/K1/K1/Goal_Pose_V7/2026-08-04-09-48-36_I3b_stance10/nn/model_200.pt
D=logs/mujoco/sig
mkdir -p "$D"
DUR="${DUR:-20}"

run() {   # 이름, 인자...
    local name="$1"; shift
    [ -f "$D/$name.csv" ] && return 0          # 이미 돈 것은 건너뛴다(재시작 안전)
    python play_mujoco_goalpose.py --duration "$DUR" --policy "$PT" --goal-hold \
        --dump-csv "$D/$name.csv" --out "$D/$name.json" "$@" > "$D/$name.log" 2>&1
}

echo "### 1단계 단일 레버 스캔 (각 ${DUR}초)"
run base

# 발목이 약하다 -- 실기 토크가 sim의 0.6-0.7배
for g in 0.3 0.5 0.7; do run ankle_g$g --ankle-gain $g; done
# Hip_Roll이 약하거나 강하다
for g in 0.5 0.7 1.5; do run hipr_g$g --hip-roll-gain $g; done
# 관절 영점 (실측 차동 1.29도지만 공통모드는 미측정)
for v in 2 3 5; do run jbias$v --joint-bias-deg $v; done
for v in -3 -5 3 5; do run hiprbias$v --hiproll-bias-deg $v; done
# 감지
for v in 3 5 10; do run imubias$v --imu-bias-deg $v; done
for v in 20 30 40; do run lag$v --sense-lag-ms $v; done
for v in 25 30 40; do run period$v --period-ms $v; done
# 지면
for f in 0.3 0.5 2.0; do run fric$f --lat-friction $f; done
# 배포 필터 / 토크
run filter --deploy-filter
run noclamp --no-torque-clamp
run urdffoot --foot-inertia urdf 2>/dev/null || true

echo "### 점수"
python3 tools/signature_score.py "$D"/*.csv 2>&1 | tail -40

echo "### 2단계: 발목 감쇠 (실기 발목 roll 속도가 sim의 2.9-4.5배, 토크는 정상)"
for d in 0.2 0.35 0.5 0.7; do run ankdamp$d --ankle-damp $d; done
for d in 0.1 0.2 0.35 0.5; do run arolldamp$d --ankle-roll-damp $d; done
echo "### 2단계 점수"
python3 tools/signature_score.py logs/mujoco/sig/*.csv 2>&1 | tail -45
