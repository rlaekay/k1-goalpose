#!/usr/bin/env bash
# 매개자 검정 두 건을 위한 원시 상태 수집 — 새 롤아웃은 이번 한 번뿐이다.
#
# ① LIP 측방 발산 (분석 세션): 주기 T 에서 착지 순간 측방 발산이 e^{(T/2)/tau} 로
#    커지는가. tau=0.214 s. 24.0 -> 26.14 ms 는 반스텝 0.250 -> 0.272 s 이므로
#    **예측 +10.6 %**. 기각: 관측이 예측의 절반 미만이거나 부호 반대.
#    ⭐ 이 검정의 가치는 **자유모수 없이 크기까지 사전에 고정**된다는 것이다 --
#    `p1` 은 "낮으면 나쁘다" 방향만 있어서 낙상의 그림자와 구별이 안 됐다.
#
# ② 낙상 직전 명령 서명 (배포 세션): 청정 25.7 대 오염 25.0(낙상 수 비슷).
#    표본 낙상 −0.5~0 s, 대조 같은 팔의 −4~−2 s.
#    종점: 명령 변화량 RMS 의 채널 분해 — 오염 팔에서 **발목이 아닌 관절**(힙)의
#    명령이 더 큰가. 크면 "오염 관측 -> 전신 오명령" 확증, 같으면 기전은 명령 밖.
#
# ⚠️ 두 검정 다 **첫 낙상 이전 분모** 규칙을 적용한다(오늘 p1 에서 겪은 것 반복 금지).
set -u

ROOT=/mnt/DATA/workspace/ws_eungkyu/k1-goalpose/htwk-gym
cd "$ROOT" || exit 1
P=${POLICY:-logs/K1/K1/Goal_Pose_V7/2026-08-04-09-48-36_I3b_stance10/nn/model_200.pt}
B="--policy $P --real-asset --goal-hold --duration 120 --armature-preset vendor --clock-mode counter"
O=logs/mujoco/mediator
mkdir -p "$O"

# ① 주기 스윕 (청정 팔은 결정론적이라 시드 1개)
for MS in 24.0 25.0 25.35 25.7 26.14; do
    python play_mujoco_goalpose.py $B --seed 0 --period-ms "$MS" \
        --out "$O/clean_${MS}.json" || exit 1
done
# ② 오염 팔 (rng 를 쓰므로 진짜 반복 3시드)
for S in 0 1 2; do
    python play_mujoco_goalpose.py $B --seed "$S" --period-ms 25.0 \
        --corrupt-lankr 0.44 --corrupt-rankr 0.21 \
        --out "$O/corr_25.0_s${S}.json" || exit 1
done
# 대조: 설계 주기, 낙상 0
python play_mujoco_goalpose.py --policy "$P" --real-asset --goal-hold --duration 120 \
    --armature-preset vendor --period-ms 20.0 --clock-mode sim --seed 0 \
    --out "$O/ctrl20.json"
echo "수집 완료: $O/*_series.npz"
