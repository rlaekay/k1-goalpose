# K1 Goal-Pose RL Learning Masterplan

## 📋 목표
평지에서 로봇 로컬 프레임 기준 목표 자세 (Δx, Δy, Δθ)를 받아, 
넘어지지 않고 그 자세에 도달해서 정지하는 K1용 관절 위치 정책.

## 🎯 범위 고정
- 목표 샘플링: Δx ∈ [-2, 2] m, Δy ∈ [-1.5, 1.5] m, Δθ ∈ [-π, π]
- 평지, 장애물 없음, 단일 목표
- 에피소드: 4~8초마다 목표 재샘플링

## ✅ 성공 기준 (초기값)
| 지표 | 게이트 |
|---|---|
| 최종 위치 오차 (median / p90) | ≤ 5 cm / ≤ 10 cm |
| 최종 heading 오차 | ≤ 10° |
| 넘어짐률 | 0% |

## 🔧 학습 환경
- Framework: htwk-gym (Isaac Gym Preview 4)
- Robot: K1
- Hardware: A6000 x2
- Server: user-ESC4000A-E12 (/mnt/DATA/workspace/ws_eungkyu/htwk-gym)

## 🧠 학습 아키텍쳐
- Approach: Warm-start (ParameterWalk) → End-to-end GoalPose
- Network: MLP [512,256,128] + history
- Action: 관절 위치 목표
- Reward: constellation + style/regularization 상속

## 📍 마일스톤
| # | 할 일 | 조건 |
|---|---|---|
| 0 | 베이스라인 (ParameterWalk 재현) | 영상 확인 |
| 1 | 평가 하네스 | 오차 분포 숫자 |
| 2 | GoalPose 태스크 골격 | 크래시 없음 |
| 3 | constellation 학습 | 게이트 통과 |
| 4 | export + MuJoCo 검증 | 배포 준비 |
