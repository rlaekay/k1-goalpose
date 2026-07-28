# K1 GoalPose 학습 히스토리 — 발표·공부용 근거 문서

작성 기준일: 2026-07-28  
범위: 저장소의 GoalPose 계열과 여기서 갈라진 GetUp / Kick / SafeFall 계열, 실제 평가 결과, 계획되었지만 아직 검증되지 않은 실험까지

---

## 0. 먼저 말할 결론

1. **가장 큰 단일 개선은 `goal_reached` 보상**이었다. v1을 그대로 연장한 armA와 딱 한 항목만 바꾼 armB를 비교하면 위치 오차 중앙값이 **7.6 → 3.9 cm**, strict success가 **13.9 → 52.8%**로 바뀌었다. “도착 후 멈춰 있는 상태”를 보상해야 한다는 인과 근거가 가장 강하다.
2. **현재 가장 좋은 유효 결과는 E0**다. 위치 오차 **2.7 cm / p90 5.0 cm**, 방위각 오차 **2.5°**, strict success **89.3%**다. 다만 120초 × 256 환경에서 **낙상 2회**이므로 최종 gate 네 개 중 낙상 gate만 통과하지 못했다.
3. **좋은 아이디어를 한꺼번에 넣는 것은 좋은 실험이 아니다.** armD는 12개 변경을 동시에 넣고 24.3 cm로 악화되어 원인을 분리할 수 없었다. v3도 armD를 일부 회복했지만, 나쁜 PD 설정과 거리 curriculum을 물려받아 너무 느리고 보수적이었다.
4. **E1/E2/V7_full의 기존 숫자는 해당 가설을 판정하는 데 쓸 수 없다.** E1은 재평가 사이 코드 의미가 바뀌었고, E2와 V7_full은 자신의 frozen config가 아니라 base config로 평가되었다. 숫자는 존재하지만 결론의 증거는 아니다.
5. **v4–v6, v8/G batch는 구현 또는 계획 단계**다. 학습 결과가 없는 항목을 성공처럼 발표하면 안 된다. 특히 현재 G4 생성기는 `V8_ARMS`를 `ALL_ARMS`에 합치지 않아 launch 경로에 `KeyError` 위험이 있다.

발표의 중심 논증은 다음 한 문장으로 정리할 수 있다.

> 목표 자세 제어에서 성능을 만든 것은 보상의 “양”이 아니라 **원하는 종착 상태를 명시한 것**, 그리고 그 효과를 **단일변수 비교와 재현 가능한 평가**로 분리해 확인한 것이다.

---

## 1. 증거를 읽는 규칙

### 1.1 증거 등급

| 표기 | 뜻 | 발표에서 허용되는 표현 |
|---|---|---|
| **유효 결과** | 해당 run의 frozen config와 checkpoint를 표준 조건으로 평가 | “관측되었다”, “통과했다” |
| **준실험** | 비교군과 변경점이 거의 같지만 둘 이상 차이가 남음 | “시사한다”, “원인 후보” |
| **무효 비교** | config/code drift 또는 잘못된 config로 평가 | “숫자는 있으나 가설 판정 불가” |
| **구현만 완료** | 코드·config 정적 검증만 완료 | “학습 결과 없음” |
| **계획** | 아직 run/report가 없음 | “검증할 예정이다” |

### 1.2 근거 우선순위

1. `report.json`: 실제 평가 숫자와 평가 조건
2. 학습 run 안의 frozen `config.yaml`과 checkpoint 경로
3. 실제 reward/runner/env 소스 코드
4. `MASTERPLAN.md`, `masterplan2.md`, `masterplan3.md`, `gbatch.md`: 의도와 사후 분석
5. 외부 논문: 아이디어의 출처. 단, **논문이 좋다는 사실은 이 로봇에서 효과가 있다는 증거가 아니다.**

### 1.3 공통 최종 gate

| 항목 | 기준 | 무엇을 보는가 |
|---|---:|---|
| 위치 오차 median | ≤ 5 cm | 보통 상황에서의 정확도 |
| 위치 오차 p90 | ≤ 10 cm | 나쁜 10%까지 포함한 신뢰성 |
| 방위각 오차 median | ≤ 10° | 목표 방향 정렬 |
| 낙상 | 0 | 안전성; 평균으로 희석하면 안 되는 hard constraint |

표준 clean 평가는 **256 environments × 120 s, deterministic policy, perturbation OFF, observation noise ON, seed 0**이다.

---

## 2. Newbie를 위한 최소 개념

### 2.1 학습 루프

정책(policy)은 관측 54개를 입력받아 다리 관절 목표 12개를 낸다. 시뮬레이터가 다음 상태를 만들고, reward가 행동의 좋고 나쁨을 한 숫자로 바꾼다. PPO는 “이전 정책에서 너무 멀리 가지 않는 범위”에서 더 높은 reward를 내는 행동의 확률을 키운다.

### 2.2 이 프로젝트에서 고정된 핵심 구조

- 입력: 54-dimensional observation, history 없는 feed-forward 정책
- 출력: 12 leg joint position actions
- actor: MLP `[256, 128, 128]`
- critic: 대체로 `[256, 256, 128]`
- 초기화: ParameterWalk의 actor를 warm-start, critic은 새로 시작
- 기본 알고리즘: PPO + GAE + adaptive KL
- v3/v7 계열: RunnerV3의 **mini-batch PPO**와 실제 **symmetry loss**

이 구조를 오래 유지한 이유는 warm-start를 보존하고, reward나 환경 변화의 효과를 비교하기 위해서다. 관측·행동 차원을 바꾸면 기존 정책의 표현을 그대로 물려받기 어렵고, 원인 분리가 더 힘들어진다.

### 2.3 용어

- **On-policy / PPO**: 지금 정책으로 모은 데이터를 짧게 사용한다. 안정적이지만 샘플을 많이 쓴다.
- **Off-policy / CrossQ**: replay buffer의 과거 데이터도 다시 쓴다. 샘플 효율이 좋지만 구현·튜닝이 더 복잡하다.
- **Reward shaping**: 최종 성공뿐 아니라 중간 상태에도 점수를 주어 학습 신호를 촘촘하게 만든다.
- **Sparse/terminal reward**: 성공한 상태에 큰 점수를 준다. 목표는 명확하지만 성공을 처음 발견하기 어렵다.
- **Warm-start**: 이미 걷는 정책에서 시작한다. 빠르지만 기존 controller의 PD/URDF 의미가 바뀌면 오히려 발목을 잡는다.
- **Ablation**: 한 번에 한 요소를 빼거나 더해 원인을 확인하는 실험.
- **Domain randomization**: 노이즈·외력·모델 오차를 학습 중 섞어 실제 환경에 강하게 만드는 방법.

---

## 3. 버전 지도

| 계열 | 상태 | 핵심 질문 |
|---|---|---|
| Seed | 유효 baseline | 걷기 정책이 목표 자세에 zero-shot으로 얼마나 가까워지는가? |
| v0 | 유효 결과 | 위치·방향·정지를 각각 보상하면 되는가? |
| v1 | 유효 결과 | SE(2) constellation 하나로 결합하면 안정성이 좋아지는가? |
| armA/B/C/D | 유효 결과 | 연장 학습, goal-reached, PD rate, 대규모 통합 변경 중 무엇이 실제로 유효한가? |
| v3 | 유효 결과 | mini-batch PPO + symmetry + 거리 curriculum이 armD를 구할 수 있는가? |
| v4 GetUp | 구현만 완료 | 넘어졌을 때 CrossQ로 빠르게 일어날 수 있는가? |
| v5 Kick | 구현만 완료 | 보행 정책 구조에 공 접근·차기 reward를 이식할 수 있는가? |
| v6 SafeFall | 구현만 완료 | 피크 충격과 머리 접촉을 줄이는 낙법을 학습할 수 있는가? |
| v7 / E0 | 유효 결과 | armB의 정확한 종착 상태 + RunnerV3 + arms-down이 강한 기준선이 되는가? |
| E1/E2/V7_full | 무효 비교 | 경로·강건성·통합안의 효과를 재평가하려 했으나 측정 계보가 오염됨 |
| E3 / F batch | 계획 후 대체 | 단순 확장안을 계획했으나 G batch로 재설계 |
| v8 / G4 | 구현·미평가 | 연속 목표에서 stop-and-go 없이 전환할 수 있는가? |
| G1/G2/G3 | 계획·미평가 | 속도, 강건성, 보호·팔 동작을 E0 위에서 하나씩 검증할 수 있는가? |

**중요:** 저장소에 독립적인 “v2 학습 모델”은 없다. 문서의 milestone 2는 task skeleton 단계였고, `armD_v2_ultimate`는 실험 이름이지 정식 v2 계열이 아니다.

---

## 4. Seed — 걷기 정책 zero-shot 기준선

### 알고리즘

- ParameterWalk actor를 GoalPose 환경에 그대로 넣은 기준선
- GoalPose용 추가 학습 없음

### reward

- 이 비교에서는 새 GoalPose reward로 최적화하지 않았다.

### 무엇을 시험했나

“이미 걷는 정책이 목표를 관측하면, GoalPose 학습 없이도 목표 위치와 방향에서 멈출 수 있는가?”

### 결과

- 위치 median / p90: **30.2 / 45.3 cm**
- 방위각 median: **12.1°**
- 낙상: **1 / 4,633 segments**
- 도착 시 속도: 약 **0.12 m/s**

### 판단

- 장점: 매우 안정적이고 warm-start 출발점으로 좋다.
- 단점: 목표 자세 정확도와 방위각이 gate 밖이다.
- 보고 싶은 항목: GoalPose 학습이 정확도를 얼마나 개선하는지, 그 대가로 낙상이 얼마나 늘어나는지.
- 내부 근거: `MASTERPLAN.md`의 baseline 결과와 초기 GoalPose 기록.

---

## 5. v0 — 위치·방향·정지를 따로 더한 보상

### 알고리즘

- 기본 PPO runner
- 54 obs / 12 actions / ParameterWalk warm-start
- 4–8초마다 목표 재샘플, 30초 episode
- gait clock은 목표 도달 뒤에도 계속 동작

### reward

핵심 GoalPose 항목:

- `goal_position = exp(-d² / 1.0)`, scale `+2.0`
- `goal_heading = exp(-θ² / 0.4)`, scale `+1.5`
- `goal_stop = -(v_xy² + ω_z²)` within 0.1 m, scale `-1.0`
- survival, base height/orientation, torque, feet swing 등 기존 보행 style/safety reward 유지
- 전체 reward는 `only_positive_rewards` 처리

### source와 채택 근거

- Source: 프로젝트 내부 설계. 역사 config는 Git commit `b15eb13`의 `Goal_Pose.yaml`, 구현은 `htwk-gym/envs/K1/goal_pose.py`.
- 채택 이유: 위치, 방향, 정지라는 요구를 사람이 이해하기 쉬운 세 항목으로 직접 대응시킬 수 있다.

### 이번 버전의 질문

“세 오차를 각각 줄이면 GoalPose가 자연스럽게 해결되는가?”

### 결과

- model 3400: 위치 **12.8 / 17.9 cm**, 방위각 **1.7°**, 낙상 **33 / 4,702**, 도착 속도 **0.117 m/s**
- seed보다 자세·방향은 크게 좋아졌지만 낙상은 약 30배, 목표 부근에서도 marching이 남았다.

### 장단점과 해석

- 장점: 무엇을 보상하는지 설명하기 쉽고, 방향 정렬은 잘 학습된다.
- 단점: 항목들이 서로 경쟁한다. “근처에서 계속 걷기”도 위치·방향 점수를 받을 수 있어 멈춘 상태가 안정된 종착점이 되지 않는다.
- 결론: **오차의 합이 성공 상태의 정의를 대신하지 못했다.**

---

## 6. v1 — SE(2) constellation reward

### 알고리즘

- 기본 PPO, v0와 같은 54 obs / 12 actions / warm-start
- 목표 category: stand 0.1, straight 0.2, lateral 0.2, turn 0.2, combined 0.3
- stand에서는 gait clock을 0으로 둠
- episode는 30초로 유지; 원 논문의 짧은 episode를 그대로 복제한 것은 아님

### reward

`No More Marching`의 constellation 형태:

`r = 3.5 · exp[-0.2 · (d² + 2·1²·(1-cos θ))]`

- `goal_position`, `goal_heading`, `goal_stop` scale은 0
- 위치와 방향을 “로봇 주위 두 점의 배치 오차”처럼 한 거리로 결합

### source와 채택 근거

- Source: [No More Marching: Learning to Navigate Humanoids in Dynamic Environments](https://arxiv.org/abs/2508.14098)
- 채택 이유: 속도 command를 따라 걷게 하는 대신 목표 SE(2) pose 자체를 직접 최적화하며, 위치와 회전의 단위를 한 기하학적 표현으로 결합한다.

### 이번 버전의 질문

“분리된 세 reward 대신 하나의 기하학적 목표로 만들면 안정성과 정밀도가 함께 좋아지는가?”

### 결과

- model 2165 조기 종료: **12.7 / 20.1 cm**, 방위각 **1.9°**, 낙상 **6**, 속도 **0.123 m/s**
- model 20000: **8.5 / 16.4 cm**, 방위각 **3.6°**, 낙상 **4**, 도착 속도 **0.10 m/s**, strict / loose **12.4 / 60.2%**

### 장단점과 해석

- 장점: v0보다 낙상이 크게 줄었고, 학습을 계속하면 위치 오차도 개선됐다.
- 단점: 목표 근처의 작은 움직임을 없애는 “absorbing terminal state”가 없다. 정확히 멈춰야 할 이유가 약하다.
- 운영 교훈: 초기 `auto_stop` 설정(2%, patience 3, min 2000)은 너무 공격적이었다. 이후 0.5%, patience 6, min 8000 수준으로 보수화하는 근거가 됐다.

---

## 7. armA–D — 무엇이 실제로 효과가 있었나

공통 평가: 256 env × 120 s, deterministic, perturbation OFF, obs noise ON, seed 0.

| 실험 | 알고리즘·변경 | 핵심 reward | pos med/p90 | heading | strict | falls | 판정 |
|---|---|---|---:|---:|---:|---:|---|
| armA_continue | v1 그대로 연장 | constellation 3.5 | 7.6/14.7 cm | 2.1° | 13.9% | 34 | 대조군 |
| armB_goal_reached | armA + 단일 변경 | `goal_reached +1.0` | **3.9/6.7 cm** | 7.3° | **52.8%** | 37 | 채택 |
| armC_200hz | armA + physics/PD rate 변경 | v1과 같음 | 8.3/16.7 cm | 1.8° | 13.0% | 10 | 기각 |
| armD_v2_ultimate | 12개 변경 동시 투입 | constellation + reached + posture | 24.3/49.5 cm | 2.8° | 6.2% | 21 | 분해 필요 |

### armA — 더 오래 학습하면 해결되는가

- 질문: reward를 바꾸지 않고 training time만 늘리면 exact stop이 생기는가?
- 결과: `not_stopped 33.4%`, `never_arrived 19.3%`.
- 해석: 단순 연장은 부족하다. reward가 정의하지 않은 상태는 시간이 자동으로 만들어주지 않는다.
- 근거: `K1_walk/armA_20260726_121213/report.json`.

### armB — goal_reached 한 항목

- reward: `d < 0.1 m`이고 `v_xy < 0.1 m/s`이면 매 step `+1.0`.
- 질문: 도착·정지를 지속적으로 보상하면 목표 부근이 안정된 종착 상태가 되는가?
- 결과: `not_stopped 0.5%`, `never_arrived 0.2%`; 위치와 strict success가 크게 개선.
- 장점: A/B가 단일변수여서 인과 주장이 가장 강하다.
- 단점: angular speed가 성공 조건에 없어서 방위각 오차와 wobble 여지가 남고, 낙상은 개선하지 못했다.
- 판단: 이후 계열의 backbone으로 채택.
- 근거: `K1_walk/armB_20260726_121222/report.json`.

### armC — 200 Hz가 더 부드러운가

- 변경: physics `dt=0.005`, decimation 4. 제어 주기는 50 Hz로 유지되지만 PD update는 200 Hz.
- 결과: 낙상은 10회로 감소했으나 위치와 strict success가 악화.
- 장점: 시각적으로 부드럽고 낙상 후보를 줄였다.
- 단점: checkpoint 길이도 16.8k로 달라 엄밀한 동일 compute 비교가 아니며, 핵심 GoalPose 성능은 나빠졌다.
- 판단: 채택하지 않음.
- 근거: `K1_walk/armC_20260726_121232/report.json`.

### armD — “좋은 것 12개”를 한 번에 넣은 실패

- 변경: perception jitter/bias/hold, goal_reached, softer stand posture, 목표 범위 확대, resample 변경, push, armature, PD gain 두 배, category weight 등.
- 결과: move category가 25–32 cm, `never_arrived 60.0%`.
- 주된 가설: PD gain 두 배가 warm-start의 action→torque 의미를 바꿨고, 넓은 goal range가 먼 거리의 약한 gradient 문제를 키웠다.
- 장점: robustness에 필요한 후보들을 빠르게 한 번 모아봤다.
- 단점: 12개가 함께 바뀌어 어느 요소가 원인인지 증명할 수 없다.
- 판단: noise, goal_reached, softer posture는 후보로 남기고 PD 변경은 폐기. 이후 반드시 ablation.
- 근거: `K1_walk/armD_20260726_121004/report.json`.

---

## 8. v3 — mini-batch PPO + symmetry + 거리 curriculum

### 알고리즘

- RunnerV3 mini-batch PPO: 5 epochs × 4 minibatches
- 실제 symmetry loss coefficient `0.5`
- adaptive goal-distance curriculum, 초기 scale 0.35
- timed reward 코드는 있으나 `final_window_s=0`, 즉 **OFF**

주의: 예전 config의 `algorithm.symmetric_coef: 10`은 runner가 읽지 않는 dead key였다. 실제 키는 `symmetry_coef`다. armA–D에서 symmetry loss가 적용됐다고 말하면 안 된다.

### reward

- constellation `+3.5`
- goal_reached `+1.0`
- stand_posture `-2.0`
- 기존 style/safety reward

### source와 채택 근거

- Mini-batch PPO: 동일 rollout을 작은 batch로 여러 번 업데이트해 데이터 활용도를 높이기 위해 채택.
- Symmetry loss: 좌우 미러 상태에서 일관된 행동을 내도록 정규화. [Symmetric Deep Reinforcement Learning for Legged Locomotion](https://www.cs.ubc.ca/~van/papers/2019-MIG-symmetry/index.html)의 아이디어 계열.
- 거리 curriculum: 가까운 목표부터 시작해 성공을 먼저 발견시키려는 목적.

### 이번 버전의 질문

“학습 알고리즘과 symmetry/curriculum stack이 armD의 복합 실패를 회복할 수 있는가?”

### 결과

- model 8400: 위치 **13.6 / 30.1 cm**, 방위각 **3.3°**, strict **17.5%**, 낙상 **3**
- armD 대비 위치 median 약 44% 개선, strict 약 2.8배, 낙상 약 86% 감소
- 그러나 `never_arrived 49.9%`, 접근 속도 약 `0.043 m/s`로 지나치게 보수적

### 판단

- 장점: RunnerV3 + symmetry stack이 학습 안정성·낙상에 유망하다는 준실험 근거.
- 단점: armD의 PD 문제를 물려받았고 거리 curriculum이 먼 목표를 충분히 학습하지 못하게 했다.
- 계승: mini-batch PPO와 symmetry는 v7으로 가져가고, distance curriculum과 PD 변경은 버렸다.
- 근거: `htwk-gym/utils/runner_v3.py`, `htwk-gym/envs/K1/goal_pose_v3.py`, `K1_walk/v3_20260726_121242/report.json`.

---

## 9. v4 GetUp — 다른 task로 갈라진 회복 정책

### 상태

**구현·정적 검증만 완료. 학습/eval 결과 없음.** GoalPose 성능 표에 섞지 않는다.

### 알고리즘

- 22 DOF K1_serial
- joint velocity command를 적분해 PD target 생성
- CrossQ: SAC 계열 off-policy, replay buffer, target network 없이 critic의 current/next batch를 함께 forward, Batch Renormalization
- asymmetric critic

### reward

- posture kernel `+2.0`
- upright hold `+1.0`
- action smoothness `+0.2`
- head contact `-1.0`
- torque / dof velocity 각 `-2e-4`
- posture kernel은 target joint pose, upright gravity, base height를 결합

### source와 채택 근거

- [FRASA](https://arxiv.org/abs/2410.08655): CrossQ 기반 빠른 fall recovery.
- [CrossQ](https://openreview.net/forum?id=Z5rhPej0V7): target network 없이 normalization으로 높은 sample efficiency를 노리는 off-policy 알고리즘.
- [HumanUP](https://arxiv.org/abs/2502.12152): 한 단계 정책이 어렵다면 discovery와 deployable motion을 분리하는 fallback 근거.

### 보고 싶은 eval

- `getup_success`: 제한 시간 내 upright 도달 비율
- `upright_hold`: 일어난 뒤 일정 시간 유지
- 자세 유형별 성공률, 머리 접촉, 회복 시간

### 장단점

- 장점: replay로 데이터를 재사용하므로 일어나기처럼 비싼 transition에 효율적일 수 있다.
- 단점: on-policy PPO와 다른 학습 stack이라 구현·튜닝 복잡도가 높고, GoalPose warm-start 논리와 직접 비교할 수 없다.
- 내부 근거: `htwk-gym/envs/K1/Get_Up.yaml`, `htwk-gym/utils/runner_crossq.py`, `htwk-gym/algorithms/crossq.py`.

---

## 10. v5 Kick — 접근과 차기를 한 정책에 담는 task

### 상태

**구현·정적 검증만 완료. 학습/eval 결과 없음.**

### 알고리즘

- 12 DOF PPO / RunnerV3
- 로컬 T1 kicking task를 K1과 ball actor에 이식

### reward

- ball velocity target direction `+10`
- approach `+10`
- body alignment `+1`
- body angle `+0.1`
- ball acceleration `+0.25`
- waiting `-1`
- survival `+0.25`, velocity tracking x/y `+1`, angular `+0.25`, base height `-200`, orientation `-20`, torque 등
- 공이 굴러가기 시작하면 reward scale을 switching

### source와 채택 근거

- Source: 저장소의 기존 T1 kick 구현을 K1로 port. 외부 논문을 그대로 재현한 버전은 아니다.
- 채택 이유: 검증된 로컬 task 구조와 observation/reward 흐름을 재사용해 구현 위험을 낮춘다.

### 보고 싶은 eval

- `kick_success`, 목표 방향 ball speed, 공까지의 접근 성공, 차기 후 낙상
- 접근 실패와 차기 실패를 분리해야 reward 어느 단계가 병목인지 알 수 있다.

### 장단점

- 장점: 접근과 kick을 한 policy에서 최적화할 수 있다.
- 단점: switching reward는 단계 경계에서 불연속이고, 높은 approach/kick scale이 보행 안정성 reward를 압도할 수 있다.
- 내부 근거: `htwk-gym/envs/K1/Kick.yaml`과 관련 K1 kick env.

---

## 11. v6 SafeFall — 넘어질 때 충격을 줄이는 정책

### 상태

**구현·정적 검증만 완료. 학습/eval 결과 없음.**

### 알고리즘

- 22 DOF PPO / RunnerV3
- position-offset action, 3.5 s episode, external push로 낙상 유도

### reward

- survival `+0.25`
- impact force `-2.0`
- head contact `-10.0`
- arm brace `+0.5`
- settle still `+1.0`
- post-impact spin `-0.05`
- torque / dof velocity 등 regularization

### source와 채택 근거

- [Self-Protective Falling with Humanoid Robots via Learning-Based Control](https://arxiv.org/abs/2512.01336) 계열의 충격·머리 보호 목적.
- 채택 이유: 낙상 횟수만 줄이는 GoalPose 정책과, 피할 수 없는 낙상에서 피해를 줄이는 정책은 별도 문제이기 때문이다.

### 보고 싶은 eval

- zero-action baseline 대비 episode peak impact force(kN)
- head contact rate, settle time, post-impact spin, 자세/밀기 방향별 결과

### 장단점

- 장점: 안전을 “넘어지지 않음”에서 “넘어져도 덜 다침”까지 확장한다.
- 단점: arm bracing은 실제 actuator/관절 한계와 충돌할 수 있고, sim contact force의 현실성이 핵심 제약이다.
- 내부 근거: `htwk-gym/envs/K1/Safe_Fall.yaml`과 safe-fall env.

---

## 12. v7 / E0 — 현재 최고 기준선

### 알고리즘

- RunnerV3 mini-batch PPO + symmetry coefficient `0.5`
- 54 obs / 12 leg actions 유지
- ParameterWalk warm-start
- arms-down URDF로 상체 yaw inertia 감소
- E0에서는 path, robust disturbance, protection reward를 OFF

E0은 엄밀한 단일변수 실험이 아니다. armB 대비 runner, symmetry, arms-down이 함께 달라졌다. 따라서 “E0이 좋다”는 결론은 강하지만, “symmetry 때문”이라는 원인 주장은 약하다.

### reward

- constellation `+3.5`
- goal_reached `+1.0`
- stand_posture `-1.0`
- protection margin: position `-0.5`, velocity `-0.02`, torque `-0.002`
- electrical power `-0.002`
- E0에서는 protection stack을 실제로 OFF한 구성으로 평가

### source와 채택 근거

- armB의 가장 강한 단일변수 근거를 backbone으로 채택.
- v3에서 유망했던 mini-batch PPO/symmetry를 가져오되, armD의 PD 변경과 distance curriculum은 제거.
- arms-down은 yaw inertia와 불필요한 상체 운동을 줄이려는 로봇 구조 가설.

### 이번 버전의 질문

“검증된 goal_reached backbone 위에 안정적인 학습 stack과 단순한 arms-down 구조를 얹으면 모든 category에서 5 cm를 달성하는가?”

### 유효 결과 — E0 model 6200

- 위치 median / p90: **2.72 / 5.01 cm**
- 방위각 median / p90: **2.52 / 5.85°**
- final speed median: **0.032 m/s**
- strict / loose success: **89.3 / 99.6%**
- 낙상: **2**
- failure mode: ok 99.3%, not_stopped 0.3%, heading_only 0.3%, never_arrived 0%
- body speed median/p90/p99/max: **0.04 / 0.58 / 1.20 / 1.87 m/s**
- segment peak speed median/p90: **0.81 / 1.37 m/s**

### 판단

- 장점: 전 category가 2.6–2.9 cm로 균일하고, 도착·정지가 거의 해결됐다.
- 단점: 낙상 2회로 hard gate 실패. 구간 피크 속도와 실제 경로 추종 성능은 별도 개선 여지가 있다.
- 다음 기준선: 이후 G 실험은 모두 E0@6200에서 warm-start해야 비교가 깨끗하다.
- 근거: `K1_walk/select_results/E0_armB_armsdown/report.json`, 해당 report가 가리키는 frozen config와 checkpoint.

---

## 13. E1 / E2 / V7_full — 숫자가 있어도 결론이 아닌 이유

### E1_path

- 의도: E0에 moving carrot/path machinery를 더해 waypoint 정확도를 유지하면서 속도를 높이는가?
- 기존 report 숫자: 위치 median 53.5 cm, 낙상 346, segment peak p90 1.95 m/s.
- **판정: 무효 비교.** 최초 평가와 재평가 사이 lookahead floor의 코드 의미가 바뀌었다. 동일 이름이 동일 처리(treatment)를 뜻하지 않는다.
- 배운 점: run마다 `ENV_CODE_SHA`를 기록하고, 평가가 학습 당시 env semantics를 재현해야 한다.

### E2_robust

- 의도: disturbance 두 종류와 perception flicker가 clean 정확도를 크게 해치지 않으면서 stress 안정성을 올리는가?
- 기존 report 숫자: 위치 median 34.5 cm, 낙상 125.
- **판정: 무효 비교.** report의 `config`가 해당 arm의 frozen config가 아니라 `envs/K1/Goal_Pose_V7.yaml` base를 가리킨다.

### V7_full

- 의도: path + robust + protection을 한 후보로 합쳤을 때 통합 성능이 유지되는가?
- 기존 report 숫자: 위치 median 41.9 cm, 낙상 2.
- **판정: 무효 비교.** E2와 같은 config 계보 문제.

### 발표에서의 올바른 표현

- 틀림: “path는 성능을 53.5 cm로 망쳤다.”
- 맞음: “해당 숫자는 코드 drift가 섞여 path 가설을 판정하지 못했다. 그래서 E0에서 한 요소씩 다시 시작한다.”

내부 근거: `masterplan3.md`, 각 `K1_walk/select_results/*/report.json`, 현재 평가의 `ENV_CODE_SHA` 기록 코드.

---

## 14. E3와 F batch — 계획했지만 결과 없이 G로 대체

### E3 wide / no curriculum

- 의도: 거리 curriculum 없이 넓은 목표 분포를 직접 학습.
- 상태: launch 전 기각/보류. adaptive velocity curriculum 문헌의 ablation을 검토한 뒤, 단순히 범위만 넓히는 것보다 task progression을 더 구조화할 필요가 있다고 판단.
- source: [Rapid Locomotion via Reinforcement Learning](https://arxiv.org/abs/2205.02824).

### F1–F4

- F1: final-window timed reward 2초 — 실제 검증 없음.
- F2: grid path/dwell — E1 측정 문제를 수정해 G1로 대체.
- F3: stress flicker 0.01 — G2로 대체.
- F4: 미정.

발표에서는 “실패한 모델”이 아니라 **가설 관리의 중간 기록**으로 짧게 다룬다.

---

## 15. v8 / G4 SmoothTurn — 연속 목표 전환

### 상태

**구현됨, 아직 학습/eval 결과 없음.**

### 알고리즘

- `Goal_Pose_V8.yaml` / `goal_pose_v8.py`
- RunnerV3 계열, E0 warm-start 계획
- 순차적인 4개 SE(2) 목표 stack
- 다음 2개 목표를 기존 dead command slot 4–9에 넣어 observation 54와 warm-start compatibility 유지
- 자동 curriculum

### reward

- `seq_goal` scale `+2.0`
- 정규화된 형태: `r = (banked progress + ρ(e)) / N`
- `e = d + λ·ρ(d)·|θ|`
- 목표 전환은 두 경로:
  - direct: `d < 0.1 m` and heading `< 5°`, 속도 조건 없음
  - loose: `d < 0.5 m`, heading `< 60°`, `v,w < 0.1`

### source와 채택 근거

- [SmoothTurn](https://arxiv.org/abs/2603.12842): sequential navigation에서 lookahead와 progress banking으로 매 목표마다 멈췄다 출발하는 현상을 줄이는 접근.
- 채택 이유: E0은 단일 목표의 정확한 정지는 잘하지만, 연속 경로에서는 그 장점이 stop-and-go 단점이 될 수 있다.

### 보고 싶은 eval

- sequence completion rate, bank rate, 목표당 완료 시간
- 목표 전환 전후 속도 dip, turn curvature, 낙상
- 비교군: single-target E0를 같은 waypoint sequence에 적용

### 장단점

- 장점: observation/action 차원을 유지해 warm-start가 가능하고, 연속 이동을 직접 목적화한다.
- 단점: reward와 전환 규칙이 복잡해져 loophole이 늘고, “정확히 멈추기”와 “멈추지 않고 통과하기”의 목표 충돌을 관리해야 한다.

### 현재 launch blocker

`htwk-gym/tools/make_v7_arms.py`에서 `V8_ARMS`에 `G4_smoothturn`이 정의되어 있지만 `ALL_ARMS = dict(**ARMS, **F_ARMS)`에는 포함되지 않는다. `--only G4_smoothturn`은 `ALL_ARMS[args.only]`를 조회하므로 현재 경로는 `KeyError`가 날 수 있다. G batch 결과가 없는 이유를 단순히 “아직 안 돌렸다”로만 설명하기 전에 이 생성 경로를 고쳐야 한다.

---

## 16. G batch — E0 위의 다음 네 질문

모두 E0 model 6200 warm-start를 사용해 기준선을 통일하는 것이 핵심이다.

| 실험 | 알고리즘/추가 요소 | reward/처리 | 특히 볼 것 | 성공 조건 | 상태 |
|---|---|---|---|---|---|
| G1_speed | E0 + path floor + dwell 0.35 + speed×curvature grid | path/speed shaping | waypoint 정확도와 segment peak speed | pos ≤5 cm 유지, p90 speed >1.37 m/s | 미실행 |
| G2_robust | E0 + 2 disturbances + perception flicker | robustness randomization | clean degradation, stress | clean +2 cm 이내, stress `|ω| p90 < 3` | 미실행 |
| G3_full | G1+G2+protection + scripted arm swing | joint margin/power + arm script | 정확도·속도·안전 동시 유지 | G1과 G2 조건 동시 | 미실행 |
| G4_smoothturn | v8 sequential goal | seq_goal + banked progress | stop-and-go 없는 turn | completion/bank/time 개선 | 구현, launch blocker |

### G3의 팔 동작

- `K1_locomotion_armswing.urdf`
- 정책 action은 여전히 다리 12개; elbow 4 DOF는 scripted
- stopped condition에서 0.25초 blend
- 장점: 학습 차원을 늘리지 않고 관성·외형 효과를 시험.
- 단점: scripted 팔이 다리 정책과 상호작용해도 policy가 팔 action을 직접 조절하지 못한다.

---

## 17. Eval 항목과 “왜 보는가”

### 17.1 Clean — 최종 선정의 권위 있는 조건

| 항목 | 의미 | 실패하면 먼저 의심할 것 |
|---|---|---|
| pos median / p90 | 평균적·꼬리 위치 정확도 | goal geometry, terminal reward, path semantics |
| heading median / p90 | 방향 정렬 | constellation 회전 비중, angular stop 조건 |
| final speed | 정지 품질 | goal_reached의 speed gate, gait clock |
| strict / loose success | 제품 요구를 한 숫자로 요약 | 어느 하위 조건이 병목인지 반드시 분해 |
| falls | hard safety | PD/URDF, 속도, push, contact, rare states |

### 17.2 Failure-mode 분해

- `ok`: 최종 성공
- `not_stopped`: 도착했지만 속도가 큼 → stop reward/terminal condition 문제
- `heading_only`: 위치·정지는 됐지만 방향 불일치 → 회전 reward/gate 문제
- `arrived_then_left`: 한 번 들어갔다가 나감 → absorbing state 또는 resampling/gait clock 문제
- `never_arrived`: 접근 자체 실패 → far-field gradient, curriculum, PD/warm-start mismatch

armA에서 `not_stopped`, armD/v3에서 `never_arrived`, armB/E0에서 거의 `ok`로 바뀌는 흐름이 reward 설계의 효과를 가장 설명하기 쉽다.

### 17.3 분포·경로 진단

- start distance bins / goal categories: 전체 median이 특정 쉬운 category에 가려지는지 확인
- closest approach, along-track, cross-track, overshoot: “못 갔다”를 방향별로 분해
- feasibility required speed: 주어진 시간에 목표 도달이 물리적으로 가능한지
- body speed median/p90/p99/max와 time share `>0.5`, `>1.0 m/s`: 순간 최고속도 한 점이 아닌 실제 운용 속도
- segment peak speed: 구간별 속도 잠재력
- commanded vs achieved speed bins, tracking ratio, path lag: speed/path 실험의 핵심
- bootstrap 95% CI: run-to-run/segment sampling 불확실성

### 17.4 Robustness — clean과 섞지 않는 세 조건

1. **Clean**: 모델 선택용 authoritative gate.
2. **Perturbed**: 외력/토크를 주고 clean 대비 degradation 측정.
3. **Stress jitter**: true goal을 매 control step ±3 m로 바꿈. 위치 gate는 의미가 없으므로 upright share, falls/env-minute, `|ω| p90`, body speed만 본다.

Stress에서 느리게 굳어 있는 정책도 넘어지지 않을 수 있다. 따라서 stability와 speed를 함께 보고, “안 넘어짐 = robust”로 단순화하지 않는다.

### 17.5 Symmetry / joint telemetry

새 eval은 다음을 기록하도록 확장되었다.

- base-frame foot fore-aft asymmetry
- lateral offset/stance width
- joint별 bias, 특히 hip yaw
- position/velocity/torque limit margin

하지만 과거 report에는 plumbing이 없었으므로 역사 모델에 대해 없는 값을 만들어 말하면 안 된다. E0의 symmetry 효과도 앞으로 좌우 ablation과 이 telemetry로 확인해야 한다.

---

## 18. 발표자가 공부해야 할 “왜”의 연결

### 왜 velocity tracking이 아니라 goal pose인가

Velocity command를 여러 초 적분하면 위치 오차가 누적되고, 마지막에 정확히 멈추는 목표와 충돌할 수 있다. GoalPose는 최종 SE(2)를 직접 관측·보상한다. [Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning](https://arxiv.org/abs/2209.12827) 계열도 목표 도달에서 시간 의존성과 endpoint 정의의 중요성을 보여준다.

### 왜 constellation인가

위치 m와 방향 rad를 별도 scale로 억지로 합치기보다, 로봇 주위의 가상 점 배치 오차로 만들면 같은 공간 단위로 해석할 수 있다. 단점은 radius와 weight가 회전·이동 trade-off를 암묵적으로 정하므로 튜닝이 사라지는 것이 아니라 더 구조화될 뿐이라는 점이다.

### 왜 goal_reached가 강했나

Dense reward는 근처로 가는 gradient를 주지만, “들어와서 계속 머무는 상태”를 최적 정책의 안정점으로 만들지 못할 수 있다. `goal_reached`의 per-step bonus는 성공 영역에 오래 머무를수록 누적 return이 커지므로 absorbing state를 만든다.

### 왜 reward coefficient만 보면 안 되나

실제 영향은 scale뿐 아니라 활성 조건, timestep, clipping, `only_positive_rewards`, episode length와 함께 결정된다. `+1`이라도 매 step 쌓이는 성공 reward는 한 번만 받는 `+10`보다 클 수 있다.

### 왜 낙상 2회를 무시할 수 없나

정확도 평균은 2회의 rare catastrophic event를 희석한다. 제품 안전 요구가 0이면 낙상은 별도 hard constraint다. 관련 설계 근거로 constraint 기반 locomotion 연구([arXiv:2308.12517](https://arxiv.org/abs/2308.12517))와 actuator constraint 연구([arXiv:2312.17507](https://arxiv.org/abs/2312.17507))를 참고할 수 있다.

---

## 19. 현재 판단과 다음 실험 순서

1. **E0@6200를 frozen baseline으로 고정**한다.
2. G4 config generator의 `ALL_ARMS` 누락과 launch smoke를 먼저 고친다.
3. G1, G2, G4를 서로 독립적으로 돌린다. G3는 이 셋 중 채택된 요소를 합치는 마지막 통합 실험이어야 한다.
4. 모든 run에 config snapshot, checkpoint, `ENV_CODE_SHA`, seed, eval command를 저장한다.
5. clean gate를 통과한 뒤 perturbed/stress를 본다. stress가 좋아도 clean 정확도가 망가지면 backbone으로 채택하지 않는다.
6. E0의 낙상 2건은 fall-context replay로 직전 goal category, 속도, contact, joint margin을 추적한다.
7. symmetry의 인과를 알고 싶다면 arms-down과 RunnerV3를 고정한 상태에서 `symmetry_coef 0 vs 0.5` 단일변수 비교를 추가한다.

---

## 20. 발표용 한 줄 요약

- Seed: “잘 걷지만 정확한 목표 자세 제어는 못했다.”
- v0: “세 오차를 따로 줄이자 방향은 맞았지만 목표 부근에서 계속 걸었다.”
- v1: “SE(2) 기하 reward로 안정성은 좋아졌지만 exact stop은 부족했다.”
- armB: “도착·정지를 매 step 보상하자 가장 큰 단일 개선이 나왔다.”
- armD: “좋은 아이디어 12개를 동시에 넣어 실패했고, 원인도 잃었다.”
- v3: “학습 stack은 armD를 일부 회복했지만 curriculum과 PD가 발목을 잡았다.”
- v4–v6: “회복·차기·낙법 task는 구현됐지만 결과는 아직 없다.”
- E0: “현재 가장 좋은 유효 baseline이지만 낙상 2회가 남았다.”
- E1/E2/V7: “숫자는 있으나 config/code drift로 가설 판정은 무효다.”
- v8/G: “이제 E0에서 속도·강건성·연속 전환을 하나씩 검증해야 한다.”

---

## 21. Source map

### 내부 1차 근거

- 프로젝트 종합 기록: `MASTERPLAN.md`, `masterplan2.md`, `masterplan3.md`, `gbatch.md`
- GoalPose reward: `htwk-gym/envs/K1/goal_pose.py`
- v3 reward/curriculum: `htwk-gym/envs/K1/goal_pose_v3.py`, `htwk-gym/envs/K1/Goal_Pose_V3.yaml`
- v7 reward/path/robustness: `htwk-gym/envs/K1/goal_pose_v7.py`, `htwk-gym/envs/K1/Goal_Pose_V7.yaml`
- v8 sequential goal: `htwk-gym/envs/K1/goal_pose_v8.py`, `htwk-gym/envs/K1/Goal_Pose_V8.yaml`
- PPO runner: `htwk-gym/utils/runner.py`
- Mini-batch PPO/symmetry: `htwk-gym/utils/runner_v3.py`
- CrossQ: `htwk-gym/utils/runner_crossq.py`, `htwk-gym/algorithms/crossq.py`
- Task configs: `Get_Up.yaml`, `Kick.yaml`, `Safe_Fall.yaml`
- 실제 평가: `K1_walk/**/report.json`
- G 생성/실행: `htwk-gym/tools/make_v7_arms.py`, `run_g_suite.sh`, `tonight.sh`

### 외부 1차 근거

- No More Marching: https://arxiv.org/abs/2508.14098
- Goal-conditioned endpoint locomotion: https://arxiv.org/abs/2209.12827
- Rapid Locomotion curriculum: https://arxiv.org/abs/2205.02824
- Symmetric RL: https://www.cs.ubc.ca/~van/papers/2019-MIG-symmetry/index.html
- FRASA: https://arxiv.org/abs/2410.08655
- CrossQ: https://openreview.net/forum?id=Z5rhPej0V7
- HumanUP: https://arxiv.org/abs/2502.12152
- Self-Protective Falling: https://arxiv.org/abs/2512.01336
- SmoothTurn: https://arxiv.org/abs/2603.12842
- Locomotion constraints: https://arxiv.org/abs/2308.12517
- Actuator constraints: https://arxiv.org/abs/2312.17507

---

## 22. 질의응답에서 피해야 할 과장

- “symmetry loss가 E0의 2.7 cm를 만들었다” → **분리 실험 없음.**
- “E1 path가 53.5 cm라 path는 실패다” → **code drift 때문에 무효.**
- “E2 robust가 나쁘다” → **자기 config로 평가되지 않음.**
- “v4–v6가 동작한다” → **정적 검증만 했고 학습 결과 없음.**
- “G batch가 진행 중이다” → **로컬 결과가 없고 G4 생성 경로 blocker가 남아 있음.**
- “E0는 최종 성공이다” → **정확도·방위각은 통과했지만 falls=2라 최종 gate 실패.**

가장 방어력 있는 태도는 결과를 작게 말하는 것이 아니라, **어떤 결론까지 데이터가 허용하는지 경계를 정확히 말하는 것**이다.
