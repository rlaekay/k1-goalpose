# 상태추정(오도메트리) 문제 — 조사 및 SOTA 서베이

> MASTERPLAN.md의 동반 문서. GoalPose 태스크를 실기(K1)에 배포할 때 반드시 풀어야 하는
> 문제를 다룬다. 2026-07-24 작성.

## 1. 문제 정의 (사용자 원 질문 요약)

Custom mode(우리 학습된 정책이 관절을 직접 제어하는 모드)는 official odometer state를
신뢰할 수 있게 제공하지 않는다(사용자 확인). 즉 우리 정책이 실기에서 동작하려면:

- **Δorientation**: gyro만으로 충분 (짧은 시간창에서 드리프트 작음, IMU 적분/AHRS로 해결됨)
- **Δposition (dead-reckoning)**: IMU + leg kinematics로 직접 만들어야 함 — 고전적
  "leg odometry" / contact-aided state estimation 문제

질문 2개:
1. 현재 학습이 오도메트리/변위를 실제로 쓰는가? 안 나오면 영향이 있는가?
2. (필요하다면) 상태추정기를 정책과 **같은 환경·같은 데이터로 동시에(concurrent)** 학습할지,
   아니면 **병렬로(GPU만 공유, 독립적으로)** 학습할지 — consensus/SOTA 조사.

## 2. K1 공식 매뉴얼 조사 결과 (`K1 Instruction Manual V1.6`, 40p, Booster Robotics)

### 2.1 모드 구조
- `DAMP` → `PREP` → `WALK`/`CUSTOM`/`SOCCER` 전이. **CUSTOM 모드는 "K1이 모든 관절 제어권을
  개발자에게 넘긴다"**(SDK로 직접 제어) — 우리가 학습된 정책을 배포할 모드가 바로 이것.

### 2.2 관련 SDK 인터페이스 (실측 페이지 근거)
- **`ResetOdometry`** (RPC, high-level, ≥v1.5.0.9): "Resets the **robot's gait odometry**"
  → Booster 내부적으로 "gait odometry"라는 자체 상태추정기가 이미 존재함을 시사.
- **`rt/odometer_state`** (subscribe 토픽, ≥v1.3.1.1): `struct Odometer { float x; float y;
  float theta; }` — **공식 문서에 x/y/theta 오도메트리 토픽이 실제로 존재**한다.
- **`GetFrameTransform`** (RPC): kBody/kHead/kLeftHand/kRightHand/kLeftFoot/kRightFoot 간
  순간 변환(정적 FK) — 이건 오도메트리가 아니라 그 순간의 자세 트리 변환일 뿐이라 우리
  문제(누적 변위)엔 직접 도움 안 됨.
- **`rt/low_state`** (subscribe, ≥v1.0.0.0): **모드 무관하게 IMU(rpy/gyro/acc) + 22개 관절의
  q/dq/ddq/tau_est를 parallel/serial 구조 둘 다 제공**. 우리가 직접 오도메트리를 만들 때
  필요한 원재료가 정확히 여기 있다.
- **저수준 인터페이스 제약**: "the low level **publish** interface will only take effect
  when the robot is in custom mode" — 이건 **우리가 보내는(publish) `rt/joint_ctrl`**에 대한
  제약이지, `rt/odometer_state`처럼 우리가 **구독(subscribe)하는** 토픽에 대한 제약이 아니다.

### 2.3 확인 안 된 것 (중요, 정직하게 기록)
매뉴얼 어디에도 "`rt/odometer_state`가 CUSTOM 모드에서도 계속 갱신되는지"를 명시하지 않는다.
"gait odometry"라는 이름 자체가 Booster의 **자체 보행 컨트롤러(WALK/SOCCER 모드)에 종속된
추정치일 가능성**을 시사한다 — 우리가 CUSTOM 모드에서 그 컨트롤러를 끄고 직접 관절을
제어하면, 이 추정기가 멈추거나 갱신을 안 할 개연성이 높다. **사용자가 이미 확인한
"custom mode는 odometer state를 지원하지 않는다"는 사실과 정합적**이다 — 매뉴얼이 이를
반박하지 않고, 오히려 "gait"라는 이름을 통해 왜 그런지에 대한 그럴듯한 이유를 제공한다.

**결론**: `rt/odometer_state`를 CUSTOM 모드에서 그대로 믿고 쓸 수 없다는 사용자 전제를
그대로 채택한다. 우리가 직접 만들어야 한다는 문제의식이 옳다.

## 3. 우리 학습 코드가 실제로 오도메트리에 의존하는가 — 예, 명확히 의존한다

[`envs/K1/goal_pose.py:630`](htwk-gym/envs/K1/goal_pose.py) `_compute_observations()`에서
정책(actor)이 실제로 받는 `obs_buf`를 확인:

```python
self.obs_buf = torch.cat((
    projected_gravity,      # IMU 방향(중력 벡터)만 있으면 됨 — 안전
    base_ang_vel,            # gyro만 있으면 됨 — 안전
    self.commands[:, :10],   # ← commands[0], commands[1] = goal_rel_x, goal_rel_y
    gait_clock (cos/sin),    # 내부 상태 — 안전
    dof_pos, dof_vel,        # 관절 엔코더 — 안전
    self.actions,            # 직전 행동 — 안전
), dim=-1)
```

`commands[0:2]`(`goal_rel_x`, `goal_rel_y`)는 [goal_pose.py:462](htwk-gym/envs/K1/goal_pose.py)
`_update_goal_state()`에서 매 스텝:
```python
to_goal = self.goal_pos_world - self.base_pos[:, :2]   # base_pos = 시뮬레이터 절대위치(ground truth)
```
로 계산된다.

**결론**: 정책 관측 벡터 10개 채널 중 **딱 2개(`goal_rel_x/y`)만** 연속적인 위치 추적(=오도메트리)이
필요하다. `heading_error`(commands[2])는 현재 yaw만 있으면 되니 gyro 적분으로 충분히 안전하다.
나머지(관절 엔코더, IMU 방향/각속도, 내부 게이트 클록, 직전 행동)는 전부 로컬 센서만으로
실기에서 그대로 재현 가능하다.

부가로 확인: `base_lin_vel`(base 속도)는 `privileged_obs_buf`에만 있고 **actor 입력이 아니라
critic 전용**(비대칭 actor-critic 표준 기법) — 즉 배포 시 base velocity 추정은 애초에 필요
없다. **실기 배포에 진짜 필요한 유일한 외부 상태추정은 (Δx, Δy) 2차원 위치뿐이다.**

## 4. SOTA/컨센서스 서베이 — "동시(concurrent)" vs "병렬(parallel, GPU만 공유)"

핵심 통찰: 우리 문제(위치 Δx,Δy — **시간에 따라 무제한 누적되는 적분값**)와 대부분의
RL 상태추정 SOTA 논문이 다루는 문제(base velocity, foot height, contact probability —
**매 순간 값이 유계(bounded)이고 다음 순간이면 리셋되는 순간량**)는 **근본적으로 다른
난이도의 문제**다. 이 구분이 아래 권고의 핵심 근거다.

### 4.1 "동시(concurrent)" 학습 — 최근 mainstream, 단 *유계(bounded)* 양에 한함
- **Ji et al., "Concurrent Training of a Control Policy and a State Estimator for Dynamic
  and Robust Legged Locomotion"** ([arXiv:2202.05481](https://arxiv.org/pdf/2202.05481),
  2022): 정책 네트워크 + 상태추정 네트워크(base linear velocity, foot height, contact
  probability 출력)를 **같은 롤아웃·같은 PPO 루프**로 동시 학습. 평지 3.75m/s, 마찰계수
  0.22 슬립 표면 3.54m/s 주행 성공.
- **CTS: Concurrent Teacher-Student RL for Legged Locomotion**
  ([arXiv:2405.10830](https://arxiv.org/abs/2405.10830), 2024): teacher/student를 **같은
  환경과의 상호작용 데이터로 동시에** 학습하는 변형 PPO 제안. 기존 2단계(teacher 먼저 RL,
  student는 나중에 지도학습으로 distill) 방식 대비 **속도 추적 오차 최대 20% 개선**.
- **RMA (Rapid Motor Adaptation)** ([arXiv:2107.04034](https://dblp.org/rec/journals/corr/abs-2107-04034.html),
  Kumar et al. 2021): base policy(특권 정보 입력)를 먼저 학습 → adaptation module을
  **별도 단계로**(엄밀히는 순차, 완전 동시는 아님) proprioceptive history만으로 같은 잠재값을
  예측하도록 지도학습. 이후 "Rapid Locomotion via RL"(RSS 2022)로 확장.

**공통점**: 전부 추정 대상이 **순간 속도/접촉/높이** — 무한 시간 누적되지 않는 양이다.
"동시 학습"이 최근 대세이고 실제로 성능도 더 좋다는 근거가 확실하다 — **단, 이 클래스의
문제에 한해서**다.

### 4.2 위치(오도메트리) — 여전히 model-based(고전) 필터 계열이 지배적
- **Hartley et al., "Contact-Aided Invariant Extended Kalman Filtering for Legged Robot
  State Estimation"** ([arXiv:1805.10410](https://arxiv.org/pdf/1805.10410), 2018): Lie
  group 기반 불변 관측기 설계로 IMU(gyro+accel) + 발 접촉 시의 순운동학(forward kinematics)을
  융합해 자세·속도·접촉점을 동시 추정. 지금까지도 사실상의 표준 베이스라인.
- **2024–2025 후속 연구 다수**, 전부 이 InEKF 계열의 확장:
  - ["Legged odometry based on fusion of leg kinematics and IMU information in a humanoid
    robot"](https://journal.hep.com.cn/bir/EN/10.1016/j.birob.2024.100196) (2024/2025) —
    휴머노이드 특화, 우리와 가장 유사한 문제 설정.
  - ["Iterated Invariant EKF for Quadruped Robot Odometry"](https://arxiv.org/pdf/2604.15449) (2025)
  - ["OCELOT: Odometry and Contact Estimation for Legged Robots"](https://arxiv.org/pdf/2605.21863) (2025)
  - ["CoCo-InEKF: State Estimation with Learned Contact Covariances"](https://arxiv.org/pdf/2605.15122) (2025)
    — InEKF 뼈대는 유지, **접촉 공분산만 학습으로 보정**(하이브리드).
  - "Legged robot state estimation with invariant EKF using neural measurement network"
    (ICRA 2025), "invariant neural-augmented Kalman filter with neural compensator"
    (IROS 2025) — 역시 **InEKF + 학습 보정 모듈**의 하이브리드 패턴이 뚜렷한 트렌드.
  - ["LEGOLAS: Deep leg-inertial odometry"](https://arxiv.org/pdf/...) (CoRL 2024) — 순수
    학습 기반 시도도 있으나, 여전히 소수파이고 InEKF 대비 우위가 범용적으로 검증되진 않음.

**패턴**: 위치 추정은 2025년 최신 논문들조차 **"InEKF 골격 + 학습은 보조/보정 역할"**을
택한다. 순수 end-to-end 학습(정책과 동시 학습해서 위치까지 뽑아내는 방식)으로 이 문제를
푼 사례는 검색에서 발견하지 못했다 — Lie-group 구조가 주는 **일관성(consistency) 보장**이
없는 순수 신경망 회귀는 30초 에피소드 동안 열린루프로 적분되는 위치오차가 무한정 커질 위험이
있고, 이를 억제하는 유효한 대안적 학습 레시피가 아직 확립되지 않았기 때문으로 보인다.

## 5. 권고 — 하이브리드, "병렬" 쪽에 무게

| 대상 | 방식 | 이유 |
|---|---|---|
| **위치 (Δx, Δy) — 우리가 필요한 것** | **병렬**: 고전 contact-aided 오도메트리(IMU gyro/accel + leg FK + 접촉 검출)를 **정책과 완전히 별개로** 구현·검증. GPU만 같이 쓰거나, 아예 안 써도 됨(가벼운 필터라 실기 CPU에서도 충분) | §4.2 — 2025년 최신 연구조차 이 구조 유지. 우리 태스크가 실제로 필요한 건 이 방식이 표적으로 하는 딱 그 문제(누적 위치). 디버깅도 분리되어 쉬움(모션캡처 등으로 필터만 독립 검증 가능). |
| **(참고, 우리는 필요 없음) base velocity** | 동시 학습이 유효한 선택지 (§4.1) | 이미 §3에서 확인했듯 우리 actor는 base velocity를 입력으로 안 씀(critic 전용) — 굳이 안 만들어도 됨. |

### 구체적 다음 단계 제안
1. **오도메트리 모듈을 지금 학습 파이프라인과 독립적으로 구현**: `rt/low_state`가 주는
   IMU(rpy/gyro/acc) + 22관절 q/dq/tau_est로 contact-aided 오도메트리(최소: 발 접촉 감지 +
   순운동학 기반 dead-reckoning, 여유 있으면 InEKF)를 짠다. 시뮬레이터에서 먼저 검증
   가능 — Isaac Gym의 ground-truth `base_pos`를 정답으로 두고 우리가 만든 추정기의 오차를
   재는 것으로 GPU 학습과 완전히 분리해서 개발할 수 있다.
2. **정책 재학습 시 "추정기가 낼 법한 오차"를 도메인 랜덤화에 반영**: 지금 시뮬 학습은
   `goal_rel_x/y`를 항상 ground truth로 계산한다. 실기에서는 이 값이 우리 추정기의 (드리프트
   + 노이즈가 섞인) 출력으로 대체되므로, **sim에서도 `goal_pos_world` 계산에 인위적 드리프트/
   바이어스/지연을 주입**해 정책이 "약간 부정확한 위치추정"에 강건해지도록 만들어야 한다
   (현재 [Goal_Pose.yaml](htwk-gym/envs/K1/Goal_Pose.yaml)의 `noise:` 섹션에 아직 없음 —
   추가 필요). 이건 별도 스윕 갈래로 넣기 좋은 항목이다.
3. **heading_error는 gyro 적분만으로 우선 진행**, 실기 테스트에서 드리프트가 문제되면
   그때 자세 보정(예: 정지 구간에서 재보정)을 추가한다 — 지금 단계에서 과설계 불필요.

## 6. 요약 답변 (원 질문 대응)

- **Q. 현재 학습이 오도메트리를 쓰는가?** → 예, `commands[0:2]`(목표까지 상대위치)가
  매 스텝 절대위치 차분으로 계산되어 정책 관측에 직접 들어간다(§3).
- **Q. odometer state가 안 나오면 영향 있는가?** → 있다. 이 2채널이 없으면 정책이 배포
  시 목표까지의 거리/방향을 전혀 알 수 없어 GoalPose 태스크 자체가 성립하지 않는다.
  반드시 자체 오도메트리로 대체해야 한다(base velocity는 critic 전용이라 무관).
- **Q. 동시 학습 vs 병렬 학습?** → **병렬(고전 contact-aided 필터를 독립 구현)을 권장**.
  "동시 학습"은 최근 연구에서 실제로 우세하지만 그 대상은 항상 유계(bounded) 순간량이고,
  우리에게 필요한 무제한 누적 위치 추정 문제는 2025년 최신 논문들조차 여전히 InEKF 계열
  구조(+ 부분적 학습 보정)를 표준으로 쓴다.
