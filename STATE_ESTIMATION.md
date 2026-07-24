# CUSTOM walk 통합 상태추정 설계 — PF, A*, GoalPose까지

> `MASTERPLAN.md`의 동반 문서. K1 CUSTOM mode에서 Booster gait odometry를 사용할 수 없다는
> 전제 아래, 보행 정책뿐 아니라 기존 particle filter와 BT 전체가 필요로 하는 egomotion을
> 다시 설계한다. 2026-07-24 재작성.

## 1. 결론부터

이 문제는 `GoalPose` actor의 `goal_rel_x/y` 두 채널만 만들어 주면 끝나지 않는다.

현재 축구 stack은 SDK `odometer_state` 콜백 하나에서 다음을 모두 수행한다.

1. 누적 local odom(`robotPoseToOdom`) 갱신
2. main PF와 relocalization PF의 motion update
3. `robotPoseToField = odomToField ⊕ robotPoseToOdom`의 연속 전파
4. TF와 `/localized_pose` 발행
5. local planner의 목표 오차·도달 판정·속도 feedback 갱신
6. orientation sentinel와 odom/field-frame 보정 유지

따라서 CUSTOM mode에서 native odometer가 멈추면 **PF delta만 사라지는 것이 아니라 field pose의
연속 전파와 planner 계산도 함께 멈춘다.** landmark가 보이는 detection callback에서는
`calibrateOdom()`으로 data pose가 간헐적으로 바뀔 수 있지만, 그 사이 dead-reckoning과 TF publish,
planner update는 멎는다. 주행 중 끊기면 BT의 명목상 100 Hz tick이 마지막 nonzero planner velocity를 계속
재전송할 수도 있다. 현재 코드에는 odom freshness watchdog이 없다.
<!--왜 끊기는걸 걱정하는거지?-->

채택할 구조는 다음 하나다.

```text
K1 low_state(IMU + q/dq/ddq/tau_est)
    → K1 joint/time adapter
    → contact-aided leg-inertial estimator
    → timestamped body-frame SE(2) delta + 3×3 covariance + health
    → PF motion update + continuous O→B odom
    → M→O landmark correction + current M→B pose
    → A* / local planner / GoalPose relative target / perception consumers
```

상위 계층이 robot-frame lookahead를 streaming으로 주면 **GoalPose policy 내부**의 적분은 생략할 수
있다. 그러나 지금처럼 field-map PF와 A*를 계속 쓸 것이라면 **시스템 차원의 odom delta는 여전히
필수**다. 직접 상대 목표 센서만으로 global localization과 field 전술을 모두 폐기하는 별도 stack이
아닌 이상 이 결론은 바뀌지 않는다.

개발 우선순위는 다음과 같다.

- 1차: contact-aided ESKF/InEKF 계열의 해석적 baseline과 공분산 출력
- 2차: 같은 데이터로 AutoOdom형 supervised `Δx,Δy` 모델을 독립 학습·비교
- 최종: replay에서 검증된 learned residual/contact covariance만 filter에 결합하는 hybrid

걷기 PPO reward로 odometry를 함께 학습하는 end-to-end 방식은 우선 경로가 아니다. 같은 simulator
rollout을 데이터로 공유하는 것은 좋지만 loss, checkpoint, 검증 지표는 분리한다.

## 2. 확인한 코드와 기준 revision

반드시 참고하라는 요청에 따라 다음을 직접 추적했다.

- 별도 저장소: `../[07]sim2real`
- 기준 branch: `sim2real`
- 기준 commit: `9ffeb143ba90060604a0dab7ee0d05a3784907cf`
- 현재 checkout `ekay-fix`와 아래 localization 관련 파일의 diff는 없음

주요 파일:

- `INHA-Player/src/brain/src/brain.cpp`
- `INHA-Player/src/brain/src/locator.cpp`
- `INHA-Player/src/brain/src/detection_utils.cpp`
- `INHA-Player/src/brain/src/local_planner.cpp`
- `INHA-Player/src/brain/include/brain.h`
- `INHA-Player/src/brain/config/config.yaml`
- `INHA-Player/src/booster_ros2_interface/msg/{Odometer,LowState,ImuState,MotorState}.msg`
- 이 저장소의 `htwk-gym/deploy/`와 `htwk-gym/resources/K1/`

## 3. 기존 sim2real localization의 실제 데이터 흐름

### 3.1 SDK odom 입력부터 PF까지

`brain.cpp:620`은 외부 bridge가 발행하는 ROS `odometer_state`를 구독한다. sim2real 저장소 안에는
이 토픽 publisher가 없고 subscriber만 있다. `Odometer.msg`는 `float32 x,y,theta`뿐이며 header,
source timestamp, sequence, frame, covariance가 없다.

`Brain::odometerCallback()`(`brain.cpp:1580`)은 다음 순서다.

1. SDK 누적 `(x,y,theta)`를 직전 sample과 차분한다.
2. 직전 raw yaw로 `dx,dy`를 old body frame으로 회전한다.
3. SDK gait odom 전용 scale을 적용한다.
   - forward `×1.30`, backward `×1.31`
   - lateral `×1.17`
   - yaw delta `×1.5`
4. ROS 도착 wall-clock `dt`와 물리 속도 한계로 delta를 clamp한다.
5. 보정된 delta를 `robotPoseToOdom`에 누적한다.
6. `Locator::predict(robotPoseToOdom, dt)`를 호출한다.

`Locator::predict()`(`locator.cpp:235`)는 누적 odom을 다시 차분해 body-frame delta를 만들고 모든
particle을 전파한다. motion noise는 현재 `odom_{x,y,theta}_dev`와 이동거리의 제곱근으로 만든
고정 경험식이다. estimator가 계산한 per-delta covariance를 받을 인터페이스는 없다.

vision correction은 `detection_utils.cpp:357`의 `locator->correct(markers_pf)`에서 수행된다.
posterior pose를 얻은 뒤 `Brain::calibrateOdom()`이

```text
odomToField = PF_pose ⊕ inverse(robotPoseToOdom)
```

를 갱신한다. 즉 현재의 `odomToField`는 아래 설계에서 말하는 `T_MO`, `robotPoseToOdom`은
`T_OB`와 같은 역할이다. 이 frame 분리는 유지할 가치가 있다.

### 3.2 odom callback의 숨은 소비자

같은 callback 뒤쪽에서 다음까지 실행된다.

- `robotPoseToField` 합성 및 TF/`localized_pose` publish
- orientation sentinel check
- second PF `relocalizePredict()`
- drift compensation
- `local_planner->compute(robotPoseToField, robotPoseToOdom)`

`LocalPlanner::compute()`은 field pose로 goal error와 stop condition을 계산하고, 누적 odom history로
body velocity를 추정해 near-controller D term에 쓴다. 반면 `Brain::tick()`은 별도 thread에서 명목상
100 Hz로 cached planner velocity를 `setVelocity()`에 보낸다. 구현은 tick 수행 뒤 고정 10 ms를 더
sleep하므로 실제 주기는 `10 ms + tick 실행시간`이다.

따라서 CUSTOM 진입 후 `odometer_state`가 무수신이면 다음 failure가 동시에 발생한다.

| 대상 | 결과 |
|---|---|
| main/reloc PF | particle motion update 중지 |
| `robotPoseToField` | 연속 전파 중지; usable landmark correction 때만 간헐적으로 jump 가능 |
| TF, localized pose | 새 publish 중지; consumer cache는 last value를 들고 TF는 만료될 수 있음 |
| local planner | goal error, stop 판정, D feedback stale |
| BT output | 마지막 planner command를 계속 보낼 수 있음 |
| sentinel/relocalization | yaw 누적 일부는 low-state에서 계속되지만 판정/predict hook은 정지 |

즉 fake zero odom이나 callback 무수신은 안전한 fallback이 아니다. 현재 PF noise 식은 이동량이 0이면
process spread도 정확히 0이 되어 particle cloud가 고정된다. 향후 delta+Q 계약에서도 zero delta에
근거 없이 작은 Q를 붙이면 **“로봇이 확실히 안 움직였다”는 잘못된 정보**가 된다.

메시지가 계속 오지만 값이 상수인 경우는 더 위험하다. timestamp가 신선해 보여도 PF mean/noise와
planner velocity feedback이 모두 0이 된다. dropout 뒤 SDK 누적값이 reset/jump하면 기존 message에는
reset flag/sequence가 없어 이전 sample과 바로 차분한다. 긴 dropout은 gate 허용량도 `dt`에 비례해
커지므로 큰 역점프가 통과할 수 있다. 새 계약의 `seq`, `epoch`, health가 필요한 이유다.

### 3.3 low_state에서 실제로 얻을 수 있는 것

현재 ROS message는 다음을 제공한다.

- IMU: `rpy[3]`, `gyro[3]`, `acc[3]`
- motor: `q`, `dq`, `ddq`, `tau_est`, temperature, lost/error/rate metadata
- `motor_state_parallel[]`, `motor_state_serial[]`

명시적인 foot force/pressure/FSR/contact bit은 없다. `LowState.msg`에도 header와 source timestamp가
없다. `brain.cpp:2019`의 low-state callback은 도착시각을 쓰며 주석상 약 380 Hz다. 현재는 IMU와
serial q/dq를 mirror하고 parallel motor temperature를 별도 저장하지만 translation estimator는 없다.
ddq/tau_est/lost/reserve는 상태추정에 쓰지 않는다.

### 3.4 현재 CUSTOM deploy 코드는 그대로 재사용할 수 없음

`htwk-gym/deploy/deploy_parameter_walk.py:123`은 `RobotMode.kCustom`으로 전환하고 direct DDS lowcmd를
발행하지만 odometer 구독/생성, ROS Brain 연결, GoalPose target adapter가 없다. 이 코드는 지금
ParameterWalk의 velocity command용 wrapper다.

추가로 timing 문제 하나와 K1 mapping의 P0 검증 gap이 확인됐다.

1. `deploy/utils/timer.py`의 시간은 monotonic clock이 아니라 **low-state callback 횟수 × 0.002 s**다.
   실제 callback이 약 380 Hz거나 packet drop/jitter가 있으면 nominal 500 Hz라고 잘못 가정한다.
   estimator 적분 `dt`로 사용하면 안 된다.
2. K1 resource와 deploy SDK slot의 대응이 검증돼 있지 않다. K1 manual/resource는 22 actuator이고
   `K1_serial.xml`에서 leg는 XML indices `10..21`이다. 현재 deploy config는 `B1JointCnt` 기반 23-slot
   배열, `parallel_mech_indexes=[15,16,21,22]`, policy wrapper는 `[11:]`를 쓴다. SDK가 index 10을
   unused/waist placeholder로 유지한다면 이 layout이 맞을 수도 있으므로 현 시점에 one-index bug라고
   단정할 수는 없다. 다만 그 계약을 보장하는 assert/adapter가 없다. sim2real motion logger도
   23-slot waist/crank 이름을 가정한다. **slice 상수를 재사용하기 전에 실제 K1 packet 길이와
   JointIndex name/sign, parallel/serial 의미를 실기에서 확인하고 boot-time assert해야 한다.** 특히
   ankle이 generalized pitch/roll인지 crank coordinate인지 확인 없이 raw q/dq를 FK에 넣으면 안 된다.

## 4. K1 공식 매뉴얼과 CUSTOM 전제

- `CUSTOM`은 모든 관절 제어권을 개발자에게 넘기는 mode다.
- `rt/odometer_state`와 `ResetOdometry`는 문서상 존재하며 “gait odometry”로 표현된다.
- 매뉴얼은 이 토픽이 CUSTOM에서도 계속 갱신된다고 보장하지 않는다.
- `rt/low_state`는 IMU와 joint state를 제공한다.
- low-level publish가 CUSTOM에서만 유효하다는 문구는 joint command에 대한 것이지 odometer 지원
  보장이 아니다.

따라서 사용자가 확인한 “CUSTOM에서는 odometer state를 사용할 수 없다”를 설계 전제로 둔다.
SDK odom은 native WALK/SOCCER에서 shadow comparison용으로만 사용한다.

## 5. 현재 GoalPose 학습도 위치 폐루프다

[`goal_pose.py`](htwk-gym/envs/K1/goal_pose.py)의 actor observation은

```text
projected gravity, gyro, commands[0:10], gait clock, q, dq, previous action
```

이고, `commands[0:2]`는 simulator ground-truth
`goal_pos_world - base_pos_world`로 control step마다 갱신된다. goal은 4–8초 고정돼도 remaining vector는
50 Hz로 줄어든다. `base_lin_vel`은 critic-only privileged observation이므로 actor 배포 입력은 아니다.

따라서 estimator/PF가 만든 현재 `T_MB`와 map goal `T_MG`로

```text
T_BG = inverse(T_MB) ⊕ T_MG
```

를 계산해 actor에 `(x_B, y_B, heading_error)`를 공급해야 한다. normalization은 이 raw meter/radian
계산 뒤 마지막에만 기존 `×0.5`, `×1/π`를 적용한다.

## 6. 문헌상 선택지와 이번 결정

### 6.1 동시에 학습하는 정책 연구

- [Ji et al. (2022)](https://arxiv.org/abs/2202.05481)는 policy와 base velocity/foot height/contact
  estimator를 같은 rollout에서 학습했다.
- [CTS (2024)](https://arxiv.org/abs/2405.10830)는 concurrent teacher-student로 velocity tracking을
  개선했다.

이 결과들은 순간 velocity/contact 같은 bounded quantity에 대한 근거다. task reward 하나로 장시간
누적 pose와 PF용 uncertainty까지 신뢰성 있게 만들었다는 근거는 아니다.

### 6.2 odometry 세 계열

1. 모델 기반: [contact-aided InEKF](https://arxiv.org/abs/1805.10410),
   [Iterated InEKF](https://arxiv.org/abs/2604.15449),
   [OCELOT](https://arxiv.org/abs/2605.21863)
2. hybrid: [CoCo-InEKF](https://arxiv.org/abs/2605.15122)의 learned contact covariance
3. learned: [LEGOLAS](https://openreview.net/forum?id=VdyIhsh1jU),
   Booster T1에서 검증한 [AutoOdom](https://arxiv.org/abs/2511.18857)

AutoOdom은 1초/50-frame history의 action, velocity command, gyro, q/dq, orientation, 이전 delta를 써서
50 Hz `Δx,Δy`를 예측하고 sim pretrain 뒤 소량 mocap으로 fine-tune한다. 다만 약 20초 trajectory의
자체 dataset 기반 preprint이며, 그 command는 우리의 remaining-goal command와 의미가 다르다.

현재 consensus는 단일 winner가 아니라 filter/hybrid/learned가 공존한다는 것이다. 하지만 성공한
learned odometry도 walking reward에 종속시키지 않고 별도 delta label과 temporal window로 학습한다.
PF가 covariance와 failure state까지 필요하므로 **filter baseline을 먼저 만들고 learned model을
독립 challenger로 두는 것**이 이 stack에는 가장 안전하다.

## 7. estimator와 PF 사이의 계약

### 7.1 frame convention

- `M`: field/map frame
- `O`: 연속적인 local odom frame; vision correction 때 절대 jump시키지 않음
- `B_k`: time `t_k`의 trunk/body frame
- `I`: IMU frame
- `F_L`, `F_R`: left/right foot frame

한 interval의 delta mean을 다음 **Pose2D group-coordinate** 계약으로 고정한다.

```text
ΔT_k = inverse(T_OB(t_k)) ⊕ T_OB(t_{k+1})
δ_k  = [translation_x(ΔT_k), translation_y(ΔT_k), yaw(ΔT_k)]
```

`dx,dy`는 **이동 전 body frame `B_k`**, `dψ`는 CCW positive다. 여기서 `(dx,dy)`는 SE(2) `Log`의
translational coordinate가 아니라 `ΔT` matrix의 실제 translation이다. 이 구분은 회전 중 중요하다.
covariance `Q_k`는 wrapped yaw를 포함한 additive error
`e=[dx_est-dx_true, dy_est-dy_true, wrap(dψ_est-dψ_true)]`의 covariance이며 순서는 `[x,y,yaw]`,
단위는 `m²`, `m·rad`, `rad²`다. 향후 Lie tangent/`Log` 계약으로 바꾸면 mean과 Q를 함께 바꿔야 한다.

권장 transport는 누적 pose만 있는 기존 `Odometer.msg`가 아니라 다음 의미의 stamped delta다.

```text
EgomotionDelta {
  seq, epoch, clock_id, t0, t1,
  dx, dy, dyaw,
  covariance[3x3],
  contact_prob_left/right,
  slip_prob_left/right,
  status = BOOTSTRAP | OK | DEGRADED | INVALID,
  reason_bits
}
```

- `seq`로 exactly-once 적용을 보장한다.
- filter reinit 때 `epoch`을 증가시킨다. 큰 반대 delta로 좌표를 0에 되돌리면 안 된다.
- covariance는 absolute pose marginal 두 개를 단순히 뺀 값이 아니라 **그 interval delta error**다.
- raw sample과 output 모두 동기화된 source timestamp가 최선이다. 현재처럼 없으면 Brain의 공통 ROS
  clock으로 receipt stamp를 부여하고 sequence-gap detection을 쓴다. estimator가 별도 host monotonic
  clock을 쓰면 ROS detection capture stamp와 직접 비교하지 말고 affine clock mapping/PTP와 `clock_id`를
  명시해야 한다. callback count × fixed dt는 금지한다.

### 7.2 PF update

각 particle은 additive Pose2D 계약에 맞춰 다음처럼 전파한다.

```text
ε_i ~ N(0, Q_k + Q_floor)
δ_i = [dx+ε_x, dy+ε_y, wrap(dψ+ε_yaw)]
T_MB_i ← T_MB_i ⊕ Pose2D(δ_i)
```

현재 `Locator::predict()`의 scalar distance-noise model은 source-neutral `predictDelta(δ,Q,dt,status)`로
바꾸는 것이 목표다. residual의 시간상관 때문에 매 20 ms independent Gaussian으로 과신할 수 있으므로
replay에서 covariance floor와 persistent scale/yaw-bias particle model도 비교한다.

runtime `Q_k`는 `P(t_k)`와 `P(t_{k+1})`를 빼서 만들 수 없다. 두 pose error는 강하게 상관돼 있다.
filter output 시각마다 previous pose clone을 state에 유지해 joint block covariance

```text
[ P_kk      P_k,k+1 ]
[ P_k+1,k   P_k+1,k+1 ]
```

를 얻고, relative-pose Jacobian으로 위 additive delta error covariance를 계산하는 것을 기본으로 한다.
동등하게 bias/contact correlation까지 유지하는 preintegrated-delta covariance accumulator를 구현해도
된다. mocap residual은 runtime Q 생성기가 아니라 이 Q의 scale/floor/coverage를 calibration하는 용도다.

### 7.3 vision capture time 정합

현재 sim2real은 capture-time robot-frame marker를 이미 current time까지 전파된 particle에 바로 넣는다.
새 odom buffer만 추가한다고 이 mismatch가 자동으로 해결되지는 않는다. 둘 중 하나를 명시적으로 한다.

1. `t_capture→t_now` odom으로 marker를 current body frame에 옮긴 뒤 current PF를 correct
2. PF를 capture time으로 rewind해 correct하고 그 이후 delta를 exactly-once replay

1번은 변경 범위가 작은 prototype 근사이고, production 기준은 2번 rewind/correct/replay다. 1번에서
원래 vision covariance `R`를 그대로 쓰면 안 된다. marker Jacobian으로 `R`를 회전시키고 capture→now
delta Q도 더해 보수적으로 inflate해야 한다. 이 odom error는 이미 PF prediction에도 들어가 measurement와
state error가 상관되므로 완전한 독립 update는 아니며, replay NEES/coverage에서 일관성을 확인해야 한다.
correction을 current time에 수행하는 prototype이라면

```text
ΔT_capture→now = inverse(T_OB(capture)) ⊕ T_OB(now)
z_Bnow = inverse(ΔT_capture→now) · z_Bcapture
```

로 marker 좌표를 먼저 옮긴다. 그 posterior로

```text
T_MO = T_MB_PF(now) ⊕ inverse(T_OB(now))
T_MB(now) = T_MO ⊕ T_OB(now)
```

로 pose를 만든다. rewind 방식을 택하면 같은 식을 capture time에서 계산한 뒤 delta replay로 현재까지
전파한다. detection stamp와 odom buffer가 같은 clock domain이어야 한다. estimator에 landmark correction을
다시 feedback하지 않아 같은 vision measurement를 중복 사용하지 않는다.

PF가 multimodal이면 arithmetic mean goal을 actor에 보내지 않는다. dominant cluster weight, ESS,
pose covariance/ambiguity를 함께 내고 confidence가 낮으면 stand/search/relocalize로 전환한다.

## 8. 권장 estimator 내부 구조

### 8.1 3D filter, planar output

외부 출력이 SE(2)여도 내부는 다음 3D contact-aided state를 둔다.

```text
R_OB, p_OB, v_OB, gyro bias, accel bias,
left/right stance-foot anchor, contact/slip state
```

- 약 380 Hz low-state마다 gyro/acc로 propagation
- calibrated `T_BI`로 IMU와 trunk frame 정렬
- q/dq의 FK와 Jacobian으로 stance foot world velocity≈0 constraint
- double support는 두 발, single support는 stance foot만 사용
- roll/pitch는 gravity/AHRS를 보조 측정으로 사용
- yaw delta는 gyro+bias가 기본이며 PF landmark가 map yaw를 정함

### 8.2 force sensor가 없는 K1의 contact/slip

contact를 gait phase 하나로 hard-code하지 않는다. 다음을 probability로 결합한다.

- FK foot height와 vertical/lateral velocity
- stance-foot velocity innovation/NIS
- joint q/dq/ddq와 `tau_est` contact proxy
- policy gait phase/action은 약한 prior
- 양발이 주장하는 base velocity의 disagreement
- IMU shock/tilt와 motor packet health

innovation이 큰 발은 slip으로 보고 measurement covariance를 키우거나 reject한다. 사람의 팔 지지,
pickup, kick, fall/get-up을 “command가 0이므로 body velocity도 0”이라는 pseudo-measurement로 누르면 안 된다.

### 8.3 learned track

AutoOdom형 network는 같은 low-state history, previous action, gait phase, previous delta로 `Δx,Δy`와
calibrated uncertainty를 예측한다. yaw는 gyro filter를 공유한다. filter와 learned model은 같은 sensor를
쓰기 때문에 독립 측정처럼 임의 Kalman fusion하지 않는다. 다음 중 하나가 replay에서 이길 때만 결합한다.

- learned residual correction
- learned contact/slip probability 또는 contact covariance
- health-dependent estimator selector

## 9. GoalPose와 A*에 연결하는 방법

map goal `T_MG`를 최신 fused pose로 body frame에 바꾼다. PF/vision update 사이에는 같은 delta로
relative target을 다음처럼 재귀 전파할 수 있다.

```text
r_next = R(-dψ) · (r_now - dp_old_body)
heading_next = wrap(heading_now - dψ)
```

streaming relative measurement가 과거 timestamp의 값이면 그 시점부터 현재까지 odom buffer로 먼저
전파한다. target에는 `goal_id`, `path_version`, timestamp, covariance, source age를 붙인다.

- fresh target + odom degraded: lookahead/속도를 줄여 제한 운용 가능
- odom OK + target 잠깐 stale: last goal을 delta로 짧게 bridge
- PF가 큰 correction/relocalization: 즉시 큰 turn command를 만들지 말고 stand → replan → 새 goal_id
- 공 근처: map goal보다 ball-relative capture measurement/RLKick으로 source를 전환

고정 waypoint를 한 번만 주고 IMU 한 frame과 reward만으로 도달시키는 것은 nominal sim에서 open-loop
timing으로 성공할 수 있어도 slip/push/손 지지를 교정하지 못한다. RNN/history policy의 implicit
odometry는 유효한 ablation이지만 PF용 covariance와 다른 consumer를 대체하지 못한다.

현재 학습 범위는 `dx∈[-2,2]`, `dy∈[-1.5,1.5]`다. 정면 3 m lookahead는 OOD이므로 우선 path point를
이 envelope 안으로 project하고, 실제 3 m가 필요하면 radial 3 m 분포로 재학습한다.

## 10. sim2real 코드에 넣을 구체적 seam

### 10.1 source-neutral egomotion consumer

`Brain::odometerCallback()`의 ROS 입력 처리와 downstream 갱신을 분리한다.

```text
SDK callback ─┐
              ├→ consumeEgomotion(delta, covariance, stamp, status)
leg estimator ┘
```

config에 `robot.odom_source = sdk | leg_filter | learned | hybrid`를 두고 동시에 두 source를 적용하지
않는다. `consumeEgomotion()`은 다음만 담당한다.

1. `T_OB` 누적
2. main/reloc PF `predictDelta`
3. `T_MB = T_MO ⊕ T_OB` 갱신
4. thread-safe pose/health snapshot 발행

기존 SDK scale(x/y/yaw factor)는 `sdk` adapter 안에만 남긴다. 새 estimator source에서는 모두 1.0으로
두고 estimator calibration/Q에 흡수한다. 그렇지 않으면 전진과 yaw를 이중 확대한다.

기존 BT `ResetOdometry`는 Booster gait-odom RPC라 CUSTOM estimator reset이 아니다. source가 SDK가
아닐 때는 estimator rebase/epoch 변경으로 route하고, PF에는 불연속 delta 대신 새 anchor event를 준다.

최소 호환 prototype은 estimator가 누적 `Odometer`를 합성해 기존 callback에 넣는 방식도 가능하다.
하지만 timestamp/covariance/health를 잃고 scale이 중복될 위험이 있어 shadow test용 이상으로 쓰지 않는다.

### 10.2 low-state ingestion

`Brain::lowStateCallback()`에서 packet 전체를 immutable sample로 ring buffer에 넣고 filter worker 또는
작은 `StateEstimator` class가 처리한다. estimator ingestion은 policy inference decimation 안이 아니라
모든 packet에서 해야 한다. worker가 PF/`odomToField`를 직접 수정하면 detection correction과 race가
난다. worker는 delta를 queue/publish하고, 현재 SingleThreadedExecutor의 Brain callback 한 곳이
`consumeEgomotion()`과 `Locator`를 single-writer로 갱신한다. 별도 `Brain::tick()` thread는 immutable
snapshot만 읽거나 명시적 mutex를 쓴다.

필수 sample:

- common ROS clock stamp 또는 명시적으로 mapping된 `clock_id`, sequence/gap
- rpy/quaternion, gyro, acc
- SDK/robot-version별 K1 index table로 변환한 leg q/dq/ddq/tau_est
- parallel/serial source 구분과 motor health
- previous policy action, gait phase, command(learned/contact prior용)

PREP→CUSTOM 전환 중 stable double support에서 bias/contact bootstrap을 마치기 전 gait를 시작하지 않는다.
mode change나 estimator reinit 때 epoch를 올리고 target freshness를 재검증한다.

### 10.3 planner와 safety watchdog

planner 계산을 odom ROS callback 존재 여부에 묶지 않는다. 최신 immutable pose snapshot을 읽는 stable
timer/tick으로 옮기거나, 적어도 estimator output callback에서 source-independent하게 호출한다.

`Brain::tick()`에는 다음 watchdog이 필요하다.

- egomotion/pose age가 예상 period의 여러 배를 넘으면 cached nonzero velocity 전송 금지
- DEGRADED면 velocity/lookahead 제한
- INVALID가 지속되면 GoalPose gait_frequency=0과 neutral stand, 이후 DAMP
- stale target도 같은 방식으로 stop

초기 threshold는 policy 50 Hz 기준 pose age 100 ms stop으로 시작하되, low-state rate/gap histogram과
실기 latency를 보고 조정한다.

### 10.4 GoalPose 전용 deployment wrapper

현재 `policy_thomas.py`는 54차원 모양만 우연히 같고 command 의미가 다르다. 별도 wrapper에서 정확히
다음을 구성한다.

```text
[gravity3, gyro3,
 goal_x, goal_y, heading_error, gait_freq, yawL, yawR, pitch, roll, offsetX, offsetY,
 clock2, K1_leg_q12, K1_leg_dq12, previous_action12]
```

LowState 배열 자체에는 joint name이 없으므로 SDK/robot version별 index table, packet count/schema,
normalization을 boot-time assert하고 1-joint physical test로 semantics를 승인한다. target invalid이면
x/y/yaw만 0으로 만들지 말고 gait frequency도 0으로 만들어 stand시킨다. UI/file polling은 debug
override로만 쓴다.

## 11. health state와 failure 처리

| state | 의미 | PF/보행 동작 |
|---|---|---|
| BOOTSTRAP | bias, joint mapping, double-support 초기화 중 | motion inhibit |
| OK | 정상 contact/innovation/timing | mean+Q로 정상 predict |
| DEGRADED | slip, one/no-contact, 작은 packet gap, 높은 innovation | IMU propagation, Q 급증, 저속/짧은 lookahead |
| INVALID | NaN, 큰 gap, joint mismatch, saturation, filter divergence | fake zero 금지, PF uncertainty 확대, stand |
| RECOVERING | stable double support로 재초기화 | epoch 증가, 새 `T_MO` anchor 후 재개 |

fall/get-up/pickup에서는 현재처럼 무조건 mean propagation을 믿지 않는다. yaw와 가능한 IMU motion만
유지하되 translation Q를 크게 하고, recovery 후 landmark correction/relocalization을 요구한다.
estimator reset은 local `O` epoch만 바꾸며 PF의 map hypothesis를 지우지 않는다.

## 12. 학습과 검증 계획

### Phase 0 — 계약과 timing부터

1. robot/SDK version별 index table을 만들고 실제 K1 packet count/order/sign/unit과 serial/parallel
   의미를 1-joint slow test로 확인
2. ankle crank↔pitch/roll mapping 또는 SDK generalized state 검증
3. `T_BI`, gyro/acc axes, rad/degree, m/s²/g 단위 확인
4. source/arrival timestamp, clock-domain mapping, packet rate, gap, latency histogram 기록
5. 회전을 포함한 random SE(2) composition으로 Pose2D group-coordinate delta/PF/goal recursion unit test
6. delayed detection을 marker-forward-transform 또는 PF rewind/replay했을 때 같은 posterior가 되는지 test

### Phase 1 — 해석적 baseline

1. static multi-pose/temperature gyro·acc bias와 Allan variance
2. double-support zero-motion drift
3. FK stance-foot dead reckoning prototype
4. contact-aided ESKF/InEKF와 slip gating
5. pose clone/preintegration cross-covariance로 runtime delta Q 생성
6. mocap residual coverage/NEES로 Q scale/floor calibration

### Phase 2 — learned challenger

sim에서는 root GT로 50 Hz old-body `Δx,Δy,Δyaw` label을 기록한다. 입력은 low-state sensor proxy,
K1 q/dq, action, gait phase, contact/slip auxiliary label이며 real-calibrated noise/delay/dropout을 넣는다.
실기 mocap/AprilTag trajectory로 autoregressive fine-tune한다. train/calibration/test trajectory를 분리한다.

### Phase 3 — PF replay가 교체 gate

동일 landmark detection log에 다음 세 입력을 replay한다.

1. mocap GT delta
2. estimator mean + 기존 고정 motion noise
3. estimator mean + calibrated per-delta Q

비교 지표는 ATE/RPE만이 아니라 PF map error, ESS, resample rate, landmark innovation, multimodal recovery,
kidnap/fall recovery, estimator reset·duplicate·out-of-order fault다. learned/hybrid는 이 replay에서 filter
baseline을 이겨야 control path에 들어간다.

### Phase 4 — closed loop와 실기 순서

- sim: reward는 true goal, actor는 estimated/noisy goal을 사용해 5 cm/10 cm/heading/hold gate 저하 측정
- native WALK: official odom과 새 estimator를 mocap 기준 shadow 비교
- CUSTOM tethered stand: estimator shadow, motion control에는 미사용
- CUSTOM short straight/lateral/backward/turn/arc/stop-go
- PF shadow → PF motion update enable → GoalPose target enable → A*/BT/RLKick handoff
- push, low-friction slip, 손 지지/release, floor/battery/day 변화를 별도 test set으로 유지

필수 log는 raw low-state, timestamp와 clock_id/mapping, action/command, estimator state/delta/Q/health, PF particles/mode,
`T_MO/T_OB/T_MB`, relative target, actor observation/action이다.

## 13. acceptance 기준

estimator 평균 RMSE 하나로 통과시키지 않는다.

- 20 ms와 1 s window delta RPE, 0.5/1/2/3 m endpoint/yaw error
- contact/slip precision-recall과 recovery time
- whitened delta error, NEES, 90/95% covariance coverage reliability
- landmark blackout 동안 PF drift와 posterior ESS
- PF correction 후 goal jump/overshoot와 relocalization 성공률
- estimated-target GoalPose의 도달·1 s hold·RLKick handoff 성공률
- packet loss, reset, NaN, joint mismatch를 INVALID로 잡고 stale command를 실제로 stop하는지

최종 허용 오차는 “odom 논문의 숫자”가 아니라 downstream budget으로 정한다. 가장 긴 정상 landmark
blackout에서 odom p90 오차가 A*/RLKick capture margin의 절반을 넘지 않게 하고, 넘으면 lookahead 축소나
stand를 선택한다.

## 14. 요약

- CUSTOM에서 native odom이 끊기면 현재 sim2real은 PF, field pose의 연속 전파, TF, planner가 멈춘다.
  landmark correction은 간헐적으로 data pose를 바꿀 수 있지만 navigation loop를 대체하지 못한다.
- 그래서 streaming lookahead만으로는 현재 축구 stack을 살릴 수 없다.
- `low_state`로 contact-aided local odom을 만들고, old-body Pose2D delta와 cross-covariance를 보존해
  계산·calibration한 per-delta Q/health를 PF에 제공해야 한다.
- vision PF는 `M→O`를 보정하고 local estimator의 `O→B`는 연속으로 유지한다.
- detection capture time과 odom clock을 맞춘 뒤 marker forward-transform 또는 PF rewind/replay를 한다.
- 기존 SDK odom scale은 새 estimator에 적용하지 않는다.
- 현재 CUSTOM deploy의 callback-count time은 고쳐야 하고, 23-slot slice가 실제 K1 SDK mapping과
  일치하는지는 versioned index table, packet/schema assert, 1-joint physical test로 먼저 검증해야 한다.
- 구현은 filter baseline → learned challenger → PF replay gate 순서로 진행한다.
