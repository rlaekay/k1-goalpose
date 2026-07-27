# CUSTOM odometry 계획 — 합성 데이터, 실측 calibration, PF/BT 통합

> `MASTERPLAN.md`의 동반 문서. K1 CUSTOM mode에서 Booster gait odometry를 사용할 수 없다는
> 전제 아래 odom을 별도 모듈로 만든다. 2026-07-25 전면 재작성.
>
> 담당 경계: 이 문서와 odometry/state-estimation 구현은 Codex 담당이다. Locomotion policy와
> 그 학습 코드는 Claude 담당이며, odom 데이터 수집 때문에 locomotion 코드를 임의로 수정하지 않는다.
> 충돌이 생기면 `MASTERPLAN.md`에 `[codex/odometry]`로 남기고 합의한 뒤 진행한다.

## 1. 결론과 이전 문서의 정정

방법론 검증은 **새 실측 데이터 없이 지금 시작할 수 있다.** 현재 policy를 Isaac Gym에서 rollout해
ground-truth body-frame delta를 만들고, 기존 실측 로그로 IMU/joint/timing 분포를 맞춘다. 동시에 기존
30개 session을 정합해 SDK odom을 약한 라벨로 쓰는 real-domain smoke test를 한다.

하지만 production 승인에는 다음 데이터가 반드시 새로 필요하다.

```text
동일 clock의 CUSTOM walk
  + raw IMU
  + q/dq/(가능하면 tau_est)
  + 실제 policy action/관절 target
  + command/gait phase
  + MoCap 6DoF ground truth
```

현재 폴더에는 `IMU+leg kinematics`와 MoCap이 동시에 기록된 session이 **0개**다. 따라서 지금 가진
데이터만으로 pipeline과 학습 방법은 검증할 수 있지만, custom odom의 실제 정확도·장기 drift·PF용
covariance가 맞다고 판정할 수는 없다.

이 요구를 이전 문서에 명시하지 않은 것은 누락이었다. 문헌 해석도 두 군데 고친다.

- AutoOdom은 `dx,dy`만 출력한다. yaw와 uncertainty/covariance는 출력하지 않으며 uncertainty는
  future work다.
- AutoOdom의 실측 학습량은 공개되지 않았다. trajectory 평균이 약 20초라고만 되어 있으므로
  “소량의 mocap이면 충분하다”고 정량적 사실처럼 쓸 수 없다.
- CTS의 약 20% 개선은 odometry가 아니라 concurrent teacher-student locomotion의 velocity tracking
  결과다. odom 동시학습의 근거로 쓰지 않는다.
- OCELOT과 Iterated InEKF의 실기 구성은 foot-force/GRF로 stance를 정한다. force topic이 없는 K1에
  그대로 적용할 후보가 아니다.

채택하는 개발 순서는 다음이다.

1. 데이터 clock/frame/joint 계약과 gap-safe 전처리
2. Isaac Gym 대규모 합성 GT와 기존 실측 분포 기반 augmentation
3. 해석적 `gyro + stance-FK` baseline
4. 같은 rollout로 `direct delta`와 `velocity→filter` 두 learned challenger 비교
5. 기존 real log에서는 pseudo-label shadow test만 수행
6. 새 paired MoCap pilot에서 zero-shot, residual, covariance를 검증
7. PF replay를 통과한 모델만 BT control path에 연결

Locomotion reward와 odom loss는 분리한다. 같은 rollout과 GPU를 공유할 수는 있지만 odom checkpoint,
optimizer, validation split, go/no-go gate는 별도다.

## 2. 사용자에게 필요한 데이터와 정보

### 2.1 지금 시작하는 데 새 녹화는 필요 없다

P0~P2의 합성 데이터 생성과 prototype 학습은 현재 파일로 시작할 수 있다. 다만 아래 metadata가 이미
다른 곳에 있다면 파일 위치나 값을 받아야 한다. 없으면 P0에서 실기 확인 항목으로 남긴다.

- K1 robot/SDK/firmware version과 `motor_state_serial`, `motor_state_parallel`의 index/name/sign/unit 표
- 23-slot 로그와 12-DoF locomotion URDF 사이의 정확한 joint mapping
- 각 실측 session의 robot mode, 사용한 walk 종류, policy/checkpoint, control Hz와 decimation
- `robot_mode` 숫자 1, 2, 12, 20 등의 의미
- MoCap rigid body가 trunk 어디에 붙었는지 나타내는 `T_body_marker`
- 실제 IMU 위치·축을 나타내는 `T_body_imu`, quaternion 순서와 가속도 부호 convention
- MoCap/robot/host clock 동기화 방법과 source timestamp의 의미
- 손 지지, pickup, fall, reset, 바닥, 배터리 상태를 기록한 별도 메모가 있다면 그 원본

이 정보가 없다고 합성 smoke test를 막지는 않는다. 대신 mapping/extrinsic이 필요한 실기 accuracy
주장은 보류한다.

### 2.2 production 전에 새로 받아야 할 paired pilot

모든 channel을 **한 recorder와 한 clock 계약**으로 기록한다.

| 분류 | 필수 channel |
|---|---|
| 시간 | source stamp, host receive stamp, sequence, clock id, mode/epoch |
| IMU | quaternion 또는 rpy, gyro, acc, 가능하면 temperature와 sensor status |
| 관절 | serial과 parallel 원본, q/dq, 가능하면 ddq/tau_est, lost/error/rate |
| 제어 | policy raw action, clip/scale 뒤 실제 joint target, 직전 action, command/relative goal, gait phase/frequency |
| GT | MoCap 6DoF pose, timestamp, tracking quality, rigid-body definition |
| 비교용 | native SDK odom이 나오는 mode에서는 함께 기록하되 GT로 쓰지 않음 |
| 사건 | stand/walk transition, reset, fall/get-up, pickup, 손 지지/release, push/slip, intervention |
| 재현성 | robot/SDK version, joint map, policy hash/config, floor, shoes/feet, battery |

첫 pilot의 engineering target은 20~30분이다. 논문에서 보장한 최소량이 아니라 coverage를 확보하기 위한
초기 계획이며, zero-shot residual을 본 뒤 늘린다.

- 정지: 여러 heading/roll/pitch 자세와 온도 구간으로 합계 2~5분
- 전진/후진, 좌/우 횡보, 양방향 제자리 회전: 낮음/중간/높음 속도 각 3~5회
- diagonal, arc, start-stop, stand↔walk, 감속 후 stand 각 3~5회
- 손 지지, 작은 push, 의도한 짧은 slip, 저마찰 또는 다른 바닥은 별도 trial
- 가능하면 native walk와 final/nearly-final CUSTOM policy를 같은 course에서 각각 수행

train/calibration/test는 frame이나 overlapping window를 무작위로 나누지 않는다. 날짜·trial·바닥을
통째로 분리한다. 첫 paired dataset은 가능하면 전부 untouched zero-shot test로 먼저 보존하고,
zero-shot이 실패할 때만 일부 session을 calibration/fine-tune으로 승격한다.

상시 foot-force sensor는 요구하지 않는다. contact 판정 검증용 subset에만 side video, pressure insole,
force plate 중 하나가 있으면 좋다. 이것은 optional validation label이지 production input이 아니다.

## 3. 현재 데이터 전수 점검

### 3.1 label의 의미부터 분리한다

| 이름 | 이 문서의 의미 |
|---|---|
| GT label | 동기화된 MoCap/root state에서 계산한 delta |
| weak/pseudo label | native SDK odom이나 PF pose에서 만든 delta |
| auxiliary label | simulator contact/force/slip/stationary 등 학습 보조값 |
| unlabeled | sensor 입력만 있고 translation GT가 없는 데이터 |

SDK odom은 비교 대상이지 GT가 아니다. PF `field_pose`는 vision correction과 기존 odom을 이미 포함하므로
학습 GT로 쓰면 순환 라벨이 된다.

### 3.2 `[log]velocity/log_odom`

ROS2 bag의 topic과 크기는 다음과 같다.

- `/odometer_state`: 73,681 messages, 약 147.63초
- `/localized_pose`: 34,131 messages, 약 148.70초
- `/booster_vision/detection`: 1,355 messages
- low_state, IMU, q/dq, policy action: 없음

TUM 파일은 다음과 같다.

- native odom `pose.tum`: 73,681 rows지만 source timestamp unique는 5,906개, 실제 약 40 Hz
- MoCap `vrpn_client_node_Tracker0_pose.tum`: 13,520 rows, 중앙값 약 100 Hz
- PF `localized_pose.tum`: 34,131 rows; pose가 실제로 바뀐 row는 13.62%

MoCap과 SDK odom trajectory를 비교하는 baseline에는 쓸 수 있다. 실제 overlap에서 큰 MoCap gap을
제외한 native-odom body velocity는 대략 `vx RMSE 0.146 m/s, corr 0.942`,
`vy RMSE 0.146 m/s, corr 0.729`, yaw-rate는 `RMSE 0.635 rad/s, corr 0.666`이었다. 그러나 이
session에는 estimator 입력인 IMU와 q/dq가 없으므로 `IMU+leg → MoCap delta`의 학습이나 평가에는
쓸 수 없다. MoCap에도 2.29초 gap과 짧은 tracker jump가 있어 GT mask가 필요하다.

### 3.3 `[log]velocity/sysid_logs`

- 31 sessions
- valid 265,827 rows, 약 17,062초 span
- schema 16열: timestamp, command 3, SDK odom 3, gyro 3, acc 3, rpy 3
- q/dq, action, gait phase, torque, contact, MoCap은 없음
- 일부 파일에 잘린 종료행과 NUL byte가 있으므로 strict parser가 필요

중요한 전처리 문제도 확인했다. 전체에서 `dt` 중앙값은 약 27.7 ms지만 p95는 약 196.8 ms이고,
100 ms 초과 gap이 86,130개다. 현재 `velocity_estimator.py`는 큰 gap에서 segment를 끊지 않고 전체
시간축을 `np.interp`하므로 약 265k 실측 row를 약 1.7M grid row로 늘리며 미관측 구간에 가짜 완만한
속도를 만든다. 이 결과와 그 위에서 fit한 model을 custom odom label로 그대로 재사용하지 않는다.

새 전처리에서는 다음을 강제한다.

```text
parse validation → source-time sort/deduplicate → clock fit
→ gap threshold로 segment 분리 → reset/jump mask
→ 각 segment 안에서만 resample/window 생성
```

### 3.4 sibling `[log]motion/motion_logs`

- 172 CSV, valid 3,109,736 rows, 합산 span 약 87,171초(24.2시간)
- IMU rpy/gyro/acc, q23/dq23, native odom 3, PF field pose 3, event/mode metadata
- command, policy action/target, gait phase, tau/contact, MoCap, 공통 absolute source stamp는 없음
- 파일별 median `dt`가 약 2 ms부터 647 ms까지 달라 단일 rate를 가정할 수 없음
- `r_crank_dn_q/dq`는 약 310만 numeric row 전체에서 0이라 dummy 또는 미측정 slot일 수 있음

이 데이터는 IMU bias/noise, q/dq 범위, timing/dropout, mode별 분포, simulator distribution matching에
유용하다. `field_pose`와 native odom은 weak label 이상으로 취급하지 않는다. FK 전에 실제 joint mapping을
검증한다.

### 3.5 지금 만들 수 있는 real pseudo-labeled dataset

31개 sysid 중 30개는 대응하는 motion CSV가 있다. IMU 6축 값을 이용해 affine clock mapping을 맞추면
대부분의 session에서 slope가 거의 1이고 residual p99가 약 1 ms 이내다. 따라서 다음 join은 가능하다.

```text
sysid: command + SDK odom + IMU
              ↕ IMU cross-match + affine clock fit
motion: q/dq + IMU + mode/event
```

이렇게 만든 30-session dataset은 다음 용도로 쓴다.

- parser, synchronization, window, gap/reset mask 검증
- `command+IMU+q/dq → SDK odom delta` student의 end-to-end smoke test
- feature/history/model ablation과 real input distribution 검사
- sim-trained model의 shadow output이 터지지 않는지 확인

이 dataset이 증명하지 못하는 것은 명확하다.

- SDK odom보다 정확한지
- CUSTOM mode에서 translation drift가 얼마인지
- slip/contact failure를 잡는지
- predictive covariance가 실제 error를 덮는지
- PF와 BT를 안전하게 구동할 수 있는지

## 4. 시스템이 실제로 필요로 하는 것

이 절의 판단은 `../[07]sim2real`의 `sim2real` branch, commit
`9ffeb143ba90060604a0dab7ee0d05a3784907cf`를 기준으로 다음 파일을 직접 추적한 결과다.

- `INHA-Player/src/brain/src/brain.cpp`
- `INHA-Player/src/brain/src/locator.cpp`
- `INHA-Player/src/brain/src/detection_utils.cpp`
- `INHA-Player/src/brain/src/local_planner.cpp`
- `INHA-Player/src/booster_ros2_interface/msg/{Odometer,LowState,ImuState,MotorState}.msg`

K1 매뉴얼/API에는 `rt/odometer_state`와 `rt/low_state`가 있지만 CUSTOM에서 gait odometry가 계속
갱신된다는 보장은 없다. `LowState`에는 IMU와 serial/parallel motor state가 있고 명시적인 foot
contact/force/pressure field는 없다. 따라서 contact topic을 가정하지 않고 proprioception에서 추론한다.

### 4.1 delta만 estimator가 만들고 pose는 consumer가 누적한다

Estimator가 field 기준 장기 pose를 직접 회귀할 필요는 없다. 매 interval의 local egomotion과 uncertainty를
만들면 된다.

```text
ΔT_k = inverse(T_OB(t_k)) ⊕ T_OB(t_{k+1})
δ_k  = [dx, dy, d_yaw]
```

`dx,dy`는 이동 전 body frame `B_k`, yaw는 CCW positive다. local odom consumer가 이를 `T_OB`로
누적하고, landmark PF가 `T_MO`를 보정한다.

```text
T_MB = T_MO ⊕ T_OB
```

그러므로 learned model의 핵심 label은 장기 global pose가 아니라 20 ms 또는 정한 output interval의
old-body-frame delta다. 장기 drift는 이 delta를 sequence로 누적해 평가한다.

### 4.2 PF와 BT 때문에 delta는 반드시 필요하다

현재 sim2real의 SDK `odometer_state` callback은 다음을 한꺼번에 수행한다.

1. 누적 local odom 갱신
2. main/relocalization PF motion update
3. `robotPoseToField` 연속 전파
4. TF와 `/localized_pose` 발행
5. local planner의 goal error, stop 판정, velocity feedback 갱신

CUSTOM에서 native odom 값이 상수이거나 callback이 사라지면 PF뿐 아니라 planner와 pose publish까지
stale해진다. 이를 “끊김”이라고 부르는 이유는 통신 dropout만 걱정한다는 뜻이 아니라, CUSTOM 진입
후 **odom source가 영구적으로 사라지는 상태**를 말한다. fake zero odom은 로봇이 확실히 정지했다는
잘못된 정보가 되므로 대안이 아니다.

A*가 0.5~1 m lookahead만 계속 주더라도 field-map pose와 obstacle/landmark transform을 유지하려면
system-level delta가 필요하다. 반대로 모든 전술을 camera-relative reactive stack으로 바꾸는 경우에만
장기 local pose를 버릴 수 있는데, 이것은 현재 scope가 아니다.

### 4.3 GoalPose 입력

현재 actor의 `commands[0:2]`는 simulator GT의
`goal_pos_world - base_pos_world`로 매 control step 갱신된다. 실기에서는 최신 fused pose로

```text
T_BG = inverse(T_MB) ⊕ T_MG
```

를 계산해 `(goal_x, goal_y, heading_error)`를 준다. actor의 `base_lin_vel`은 critic-only라 배포 입력이
아니다. map target이 잠시 stale해도 local delta로 짧게 전파할 수 있지만, PF ambiguity가 크면 평균 goal을
억지로 주지 않고 stand/search/relocalize로 전환한다.

현재 production 범위는 `dx∈[-2,2]`, `dy∈[-1.5,1.5]`다. Claude 측 3 m 실험은 아직 production
승격 전이므로 A* lookahead는 당분간 이 envelope 안으로 제한한다. GoalPose goal-noise mechanism은 이미
actor observation에만 jitter/bias/staleness를 넣는 구조로 구현돼 있으므로, paired real residual이 나오면
그 분포를 Claude에게 전달해 수치만 맞춘다. sustained 1초 hold gate는 현재 eval에 없으므로 RLKick handoff가
실제로 이를 요구하면 별도 locomotion 합의 항목이다.

## 5. 논문 원문에서 가져갈 것과 버릴 것

| 방법 | 원문에서 실제로 한 일 | 이번 계획의 판단 |
|---|---|---|
| [LEGOLAS](https://proceedings.mlr.press/v270/wasserman25a.html) | 같은 locomotion policy의 Isaac Gym data 13.7M samples, 50 Hz 1초 history; incremental SE(3), 성분별 variance, stationary head; real fine-tune 없음 | direct-delta challenger의 주 근거. policy가 바뀌면 다시 rollout해야 함. variance를 바로 calibrated PF Q라 부르지는 않음 |
| [AutoOdom](https://arxiv.org/abs/2511.18857) | Isaac Gym stage 후 real MoCap fine-tune; 50 Hz `dx,dy`; real에서 accel과 이전 1초 누적 예측을 feedback | Booster 계열 참고. yaw/Q 없음, real 총량 미공개, input 차원 표기도 불일치하여 그대로 복제하지 않음 |
| [GAIT](https://arxiv.org/abs/2606.14160) | force/contact input 없이 proprioception으로 body velocity와 diagonal covariance를 예측해 IEKF에 사용; sim-only train | force-less velocity→filter challenger의 가장 직접적인 최신 근거. 단 단일 Go1 preprint이고 direct delta가 아님 |
| [CoCo-InEKF](https://arxiv.org/abs/2605.15122) | gyro/acc, q/dq, torque, FK 후보점으로 contact-velocity covariance를 학습; differentiable InEKF | 유력한 2차 hybrid. 600 Hz·1280 env·최대 5일 학습이므로 첫 smoke에는 무거움 |
| [Hartley contact-aided InEKF](https://arxiv.org/abs/1805.10410) | IMU, encoder/FK와 binary contact indication으로 pose/velocity/bias/contact anchor 추정 | 해석적 baseline 골격. K1용 contact frontend를 별도로 만들어야 함. global x/y/yaw는 원래 unobservable |
| [OCELOT](https://arxiv.org/abs/2605.21863) | per-foot force sensor와 kinematic GLRT를 결합해 ESEKF measurement covariance 조정 | force sensor 없는 K1에는 as-is 적용하지 않음. GLRT/slip gate만 참고 |
| [Youm NMN](https://arxiv.org/abs/2402.00366) | sim GT로 contact probability와 body velocity 학습; force input 없음 | auxiliary contact/velocity head 참고. 과도한 dynamics randomization이 악화된 ablation 때문에 DR을 단계별로 검증 |
| [Lin et al.](https://proceedings.mlr.press/v164/lin22b.html) | force input 없이 kinematic self-label/real data로 contact representation 학습 | contact topic이 없어도 추론 가능하다는 근거. robot-specific validation은 여전히 필요 |

LEGOLAS의 정확한 1-frame 입력은 `q12, qdot12, previous desired action12, command3, gyro3,
roll/pitch2 = 44D`이고 accelerometer는 없다. 독립 uniform noise만 사용했으며 bias random walk, latency,
jitter, dropout까지 다루지는 않았다. 따라서 그대로 베끼지 않고 K1 실측 분포를 추가한다.

AutoOdom stage 1은 simulator acceleration mismatch 때문에 acceleration을 제외하고 noise도 넣지 않았다.
실측 fine-tune stage에서만 acceleration을 추가했다. 이 결과를 따라 첫 direct-delta 모델도 accel 없이
시작하고, Isaac Sim/Lab과 paired real calibration이 준비된 뒤 accel ablation을 연다.

Walking policy와 state estimator를 같은 rollout에서 학습한
[Ji et al.](https://arxiv.org/abs/2202.05481)의 출력도 base velocity, foot height, contact처럼 bounded된
값이다. 장기 pose와 PF covariance의 근거는 아니다. 따라서 odom을 PPO reward에 종속시키지 않는다.

## 6. canonical dataset 계약

Gym, Lab/Sim, pseudo-labeled real, paired real이 모두 같은 schema로 변환돼야 한다. 원본은 절대 덮어쓰지
않고 derived dataset에는 converter version과 source hash를 남긴다.

### 6.1 공통 key와 mask

```text
source_id, session_id, episode_id, env_id
seq, epoch, clock_id, t_source, t_receive, dt
valid, reset, gap, fall, pickup, mode_transition
policy_hash, policy_mode, simulator, randomization_seed
```

### 6.2 estimator 입력

```text
imu_quaternion/rpy, imu_gyro[3], imu_acc[3]
leg_q[12], leg_dq[12], optional leg_tau_est[12]
previous_action[12], applied_joint_target[12]
command_or_relative_goal, gait_phase_sin_cos, gait_frequency
motor_validity, packet_age
```

원본 23-slot serial/parallel 배열도 debug column으로 보존하고, versioned adapter가 12-DoF generalized
coordinate를 만든다. raw 배열 slice를 FK에 바로 넣지 않는다.

### 6.3 GT와 auxiliary label

```text
label_delta_body = [dx, dy, d_yaw]
label_body_velocity = [vx, vy, vz, wx, wy, wz]
label_stationary
label_contact_left/right
label_slip_left/right
label_valid

debug_gt_world_position/quaternion
debug_gt_world_linear/angular_velocity
debug_foot_force_left/right
debug_domain_randomization_parameters
```

합성 label은 다음 식 하나로 고정한다.

```text
delta_xy_Bk = Rz(yaw_k)^T · (p_W,k+1 - p_W,k)
delta_yaw   = wrap(yaw_k+1 - yaw_k)
```

Contact force와 simulator contact bit는 **label/debug에만** 둔다. production input에 넣으면 CUSTOM에서
재현할 수 없다.

### 6.4 split과 leakage 방지

- 동일 episode의 overlapping window가 train과 validation에 갈라지지 않음
- simulator는 terrain, dynamics seed, command sequence, policy checkpoint 기준으로 holdout
- real은 session/day/floor 단위 holdout
- normalization 통계는 train split만 사용
- MoCap zero-shot set을 augmentation 범위 선택에 반복 사용하지 않음; calibration과 final test 분리

## 7. 시뮬레이터 증강 계획

### 7.1 세 제품을 세 독립 물리엔진처럼 취급하지 않는다

현재 repo는 Isaac Gym Preview 4 task다. Isaac Lab은 Isaac Sim 위의 framework이고 Gym/Lab/Sim 모두
PhysX 계열이므로 “세 simulator 데이터를 섞었다”는 것만으로 현실 domain diversity가 생기지 않는다.

| 단계 | 역할 | 이유 |
|---|---|---|
| Isaac Gym | 즉시 사용할 primary 대규모 generator | 현재 final policy와 environment를 그대로 50 Hz rollout 가능 |
| Isaac Lab/Sim | sensor/timing/USD 구현을 가진 challenger와 cross-implementation holdout | high-rate IMU/contact sensor, ROS2 timestamp, richer sensor pipeline 검증 |
| 별도 engine | 필요할 때만 Newton/MuJoCo 등의 physics OOD test | PhysX 공통 bias가 의심될 때 추가하며 P0 blocker가 아님 |

`k1/k1.usd`에는 trunk의 `IsaacImuSensor` prim이 있지만 현재 repo에 Lab task/runtime는 없고 일부 Omni
material dependency도 정리되지 않았다. 따라서 Gym을 기다리지 않고 먼저 쓴다.

### 7.2 Gym primary generator

현재 `goal_pose.py`의 physics step은 0.002초, decimation 10이므로 control/output은 50 Hz다. environment에는
root pose/velocity, q/dq, contact force, foot state가 이미 GPU tensor로 있다. 별도 odom collector가
eval loader로 policy/environment를 불러 `env.step()` 전후 tensor를 복사한다. 이 방식은 locomotion
source를 수정하지 않는다.

첫 synthetic budget은 engineering smoke 기준으로 잡는다.

- P1-smoke: 1~2M valid 50 Hz transitions
- P1-scale: 결과가 좋으면 10~15M transitions; LEGOLAS의 13.7M과 비슷한 order
- final policy/checkpoint가 바뀌면 적어도 smoke set을 다시 만들고 policy-specific degradation 확인

500 Hz substep logging을 현재 `step()` 안에 임의로 삽입하지 않는다. high-rate IMU가 필요해지면 독립
Isaac Sim/Lab generator에서 만든다. Gym의 synthetic accelerometer는
`R_WB^T(a_W-g_W)`로 만들고 stationary body-z가 실제 로그의 약 `+9.81 m/s²` convention과 맞는지 먼저
검증한다. 초기 direct-delta model에는 accel을 넣지 않는다.

### 7.3 command와 motion coverage

무작위 goal만 던져 평균적으로 걷게 하지 않는다. 조건별 quota를 둔다.

- stationary, stand↔walk, 감속과 완전 정지
- 전진/후진, 좌/우 lateral, diagonal
- CW/CCW turn-in-place와 다양한 radius의 arc
- 낮음/중간/높음 속도, 0-crossing, rapid command change
- push, hand-support에 해당하는 external wrench, pickup/fall/recovery mask
- flat, slope, small step, low/high friction과 foot slip
- production goal envelope와 A* lookahead 분포

평가 set에는 train보다 강한 일부 조건을 남겨 OOD health detection을 본다.

### 7.4 실측으로 정하는 augmentation

각 augmentation parameter를 무작정 넓히지 않는다. 먼저 단일-factor ablation으로 sensitivity를 보고,
실측 로그에서 근거가 있는 범위를 결합한다.

| 층 | augmentation |
|---|---|
| IMU episode-persistent | gyro/acc bias, scale, axis misalignment, `T_BI` 오차 |
| IMU temporal | white noise, bias random walk, bandwidth/low-pass, saturation |
| timing | sample jitter, packet hold/drop/burst gap, source/receive latency, clock offset/drift |
| joint | q/dq noise, quantization, delay, missing slot, joint zero/sign 오차의 fail-fast test |
| actuator | applied target delay, PD gain, motor strength, backlash/ankle mapping uncertainty |
| rigid body | mass, COM, inertia, payload/arm support external wrench |
| contact | friction, restitution, compliance, terrain height, stance slip |
| operation | battery/day/floor, stand/walk transition, push/pickup/fall masks |

Bias, scale, delay처럼 session 동안 지속되는 오차는 sample마다 독립 재추출하지 않고 episode-persistent로
둔다. dropout도 독립 Bernoulli만이 아니라 실측 burst-length 분포를 보존한다. `[log]motion`과 sysid의
session별 rate/gap histogram, stationary IMU PSD/Allan, q/dq/action tracking 분포를 P0에서 다시 산출해
범위를 고정한다. 기존 README의 “항상 485 Hz/71~75 Hz” 숫자는 전체 dataset에 일반화하지 않는다.

Unlabeled real q/dq/action marginal을 synthetic와 비교하는 proprioceptive distribution matching은
simulator parameter 우선순위를 정하는 데 쓸 수 있다. 이것은 GT odom을 만드는 방법이 아니라 sim
input distribution을 현실에 가깝게 하는 보조 절차다.

## 8. 비교할 estimator

### 8.1 F0 — 해석적 baseline

```text
IMU propagation
  + gyro/acc bias
  + FK/Jacobian stance-foot zero-velocity constraint
  + contact/slip probability에 따른 measurement covariance
  → 3D pose/velocity state
  → 50 Hz SE(2) delta + Q + health
```

Internal state는 `R,p,v,gyro bias,acc bias,left/right foot anchor`를 둔다. yaw delta는 gyro+bias가
기본이고 map yaw는 PF landmark가 보정한다. K1에 force/contact topic이 없으므로 contact를 hard-coded
gait phase 하나로 정하지 않는다.

- FK foot height/velocity
- q/dq와 가능하면 tau_est
- gait phase/action은 약한 prior
- 양발이 주장하는 base velocity disagreement
- stance-foot innovation/NIS
- IMU shock, tilt, packet/motor health

큰 innovation의 발은 slip으로 보고 update covariance를 키우거나 reject한다. tau_est가 없으면 모델은
그 입력 없이 동작해야 한다.

### 8.2 L0 — LEGOLAS-lite direct delta

첫 learned model은 exact deployed policy의 1초 history를 받아 50 Hz `dx,dy`와 stationary probability를
출력한다. yaw는 F0 gyro+bias를 공유한다. 입력 baseline은 다음이다.

```text
q12, qdot12, previous desired action12,
command/relative-goal summary, gyro3, roll/pitch2
```

PF에 Q가 필요하므로 원 논문을 넘어 `dx,dy` covariance head를 붙이는 실험은 가능하다. 다만 NLL로
학습했다는 이유만으로 calibrated라고 부르지 않는다. held-out sim과 paired real calibration에서
coverage/NLL/NEES를 맞춘 뒤에만 Q로 사용한다. 실패하면 runtime Q는 empirical condition bin과
covariance floor로 만든다. 초기 3×3 Q는 검증된 planar `Q_xy`와 F0 yaw variance의 block diagonal로
구성하고, x/y/yaw cross term은 paired real residual로 정당화되기 전에는 임의로 학습값을 쓰지 않는다.

Loss는 one-step MSE 하나가 아니다.

```text
per-step old-body delta NLL/MSE
+ stationary/contact auxiliary loss
+ 0.1/0.5/1/2 s SE(2) composition loss
+ autoregressive rollout drift loss
+ covariance calibration penalty(on calibration split only)
```

### 8.3 V0 — GAIT-lite velocity measurement

두 번째 challenger는 `gyro,acc,q,qdot,q_des-q` history로 body velocity와 uncertainty를 예측하고
IEKF pseudo-measurement로 넣는다. direct delta와 달리 IMU/filter dynamics를 명시적으로 보존한다.
첫 단계에서는 simulator acceleration fidelity를 별도 검증하고, acceleration convention이 맞지 않으면
사용을 보류한다.

### 8.4 H1 — hybrid는 뒤에 둔다

CoCo-InEKF식 learned contact-velocity covariance나 learned residual은 F0/L0/V0 비교 뒤에 연다. 같은
sensor에서 나온 두 estimator를 독립 측정처럼 임의 Kalman fusion하지 않는다. 결합 후보는 다음뿐이다.

- learned contact/slip probability 또는 contact covariance를 F0에 제공
- F0 mean에 condition-dependent learned residual 적용
- health에 따른 selector

Corrected mean과 covariance가 일치하지 않으면 PF에 넣지 않는다.

## 9. 학습·검증 단계와 go/no-go

### P0 — 데이터와 좌표계, 지금 수행 가능

1. sysid/motion 30-session affine join manifest 생성
2. NUL/잘린 row 제거, deduplicate, gap segmentation, reset/jump mask
3. joint index/name/sign/unit과 serial/parallel mapping 검증
4. IMU axis/unit/gravity convention, `T_BI`, MoCap `T_BM` metadata 확인
5. session별 rate, jitter, burst gap, stationary bias/PSD/Allan 산출
6. old-body delta composition과 frame unit test

Gate: 30개 join의 clock residual, segment mask, 12-DoF mapping이 보고서와 sample plot으로 재현돼야 한다.
현재 `velocity_estimator.py`의 full-span interpolation은 이 gate를 통과하지 못한다.

### P1 — Isaac Gym synthetic, 지금 수행 가능

1. locomotion 코드를 수정하지 않는 외부 collector
2. P1-smoke 1~2M transitions 생성
3. no-noise oracle, LEGOLAS noise, real-calibrated noise를 분리
4. F0/L0/V0의 가능한 부분 비교
5. stationary/axis-motion/arc/slip condition별 held-out rollout

Gate: oracle 입력에서 frame/sign bug 없이 multi-horizon delta가 수렴해야 한다. noise를 넣기 전 oracle이
실패하면 모델을 키우지 않고 data/label pipeline을 고친다.

### P2 — augmentation과 Isaac Lab/Sim challenger, 지금 설계 가능

1. 단일-factor augmentation sensitivity
2. 기존 real log의 input distribution과 synthetic distribution 비교
3. validated factor만 조합
4. Lab/Sim에서 high-rate IMU/contact/timing generator 구현
5. Gym train → Lab/Sim holdout과 반대 방향 test

Gate: 더 많은 DR가 nominal과 OOD를 함께 개선하는지 condition별로 확인한다. 평균 하나만 좋아지고
stationary, lateral, slip 중 하나가 무너지면 범위를 축소한다.

### P3 — 기존 real pseudo-label shadow, 지금 수행 가능

30개 joined session에서 다음을 비교한다.

1. native SDK odom delta
2. F0/L0/V0 output
3. command-only integration baseline

이 단계는 runtime, NaN, OOD, temporal drift 형태를 보는 smoke test다. SDK student가 SDK label을 잘
따랐다는 이유로 정확하다고 판정하지 않는다.

### P4 — 새 paired MoCap pilot, 데이터 수집 후 가능

1. clock offset/drift와 extrinsic을 먼저 추정
2. untouched paired sessions에서 zero-shot
3. condition별 20 ms/0.1/0.5/1/2 s delta residual과 sequence drift
4. Q coverage/NLL/NEES calibration
5. 부족할 때만 일부 session으로 scale/bias/uncertainty calibration 또는 fine-tune
6. 마지막 day/floor/session은 final test로 봉인

Gate: learned model이 F0와 native odom baseline을 condition별로 비교해 이겨야 한다. 평균 RMSE가 좋아도
stationary drift, lateral, turn, stop, support/slip failure가 더 나쁘면 승격하지 않는다.

### P5 — PF replay와 BT integration

동일 landmark log에 mocap delta, estimator mean+fixed noise, estimator mean+Q를 replay한다. 비교 대상은
ATE만이 아니다.

- 20 ms/1 s delta RPE, 1 m와 5 m trajectory RPE, stop drift
- covariance 90/95% coverage, NLL/NEES, whitened residual autocorrelation
- PF map error, ESS, resample rate, landmark innovation, relocalization
- vision blackout과 delayed detection에서 drift
- reset, duplicate, out-of-order, packet gap, fall/pickup fault injection
- GoalPose reach/overshoot, PF correction 뒤 goal jump, RLKick handoff

PF replay를 통과하기 전에는 estimator를 control path에 연결하지 않는다.

## 10. runtime 계약

### 10.1 estimator output

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

Covariance는 `[x,y,yaw]` interval error의 3×3 covariance다. absolute pose marginal 두 개를 단순히
빼지 않는다. filter pose clone의 cross-covariance나 preintegrated-delta covariance를 쓰고, paired real
residual은 scale/floor/coverage calibration에 쓴다.

### 10.2 source-neutral PF adapter

기존 downstream을 SDK callback에 묶어 두지 않는다.

```text
SDK adapter ─────┐
                 ├→ consumeEgomotion(delta, Q, stamp, health)
custom estimator ┘
```

`consumeEgomotion`은 `T_OB` 누적, main/reloc PF predict, `T_MB` snapshot publish만 담당한다. config에서
`sdk | leg_filter | learned | hybrid` 중 하나만 active source로 둔다. 기존 SDK의 forward/lateral/yaw
scale은 SDK adapter 안에만 남기고 custom estimator에 중복 적용하지 않는다.

초기 호환 test에서는 estimator delta를 누적 Pose2D로 바꿔 기존 `Locator::predict()`에 넣을 수 있다.
Production에서는 `predictDelta(delta,Q,dt,status)`를 추가해 per-delta Q와 exactly-once sequence를
보존한다.

### 10.3 timing과 delayed vision

LowState message 자체에는 source header가 없고 현재 deploy timer의 `callback_count×0.002`는 실제 시간이
아니다. receipt monotonic timestamp와 sequence를 우선 기록하고, 가능하면 SDK source stamp를 transport에
추가한다. estimator clock과 camera capture clock은 affine mapping/PTP 등 명시된 계약으로 맞춘다.

Production은 detection capture time으로 PF를 rewind/correct한 뒤 이후 delta를 exactly-once replay한다.
Prototype에서 marker를 current frame으로 forward-transform할 수는 있지만, 그때는 capture→now odom Q를
measurement covariance에 반영해야 한다.

### 10.4 health와 safety

| state | PF/보행 동작 |
|---|---|
| BOOTSTRAP | bias/joint/contact 초기화, motion inhibit |
| OK | mean+Q로 predict |
| DEGRADED | slip/gap/innovation, Q 확대, 저속·짧은 lookahead |
| INVALID | fake zero 금지, uncertainty 확대, nonzero cached command stop, stand |
| RECOVERING | stable double support에서 epoch 갱신 후 재개 |

Planner는 odom ROS callback 존재 여부가 아니라 최신 immutable pose snapshot을 읽는다. pose age가 threshold를
넘으면 마지막 nonzero planner command를 재전송하지 않는다. 초기 stop threshold는 100 ms로 시작하되 P0의
실측 rate/gap과 end-to-end latency를 보고 확정한다.

## 11. 구현 경계와 바로 다음 작업

Codex가 수정할 수 있는 범위는 다음이다.

- odom data audit/converter/collector와 dataset schema
- estimator 학습 및 evaluation 코드
- sim2real의 odom adapter, PF delta/Q/health seam
- odom replay와 health watchdog
- 이 문서와 `[codex/odometry]` coordination comment

Claude 소유의 locomotion environment, reward, policy architecture, checkpoint 승격 코드는 수정하지 않는다.
Synthetic collector는 외부 wrapper로 만든다. 500 Hz hook이나 추가 observation처럼 locomotion source 변경이
필요해지면 먼저 `MASTERPLAN.md`에 interface request만 남긴다.

실행 순서는 다음으로 고정한다.

1. P0 join/gap/joint-map audit
2. Gym external collector와 oracle label test
3. 1~2M synthetic smoke set
4. F0와 L0 최소 baseline
5. 기존 30-session pseudo-label shadow
6. 결과를 보고 Lab/Sim, V0/H1, 새 paired pilot의 우선순위 결정

## 12. 한 줄 요약

현재 데이터는 버릴 데이터가 아니다. 30개 real session 정합, simulator noise/timing calibration, SDK-odom
pseudo-label smoke test에 충분하다. 다만 MoCap 세션에는 IMU와 관절이 없으므로 custom odom accuracy의
정답은 아직 없다. 따라서 **현재 final policy의 Isaac Gym GT로 먼저 방법을 검증하고, 기존 real 분포로
증강하며, 새 동기화 CUSTOM-walk MoCap pilot에서 zero-shot과 Q를 통과시킨 뒤 PF/BT에 넣는다.**
