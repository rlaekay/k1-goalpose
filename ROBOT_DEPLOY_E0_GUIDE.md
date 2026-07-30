# E0 GoalPose Policy Real-Robot Deployment Guide

이 문서는 현재 repo 상태에서 `E0_armB_armsdown` policy를 Booster K1 실기 로봇에 올리기 위한 연결 가이드다.

결론부터 말하면, E0는 "속도 명령 walk"가 아니라 "목표 pose error를 먹는 12-DOF leg policy"다. 따라서 기존 `deploy_parameter_walk.py`에 `.pt`만 갈아끼우면 안 된다. 기존 deploy wrapper는 `vx, vy, vyaw`를 obs[6:8]에 넣지만, E0는 `goal_rel_x, goal_rel_y, heading_error`를 obs[6:8]에 넣어야 한다. BT/local planner는 최종 속도를 만들지 말고, 로봇 로컬 프레임의 `(dx, dy, dtheta)`만 policy wrapper에 넘겨야 한다.

## 1. Source Base

로컬 repo 근거:

- E0 best checkpoint: `logs/K1/K1/Goal_Pose_V7/2026-07-26-19-36-15_E0_armB_armsdown/nn/model_6200.pth`
- E0 성능 요약: `masterplan3.md`
  - 자기 config 기준 4633 segments
  - position median / p90: 2.7 cm / 5.0 cm
  - heading median: 2.5 deg
  - strict success: 89%
  - falls: 2
- E0 config: `htwk-gym/envs/K1/Goal_Pose_V7.yaml`
  - `num_observations: 54`
  - `num_actions: 12`
  - asset: `resources/K1/K1_locomotion_armsdown.urdf`
  - `control.decimation: 10`, `sim.dt: 0.002` -> policy inference interval 20 ms = 50 Hz
  - `command_order`: `[goal_rel_x, goal_rel_y, heading_error, gait_frequency, foot_yaw_L, foot_yaw_R, body_pitch_target, body_roll_target, feet_offset_x_target, feet_offset_y_target]`
- 실제 observation 구성: `htwk-gym/envs/K1/goal_pose.py`
  - projected gravity 3
  - base angular velocity 3
  - commands 10
  - gait clock cos/sin 2
  - leg dof position error 12
  - leg dof velocity 12
  - previous action 12
- 현재 deploy wrapper: `htwk-gym/deploy/utils/policy.py`, `htwk-gym/deploy/utils/policy_thomas.py`
  - TorchScript actor `.pt`를 `torch.jit.load()`로 로드
  - LowState의 IMU, gyro, motor q/dq로 observation 구성
  - action을 clip한 뒤 `default_qpos + action_scale * action`으로 leg target 생성
- 현재 SDK publish 구조: `htwk-gym/deploy/deploy_parameter_walk.py`
  - `LowState` subscribe
  - `LowCmd` publish
  - `RobotMode.kCustom` 진입
  - 2 ms publish loop, 20 ms inference loop

외부 공식/사용자 경험 근거:

- Booster 공식 open-source 페이지는 SDK, Booster Gym, Booster Deploy를 별도 구성요소로 두고, SDK를 로봇 2차 개발용 suite라고 설명한다: https://www.booster.tech/open-source/
- Booster Robotics SDK GitHub README는 Python SDK가 pip package로 제공된다고 설명한다: https://github.com/BoosterRobotics/booster_robotics_sdk
- Booster Deploy 공식 repo는 real robot deployment 전에 firmware `>= v1.4`, SDK Python binding 설치, Sim2Sim 확인 후 robot copy, `scripts/deploy.py --task <TASK_NAME>` 흐름을 제시한다: https://github.com/BoosterRobotics/booster_deploy
- FastTD3/BoosterGym 사용자 경험 문서는 real hardware 배포 전에 sim-to-sim 확인, TorchScript/JIT export, damping/stiffness/torque limit 재확인을 강하게 경고한다: https://git.dominik-roth.eu/dodox/FastTD3/src/commit/258bfe67dd3446bb918c67f6fc250f9b55a98bb2/sim2real.md
- B-Human Booster handling 문서는 K1/T1 사용 시 learned policy가 켜진 WALK 상태에서 로봇을 들지 말고, prepare/damping/emergency 흐름을 명확히 둘 것을 강조한다: https://docs.b-human.de/master/handling-the-booster/

## 2. E0가 먹는 입력

E0 actor 입력은 54차원이다.

```text
0:3    projected_gravity
3:6    base_ang_vel
6:16   commands
16:18  gait clock cos/sin
18:30  leg dof_pos - default_dof_pos
30:42  leg dof_vel
42:54  previous action
```

`commands` 10개 중 실기에서 BT가 매 tick 공급해야 하는 것은 앞 3개뿐이다.

```text
commands[0] = goal_rel_x       # robot local frame, meters
commands[1] = goal_rel_y       # robot local frame, meters
commands[2] = heading_error    # wrapped to [-pi, pi], radians
```

나머지 7개는 E0 기준 상수로 둔다.

```text
commands[3] = gait_frequency   # E0 deploy 초기값은 2.0 Hz 근처 권장
commands[4] = foot_yaw_L       # 0
commands[5] = foot_yaw_R       # 0
commands[6] = body_pitch       # 0
commands[7] = body_roll        # 0
commands[8] = feet_offset_x    # 0
commands[9] = feet_offset_y    # 0
```

정규화는 E0 config와 같아야 한다.

```text
goal_rel_x/y   * 0.5
heading_error  * 0.31831
gait_frequency * 1.0
dof_pos        * 1.0
dof_vel        * 0.1
action clip    [-1, 1]
```

중요한 판단:

- 기존 SDK walk처럼 `vx, vy, vyaw`를 넣는 구조가 아니다.
- local planner가 error를 계산하고 `setVel()`을 부르는 구조도 아니다.
- local planner/BT는 goal pose를 정하고, policy wrapper가 현재 robot pose와 goal pose로 `(dx,dy,dtheta)`를 매 20 ms 재계산한다.
- policy가 action 12개를 내면 deploy controller가 leg joint target 12개로 바꿔 `LowCmd`를 publish한다.

## 3. 현재 Deploy 코드와 맞지 않는 부분

현재 `htwk-gym/deploy/utils/policy.py`는 observation layout이 ParameterWalk 계열이다.

```text
obs[6] = vx
obs[7] = vy
obs[8] = vyaw
obs[9:11] = gait clock
obs[11:23] = leg dof_pos
obs[23:35] = leg dof_vel
obs[35:47] = previous action
```

E0는 commands 10개가 들어가므로 layout이 다르다.

```text
obs[6:16] = commands 10
obs[16:18] = gait clock
obs[18:30] = leg dof_pos
obs[30:42] = leg dof_vel
obs[42:54] = previous action
```

따라서 필요한 것은 새 wrapper다.

권장 이름:

```text
htwk-gym/deploy/utils/policy_goal_pose.py
htwk-gym/deploy/configs/Goal_Pose_E0.yaml
htwk-gym/deploy/deploy_goal_pose.py
```

이 세 개를 만들면 기존 deploy 구조를 유지하면서 E0 전용 observation만 바꿀 수 있다.

## 4. Export Flow

학습 checkpoint는 `.pth`이고 실기 deploy는 actor-only TorchScript `.pt`가 편하다.

명령 흐름:

```bash
cd htwk-gym
python export_model.py \
  --task K1/Goal_Pose_V7 \
  --checkpoint logs/K1/K1/Goal_Pose_V7/2026-07-26-19-36-15_E0_armB_armsdown/nn/model_6200.pth
```

예상 산출물:

```text
logs/K1/K1/Goal_Pose_V7/2026-07-26-19-36-15_E0_armB_armsdown/nn/model_6200.pt
```

현재 주의점:

- `export_model.py`가 `utils.model_thomas`를 import하고 있는데 현재 repo에는 `utils/model_thomas.py`가 없다.
- 실제 모델 정의는 `htwk-gym/utils/model.py`의 `ActorCritic`이다.
- 즉 export 전에 `export_model.py` import를 `from utils.model import *`로 고치거나, 같은 class를 re-export하는 호환 파일을 추가해야 한다.

Export 후 smoke:

```bash
python - <<'PY'
import torch
p = "logs/K1/K1/Goal_Pose_V7/2026-07-26-19-36-15_E0_armB_armsdown/nn/model_6200.pt"
m = torch.jit.load(p, map_location="cpu")
y = m(torch.zeros(1, 54))
print(y.shape, y.min().item(), y.max().item())
PY
```

정상 출력:

```text
torch.Size([1, 12]) ...
```

## 5. BT / Local Planner Integration

기존 구조:

```text
BT/chase -> local_planner goalpose -> pose error -> setvel(vx,vy,wz) -> SDK walk
```

E0 권장 구조:

```text
BT/chase -> goal pose selector -> GoalPoseAdapter -> policy -> LowCmd joint targets
```

BT가 넘길 최소 인자:

```cpp
struct GoalPoseCommand {
  double goal_x_field;
  double goal_y_field;
  double goal_yaw_field;
  double stamp;
  bool valid;
};
```

GoalPoseAdapter가 매 policy tick 계산:

```text
dx_world = goal_x_field - robot_x_field
dy_world = goal_y_field - robot_y_field

goal_rel_x =  cos(yaw) * dx_world + sin(yaw) * dy_world
goal_rel_y = -sin(yaw) * dx_world + cos(yaw) * dy_world
heading_error = wrap_pi(goal_yaw_field - robot_yaw_field)
```

그리고 clamp:

```text
goal_rel_x clamp: [-2.0, 2.0]
goal_rel_y clamp: [-1.5, 1.5]
heading_error clamp/wrap: [-pi, pi]
```

왜 이 구조가 맞는가:

- E0 학습 환경의 `_update_goal_state()`가 정확히 이 계산을 한다.
- error는 현재 pose 기준으로 매 step 재투영되어야 한다. 한 번 계산한 local error를 계속 들고 있으면 odometry drift와 회전 때문에 의미가 깨진다.
- `setVel`은 velocity controller를 전제로 한 API다. E0는 velocity controller가 아니라 joint-target policy이므로 `setVel` 뒤에 붙이는 방식은 목적함수가 바뀐다.

## 6. Robot Runtime Flow

실기 실행 순서:

```text
1. robot PREP mode
2. deploy_goal_pose.py 실행
3. SDK channel init
4. LowState subscribe 시작
5. 안전 조건 확인
6. Custom mode 진입
7. 첫 frame command로 default_qpos publish
8. 500 Hz LowCmd publish loop 시작
9. 50 Hz policy inference loop 시작
10. BT/PF가 goal pose update
11. policy wrapper가 매 tick local goal error 재계산
12. actor output -> leg target -> LowCmd publish
```

필수 watchdog:

- `LowState`가 끊기면 DAMP 또는 PREP 복귀
- roll/pitch 절댓값이 1 rad 근처로 가면 중단
- command stamp가 stale이면 `goal_rel_x/y/theta = 0`으로 두고 gait clock을 끄거나 매우 낮춘다
- action NaN/Inf면 즉시 중단
- target jump limit을 둔다
- torque limit은 config 값으로 clamp한다

현재 deploy code는 roll/pitch가 1 rad를 넘으면 running false로 내리는 safety check가 이미 있다. E0 wrapper에서도 유지한다.

## 7. E0의 팔 문제

E0는 `K1_locomotion_armsdown.urdf`로 학습했다. 이 URDF에서는 shoulder/elbow joints가 `fixed`이고, `collapse_fixed_joints: true`라 팔이 정책이 제어하는 DOF가 아니다. 정책 action은 12개 leg joint만 나온다.

실제 로봇에서는 팔이 물리적으로 존재하고, SDK `LowCmd`는 전체 joint에 명령을 보낸다. 따라서 팔은 "policy가 알아서 하는 것"이 아니라 deploy layer가 별도로 posture command를 줘야 한다.

E0 기준으로 가장 보수적인 실기 팔 전략:

- policy action은 legs 12개에만 적용한다.
- arms/head/waist는 `common.default_qpos` 또는 별도 arms-down qpos로 고정한다.
- 팔 joint target은 갑자기 바꾸지 말고 low-pass filter로 천천히 보낸다.
- 처음 실기에서는 팔을 움직이는 script를 켜지 않는다.

주의:

- `make_v7_arms.py` 주석상 E0가 실제로 학습한 geometry는 당시 80-degree splay arms-down이고, 이후 URDF가 90-degree + rearward shoulder-pitch tuck으로 바뀌었다.
- 그래서 real robot 팔을 "현재 URDF의 새 armsdown"으로 강하게 고정하면 E0가 본 동역학과 조금 다를 수 있다.
- 하지만 policy가 팔을 observe/control하지 않으므로 팔을 흔들거나 BT 상태에 따라 움직이는 것은 E0 배포 첫 단계에서 금지하는 게 맞다.

실기에서 팔은 어떻게 되어 있을까?

- 현재 deploy config의 `default_qpos`는 전체 23 joints를 갖고 있고, 앞쪽 indices가 arms/head/waist, 뒤쪽 12개가 legs다.
- `policy.py`는 `dof_targets[11:] += action`을 쓰므로, 실기에서도 action은 11번 이후 leg joints에만 들어간다.
- arms/head/waist는 config의 default posture로 유지된다.
- 즉 "E0의 팔이 뒤로 가 있다"는 것은 actor가 출력하는 행동이 아니라 asset/default posture 문제다. 실기에서는 `Goal_Pose_E0.yaml`의 `common.default_qpos[0:11]`이 실제 팔 자세를 결정한다.

권장 실험 순서:

```text
Stage A: hoist, feet barely touching, goal=(0,0,0), 10 s
Stage B: hoist, tiny goal=(0.2,0,0), 10 s
Stage C: hoist, goal=(0.5,0,0), 30 s
Stage D: ground but human ready, goal=(0.3,0,0), 10 s
Stage E: ground, waypoint box within 0.5 m
Stage F: BT/local planner live goal
```

각 stage에서 남길 log:

```text
timestamp
robot mode
LowState imu rpy/gyro
motor q/dq/tau
policy obs[0:54]
raw action[0:12]
clipped action[0:12]
full dof target[0:23]
goal field pose
robot field pose
goal_rel_x/y/theta
command age
fall/safety stop reason
```

## 8. Minimal Implementation Checklist

1. Fix/export actor:

```bash
cd htwk-gym
python export_model.py --task K1/Goal_Pose_V7 --checkpoint <E0_model_6200.pth>
```

2. Add deploy config:

```text
deploy/configs/Goal_Pose_E0.yaml
```

Must include:

```text
policy_path: ./models/goal_pose_e0.pt
num_observations: 54
num_actions: 12
normalization.goal_pos: 0.5
normalization.goal_heading: 0.31831
control.decimation: 10
control.action_scale: 1.0
common.dt: 0.002
```

3. Add `policy_goal_pose.py`:

- load `.pt`
- build E0 observation layout
- accept `(goal_rel_x, goal_rel_y, heading_error)`
- maintain gait clock
- keep previous action
- output full 23-DOF target with only leg slice modified

4. Add `deploy_goal_pose.py`:

- copy `deploy_parameter_walk.py`
- replace `RemoteControlService` velocity command path with BT/local planner goal source
- keep LowState/LowCmd and mode handling
- keep safety stop

5. Connect BT:

- BT sends field goal pose and validity/stamp
- adapter computes local error each policy tick
- no `setVel()` in the E0 path

6. Sim2Sim or dry-run before hardware:

- TorchScript smoke: input `[1,54]` -> output `[1,12]`
- recorded LowState replay: no robot, run policy wrapper and inspect targets
- MuJoCo/Booster Deploy if available
- hoist test only after logs are clean

## 8. Current ekay-fix Progress — 2026-07-30

이번 `ekay-fix` 작업에서 real robot로 이어붙이기 위한 상단 harness는 INHA-Player 쪽에 들어갔다.

구현된 것:

- `src/brain/behavior_trees/locomotion_test.xml`
  - `RobocupWalk`
  - `ResetOdometry`
  - `SelfLocateEnterField`
  - `LocomotionTest`
- `src/brain/include/mission_test.h`, `src/brain/src/mission_test.cpp`
  - `/locomotion_test/command` subscribe
  - `/locomotion_test/mission_id` subscribe (`std_msgs/msg/Int32`, 0=stop, 1..5=mission)
  - `/locomotion_test/status` publish
  - `/locomotion_test/telemetry` publish (`std_msgs/msg/String` JSON, all angles in degree)
  - `/locomotion_test/goal_pose` publish
  - `prep / ready / playing / finished` FSM
  - mission별 map-frame path 생성
  - pure-pursuit style lookahead target 생성
- `src/brain/config/config.yaml`
  - `locomotion_test.goal_reached_xy_m: 0.10`
  - `locomotion_test.goal_reached_theta_deg: 6.0`
  - mission별 repeat/goal/random seed/lookahead 설정

현재 중요한 한계:

```text
현재 경로: LocomotionTest -> LocalPlanner -> RobotClient::setVelocity() -> SDK walk
목표 경로: LocomotionTest -> GoalPoseAdapter -> E0 actor -> LowCmd
```

즉 지금 들어간 것은 BT mission harness와 lookahead generator다. E0 actor를 실기 low-level controller로 직접 호출하는 refactor는 아직 남아 있다.

다음 refactor 단위:

1. `export_model.py` import 수정 또는 호환 shim 추가.
2. E0 actor-only TorchScript `.pt` export.
3. `htwk-gym/deploy/utils/policy_goal_pose.py` 추가.
4. `htwk-gym/deploy/configs/Goal_Pose_E0.yaml` 추가.
5. `htwk-gym/deploy/deploy_goal_pose.py` 추가.
6. INHA-Player `LocomotionTest`가 publish하는 `/locomotion_test/goal_pose`를 GoalPoseAdapter에서 subscribe.

권장 연결 방식:

```text
/locomotion_test/status        # 사람 확인용
/locomotion_test/telemetry     # walk 비교/debug용 JSON stream
/locomotion_test/goal_pose     # adapter 입력용, geometry/msg 또는 custom msg
```

현재는 `geometry_msgs/msg/PoseStamped`로 map-frame carrot을 낸다. 처음에는 topic이 가장 안전하다. 이유는 BT와 policy process를 분리해 한쪽 crash가 다른 쪽을 바로 죽이지 않고, `ros2 topic echo/bag`로 goal stream을 검증할 수 있기 때문이다. latency가 문제가 되면 같은 message contract를 유지한 채 shared memory 또는 in-process adapter로 옮긴다.

policy 교체 방법:

1. `locomotion_test.active_policy` 값을 `e0`, `e1_path`, `g1_speed` 등으로 바꾼다.
2. adapter가 policy별 checkpoint/config/profile을 선택한다.
3. `lookahead_default_m`, `lookahead_max_m`, `heading_blend_distance_m`만 policy profile에 맞춰 바꾼다.
4. `locomotion_test.xml`과 mission sequence는 바꾸지 않는다.

E0 기준 첫 값:

```yaml
locomotion_test:
  active_policy: "e0"
  lookahead_min_m: 0.25
  lookahead_default_m: 0.55
  lookahead_max_m: 0.90
  heading_blend_distance_m: 0.60
```

판단 근거:

- E0는 waypoint precision baseline이다. 현재 유효 결과는 위치 2.7 cm / p90 5.0 cm, heading 2.5°, strict 89.3%지만 path 학습은 없다.
- `STATE_ESTIMATION.md` 기준 production GoalPose 입력 범위는 `dx∈[-2,2]`, `dy∈[-1.5,1.5]`다. 첫 실기는 이 범위를 벗기지 않는 짧은 lookahead가 맞다.
- pure-pursuit 계열 문헌도 lookahead가 크면 smooth하지만 corner cutting과 tracking error가 커지고, 작으면 oscillation이 커지는 tradeoff를 말한다. 그래서 E0는 짧고 보수적인 값에서 시작한다.

## 9. What Not To Do

- Do not replace `parameter_walk.pt` with E0 `.pt` while keeping `policy.py`.
- Do not feed velocity command into E0 obs[6:8].
- Do not let `setVel()` clamp E0 commands. E0 command clamp is pose-error range, not velocity range.
- Do not move arms dynamically in the first real test.
- Do not call SDK WALK/native locomotion and CUSTOM low-level policy at the same time.
- Do not trust SDK odom as ground truth for final localization evaluation. It can be logged as a baseline, but PF/vision/MoCap calibration is separate.

## 10. Recommended First Real-Robot Target

For E0, the first real target should not be path following. Use waypoint only:

```text
goal_rel_x = 0.2 m
goal_rel_y = 0.0 m
heading_error = 0.0 rad
```

Then:

```text
0.3 m forward
0.3 m lateral
0.3 m forward + 10 deg heading
0.5 m forward
```

E0's strength is precise short-range arrival and stop. It was not trained as the final high-speed path follower. Use it to validate the real robot observation/action bridge first. After that, G-batch policies can be considered for speed/path behavior.
