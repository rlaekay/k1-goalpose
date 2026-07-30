# Mission-only 실기 배포 감사 및 실행 런북 — 2026-07-30

## 결론

Claude Code가 잡은 큰 방향은 맞다.

```text
Mac mission number
  -> /locomotion_test/mission_id (Int32, 0..5)
  -> Brain locomotion_test BT/FSM + camera-PF localization
  -> /locomotion_test/goal_rel (Vector3Stamped, robot frame)
  -> E0 actor obs[6:9]
  -> LowCmd (23 joints; actor controls leg indices 11..22)
```

그러나 2026-07-30 확인 시점의 상태를 **실기 mission-ready**라고 부를 수는 없다.
E0 checkpoint `model_6200.pth`의 기대 서버 경로만 알고 있을 뿐 실제 존재/hash와
TorchScript export 결과를 재확인하지 못했고, 로봇에는 `goal_pose_e0.pt`가 없다. 서버
SSH가 Mac에서는 timeout, 로봇에서는 port 22 connection refused였기 때문이다. 따라서
이번 세션에는 checkpoint export와 서버→로봇 policy 복사를 완료하지 못했다. 정책 파일
없이 로봇을 CUSTOM/LowCmd로 진입시키면 안 된다.

코드 측면에서는 Claude 반영분에 있던 Ctrl-C 종료 안전성, LowState freshness,
policy bridge 가시성을 보강했다. `deploy_goal_pose.py`는 이제 CUSTOM 진입 후 어떤 종료
경로로 빠져도 DAMPING을 요청하고, LowState가 0.2초 이상 끊기면 LowCmd publish를 멈추며,
`/locomotion_test/policy_debug`를 10 Hz로 낸다.

## Claude 반영분 critic

| Claude 주장/변경 | 코드·현장 근거 | 판정 및 조치 |
|---|---|---|
| BT에서 LocalPlanner/`setVelocity()`를 제거하고 E0 goal을 publish | Mac의 sibling repo `INHA-Player` commit `afb731d2`; `locomotion_test.xml`에는 `RobocupWalk` 없음; `mission_test.cpp`는 `goal_pose`와 `goal_rel`만 publish | 방향과 구현 모두 맞음 |
| `/locomotion_test/goal_rel`을 E0 deploy가 ROS로 구독 | Brain은 `Vector3Stamped`, deploy도 같은 타입/단위를 구독 | contract는 맞음 |
| stale goal이면 “E0 정지” | 실제 코드는 `(0,0,0)` goal로 바꿀 뿐이고 gait clock/frequency와 LowCmd는 계속 돈다 | 과장. 문구를 “learned zero-goal 요청”으로 수정. 하드웨어 정지는 DAMPING/E-stop만 해당 |
| mission-only live 경로에서 hoist 생략 | E0@6200은 sim full eval에서 2 falls이고 실기 미검증; 로봇에 model도 없음 | 거부. fixed/hoist는 mission 기능이 아니라 최초 1회 안전 승격 gate이므로 생략 불가 |
| Ctrl-C로 안전 종료 | 기존 signal handler는 `SystemExit`만 내고 channel을 닫았으며 DAMPING 전환을 보장하지 않았음 | 수정: idempotent cleanup에서 publish thread 정지 후 DAMPING 요청 |
| NaN/rpy watchdog이면 충분 | LowState가 끊겨도 마지막 sensor 값으로 추론·publish를 계속할 수 있었음 | 수정: `low_state_timeout_s: 0.20` watchdog 추가 |
| telemetry로 end-to-end 확인 가능 | Brain telemetry만으로는 deploy가 topic을 실제 수신했는지, goal stale인지, actor action 범위가 어떤지 알 수 없음 | 수정: `/locomotion_test/policy_debug` 추가 |
| 문서가 현재 구현을 설명 | `missions.md`는 여전히 LocalPlanner/velocity와 “adapter 미구현”을 설명하고, 새 Brain 코드는 polyline projection이 아니라 현재 waypoint 방향의 radial carrot을 사용 | 이 보고서와 `missions.md`를 현재 코드 기준으로 정정 |

## 실제로 확인한 환경

### Mac

- `k1-goalpose`: branch `ekay-fix`; 기존 local amend와 cached `origin/ekay-fix`의 갈라진
  이력은 tree를 유지한 merge로 정리했다. mission hardening commit은 `fb8130f`, merge
  commit은 `03f2f5d`다.
- `INHA-Player`: `/Users/dmdrb/RoboCup/[07]sim2real/INHA-Player`, branch `ekay-fix`,
  `origin/ekay-fix`보다 `afb731d2` 1 commit ahead, worktree clean.
- 두 repo 모두 GitHub hostname DNS 실패로 `git fetch/push`가 실패했다. 위 commit들은
  로컬에만 있으며 서버/로봇이 pull할 수 있는 원격 branch에는 아직 올라가지 않았다.
  robot staging에는 검증 대상 source를 SCP로 복사했기 때문에 이번 비구동 build 결과에는
  영향이 없다.
- `ros2`, `colcon`, PyTorch, PyYAML은 Mac 기본 Python에 없음. 따라서 Mac은 ROS node를
  직접 실행하지 않고 `missionctl.sh`가 SSH로 로봇의 ROS CLI를 호출한다.

### Robot `booster@192.168.10.102`

- Ubuntu 22.04, aarch64, real-time Tegra kernel.
- ROS 2 Humble, `ROS_DOMAIN_ID=0`, `ros2`와 `colcon` 설치됨.
- Python 3.10.12, PyTorch 2.7.0, NumPy 1.26.3, PyYAML 5.4.1.
- Booster Python SDK import 성공:
  `/usr/local/lib/python3.10/dist-packages/booster_robotics_sdk_python...so`.
- `rclpy` import 성공.
- 기존 경기 repo `/home/booster/Workspace/INHA-Soccer/INHA-Player`는 branch
  `sim2real`이며 `config.yaml`, `game.xml` 등 사용자 수정 7개가 있다. 이 worktree에는
  pull/switch/overwrite를 하지 않는다.
- 기존 `/home/booster/Workspace/INHA-RL/deploy`는 동작 중인 velocity walk 환경이며
  `base_walk.pt`, `parameter_walk*.pt`, `velocity_command_walk_k1.pt`만 있다.
- `/home/booster/Workspace/deploy/tasks/goto`에는 별도 `k1_goto_jit.pt`가 있으나 E0@6200이
  아니고 observation/goal-latching contract도 다르므로 대체품으로 쓰지 않는다.
- `goal_pose_e0.pt`, `model_6200.pth`, `deploy_goal_pose.py`는 확인 당시 로봇에 없었다.
- 기본 Booster agent/ROS bridge들은 실행 중이었지만 Brain과 E0 deploy는 실행 중이 아니었다.

### Training server

- 기대 경로:
  `/mnt/DATA/workspace/ws_eungkyu/k1-goalpose/htwk-gym/logs/K1/K1/Goal_Pose_V7/2026-07-26-19-36-15_E0_armB_armsdown/nn/model_6200.pth`
- `165.246.193.194:22`: Mac SSH timeout.
- 같은 주소: 로봇에서 connection refused.
- 따라서 서버 파일 존재/hash/export 결과를 이번 세션에는 재확인하지 못했다.

### 이번 세션에서 실제로 만든 robot staging과 검증 결과

기존 두 worktree는 수정하지 않고 새 경로
`/home/booster/Workspace/k1-goalpose-mission`만 만들었다.

- `deploy/`: 이 repo의 전체 deploy 복사본을 배치했고, 최신
  `deploy_goal_pose.py`/`policy_goal_pose.py`의 robot-side `py_compile`을 통과했다. 최신
  `deploy_goal_pose.py`의 Mac/robot SHA-256은 모두
  `83f292fa5f6aa11dde0ef9591d37d9e016165d136e984c07da2fc07904afa7b2`다.
- `brain_ws/src/brain`: Mac sibling `INHA-Player`의 `afb731d2` Brain source를 복사했다.
- 기존 경기 install을 dependency underlay로 source한 뒤 새 workspace에서
  `colcon build --packages-select brain --executor sequential --parallel-workers 1` 실행:
  `1 package finished [12min 47s]`.
- 새 overlay의 `ros2 pkg prefix brain`은
  `/home/booster/Workspace/k1-goalpose-mission/brain_ws/install/brain`, vision은 기존 read-only
  underlay `/home/booster/Workspace/INHA-Soccer/INHA-Player/install/vision`을 가리킨다.
- source/install의 `locomotion_test.xml` SHA-256이 일치했고, 설치 binary에
  `/locomotion_test/goal_rel` 문자열이 있으며 `ldd` missing dependency는 없었다.
- `ros2 launch brain launch.py --show-args`가 성공했다. 실제 Brain/vision node는 시작하지
  않았다.
- 임시 ROS topic을 사용한 비구동 bridge smoke에서 finite `Vector3Stamped` 수신,
  stale/received counter, `policy_debug` JSON publish를 확인했다.
- 기존 54→12 `parameter_walk.pt`를 **shape-only 대체물**로 쓴 wrapper smoke는
  `(54,) obs -> (12,) action -> (23,) target`, finite를 통과했다. 이것은 E0@6200의 policy
  의미/안전 검증이 아니므로 승격 근거로 쓰지 않는다.
- CUSTOM 진입 직전 LowState gate unit smoke는 fresh 입력을 허용하고 0.3초 stale 입력을
  거부했다.
- 기존 경기 repo의 `git status --short` 7개 항목은 감사 전후 동일하다. 기존 파일에는
  pull/switch/edit를 하지 않았다.

따라서 현재 로봇은 **코드와 Brain binary까지 staging 완료**, E0 actor와 실기 safety gate는
미완료 상태다. 정책 없이 CUSTOM/LowCmd를 실행하지 않았다.

## 사용되는 full source

아래 파일들이 mission-only 경로의 전체 구현이다. 일부 코드 조각만 복사해 조합하지 말고
각 파일을 그대로 사용한다.

| 계층 | full source |
|---|---|
| BT | sibling `INHA-Player/src/brain/behavior_trees/locomotion_test.xml` @ `afb731d2` |
| mission/FSM/topic | sibling `INHA-Player/src/brain/include/mission_test.h`, `src/brain/src/mission_test.cpp` @ `afb731d2` |
| Brain config/registration | sibling `src/brain/config/config.yaml`, `src/brain/CMakeLists.txt`, `src/brain/src/brain.cpp`, `src/brain/src/brain_tree.cpp` |
| actor export | `htwk-gym/export_model.py` |
| observation/action adapter | `htwk-gym/deploy/utils/policy_goal_pose.py` |
| LowState/LowCmd/ROS bridge | `htwk-gym/deploy/deploy_goal_pose.py` |
| gains, pose, topics, safety | `htwk-gym/deploy/configs/Goal_Pose_E0.yaml` |
| Mac control | `missionctl.sh` |

Brain `goal_rel` contract:

```text
dx_world = goal_x_map - robot_x_map
dy_world = goal_y_map - robot_y_map
goal_rel_x =  cos(yaw) * dx_world + sin(yaw) * dy_world   # forward [m]
goal_rel_y = -sin(yaw) * dx_world + cos(yaw) * dy_world   # left [m]
heading_error = wrap_pi(goal_yaw_map - robot_yaw_map)      # [rad]
```

ROS message:

```text
topic: /locomotion_test/goal_rel
type:  geometry_msgs/msg/Vector3Stamped
vector.x: goal_rel_x [m]
vector.y: goal_rel_y [m]
vector.z: heading_error [rad]
```

E0 observation:

```text
0:3    projected gravity
3:6    angular velocity
6:16   commands = [goal_rel_x, goal_rel_y, heading_error, gait_frequency,
                    foot_yaw_L, foot_yaw_R, pitch, roll, feet_offset_x, feet_offset_y]
16:18  gait clock cos/sin
18:30  12 leg position errors
30:42  12 leg velocities
42:54  previous 12 actions
```

BT가 제공하는 값은 앞의 3개뿐이다. 나머지는 onboard sensor, internal state, config
constant다.

## 서버가 다시 열렸을 때: export와 복사

서버에서 checkpoint를 TorchScript actor로 export한다. **현재 base YAML이 아니라 checkpoint의
frozen run config가 최종 기준**이어야 한다. architecture는 54→12로 같아도 normalization,
joint default, action scale은 frozen config와 다시 대조한다.

```bash
cd /mnt/DATA/workspace/ws_eungkyu/k1-goalpose/htwk-gym
conda activate k1goalpose

CKPT=logs/K1/K1/Goal_Pose_V7/2026-07-26-19-36-15_E0_armB_armsdown/nn/model_6200.pth
python export_model.py --task K1/Goal_Pose_V7 --checkpoint "$CKPT"

python - <<'PY'
import torch
p = "logs/K1/K1/Goal_Pose_V7/2026-07-26-19-36-15_E0_armB_armsdown/nn/model_6200.pt"
m = torch.jit.load(p, map_location="cpu").eval()
with torch.inference_mode():
    y = m(torch.zeros(1, 54))
assert tuple(y.shape) == (1, 12), y.shape
assert torch.isfinite(y).all()
print(p, tuple(y.shape), float(y.abs().max()))
PY

sha256sum "${CKPT%.pth}.pt"
```

서버에서 robot private IP로 직접 갈 수 있을 때:

```bash
scp logs/K1/K1/Goal_Pose_V7/2026-07-26-19-36-15_E0_armB_armsdown/nn/model_6200.pt \
  booster@192.168.10.102:/home/booster/Workspace/k1-goalpose-mission/deploy/models/goal_pose_e0.pt
```

직접 route가 없으면 Mac relay를 쓴다:

```bash
scp user@165.246.193.194:/mnt/DATA/workspace/ws_eungkyu/k1-goalpose/htwk-gym/logs/K1/K1/Goal_Pose_V7/2026-07-26-19-36-15_E0_armB_armsdown/nn/model_6200.pt \
  /tmp/goal_pose_e0.pt

scp /tmp/goal_pose_e0.pt \
  booster@192.168.10.102:/home/booster/Workspace/k1-goalpose-mission/deploy/models/goal_pose_e0.pt
```

복사 후 양쪽 `sha256sum`이 같아야 한다.

## 로봇 staging 원칙

기존 dirty 경기 repo와 기존 INHA-RL deploy를 덮어쓰지 않는다. 별도 경로만 쓴다.

```text
/home/booster/Workspace/k1-goalpose-mission/
  brain_ws/                 # clean ekay-fix mission Brain build
  deploy/
    deploy_goal_pose.py
    configs/Goal_Pose_E0.yaml
    models/goal_pose_e0.pt
    utils/...
```

허용된 준비 작업은 이 새 staging 경로로 copy, clean branch pull/clone, build뿐이다.
기존 `/home/booster/Workspace/INHA-Soccer/INHA-Player`의 사용자 변경은 보존한다.

## 최초 1회 승격 gate

아래는 다른 기능 테스트가 아니라 mission을 지면에서 실행하기 위한 안전 전제다.

1. robot에서 `goal_pose_e0.pt` hash 일치.
2. `GoalPosePolicy` load 시 자동 zero-observation smoke가 `(1,54)->(1,12)`, finite로 통과.
3. Brain clean build 통과.
4. `--goal-source fixed --goal "0,0,0"`를 hoist에서 실행.
5. hoist에서 0.2 m forward goal, joint order/IMU sign/action range 확인.
6. Ctrl-C 직후 DAMPING 진입 확인.
7. LowState를 끊는 fault injection 없이도 debug topic에서 age가 계속 0.2초 미만인지 확인.

E0@6200 full sim report는 position median 2.72 cm, p90 5.01 cm, heading median 2.52°,
strict 89.29%였지만 4633 segment에서 2 falls였다. 이 수치는 waypoint 정확성 근거이지
지면 무검증 실기 안전 보증이 아니다.

## Mission-only 실행

### Robot terminal A — camera-PF vision

mission에는 camera-PF localization이 필요하다. 기존 install의 vision package는 underlay로
읽기만 하고, 새 mission Brain overlay를 함께 source해 `ekay_odom=true` contract를 맞춘다.

```bash
source /opt/ros/humble/setup.bash
source /home/booster/Workspace/INHA-Soccer/INHA-Player/install/setup.bash
source /home/booster/Workspace/k1-goalpose-mission/brain_ws/install/setup.bash
ros2 launch vision launch.py vision_config_path:=/opt/booster \
  ekay_odom:=true save_data:=false show_det:=false
```

### Robot terminal B — mission-only Brain

경기용 `scripts/start.sh`는 사용하지 않는다. 그것은 mission에 불필요한 whistle,
game-controller, sound를 함께 시작한다.

```bash
source /opt/ros/humble/setup.bash
source /home/booster/Workspace/INHA-Soccer/INHA-Player/install/setup.bash
source /home/booster/Workspace/k1-goalpose-mission/brain_ws/install/setup.bash
ros2 launch brain launch.py tree:=locomotion_test \
  vision_config_path:=/opt/booster disable_com:=true
```

marker가 보이지 않아 `odom_calibrated=false`이면 BT는 goal을 publish하지 않고 기다리는
것이 정상이다.

### Robot terminal C — E0 deploy

```bash
cd /home/booster/Workspace/k1-goalpose-mission/deploy
source /opt/ros/humble/setup.bash
python3 deploy_goal_pose.py \
  --config Goal_Pose_E0.yaml \
  --goal-source ros \
  --net 127.0.0.1
```

로봇은 PREP/hoist 상태에서 시작하고 remote-control prompt를 따른다. 경기용 BT의
`RobocupWalk` 또는 다른 CUSTOM/LowCmd publisher를 동시에 실행하지 않는다.

### Mac terminal — 확인과 mission 번호

Mac에는 ROS 2를 설치할 필요가 없다.

```bash
cd /Users/dmdrb/RoboCup/k1-goalpose

./missionctl.sh check
./missionctl.sh once telemetry
./missionctl.sh watch policy

./missionctl.sh 1       # mission 1
./missionctl.sh 0       # BT stop
```

다른 terminal에서 동시에:

```bash
./missionctl.sh watch telemetry
./missionctl.sh watch goal-rel
```

mission 번호 의미:

| id | mission |
|---:|---|
| 0 | BT mission stop/reset; emergency stop은 아님 |
| 1 | 제자리 CW 360° + CCW 360°, 3회 |
| 2 | x=+3 m ↔ -3 m, heading 0 유지, 3회 |
| 3 | y=-2 m ↔ +2 m, heading 0 유지, 3회 |
| 4 | 중심 (-3,0), 반지름 6 m 원 위 seeded random pose 4개 |
| 5 | (-2,-0.5) 시작, 1 m spacing의 9 m lawnmower |

현재 Brain의 carrot은 옛 문서의 polyline projection pure-pursuit가 아니다. 현재 waypoint가
2 m보다 멀면 waypoint 방향으로 2 m radial carrot을 만들고, 가까우면 waypoint 자체를 낸다.
deploy에서 x는 ±2 m, y는 ±1.5 m로 다시 clamp한다.

## 실시간 검증 topic

| topic | type | 반드시 볼 값 |
|---|---|---|
| `/locomotion_test/status` | `std_msgs/String` | FSM, mission, waypoint, detail |
| `/locomotion_test/telemetry` | `std_msgs/String` JSON | `odom_calibrated`, ego pose/velocity, active goal/waypoint, errors |
| `/locomotion_test/goal_pose` | `PoseStamped` | map-frame carrot 시각화 |
| `/locomotion_test/goal_rel` | `Vector3Stamped` | E0로 들어가기 전 robot-frame 3값 |
| `/locomotion_test/policy_debug` | `std_msgs/String` JSON | `goal_stale=false`, received count 증가, `low_state_age_sec<0.2`, finite action min/max |

mission 실행 중 acceptance:

- `odom_calibrated=true` 전에는 goal이 없어야 한다.
- PLAYING 동안 `goal_messages_received`가 계속 증가해야 한다.
- PLAYING 동안 `goal_stale=false`여야 한다.
- `low_state_age_sec < 0.20`이어야 한다.
- `action_min/max`가 finite이고 [-1,1] 안이어야 한다.
- mission 0/finished/odom loss 후 0.5초 안에 `goal_stale=true`, applied goal은 `(0,0,0)`이어야 한다.
- 이 stale 동작은 E-stop이 아니다. 비정상 자세/통신/actor 문제는 deploy cleanup DAMPING 또는
  물리 remote/E-stop으로 처리한다.

## 아직 남은 blocker

1. 서버 접속 복구.
2. Mac의 GitHub DNS/network 복구 후 두 repo `ekay-fix` push; 그 다음 server clean pull.
3. frozen run config와 deploy config의 normalization/default pose/action scale 최종 diff.
4. E0@6200 export, hash 기록, robot staging copy와 **정확한 E0 actor** smoke.
5. 사람이 로봇 옆에서 hoist 승격 gate 수행.

이 다섯 항목 전에는 “mission 수행용으로 준비 완료”가 아니라 **코드/Brain staging 완료**다.
