# Locomotion Test Missions

이 문서는 real robot에서 `E0_armB_armsdown`부터 시작해 여러 GoalPose policy를 갈아끼워 시험하기 위한 BT mission harness 계획과 실행법이다.

> 2026-07-30 실기 환경 감사, Claude critic, 서버→로봇 export/copy, 안전 gate와
> 실제 경로는 `MISSION_DEPLOY_AUDIT_20260730.md`를 기준으로 한다. 이 문서는 mission/FSM
> 정의를 설명한다.

현재 구현 위치는 두 repo로 나뉜다.

| 영역 | repo / branch | 수정 파일 | 역할 |
|---|---|---|---|
| 학습·배포 문서 | `/Users/dmdrb/RoboCup/k1-goalpose`, `ekay-fix` | `missions.md`, `ROBOT_DEPLOY_E0_GUIDE.md` | 실험 설계, policy 교체 기준, real deploy refactor 기록 |
| BT·로봇 실행 | `/Users/dmdrb/RoboCup/[07]sim2real/INHA-Player`, `ekay-fix` | `src/brain/behavior_trees/locomotion_test.xml`, `src/brain/include/mission_test.h`, `src/brain/src/mission_test.cpp`, `src/brain/config/config.yaml` | mission FSM, command/status topic, map-frame lookahead 생성 |

## 결론 구조

BT는 mission을 고르고 map-frame waypoint/carrot을 생성한다. `LocomotionTest`가 최신
camera-PF pose로 이를 robot-frame `(goal_rel_x, goal_rel_y, heading_error)`로 변환해
ROS topic으로 내고, 별도 E0 deploy process가 actor와 LowCmd를 소유한다. 이 tree에는
LocalPlanner, `setVelocity()`, SDK `RobocupWalk`가 없다.

```text
Laptop terminal
  -> SSH robot-side ROS CLI -> /locomotion_test/mission_id (Int32 0..5)
  -> BT LocomotionTest FSM
  -> map-frame mission waypoint + radial carrot
  -> /locomotion_test/goal_pose (viz)
  -> /locomotion_test/goal_rel (robot-frame E0 input)
  -> /locomotion_test/telemetry
  -> deploy_goal_pose.py -> E0 actor obs[6:16] -> LowCmd joint targets
  -> /locomotion_test/policy_debug
```

이렇게 나눈 이유는 policy를 E0에서 E1/G1/G3 등으로 갈아껴도 BT XML을 다시 만들지 않기 위해서다. 바뀌어야 하는 것은 `locomotion_test.active_policy`와 하단 adapter/profile이다.

## 실행 계획과 현재 반영 상태

| 순서 | 하위 태스크 | 상태 | 판단 근거 |
|---|---:|---|---|
| 1 | 기존 ekay odom/global PF 연결 확인 | 완료 | `ResetOdometry`는 ekay mode에서 `requestOdomRebase(0.0)`, `SelfLocateEnterField`는 `globalInit()`을 호출한다. mission tree는 LocalPlanner를 호출하지 않는다. |
| 2 | `locomotion_test.xml` 추가 | 완료 | 경기용 `game.xml`과 분리해 실험 중 role/chase/kick 로직이 섞이지 않게 한다. |
| 3 | mission FSM + command/status topic 추가 | 완료 | Mac의 `missionctl.sh`가 robot-side ROS CLI를 통해 숫자 mission topic을 보낸다. |
| 4 | mission/repeat/threshold/lookahead config화 | 완료 | `src/brain/config/config.yaml`의 `locomotion_test.*`에 모았다. 초기 도달 threshold는 10 cm, 6°다. |
| 5 | E0 real LowCmd adapter | 코드 구현 | `policy_goal_pose.py`와 `deploy_goal_pose.py`가 ROS `goal_rel`→54 obs→12 action→23-joint LowCmd를 연결한다. 실기 policy 검증은 미완료. |
| 6 | 로봇/server 준비 | 일부 완료/차단 | clean Brain build와 ROS bridge smoke 통과. server `:6666` 접속 및 E0@6200 `.pth`/frozen config hash 확인. `.pt` 미export·robot 미복사이므로 export/copy/hash/E0 smoke 전에는 실행 금지. |

## 실행법

로봇의 **clean mission staging workspace**에서:

```bash
cd <ROBOT_WS>/brain_ws
source /opt/ros/humble/setup.bash
source <ROBOT_GAME_WS>/install/setup.bash
colcon build --packages-select brain --executor sequential --parallel-workers 1
source install/setup.bash
```

그 다음 mission에 필요한 process만 서로 다른 robot terminal에서 실행한다. 경기용
`start.sh`는 whistle/game-controller/sound까지 띄우므로 쓰지 않는다.

```bash
# terminal A: camera-PF localization (기존 install의 vision package를 underlay로 사용)
source /opt/ros/humble/setup.bash
source <ROBOT_GAME_WS>/install/setup.bash
source <ROBOT_WS>/brain_ws/install/setup.bash
ros2 launch vision launch.py vision_config_path:=/opt/booster \
  ekay_odom:=true save_data:=false show_det:=false

# terminal B: mission-only Brain
source /opt/ros/humble/setup.bash
source <ROBOT_GAME_WS>/install/setup.bash
source <ROBOT_WS>/brain_ws/install/setup.bash
ros2 launch brain launch.py tree:=locomotion_test \
  vision_config_path:=/opt/booster disable_com:=true
```

terminal C의 E0 deploy 명령은 `MISSION_DEPLOY_AUDIT_20260730.md`의
`Mission-only 실행` 절을 그대로 따른다.

Mac에는 ROS 2가 없으므로 repo의 SSH helper를 쓴다:

```bash
./missionctl.sh check
./missionctl.sh watch telemetry
./missionctl.sh watch goal-rel
./missionctl.sh watch policy
```

mission 실행은 숫자 topic을 기본으로 쓴다. 숫자는 사용자가 지정한 mission 순서 그대로다.

```bash
./missionctl.sh 1
./missionctl.sh 2
./missionctl.sh 3
./missionctl.sh 4
./missionctl.sh 5
```

수동 select/play가 필요하면 robot의 ROS terminal에서:

```bash
ros2 topic pub --once /locomotion_test/command std_msgs/msg/String "{data: select mission2}"
ros2 topic pub --once /locomotion_test/command std_msgs/msg/String "{data: play}"
```

정지/초기화:

```bash
./missionctl.sh 0
```

`mission_id=0`은 BT goal stream을 끊는 명령이지 emergency stop이 아니다. 실기 fault는
deploy의 DAMPING cleanup과 물리 remote/E-stop으로 처리한다.

## FSM 정의

| 상태 | 조건 | 동작 |
|---|---|---|
| `prep` | mission 없음, 시작점 이동 중, 또는 `finished` 후 5초 경과 | 선택된 mission의 시작 pose가 있으면 그쪽으로 이동. 없으면 정지. |
| `ready` | 시작 pose 도달 | `select missionN`으로 들어온 경우 대기. `missionN` 명령은 기본 autoplay라 바로 `playing`으로 넘어간다. |
| `playing` | mission goal sequence 수행 중 | 현재 waypoint 방향의 map-frame radial carrot을 계속 갱신해 E0 adapter에 전달한다. |
| `finished` | 모든 waypoint 완료 | 정지 명령을 유지하고 5초 후 `prep`으로 복귀한다. |

## Config 항목

초기값은 `src/brain/config/config.yaml`에 들어갔다.

```yaml
locomotion_test:
  command_topic: "/locomotion_test/command"
  mission_id_topic: "/locomotion_test/mission_id"
  status_topic: "/locomotion_test/status"
  telemetry_topic: "/locomotion_test/telemetry"
  goal_topic: "/locomotion_test/goal_pose"
  active_policy: "e0"
  goal_reached_xy_m: 0.10
  goal_reached_theta_deg: 6.0
  lookahead_min_m: 0.25
  lookahead_default_m: 0.55
  lookahead_max_m: 2.0
  heading_blend_distance_m: 0.60
  finished_hold_sec: 5.0
```

반복횟수와 mission별 목표도 같은 section에 있다. 예를 들어 mission2의 앞/뒤 거리는:

```yaml
mission2:
  repeat: 3
  forward_x_m: 3.0
  backward_x_m: -3.0
```

주의: `command_topic`, `mission_id_topic`, `status_topic`, `telemetry_topic`, `goal_topic`은 subscription/publisher 생성 시점에 결정되므로 restart-only다. threshold, repeat, lookahead 숫자는 runtime `ros2 param set`으로 바꿔도 다음 tick부터 반영되도록 연결했다.

## Telemetry stream

`/locomotion_test/telemetry`는 `std_msgs/msg/String` JSON이다. 비교 실험에서 walk별 log를 같은 schema로 저장하기 위한 stream이다. 모든 telemetry 각도는 degree 단위로 낸다.
실제 E0 입력 topic인 `/locomotion_test/goal_rel`의 heading은 radian이다.

주요 필드:

| field | 의미 |
|---|---|
| `original_timestamp` | Brain ROS clock 기준 원본 송출 timestamp |
| `fsm_state` | `prep`, `ready`, `playing`, `finished` |
| `mission.id` | 숫자 mission id. `1..5`, `0`은 없음/stop |
| `mission.elapsed_sec` | 단일 mission playing duration. finished 상태에서는 완료 시 duration으로 고정 |
| `waypoint.index/count` | 현재 수행 중인 waypoint index와 전체 count |
| `ego_pose_map.{x_m,y_m,theta_deg}` | localization 기준 robot map pose |
| `ego_velocity_map_diff.{vx_mps,vy_mps}` | ego pose 차분으로 계산한 실제 이동 속도. **map(world) frame**. |
| `ego_velocity_map_diff.{vx_body_mps,vy_body_mps}` | 같은 실측 속도를 body frame(forward/left)으로 회전한 값. |
| `ego_velocity_map_diff.{speed_mps,vtheta_degps}` | frame 무관 speed와 yaw rate |
| `active_goal_map.pose` | E0 adapter에 넘기는 carrot goal (map frame). velocity 제어기는 제거됨 — LocomotionTest는 goal pose만 낸다. |
| `active_waypoint_map.pose` | mission sequence상 현재 완료해야 하는 waypoint |
| `pose_error_to_goal` | carrot goal 기준 error. robot-frame `goal_rel_x/y` + `heading_error`가 **곧 E0 policy 입력**(obs index 6,7,8)이다. |
| `pose_error_to_waypoint` | 최종 waypoint 기준 error |
| `thresholds` | 도달 판정 threshold와 lookahead 설정 |

Mac에서 저장:

```bash
./missionctl.sh watch telemetry | tee locomotion_test_telemetry.log
```

JSON만 뽑아 CSV로 바꾸려면 `data:` 라인만 추출해서 Python `json.loads()`로 처리하면 된다. localization이 순간적으로 NaN/Inf를 내도 해당 숫자 field는 `null`로 나가므로 한 줄이 깨져서 capture 전체 parsing이 실패하는 일은 없다.

## Mission 정의

| mission | 시작 pose | 목표 sequence | heading 처리 |
|---|---|---|---|
| 1 | `(0,0,0)` | CW 360°, CCW 360°를 3회 | 위치는 현재 pose에 두고 목표 heading을 회전 방향 60° 앞세운다. 완료는 wrapped pose가 아니라 누적 unwrapped yaw로 판정한다. |
| 2 | `(0,0,0)` | `(3,0,0)` ↔ `(-3,0,0)` 3회 | 모든 carrot에서 map heading `0`을 고정하므로 뒤쪽 목표에는 몸을 돌리지 않고 후진한다. |
| 3 | `(0,0,0)` | `(0,-2,0)` ↔ `(0,2,0)` 3회 | 모든 carrot에서 map heading `0`을 고정하므로 좌우 측면 보행을 요구한다. |
| 4 | 현재 pose | 중심 `(-3,0)`, 반지름 `6m` 원 위 random point/heading 4개 | seed 고정. point angle과 heading을 모두 random으로 생성한다. |
| 5 | `(-2,-0.5,0)` | 1m 간격 ㄹ자, 총 9m | heading irrelevant. 완료 판정은 position만 본다. 이동 중 target heading은 tangent로 준다. |

mission5의 기본 path는 다음처럼 1m segment 9개다.

```text
(-2,-0.5) -> (-1,-0.5) -> (-1,0.5) -> (0,0.5) -> (0,-0.5)
          -> (1,-0.5) -> (1,0.5) -> (2,0.5) -> (2,-0.5) -> (3,-0.5)
```

## 현재 carrot 생성 원리

`afb731d2`의 현재 코드는 이전 polyline-projection pure-pursuit 구현을 제거했다.

1. 현재 waypoint까지 map-frame `(dx,dy)`와 거리를 구한다.
2. 2 m 이내면 waypoint 자체를 carrot으로 쓴다.
3. 2 m보다 멀면 waypoint 방향으로 정확히 2 m인 radial carrot을 쓴다.
4. spin은 위치를 현재 pose로 고정하고 회전 방향으로 heading을 60° 앞세운다.
5. Brain이 carrot을 robot frame으로 변환해 `/locomotion_test/goal_rel`을 publish한다.
6. deploy가 x±2 m/y±1.5 m를 per-axis clamp한 뒤 E0 observation에 넣는다.

현재 2.0 m cap의 근거와 한계:

- `STATE_ESTIMATION.md`는 production GoalPose 범위를 `dx∈[-2,2]`, `dy∈[-1.5,1.5]`로 제한하고, A* lookahead도 당분간 이 envelope 안으로 제한하라고 정리했다.
- E0는 path policy가 아니라 single waypoint precision baseline이다. `K1_LEARNING_HISTORY_KO.md`와 `masterplan3.md` 기준 E0@6200은 위치 2.7cm/p90 5.0cm, heading 2.5°, strict 89.3%로 가장 좋은 유효 baseline이지만, path/speed 가설은 아직 분리 검증되지 않았다.
- x 방향 training envelope가 ±2 m여서 radial cap을 2 m로 뒀다.
- 순수 lateral radial carrot은 y=±2 m까지 나올 수 있으므로 deploy에서 training envelope인
  ±1.5 m로 clamp한다. 이때 방향이 조금 변형된다는 점을 telemetry로 확인해야 한다.
- `lookahead_min_m`, `lookahead_default_m`, `heading_blend_distance_m`,
  `mission1.yaw_step_deg`는 현재 구현에서는 config compatibility를 위해 남아 있지만 carrot
  계산이나 spin 완료 판정에는 쓰이지 않는다.

## Orientation이 상관없는 goal 처리

heading이 상관없는 mission5 같은 경우에도 E0 actor의 observation에는 `heading_error` 채널이 있다. 따라서 “theta를 아예 안 준다”는 선택지는 없다.

현재 처리:

- 완료 판정: position threshold만 사용한다.
- 이동 중 target heading: path tangent를 준다.
- near-final heading gate: 꺼진다.

이 방식의 장점은 actor 입력 layout을 유지하면서도, mission 평가에서는 heading을 성공 조건으로 삼지 않는다는 점이다. 단점은 tangent heading 때문에 robot이 path 방향으로 몸을 돌리려는 bias가 남는다는 점이다. 만약 실기에서 lawnmower time이 더 중요하고 heading 안정성이 충분하면 `HeadingMode::CURRENT` profile을 추가해 현재 heading을 유지하게 바꾸는 것이 다음 후보가 된다.

## E0와 다른 policy에서 goal/lookahead setting이 달라지는가?

달라질 수 있다. 다만 BT mission interface는 바꾸지 않는 것이 맞다.

| policy | 현재 판단 | lookahead/profile |
|---|---|---|
| E0 | 실기 첫 기준선. waypoint precision 검증용. path 학습 없음. | 짧은 lookahead, final heading 엄격, speed 욕심 금지. |
| E1_path | moving carrot/path machinery를 시험하려 했지만 기존 비교는 코드 의미 drift 때문에 무효. 재평가 report에서는 path_lag median/p90이 0cm였지만 낙상 94회라 안전 확인 필요. | 별도 실기 승격 전까지 E0와 같은 보수 profile. |
| V7_full | path_lag p90 28.1cm, max 396.7cm로 E1보다 추종 tail이 큼. 기존 수치는 frozen config 오염 때문에 결론으로 쓰면 안 됨. | 실기 첫 후보 아님. |
| G1/G3 이후 | E0 warm-start 위에서 속도/강건성/연속전환을 다시 검증하는 batch. | 성공하면 longer/adaptive lookahead를 profile로 분리. |

즉 “policy별로 lookahead 값은 달라질 수 있지만 mission BT는 고정”이 결론이다. policy를 갈아낄 때는:

1. `locomotion_test.active_policy` 값을 바꾼다.
2. 하단 GoalPoseAdapter가 해당 policy의 checkpoint/config/profile을 로드한다.
3. 현재 구현에서는 `lookahead_max_m` radial cap과 deploy x/y clamp를 profile별로 조정한다.
4. mission XML은 그대로 둔다.

## 현재 연결 상태와 남은 배포 단계

코드 연결은 다음 형태로 완료됐다.

```text
LocomotionTest -> /locomotion_test/goal_rel -> GoalPosePolicy -> E0 actor -> LowCmd
```

남은 것은 server의 E0@6200 export, robot staging copy/hash, 정확한 E0 actor smoke,
hoist safety gate다. Brain clean build와 ROS topic bridge smoke는 통과했다. 정확한 명령과
2026-07-30 현장 근거는
`MISSION_DEPLOY_AUDIT_20260730.md`에 있다.
