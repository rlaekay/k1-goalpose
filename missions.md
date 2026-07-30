# Locomotion Test Missions

이 문서는 real robot에서 `E0_armB_armsdown`부터 시작해 여러 GoalPose policy를 갈아끼워 시험하기 위한 BT mission harness 계획과 실행법이다.

현재 구현 위치는 두 repo로 나뉜다.

| 영역 | repo / branch | 수정 파일 | 역할 |
|---|---|---|---|
| 학습·배포 문서 | `/Users/dmdrb/RoboCup/k1-goalpose`, `ekay-fix` | `missions.md`, `ROBOT_DEPLOY_E0_GUIDE.md` | 실험 설계, policy 교체 기준, real deploy refactor 기록 |
| BT·로봇 실행 | `/Users/dmdrb/RoboCup/[07]sim2real/INHA-Player`, `ekay-fix` | `src/brain/behavior_trees/locomotion_test.xml`, `src/brain/include/mission_test.h`, `src/brain/src/mission_test.cpp`, `src/brain/config/config.yaml` | mission FSM, command/status topic, map-frame lookahead 생성 |

## 결론 구조

BT는 mission을 고르고 map-frame path/goal을 생성한다. 그 아래 계층은 현재 LocalPlanner를 통해 `setVelocity()`로 이어지고, E0 실기 배포가 붙으면 같은 lookahead target을 policy adapter가 robot-frame `(goal_rel_x, goal_rel_y, heading_error)`로 변환해 actor에 넣는다.

```text
Laptop terminal
  -> /locomotion_test/command
  -> BT LocomotionTest FSM
  -> map-frame mission path + lookahead target
  -> /locomotion_test/goal_pose
  -> current: LocalPlanner -> RobotClient::setVelocity -> SDK walk
  -> E0 target: GoalPoseAdapter -> actor obs[6:16] -> LowCmd joint targets
```

이렇게 나눈 이유는 policy를 E0에서 E1/G1/G3 등으로 갈아껴도 BT XML을 다시 만들지 않기 위해서다. 바뀌어야 하는 것은 `locomotion_test.active_policy`와 하단 adapter/profile이다.

## 실행 계획과 현재 반영 상태

| 순서 | 하위 태스크 | 상태 | 판단 근거 |
|---|---:|---|---|
| 1 | 기존 ekay odom/global PF/LocalPlanner 연결 확인 | 완료 | `ResetOdometry`는 ekay mode에서 `requestOdomRebase(0.0)`, `SelfLocateEnterField`는 `globalInit()`을 호출한다. |
| 2 | `locomotion_test.xml` 추가 | 완료 | 경기용 `game.xml`과 분리해 실험 중 role/chase/kick 로직이 섞이지 않게 한다. |
| 3 | mission FSM + command/status topic 추가 | 완료 | 노트북에서 BT가 돌고 있어도 `/locomotion_test/command`로 mission을 선택할 수 있다. |
| 4 | mission/repeat/threshold/lookahead config화 | 완료 | `src/brain/config/config.yaml`의 `locomotion_test.*`에 모았다. 초기 도달 threshold는 10 cm, 6°다. |
| 5 | E0 real LowCmd adapter | 미구현, 다음 단계 | 현재 INHA-Player 경로는 여전히 LocalPlanner → `setVelocity()`다. E0 actor를 실제로 쓰려면 `ROBOT_DEPLOY_E0_GUIDE.md`의 `policy_goal_pose.py`/`deploy_goal_pose.py` 작업이 필요하다. |
| 6 | 로봇/서버 build 검증 | 로컬 미검증 | 현재 로컬 Mac에는 `colcon`이 없어서 build 불가. 로봇/서버 ROS2 환경에서 아래 build 명령으로 확인해야 한다. |

## 실행법

로봇/서버에서:

```bash
cd /Users/dmdrb/RoboCup/[07]sim2real/INHA-Player
scripts/codex_build.sh --codex-brain --packages-select brain
source install/setup.bash
ros2 launch brain launch.py tree:=locomotion_test
```

노트북 터미널에서 같은 ROS domain/network에 붙은 뒤:

```bash
ros2 topic echo /locomotion_test/status
ros2 topic echo /locomotion_test/goal_pose
```

mission 실행:

```bash
ros2 topic pub --once /locomotion_test/command std_msgs/msg/String "{data: mission1}"
ros2 topic pub --once /locomotion_test/command std_msgs/msg/String "{data: mission2}"
ros2 topic pub --once /locomotion_test/command std_msgs/msg/String "{data: mission3}"
ros2 topic pub --once /locomotion_test/command std_msgs/msg/String "{data: mission4}"
ros2 topic pub --once /locomotion_test/command std_msgs/msg/String "{data: mission5}"
```

수동 시작이 필요하면:

```bash
ros2 topic pub --once /locomotion_test/command std_msgs/msg/String "{data: select mission2}"
ros2 topic pub --once /locomotion_test/command std_msgs/msg/String "{data: play}"
```

정지/초기화:

```bash
ros2 topic pub --once /locomotion_test/command std_msgs/msg/String "{data: stop}"
ros2 topic pub --once /locomotion_test/command std_msgs/msg/String "{data: status}"
```

## FSM 정의

| 상태 | 조건 | 동작 |
|---|---|---|
| `prep` | mission 없음, 시작점 이동 중, 또는 `finished` 후 5초 경과 | 선택된 mission의 시작 pose가 있으면 그쪽으로 이동. 없으면 정지. |
| `ready` | 시작 pose 도달 | `select missionN`으로 들어온 경우 대기. `missionN` 명령은 기본 autoplay라 바로 `playing`으로 넘어간다. |
| `playing` | mission goal sequence 수행 중 | map-frame path 위 lookahead를 계속 갱신해 하단 planner/adapter에 전달한다. |
| `finished` | 모든 waypoint 완료 | 정지 명령을 유지하고 5초 후 `prep`으로 복귀한다. |

## Config 항목

초기값은 `src/brain/config/config.yaml`에 들어갔다.

```yaml
locomotion_test:
  command_topic: "/locomotion_test/command"
  status_topic: "/locomotion_test/status"
  goal_topic: "/locomotion_test/goal_pose"
  active_policy: "e0"
  goal_reached_xy_m: 0.10
  goal_reached_theta_deg: 6.0
  lookahead_min_m: 0.25
  lookahead_default_m: 0.55
  lookahead_max_m: 0.90
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

주의: `command_topic`, `status_topic`, `goal_topic`은 subscription/publisher 생성 시점에 결정되므로 restart-only다. threshold, repeat, lookahead 숫자는 runtime `ros2 param set`으로 바꿔도 다음 tick부터 반영되도록 연결했다.

## Mission 정의

| mission | 시작 pose | 목표 sequence | heading 처리 |
|---|---|---|---|
| 1 | `(0,0,0)` | CW 360°, CCW 360°를 3회 | XY path가 아니라 heading waypoint를 45° 간격으로 생성한다. pose theta는 wrap되므로 한 번에 `2π`를 주면 회전 한 바퀴가 사라진다. |
| 2 | `(0,0,0)` | `(3,0,0)` ↔ `(-3,0,0)` 3회 | 최종 heading fixed `0`. 멀 때는 tangent heading, 가까우면 final heading. |
| 3 | `(0,0,0)` | `(0,-2,0)` ↔ `(0,2,0)` 3회 | mission2와 동일. |
| 4 | 현재 pose | 중심 `(-3,0)`, 반지름 `6m` 원 위 random point/heading 4개 | seed 고정. point angle과 heading을 모두 random으로 생성한다. |
| 5 | `(-2,-0.5,0)` | 1m 간격 ㄹ자, 총 9m | heading irrelevant. 완료 판정은 position만 본다. 이동 중 target heading은 tangent로 준다. |

mission5의 기본 path는 다음처럼 1m segment 9개다.

```text
(-2,-0.5) -> (-1,-0.5) -> (-1,0.5) -> (0,0.5) -> (0,-0.5)
          -> (1,-0.5) -> (1,0.5) -> (2,0.5) -> (2,-0.5) -> (3,-0.5)
```

## Lookahead 생성 원리

코드는 pure-pursuit 계열 방식으로 구현했다.

1. mission path를 map 좌표계 polyline으로 만든다.
2. 현재 robot map pose를 path 위에 projection한다.
3. projection station에서 `lookahead_default_m`만큼 앞의 carrot point를 잡는다.
4. carrot point를 LocalPlanner target으로 준다.
5. E0 adapter가 붙으면 같은 map-frame carrot을 robot frame으로 바꾼다.

외부 근거:

- MathWorks Pure Pursuit 문서는 robot이 waypoint list 위의 lookahead point를 추종한다고 설명한다. 또한 lookahead가 크면 path가 부드러워지지만 corner cutting이 커지고, 작으면 path를 잘 회복하지만 oscillation이 커질 수 있다고 정리한다: [Pure Pursuit](https://www.mathworks.com/help/nav/ref/purepursuit.html), [Pure Pursuit Controller](https://www.mathworks.com/help/nav/ug/pure-pursuit-controller.html).
- KAIST의 pure-pursuit lookahead 최적화 연구는 smoothness와 tracking error의 tradeoff 때문에 상황에 맞는 lookahead 조정이 필요하다고 본다: [Vehicle Path Tracking Control Using Pure Pursuit with MPC-Based Look-Ahead Optimization](https://pure.kaist.ac.kr/en/publications/vehicle-path-tracking-control-using-pure-pursuit-with-mpc-based-l/).
- SNU의 lookahead 조정 연구도 pure pursuit의 장점이 localization error robustness와 실시간성에 있지만, lookahead 선택이 성능을 좌우한다고 본다: [Accurate Path Tracking by Adjusting Look-Ahead Point in Pure Pursuit](https://snu.elsevierpure.com/en/publications/accurate-path-tracking-by-adjusting-look-ahead-point-in-pure-purs/).

이번 기본값을 0.55m로 둔 이유:

- `STATE_ESTIMATION.md`는 production GoalPose 범위를 `dx∈[-2,2]`, `dy∈[-1.5,1.5]`로 제한하고, A* lookahead도 당분간 이 envelope 안으로 제한하라고 정리했다.
- E0는 path policy가 아니라 single waypoint precision baseline이다. `K1_LEARNING_HISTORY_KO.md`와 `masterplan3.md` 기준 E0@6200은 위치 2.7cm/p90 5.0cm, heading 2.5°, strict 89.3%로 가장 좋은 유효 baseline이지만, path/speed 가설은 아직 분리 검증되지 않았다.
- 따라서 첫 실기에서는 긴 lookahead로 속도를 욕심내기보다 0.25–0.90m 안에서 짧고 보수적으로 시작한다.

## Orientation이 상관없는 goal 처리

heading이 상관없는 mission5 같은 경우에도 E0 actor의 observation에는 `heading_error` 채널이 있다. 따라서 “theta를 아예 안 준다”는 선택지는 없다.

현재 처리:

- 완료 판정: position threshold만 사용한다.
- 이동 중 target heading: path tangent를 준다.
- near-final heading gate: 꺼진다.

이 방식의 장점은 actor/LocalPlanner 입력 layout을 유지하면서도, mission 평가에서는 heading을 성공 조건으로 삼지 않는다는 점이다. 단점은 tangent heading 때문에 robot이 path 방향으로 몸을 돌리려는 bias가 남는다는 점이다. 만약 실기에서 lawnmower time이 더 중요하고 heading 안정성이 충분하면 `HeadingMode::CURRENT` profile을 추가해 현재 heading을 유지하게 바꾸는 것이 다음 후보가 된다.

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
3. `lookahead_default_m`, `lookahead_max_m`, `heading_blend_distance_m`만 profile별로 조정한다.
4. mission XML은 그대로 둔다.

## 다음 구현 단계: E0 adapter 연결

현재 `locomotion_test.xml`은 mission/odom/lookahead harness다. 실제 E0를 쓰려면 하단 경로를 바꿔야 한다.

현재:

```text
LocomotionTest -> LocalPlanner -> RobotClient::setVelocity() -> SDK walk
```

목표:

```text
LocomotionTest -> GoalPoseAdapter -> E0 actor -> LowCmd
```

필수 작업은 `ROBOT_DEPLOY_E0_GUIDE.md`에 이어서 관리한다.
