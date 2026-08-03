# Locomotion Test Missions

이 문서는 real robot에서 `E0_armB_armsdown`부터 시작해 여러 GoalPose policy를 갈아끼워 시험하기 위한 BT mission harness 계획과 실행법이다.

> 2026-07-30 실기 환경 감사, Claude critic, 서버→로봇 export/copy, 안전 gate와
> 실제 경로는 `MISSION_DEPLOY_AUDIT_20260730.md`를 기준으로 한다. 이 문서는 mission/FSM
> 정의를 설명한다.
>
> **2026-08-04 변경**: ekay-odom(카메라 PF만으로 x/y)을 전부 제거하고 normal PF +
> IMU 적분 fallback 구조로 롤백했다. mission2/3은 왕복으로, FSM에는 timeout/FAILED,
> waypoint별 시각 telemetry, spin aliasing 가드, mission별 threshold를 추가했다.
> 롤백 이전 버전은 `missions_v1_pre_rollback_20260804.md`에 보존.

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
| 1 | odom/global PF 연결 확인 | 완료 (2026-08-04 롤백) | ekay-odom 제거. `ResetOdometry`는 `requestOdomRebase()`, `SelfLocateEnterField`는 `globalInit()`을 호출한다. odom source는 SDK `odometer_state`이고, CUSTOM 모드에서 끊기면 IMU 적분이 인계한다(아래 절). mission tree는 LocalPlanner를 호출하지 않는다. |
| 2 | `locomotion_test.xml` 추가 | 완료 | 경기용 `game.xml`과 분리해 실험 중 role/chase/kick 로직이 섞이지 않게 한다. |
| 3 | mission FSM + command/status topic 추가 | 완료 | Mac의 `missionctl.sh`가 robot-side ROS CLI를 통해 숫자 mission topic을 보낸다. |
| 4 | mission/repeat/threshold/lookahead config화 | 완료 | `src/brain/config/config.yaml`의 `locomotion_test.*`에 모았다. 초기 도달 threshold는 10 cm, 6°다. |
| 5 | E0 real LowCmd adapter | 코드 구현 | `policy_goal_pose.py`와 `deploy_goal_pose.py`가 ROS `goal_rel`→54 obs→12 action→23-joint LowCmd를 연결한다. 실기 policy 검증은 미완료. |
| 6 | 로봇/server 준비 | 일부 완료/차단 | clean Brain build와 ROS bridge smoke 통과. server `:6666` 접속 및 E0@6200 `.pth`/frozen config hash 확인. `.pt` 미export·robot 미복사이므로 export/copy/hash/E0 smoke 전에는 실행 금지. |

## Odometry: CUSTOM 모드에서 무엇이 PF를 미는가

이게 이 미션 전체의 급소다. Brain의 particle filter는 `predict(odom)` + `correct(camera)`
구조라서 odom이 없으면 카메라 프레임 사이에 위치가 아예 안 움직인다.

- `/odometer_state`는 **Booster SDK 보행 컨트롤러가** 내는 다리 기구학 odom이다.
  IMU 적분이 아니다.
- E0 deploy는 `ChangeMode(RobotMode.kCustom)`으로 LowCmd를 직접 잡는다. 그러면 SDK
  보행 컨트롤러가 꺼지므로 **`/odometer_state`가 멈출 수 있다.** (codex는 멈춘다고
  단언했지만 하드웨어에서 확인된 적은 없다.)

모든 walk가 CUSTOM에서 도는 이상 SDK odometer는 사실상 항상 없다. 그래서 기본값은
`robot.imu_odom_mode: "on"`이다.

| mode | 동작 |
|---|---|
| `off` | SDK `odometer_state`만. SDK walk로 경기 돌릴 때만 |
| **`on` (기본)** | 항상 IMU 적분. SDK 구독 자체를 안 만든다 |
| `auto` | SDK 쓰다가 `imu_odom_takeover_sec`(0.3s) 이상 끊기면 IMU 인계, 복귀하면 반납 |

⚠️ `config.yaml`은 경기 tree와 공유된다. **`on`이면 SDK walk 경기에서도 IMU odom을
쓴다.** 경기용으로 되돌릴 때는 `off`(또는 `auto`)로 바꿔야 한다.

### odom frame 원점 = anchor pose

기동 시점을 원점으로 잡으면 그 원점 자체가 아직 확정되지 않은 pose다. 그래서
**초기 pose가 확정되는 anchor 시점에 odom frame 원점을 그 pose로 다시 잡는다.**

anchor 조건은 `locator.startup_correction_frames`(50) + `locator.sentinel_anchor_extra_frames`(20)
= **camera correction 70회**다. (`startup_correction_frames`를 20 → 50으로 올렸다.
정확히 50회에 잡고 싶으면 `sentinel_anchor_extra_frames`를 0으로.)

anchor 이전에도 IMU propagation은 돈다 — 다만 그 구간의 원점은 의미가 없다. anchor
순간에 하는 일:

- `robotPoseToOdom = (0,0,0)`, `odomToField = anchor pose` → `robotPoseToField`는
  anchor 그대로라 pose가 튀지 않는다
- `locator->lastOdom` / `relocPf->lastOdom` 동기화 → predict가 리셋을 이동으로 오해하지 않음
- `PlanarImuOdometry::rebaseOrigin()` → 적분기 원점도 이동. **가속도계 bias 추정치는 유지**
  (`reset()`을 쓰면 2초 정지 calibration을 다시 해야 한다)

`imu_odom_rebase_on_anchor: false`로 끄면 기동 시점이 원점이 된다.

```text
[orientation-sentinel] anchor correction=70 pose_theta=12.3deg imu_yaw=11.9deg
[odom-anchor] origin set at (0.04, -0.11, 12.30deg); odom_eval epoch=epoch001_t1722 logging
```

### odom reset은 CUSTOM-safe 노드로 교체됨

기존 `ResetOdometry` BT 노드는 `client->resetOdometry()`(**SDK loco API**)를 보내고
`locoApiSubscriberCount() > 0`을 기다린다. CUSTOM에서는 그 subscriber가 안 생기므로
**3초 timeout 후 FAILURE**가 나고, `Sequence` 안에 있으므로 뒤의
`SelfLocateEnterField`/`LocomotionTest`까지 같이 무너진다. 게다가 CUSTOM에서 SDK
odometer는 odom source도 아니라 리셋해봐야 의미가 없다.

`locomotion_test.xml`은 이제 `ResetOdometryLocal`을 쓴다 — brain-local odom epoch만
rebase하고 항상 SUCCESS. 경기 tree의 `ResetOdometry`는 건드리지 않았다.

IMU 적분의 한계와 그에 대한 방어:

- 가속도 2중적분은 오차가 `t²`로 자란다. 그래서 **마지막 camera correction 이후
  `imu_odom_vision_horizon_sec`(0.5s)을 넘으면 병진 적분을 얼리고 회전만 남긴다.**
  즉 "카메라 사이를 잇는 짧은 다리"지 독립 항법이 아니다.
- 기동 시 `imu_odom_calibration_sec`(2s)만큼 **로봇을 정지**시켜 가속도계 bias를
  추정한다. 안 되어 있으면 5초마다 로그가 뜬다.
- 정지 판정(gyro + 다리 관절 속도)일 때 속도를 0으로 리셋(ZUPT)한다.
- SDK용으로 측정한 `odom_factor_*` / `odom_bias_theta`는 IMU sample에는 적용하지
  않는다. 그건 SDK 추정치의 스케일 오차 보정값이라 적분값에 곱하면 이중보정이다.

디버그 스트림: `debug/imu_odom/{x,y,vx,vy,yaw,stationary,translation_enabled,calibrated,
vision_correction_age}`. `on` 모드에서는 합성 odom이 `odometer_state`로 나가고,
`auto`에서는 SDK 토픽을 오염시키지 않도록 `imu_odometer_state`로 나간다.

### odom eval: PF 궤적 vs IMU dead-reckoning

anchor를 **공통 시작점**으로 놓고 두 궤적을 CSV로 남긴다. 둘이 같은 곳에서 출발하므로
벌어지는 양이 곧 누적 IMU bias/noise다.

- reference = camera PF 추정 (`calibrateOdom`이 호출되는 매 correction)
- test = `anchor ⊕ robotPoseToOdom` = anchor 이후 **순수 dead-reckoning**
  (`odomToField`는 correction마다 갱신되므로 anchor 스냅샷을 따로 쓴다)

epoch은 anchor에서 열리고 **odom reset**에서 닫힌다. 닫힐 때 plot과 지표가 자동으로 구워진다.

```text
odom_eval/
  epoch001_t1722.csv            # t_sec,pf_x,pf_y,pf_theta,dr_x,dr_y,dr_theta
  epoch001_t1722_metrics.json   # APE/RPE/drift 지표 (항상 생성)
  epoch001_t1722.png            # 4-panel plot (matplotlib 있을 때만)
```

지표는 SLAM 관례(evo 스타일)를 따르고, **두 궤적이 원점을 공유하므로 Umeyama 정렬은
하지 않는다** — 정렬하면 찾으려는 bias가 지워진다.

| 지표 | 의미 |
|---|---|
| APE | `E_i = P_ref_i⁻¹ · P_test_i`. 누적 drift. 초반 오차 하나가 전체를 지배할 수 있음 |
| RPE | 고정 시간 간격(`odom_eval_rpe_delta_sec`, 1.0s) 상대 오차. **drift rate**, 초기 오차에 둔감 |
| `drift_per_metre` | 최종 오차 / 이동거리. 스케일 오차의 대표값 |
| `mean_err_forward/lateral_m` | reference body frame 기준 오차 성분. **한쪽으로 치우쳐 있으면 random noise가 아니라 bias** |

수동 실행:

```bash
python3 <brain_share>/tools/plot_odom_eval.py odom_eval/epoch001_t1722.csv --rpe-delta 1.0
python3 <brain_share>/tools/plot_odom_eval.py odom_eval/ --all
```

matplotlib이 없으면 PNG 없이 metrics JSON만 나오고 그 사실을 stdout에 알린다.
plot 실행은 detach된 thread라 제어 경로를 막지 않는다.

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
  save_data:=false show_det:=false

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
| `prep` | mission 없음, 시작점 이동 중, 또는 `finished`/`failed` 후 5초 경과 | 선택된 mission의 시작 pose가 있으면 그쪽으로 이동. 없으면 정지. |
| `ready` | 시작 pose 도달 | `select missionN`으로 들어온 경우 대기. `missionN` 명령은 기본 autoplay라 바로 `playing`으로 넘어간다. |
| `playing` | mission goal sequence 수행 중 | 현재 waypoint 방향의 map-frame radial carrot을 계속 갱신해 E0 adapter에 전달한다. |
| `finished` | 모든 waypoint 완료 | goal stream을 끊고 5초 후 `prep`으로 복귀한다. |
| `failed` | prep 또는 waypoint watchdog 초과 | goal stream을 즉시 끊고 실패 사유를 status/telemetry에 남긴 뒤 5초 후 `prep`으로 복귀한다. |

### Watchdog

이전 구현에는 timeout이 없어서, 시작 pose에 못 닿거나 waypoint 하나를 못 찍으면 FSM이
영원히 carrot을 쏘고 로봇은 계속 제자리걸음했다. mission4·5는 **시간 측정 과제**라
종료 보장이 없으면 결과 자체가 안 나온다.

| config | 기본값 | 의미 |
|---|---:|---|
| `prep_timeout_sec` | 60.0 | 시작 pose 도달까지 허용 시간. 0이면 끔 |
| `waypoint_timeout_sec` | 60.0 | waypoint 하나당 허용 시간. 0이면 끔 |

`odom_calibrated=false`인 동안(localization 대기)은 두 timer가 매 tick 밀린다.
localization 공백은 mission 실패가 아니기 때문이다.

실패 사유는 `mission_failed:waypoint_timeout_60.0s_at_wp4/7_forward_dist=1.85m`처럼
어느 waypoint에서 얼마나 남기고 멈췄는지까지 담긴다.

## Config 항목

초기값은 `src/brain/config/config.yaml`에 들어갔다.

```yaml
locomotion_test:
  command_topic: "/locomotion_test/command"
  mission_id_topic: "/locomotion_test/mission_id"
  status_topic: "/locomotion_test/status"
  telemetry_topic: "/locomotion_test/telemetry"
  goal_topic: "/locomotion_test/goal_pose"
  active_policy: "e0"        # telemetry 문자열일 뿐, policy를 선택하지 않는다
  goal_reached_xy_m: 0.10    # 전역 기본. mission별 override가 우선
  goal_reached_theta_deg: 6.0
  lookahead_min_m: 0.25
  lookahead_default_m: 0.55
  lookahead_max_m: 2.0
  heading_blend_distance_m: 0.60
  finished_hold_sec: 5.0
  prep_timeout_sec: 60.0
  waypoint_timeout_sec: 60.0
  spin_max_yaw_rate_radps: 6.0
```

반복횟수와 mission별 목표도 같은 section에 있다. 예를 들어 mission2의 앞/뒤 거리는:

```yaml
mission2:
  repeat: 3
  forward_x_m: 3.0   # 목표점
  return_x_m: 0.0    # 복귀점(=시작 pose)
```

주의: `command_topic`, `mission_id_topic`, `status_topic`, `telemetry_topic`, `goal_topic`은 subscription/publisher 생성 시점에 결정되므로 restart-only다. threshold, repeat, lookahead 숫자는 runtime `ros2 param set`으로 바꿔도 다음 tick부터 반영되도록 연결했다.

## Telemetry stream

`/locomotion_test/telemetry`는 `std_msgs/msg/String` JSON이다. 비교 실험에서 walk별 log를 같은 schema로 저장하기 위한 stream이다. 모든 telemetry 각도는 degree 단위로 낸다.
실제 E0 입력 topic인 `/locomotion_test/goal_rel`의 heading은 radian이다.

주요 필드:

| field | 의미 |
|---|---|
| `original_timestamp` | Brain ROS clock 기준 원본 송출 timestamp |
| `fsm_state` | `prep`, `ready`, `playing`, `finished`, `failed` |
| `mission.id` | 숫자 mission id. `1..5`, `0`은 없음/stop |
| `mission.elapsed_sec` | 단일 mission playing duration. finished/failed 상태에서는 종료 시 duration으로 고정 |
| `waypoint.index/count` | 현재 수행 중인 waypoint index와 전체 count |
| `waypoint.started_sec` | 현재 waypoint가 활성화된 시각(mission 시작 기준 상대초) |
| `waypoint.elapsed_sec` | 현재 waypoint에 매달린 시간 |
| `waypoint.reached_sec[]` | **waypoint별 도달 시각 배열**(mission 시작 기준 상대초). mission4의 4회 개별 시간, mission5의 9m 완주 시간이 후처리 없이 여기서 바로 나온다 |
| `health.spin_alias_rejects` | spin 적분에서 기각한 비물리적 heading delta 수. **0이 아니면 mission1 회전 적분이 오염된 run이므로 결과로 쓰면 안 된다** |
| `health.fail_reason` | `failed` 상태의 사유 문자열. 그 외에는 `null` |
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
| 1 | `(0,0,0)` | CW 360°, CCW 360°를 3회 | 위치는 현재 pose 고정. heading carrot은 **남은 회전량만큼** 앞세우되 `spin_max_lead_deg`(60°)로 포화. 완료는 wrapped pose가 아니라 누적 unwrapped yaw로 판정. |
| 2 | `(0,0,0)` | `(3,0,0)` 도달 후 `(0,0,0)` 복귀, 3회 | **왕복**. 구간당 이동거리 3m. 모든 carrot에서 map heading `0`을 고정하므로 복귀 구간은 몸을 돌리지 않고 후진한다. |
| 3 | `(0,0,0)` | `(0,-2,0)` 도달 후 `(0,0,0)` 복귀, 3회 | **왕복**. 구간당 이동거리 2m. map heading `0` 고정이라 양방향 모두 측면 보행. |
| 4 | 원의 중심 `(0,3)` | 중심에서 반지름 `6m` 원 위 random pose 4개, 매 시행마다 중심 복귀 | seed 고정. 필드 안에 남는 호에서만 각도를 뽑고 heading은 완전 random. |
| 5 | `(-2,-0.5,0)` | 1m 간격 ㄹ자, 총 9m | heading irrelevant. 경로 waypoint는 IGNORE 모드라 position만 본다. 시작 pose만 FIXED(20° 허용). |

mission2/3이 왕복인 이유: 이미지 스펙("앞 3m 도달후, 뒤 3m 도달")대로 구간당 이동거리를
3m/2m로 맞춰야 속도팀의 `1.5 m/s × 2초 = 3m`와 직접 비교된다. 이전 구현은
`+3 ↔ -3`(구간당 6m), `-2 ↔ +2`(구간당 4m)였다.

### 도달 판정 threshold

전역값 하나(10cm/6°)를 9개 waypoint에 직렬로 적용하면 하나만 못 넘어도 mission 전체가
멈춘다. E0@6200의 sim 성적이 position median 2.7cm / p90 5.0cm이므로 실기
camera-PF 노이즈까지 얹으면 10cm AND 6°는 정밀도 과제(1~3)에나 맞는 값이다.
그래서 mission별 override를 뒀다. `<=0`이면 전역값을 쓴다.

| mission | `goal_reached_xy_m` | `goal_reached_theta_deg` | 근거 |
|---|---:|---:|---|
| 1~3 | -1 (전역 0.10) | -1 (전역 6.0) | 정밀도 과제 |
| 4 | 0.20 | 10.0 | 시간 최소화 과제. 정밀도로 시간을 깎으면 안 됨 |
| 5 | 0.20 | 20.0 | heading 무관. 시작 정렬에 시간 낭비 방지 |

### mission4: 필드 안에 남는 호에서만 샘플링

필드는 `s_field` **9×6** (`x∈[-4.5,4.5]`, `y∈[-3,3]`)이다. 중심 `(0,3)`(터치라인
중점)에서 반지름 6m 원을 그리면 **대부분이 필드 밖**이고, 안에 남는 건 아래쪽
호뿐이다. 원 전체에서 균등 샘플링하면 4개 중 3개가 필드 밖으로 나온다.

그래서 각도를 0.5° 간격으로 훑어 **필드 안(경계에서 `arc_margin_m`=0.5m 안쪽)에
들어오는 각도만 모아** 그 집합에서 균등하게 뽑는다. 실제 실행 결과:

```text
feasible arc: -131.5deg .. -48.5deg  (83deg span)
  #1 ang= -58.50deg -> ( 3.135, -2.116) head=-113.96deg
  #2 ang= -48.50deg -> ( 3.976, -1.494) head=  34.87deg
  #3 ang=-120.00deg -> (-3.000, -2.196) head=-172.59deg
  #4 ang= -58.00deg -> ( 3.180, -2.088) head=  79.92deg
```

모두 중심에서 정확히 6.000m, 전부 필드 안, seed 42면 매번 같은 4개다.
heading은 호와 무관하게 `[-π,π]` 전 범위에서 뽑는다("random한 heading").

**시작 pose = 원의 중심 = ready pose.** `requires_start: true`라서 PREP가 로봇을
`(0,3)`으로 데려간 뒤 시작한다. 그리고 `return_to_center: true`라서 경로는

```text
centre → outbound_1 → return_1 → outbound_2 → return_2 → outbound_3 → return_3 → outbound_4
```

이 된다. `outbound_*` 구간의 `segment_sec`이 곧 **"중심에서 6m 떨어진 random pose까지
걸린 시간"** 4개다 — 이게 이 미션이 재려는 값이다.

> 주의: 중심 `(0,3)`은 터치라인 **위**다. PREP가 로봇을 라인 위에 세운다는 뜻이니
> 실기에서 공간이 나오는지 먼저 확인할 것. 안 되면 `center_y_m`을 조금 안쪽으로.

field 밖 waypoint는 여전히 `select` 시점에 거부된다:

```text
mission_build_failed_mission4:waypoint2_outbound_1_(-8.31,2.44)_outside_field_9.00x6.00_margin1.00
```

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
4. spin은 위치를 현재 pose로 고정하고, heading을 **남은 회전량만큼**(최대
   `spin_max_lead_deg`) 앞세운다.

   예전에는 상수 60°를 계속 앞세웠는데, 그러면 policy가 **절대 줄어들지 않는
   heading error**를 받는다. E0는 "도달하면 오차가 0으로 수렴하는" goal만 학습했으므로
   이건 학습 분포 밖 입력이다. 남은 회전량을 쓰면 회전이 많이 남았을 때는 ±60°로
   포화해 연속 회전을 만들고, 마지막에는 오차가 0으로 수렴해서 **정지 목표와 같은
   분포**가 된다. 상수를 config로 빼는 것과는 다른 얘기다 — 값이 아니라 신호의
   모양이 바뀐 것이다.
5. Brain이 carrot을 robot frame으로 변환해 `/locomotion_test/goal_rel`을 publish한다.
6. deploy가 x±2 m/y±1.5 m를 per-axis clamp한 뒤 E0 observation에 넣는다.

현재 2.0 m cap의 근거와 한계:

- `STATE_ESTIMATION.md`는 production GoalPose 범위를 `dx∈[-2,2]`, `dy∈[-1.5,1.5]`로 제한하고, A* lookahead도 당분간 이 envelope 안으로 제한하라고 정리했다.
- E0는 path policy가 아니라 single waypoint precision baseline이다. `K1_LEARNING_HISTORY_KO.md`와 `masterplan3.md` 기준 E0@6200은 위치 2.7cm/p90 5.0cm, heading 2.5°, strict 89.3%로 가장 좋은 유효 baseline이지만, path/speed 가설은 아직 분리 검증되지 않았다.
- x 방향 training envelope가 ±2 m여서 radial cap을 2 m로 뒀다.
- **carrot이 deploy clamp를 넘던 문제는 Brain 쪽에서 막았다.** 예전에는 등방
  radial cap(2.0m)만 걸어서 순수 측면 carrot이 `goal_rel_y = ±2.0`으로 나갔고,
  deploy가 그걸 ±1.5로 잘랐다 — 실효 lookahead가 조용히 1.5m가 되고 telemetry와
  policy 실제 입력이 어긋났다. 이제 carrot을 **robot frame에서 policy envelope
  상자(`policy_envelope_x_m` 2.0 / `policy_envelope_y_m` 1.5) 안으로 방향 보존
  스케일링**한 뒤 publish한다:

  ```
  s = min(1, cap/dist, Xmax/|rel_x|, Ymax/|rel_y|)
  carrot_rel = s * (rel_x, rel_y)
  ```

  축별로 자르는 게 아니라 **하나의 스칼라로 같이 줄이므로 방향이 정확히 보존된다.**
  결과가 이미 envelope 안이라 deploy clamp는 no-op이고, telemetry의
  `pose_error_to_goal`이 곧 policy가 받는 값이다. status의 `_scale=` 필드로 이번
  tick에 얼마나 줄었는지 볼 수 있다.
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

즉 “policy별로 lookahead 값은 달라질 수 있지만 mission BT는 고정”이 결론이다.

⚠️ **`locomotion_test.active_policy`는 telemetry에 찍히는 문자열일 뿐 아무것도 선택하지
않는다.** 이 값을 바꿔도 로드되는 policy는 안 바뀐다. 실제 policy 교체는 deploy 쪽에서 한다:

1. `deploy/models/<name>.pt`와 `deploy/configs/<Name>.yaml`을 쌍으로 배치한다.
   `.pt`만으로는 policy가 아니다 — frozen run config의 normalization scale,
   `default_qpos`, `action_scale`, `decimation`, PD gain, `gait_frequency`가 YAML에
   미러링돼 있어야 한다.
2. `python3 deploy_goal_pose.py --config <Name>.yaml` 로 실행한다.
3. `lookahead_max_m` radial cap과 deploy의 x/y clamp를 그 policy의 training envelope에
   맞춘다.
4. mission XML과 `active_policy` 문자열은 그대로 둔다(로그 라벨 용도로만 갱신).

## 현재 연결 상태와 남은 배포 단계

코드 연결은 다음 형태로 완료됐다.

```text
LocomotionTest -> /locomotion_test/goal_rel -> GoalPosePolicy -> E0 actor -> LowCmd
```

남은 것은 server의 E0@6200 export, robot staging copy/hash, 정확한 E0 actor smoke,
hoist safety gate다. Brain clean build와 ROS topic bridge smoke는 통과했다. 정확한 명령과
2026-07-30 현장 근거는
`MISSION_DEPLOY_AUDIT_20260730.md`에 있다.
