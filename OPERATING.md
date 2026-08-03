# 실기 테스트 운영 절차

테스트 진행자용. 순서대로 따라가면 된다. 설계 배경은 [missions.md](missions.md),
안전 gate 근거는 [MISSION_DEPLOY_AUDIT_20260730.md](MISSION_DEPLOY_AUDIT_20260730.md).

> **아직 로봇에서 빌드/실행된 적 없는 코드다.** Mac에 colcon/ROS2가 없어 Brain은
> 컴파일 검증이 안 됐다. 첫 `colcon build`에서 에러가 날 수 있고, 그건 예상된 일이다.

---

## 0. 한 번만: 연결 설정

```bash
cd /Users/dmdrb/RoboCup/k1-goalpose
cp tools/deploy_env.sh.example tools/deploy_env.sh
$EDITOR tools/deploy_env.sh     # SERVER / ROBOT / 경로 채우기
```

`tools/deploy_env.sh`는 gitignore돼 있다. 비밀번호는 넣지 말고 SSH 키를 쓴다.

---

## 1. 코드 배포 (Mac → GitHub → 서버/로봇)

두 repo 모두 `ekay-fix` 브랜치다.

```bash
cd /Users/dmdrb/RoboCup/k1-goalpose && git push origin ekay-fix
cd "/Users/dmdrb/RoboCup/[07]sim2real" && git push origin ekay-fix
```

서버와 로봇에서 각각 `git pull`.

---

## 2. Policy 설치 (딸깍)

서버 checkpoint 하나만 지정하면 export → 검증 → 계약 확인 → 로봇 복사 → 해시 대조
→ 로봇에서 load 스모크까지 한 번에 돈다.

```bash
cd /Users/dmdrb/RoboCup/k1-goalpose
./tools/install_policy.sh \
  --checkpoint logs/K1/K1/Goal_Pose_V7/2026-07-26-19-36-15_E0_armB_armsdown/nn/model_6200.pth
```

먼저 `--dry-run`으로 뭘 할지 보고 돌리는 걸 권한다.

**계약 불일치가 나면 멈춘다.** 이건 기능이다 — `.pt`만 맞고 YAML이 틀리면 로봇이
조용히 다르게 움직인다(실제로 PD gain 200/5 ↔ 100/2 사고가 있었다). 확인하려면:

```bash
./tools/check_policy_contract.py \
  --frozen-config htwk-gym/envs/K1/Goal_Pose_V7.yaml \
  --deploy-config htwk-gym/deploy/configs/Goal_Pose_E0.yaml
```

의도한 차이라면 `--force`.

### 새 policy를 추가할 때

`models/<name>.pt` + `configs/<Name>.yaml` **쌍**이 policy 하나다.

```bash
cp htwk-gym/deploy/configs/Goal_Pose_E0.yaml htwk-gym/deploy/configs/Goal_Pose_G1.yaml
$EDITOR htwk-gym/deploy/configs/Goal_Pose_G1.yaml   # policy_path, gain, normalization
./tools/install_policy.sh --checkpoint <ckpt> --name goal_pose_g1 --config Goal_Pose_G1.yaml
```

`locomotion_test.active_policy`는 **telemetry 라벨일 뿐** 아무것도 선택하지 않는다.
선택은 deploy의 `--config`가 한다.

---

## 3. Brain 빌드 (로봇)

```bash
cd <ROBOT_WS>/brain_ws
source /opt/ros/humble/setup.bash
source <ROBOT_GAME_WS>/install/setup.bash
colcon build --packages-select brain --executor sequential --parallel-workers 1
source install/setup.bash
colcon test --packages-select brain --ctest-args -R 'planar_imu_odometry_test|odom_eval_logger_test'
```

deploy 쪽(Python)은 빌드 불필요. `install_policy.sh`가 파일을 넣어주면 끝이다.

플롯을 원하면 로봇에 한 번만:

```bash
pip3 install matplotlib
```

없어도 CSV와 metrics JSON은 나온다.

---

## 4. 실행 (로봇 터미널 3개)

세 터미널 모두 앞에 아래를 source한다.

```bash
source /opt/ros/humble/setup.bash
source <ROBOT_GAME_WS>/install/setup.bash
source <ROBOT_WS>/brain_ws/install/setup.bash
```

**A — vision (camera-PF localization)**

```bash
ros2 launch vision launch.py vision_config_path:=/opt/booster save_data:=false show_det:=false
```

**B — Brain (mission FSM)**

```bash
ros2 launch brain launch.py tree:=locomotion_test vision_config_path:=/opt/booster disable_com:=true
```

시작 직후 이 줄을 반드시 확인한다:

```text
[startup-check] robot/odom/map identity at t0: PASS
```

`FAIL`이면 robot/odom/map이 (0,0,0)에서 시작하지 않은 것이고, **이후 모든 pose가
그만큼 틀어진다.** 여기서 멈추고 원인을 본다.

그다음, 마커가 보이면 correction이 쌓이고 70회째에:

```text
[orientation-sentinel] anchor correction=70 ...
[odom-anchor] origin set at (x, y, θdeg); odom_eval epoch=... logging
```

이게 뜨기 전에는 BT가 goal을 안 낸다(정상).

**C — E0 deploy**

```bash
cd <ROBOT_WS>/deploy
python3 deploy_goal_pose.py --config Goal_Pose_E0.yaml --goal-source ros --net 127.0.0.1
```

리모컨 프롬프트를 따른다. mode 전환 로그(`[mode-timing] ...`)는 4절 참고.

---

## 5. 미션 실행 (Mac)

```bash
cd /Users/dmdrb/RoboCup/k1-goalpose
./missionctl.sh check
./missionctl.sh watch telemetry | tee run_$(date +%Y%m%d_%H%M%S).log   # 별도 터미널
./missionctl.sh watch policy                                          # 별도 터미널

./missionctl.sh 1     # 제자리 CW/CCW 3회
./missionctl.sh 2     # 앞 3m 갔다 복귀, 3회
./missionctl.sh 3     # 오른쪽 2m 갔다 복귀, 3회
./missionctl.sh 4     # 중심(0,3)에서 6m 원 위 random pose 4회 (시간 측정)
./missionctl.sh 5     # 1m 간격 ㄹ자 9m (시간 측정)
./missionctl.sh 0     # 정지
```

`0`은 BT goal stream을 끊는 것이지 E-stop이 아니다. 실기 fault는 deploy Ctrl-C
(DAMPING 진입)나 물리 리모컨/E-stop으로 처리한다.

### 미션이 도는 동안 볼 것

| 스트림 | 확인 |
|---|---|
| `telemetry.fsm_state` | `prep → ready → playing → finished` |
| `telemetry.health.spin_alias_rejects` | **0이어야 함.** 0이 아니면 mission1 회전 적분 오염 → 그 run 폐기 |
| `telemetry.health.fail_reason` | `null`이어야 함 |
| `telemetry.waypoint.reached[]` | 도달할 때마다 늘어남. **결과 숫자가 여기 있다** |
| `policy_debug.goal_stale` | `false` |
| `policy_debug.low_state_age_sec` | `< 0.2` |
| `policy_debug.action_min/max` | finite, `[-1,1]` 안 |

`fsm_state: failed`면 60초 안에 진행 못 한 것이고, `fail_reason`에 어느 waypoint에서
얼마 남기고 멈췄는지 들어 있다. 무한 대기는 이제 발생하지 않는다.

### 결과 뽑기

mission4/5의 답은 `waypoint.reached[]`에 바로 있다.

```bash
grep '"data"' run_*.log | sed 's/^data: //' | python3 -c "
import sys, json
for line in sys.stdin:
    try: t = json.loads(line)
    except Exception: continue
    if t['fsm_state'] != 'finished': continue
    print('mission', t['mission']['key'], 'total %.2fs' % t['mission']['elapsed_sec'])
    for w in t['waypoint']['reached']:
        print('  %-14s reached=%7.2fs  segment=%6.2fs' % (w['label'], w['reached_sec'], w['segment_sec']))
    break
"
```

`outbound_*`의 `segment_sec` 4개가 mission4의 결과다.

---

## 6. Odom 품질 확인

미션 1회 = odom-eval epoch 1개 = plot 1장. 미션 시작/종료마다 자동으로 굽는다.

```bash
ls <ROBOT_WS>/brain_ws/odom_eval/
#   epoch002_mission4_t1830.csv / .png / _metrics.json
```

`_metrics.json`에서 볼 값:

| 필드 | 의미 |
|---|---|
| `drift_per_metre` | 이동거리 대비 오차 비율. 스케일 오차 대표값 |
| `rpe_trans_m.rmse` | 1초당 drift. 초기 오차에 둔감한 진짜 drift rate |
| `mean_err_forward_m` / `mean_err_lateral_m` | **한쪽으로 치우쳐 있으면 noise가 아니라 bias** |

Mac으로 가져와 보려면:

```bash
scp -r <ROBOT>:<ROBOT_WS>/brain_ws/odom_eval ./odom_eval_$(date +%m%d)
```

---

## 7. 미션 사이 / 재시작

- 미션은 연속으로 바로 다음 번호를 보내면 된다. `finished` 5초 뒤 자동으로 `prep`.
- **로봇을 손으로 옮겼거나 localization이 튀었으면** Brain을 재시작한다. odom frame
  원점(anchor)은 Brain 기동 시 1회만 잡히므로, 크게 어긋난 채로 계속 돌리면 안 된다.
- deploy는 Ctrl-C로 끄면 DAMPING으로 들어간다. `kill`(SIGTERM)은 DAMPING을 안 거치니
  쓰지 말 것.

---

## 8. 최초 1회 안전 gate (지면 미션 전 필수)

E0는 이 로봇에서 obs/action bridge(관절 순서·IMU 부호·action scale)가 한 번도 검증된
적이 없다. bridge가 틀리면 지면에서 첫 policy step에 격하게 넘어진다. 매달아 놓으면
같은 오류가 무해하고 관측 가능하다. 2분이면 끝난다.

```bash
cd <ROBOT_WS>/deploy
python3 deploy_goal_pose.py --config Goal_Pose_E0.yaml --goal-source fixed --goal "0,0,0"
```

1. hoist에서 `goal=(0,0,0)` — 제자리걸음인지 정지인지 관측 (열린 질문)
2. `--goal "0.2,0,0"` — 관절 순서/IMU 부호/action 범위 확인
3. Ctrl-C 직후 DAMPING 진입 확인

통과 후 지면 미션으로 넘어간다.
