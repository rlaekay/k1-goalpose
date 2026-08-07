# 실기 세션 기록 — InitChannel 데드락 해결 (2026-08-07 15:3x–15:5x)

작성: 실기 담당 세션. 사용자가 "일단 정리한 뒤에 진행"하기로 해서 **여기서 멈춘 상태**를 적는다.
다음 세션은 **§1(정리해야 할 것)을 먼저 처리하고** §4로 간다.

---

## 1. ⛔ 정리해야 할 것 — 로봇에 프로세스 두 개가 살아 있다

| PID | 실행 | 비고 |
|---|---|---|
| **33763** | `deploy_goal_pose.py ... --hold-diag 15` | **다른 세션**이 띄운 것. RL gait 프롬프트까지 도달했던 그 실행 |
| **42133** | `deploy_goal_pose.py ... --goal 0,0,0` | 내 초기화 시험. `kill` 보냈으나 확인 시점에 아직 남아 있었다 |

**둘 다 LowCmd 채널을 쥐고 있다. 발행자 둘이 동시에 살아 있는 것 자체가 위험하다.**

확인 시점 로봇 상태: **직립, 기울기 1.6°**, 다리 관절 `L: 0.00 0.00 0.00 +0.10 -0.10 0.00`
— 무릎 0.10 rad은 **로봇 자체 standing 자세**다(정책 자세면 0.40). 즉 **어느 프로세스도
관절을 몰고 있지 않다.**

⚠️ **`33763`은 CUSTOM에 들어갔던 프로세스라, 죽이면 `cleanup()`이
`ChangeMode(kDamping)`을 요청한다** — 로봇이 힘을 잃는다. 매달려 있으면 무해하고,
바닥에 서 있으면 주저앉는다. **로봇이 어떤 상태인지 확인하고 죽일 것.**

`42133`은 CUSTOM에 들어간 적이 없어(`_custom_mode_started` False) 모드 변경을 요청하지
않는다 — 언제 죽여도 안전하다.

```bash
ssh booster@192.168.10.102 'pgrep -af deploy_goal_pose'
ssh booster@192.168.10.102 'kill <pid>'      # 33763은 위 경고 확인 후
```

---

## 2. InitChannel GIL 데드락 — 원인과 수정

### 무엇이 일어나는가

`deploy_goal_pose.py:_init_communication`

```python
self.low_state_subscriber = B1LowStateSubscriber(self._low_state_handler)  # 파이썬 콜백 등록
...
self.low_state_subscriber.InitChannel()                                    # C++에서 블록
```

그 창(window) 안에 LowState가 **한 개라도** 배달되면 DDS 스레드가 콜백을 부르려고
GIL을 요구하는데, GIL은 `InitChannel()`을 부른 스레드가 쥐고 있다. 순환 대기 →
전 스레드 `futex_wait`.

### ⛔ 앞선 인계 노트의 두 결론을 정정한다

**① "LowState 발행 중이면 매번 데드락, 재시도는 의미 없다" — 틀렸다.**

사용자가 붙여준 **성공 로그가 반증**이다. 그 로그의 순서:

```
INFO:[joint-layout] 22 joints ... matches hardware      <- 콜백이 찍는다
WARNING:ModeMonitor disabled ...                        <- 508행, _init_communication 이후
```

`[joint-layout]`이 `_init_communication` **완료 후**에 찍혔다. 그 실행은 로봇이 정상
발행 중이었는데도 **창에 메시지가 안 들어와서 통과**했다. **결정적 실패가 아니라 경합이다.**

**② 추천안 A(`InitChannel()`을 별도 스레드에서) — 안 통한다.**

**GIL은 프로세스 전역이다.** 어느 스레드가 쥐고 블록하든 메인도 DDS도 못 돈다.
추천안 B(콜백을 나중에 등록)도 불가하다 — SDK API를 조회했더니:

```
__init__(self: B1LowStateSubscriber, handler: function) -> None
InitChannel(self: B1LowStateSubscriber) -> None
```

**핸들러를 받는 곳은 생성자뿐**이고 `InitChannel`은 인자를 안 받는다.

### 채택한 수정 — 경합을 인정하고 걸린 실행만 잘라낸다

자르는 수단이 핵심이다. **파이썬 인터프리터가 멈춰 있으므로 파이썬으로는 못 자른다:**

| 수단 | 되는가 | 이유 |
|---|---|---|
| `signal.alarm` | ✗ | 파이썬 시그널 핸들러는 바이트코드 사이에서만 돈다 |
| `threading.Timer` | ✗ | 깨어날 때 GIL이 필요하다 |
| **`faulthandler.dump_traceback_later(..., exit=True)`** | ✅ | **워치독이 C 스레드**라 GIL 없이 `_exit()`를 부른다 |

- `deploy_goal_pose.py`: `INIT_WATCHDOG_S = 8.0`으로 **`InitChannel()` 호출만** 감쌌다.
  걸리면 `/tmp/e0_init_deadlock.log`에 전 스레드 스택을 남기고 죽는다. 통과하면 그 파일을 지운다.
- `run_e0.sh`: **그 파일이 비어 있지 않을 때만** 최대 5회 재시도.
  faulthandler는 종료코드를 1로 고정하므로(커스텀 불가) 코드로는 구분할 수 없다.
  다른 실패는 그대로 올린다.

커밋 `b71dc13`. 로봇에도 배포했다(`~/Workspace/deploy/`, 78,033 B, 08-07 15:43).

### 검증 상태

배포 후 시험에서 **`_init_communication`을 통과**했다 — `[joint-layout]`과
`FallMonitor subscribed`가 찍혔다. **워치독이 실제로 발동하는 것은 아직 못 봤다**
(그 사이 데드락이 안 걸렸다). 경합이라 재현이 보장되지 않는다.
**다음에 걸리면 `/tmp/e0_init_deadlock.log`가 증거가 된다.**

---

## 3. 내가 시험에서 틀린 것

초기화 6회 반복 시험을 짜면서 `Press 'b'` 문자열로 성공을 판정했는데, **6회 전부
"미도달"로 나왔다.** 프로그램 문제가 아니라 **내 판정이 틀렸다**:

**파이썬 stdout이 `tee`로 파이프되면 블록 버퍼링**이라 `print()` 출력이 안 나온다.
`logging`은 stderr라 그대로 보였고, 그래서 "init은 통과했는데 프롬프트가 안 뜬다"는
잘못된 그림이 나왔다. 앞선 노트의 실행 명령이 `python3 -u`를 쓴 이유가 이것이다.

**교훈**: 실기 판정 스크립트는 `PYTHONUNBUFFERED=1` 또는 `-u`를 붙인다.

---

## 4. 로봇에 대해 확정된 사실 (다음 세션이 다시 안 캐도 되게)

- **배포 위치**: `/home/booster/Workspace/deploy` — `mission_ws`는 **존재하지 않는다**
- ⛔ `tools/deploy_env.sh`의 `ROBOT_WS`가 없는 경로를 가리키고 있었다 → **고쳤다**
  (`install_policy.sh`가 엉뚱한 데 설치할 뻔했다)
- **의존성**: ROS humble ✅, torch 2.7.0 ✅, numpy 1.26.3 ✅, sshkeyboard ✅,
  `booster_robotics_sdk_python` ✅
- **`rclpy`는 ROS를 source해야 뜬다** — `run_e0.sh`가 해준다
- **`ModeMonitor` 비활성**: `booster_interface.msg`에 `RobotStatesMsg`가 없다.
  → CUSTOM 진입 시 "서 있는지" 검증 불가(`_ensure_standing_before_custom`이 경고만 찍고 통과)
- **`FallMonitor`는 동작**: `/fall_down` 구독 ✅ (`fall_down_state: 0`)
- **조이스틱 없음** → 키보드 모드(`b`/`r`). `sshkeyboard`는 **TTY가 필요**하므로
  비대화형 ssh로는 못 띄운다 → **tmux 또는 직접 터미널**
- 설치된 정책: `deploy/models/goal_pose_i3b.pt`, config가 그것을 가리킴 ✅
- **접속**: `ssh booster@192.168.10.102` (키 등록 완료)
- ⚠️ **로봇 이더넷이 올라오면 서버가 끊긴다** (Mac에 기본 경로 두 개).
  서버 접속은 `ssh -b $(ipconfig getifaddr en0) a6000`.
  영구 해법은 시스템 설정 → 네트워크 → 서비스 순서에서 **Wi-Fi를 이더넷 위로**.

---

## 5. 다음 — 정리 후 바로 할 것

§1을 정리한 뒤, tmux에서:

```bash
cd ~/Workspace/deploy
tmux new -s e0
python3 -u deploy_goal_pose.py --config Goal_Pose_E0.yaml --rate-fixed-filter \
  --goal-source fixed --goal "0,0,0" --log-timing /tmp/t_a.csv
# b (CUSTOM) → 램프 후 r (RL gait) → --goal "0.2,0,0" 10s → "0.5,0,0" 30s
```

**받아야 할 숫자 하나: `pub_hz`** (timing 로그 컬럼에 이미 있음).

⚠️ 다만 `HANDOFF_NIGHT_20260806.md`의 MuJoCo 근거(50 Hz에서 낙상 66/68 → 수정 필터로
전 주파수 0, 500 Hz에서는 켜나 안 켜나 동일)가 충분히 강하다. **수정을 적용할 것이라면
`pub_hz`는 원인 규명용이지 판정용이 아니다.** `FILTER_AUDIT-non-codex.md`가
"`pub_hz` 실측이 유일한 판별 측정"이라고 쓴 것과 이 점에서 갈리므로, 둘을 같이 읽을 것.

---

## 확인 (ACK)

```
읽은 세션:
읽은 날짜:
§1 정리:  [ ] 42133 종료  [ ] 33763 종료(로봇 상태 확인 후)
워치독이 실제로 발동한 적이 있는가:  [ ] 있음(로그 첨부)  [ ] 아직 없음
```
