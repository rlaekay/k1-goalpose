# CUSTOM 진입 자세와 복구 경로 (2026-08-07 저녁)

증상: **`b`(CUSTOM 진입)를 누르면 몸이 앞으로 쏠린다. `r`(보행 시작)을 누르면 멀쩡해진다.**
첫 deploy code에서는 없던 문제다.

이 문서 하나만 읽으면 무엇이 바뀌었고 다음에 무엇을 재야 하는지 알 수 있게 썼다.
`HANDOFF_ROBOT_20260807.md`(InitChannel 데드락)와는 다른 건이고, 그 문서 §4의
`ModeMonitor` 항목을 **이 문서가 정정한다**(아래 §5).

---

## 0. 다음 한 걸음 — 이것만 하면 된다

```bash
cd ~/Workspace/deploy && tmux new -s e0
python3 -u deploy_goal_pose.py --config Goal_Pose_E0.yaml \
  --goal-source fixed --goal "0,0,0" --hold-diag 15
```

`b`를 누르고 **`hold-diag`의 `tilt` 한 값**만 보면 판정된다.

| tilt | 뜻 |
|---|---|
| **2° 안쪽** | 자세 수정이 맞았다. `r`로 진행 |
| **4.9–13.8° 그대로** | 자세가 원인이 아니다 → 관절 영점 또는 IMU. `DEPLOY_REQUESTS_FROM_TRAINING.md` R4로 간다 |

⚠️ **다른 터미널을 미리 열어 둘 것.** `sshkeyboard`가 tty를 잡으면 Ctrl-C·SIGINT·
SIGTERM이 전부 안 먹는다(2026-08-05에 SIGKILL로 죽여 DAMPING 전환을 못 했다).
정지는 **`touch /tmp/e0_abort`**.

---

## 1. 원인 — 개루프로 설 수 없는 자세를 개루프로 붙잡고 있었다

### 측정이 범위를 좁혔다

`--hold-diag 15` 실측(2026-08-07):

| 측정 | 값 | 읽기 |
|---|---|---|
| 발목 추종오차 | −0.026 ~ +0.005 rad (**최대 1.5°**) | 정상 |
| 보행 시작 시 전 관절 | max **0.0108 rad (0.6°)**, rms 0.27° | **매우 정상** |
| drift vs peak-to-peak | 0.012/0.019 대 0.019/0.024 | 어느 쪽도 지배 안 함 → 발목 정상 |
| **몸통 tilt** | **4.9° ~ 13.8°, 15초 내내** | ⛔ |

**관절은 명령대로 0.6° 오차로 가 있는데 몸은 최대 13.8° 기울어 있다.**
서보/게인/토크 문제가 아니다. 램프 속도 문제도 아니다(tilt가 t=0.05초에 이미
10.3°이고 15초 뒤에도 남는다). **명령한 자세 자체가 곧게 서지 못한다.**

### 왜 그 자세가 못 서는가

평면 다리에서 발이 바닥에 평평하고 몸통이 수직이려면

```
hip_pitch + knee_pitch + ankle_pitch = 0
```

| | Hip | Knee | Ankle | 합 |
|---|---|---|---|---|
| **첫 deploy** (`73e71f3` 이전, prepare 자세) | −0.1 | +0.2 | −0.1 | **0.000** |
| **문제 시점** (`common.default_qpos`, RL 자세) | −0.2 | +0.4 | −0.25 | **−0.050** (2.9°) |

그리고 **`b`~`r` 구간에는 균형을 닫는 주체가 없다** — 정책이 균형 제어기인데
아직 안 돈다. 남는 것은 관절 위치 서보뿐이고, 그것은 역진자를 안정화하지 못한다.

첫 deploy는 합이 0인 자세를 잡아 개루프로 버텼고, 문제 시점에는 **정책이 있어야만
설 수 있는 자세를 정책 없이** 붙잡고 있었다. **`r`이 낫게 하는 이유가 이것이다** —
그때 비로소 루프가 닫힌다.

### 여유가 얼마나 없었나

발목에서 앞코까지 **12.15 cm**(`feet_edge_pos` x=+0.1215), CoM 높이 ~0.45–0.5 m.

| tilt | CoM 전방 이동 | 앞코까지 여유의 |
|---|---|---|
| 4.9° | ~4.1 cm | 34% |
| 10.3° | ~8.7 cm | 72% |
| **13.8°** | **~11.6 cm** | **96%** |

15초 동안 지지다각형 앞 끝을 반복해서 왕복했다.

---

## 2. 넣은 수정 ① — `b`~`r`은 합=0인 자세를 잡는다

- `prepare.default_qpos`의 **다리 12채널**을 `-0.1 / 0 / 0 / 0.2 / -0.1 / 0`으로
  되돌렸다. **팔은 RL 값 그대로** — 같이 바꾸면 CoM이 움직여 변수가 둘이 된다.
- `_enter_custom_latched`가 **prepare 자세**로 램프한다(RL 자세가 아니라).
- `start_rl_gait_conditionally`가 `r` 뒤에 **prepare → RL을 1초 램프**(다리 최대
  이동 **0.200 rad**)하고 게인을 교체한 뒤 추론을 시작한다. 이동을 **정책이 루프를
  닫기 직전**으로 미룬 것이다.
- 그 램프는 `dof_target`만 움직인다 — 발행 스레드가 500 Hz로 읽어 내보내므로
  `low_cmd` 경합이 없다.

`prepare.default_qpos`는 이 경로에서 **지금까지 죽은 설정이었다**
(`create_prepare_cmd`는 `base_walk`/`parameter_walk`만 부른다). 이 수정이 살렸다.

### 딸려온 정리 — `_log_joint_deviation`

호출이 하나뿐이었고 항상 **RL 자세** 기준이었다. 자세를 바꾸면 `b` 직후에
**0.20 rad**이 찍히는데(무릎 0.2 차이) 그건 **정상**이다. 그대로 두면 (a) 로그를
오독하고 (b) **값이 항상 크므로 진짜 이상을 구분할 수 없다** — 유일한 진단이 눈이 먼다.

그래서 단계별로 나눴다:

| 언제 | 기준 | 정상 | 크면 |
|---|---|---|---|
| `at CUSTOM entry (vs prepare pose)` | prepare 자세 | 작음 | 진입 램프가 목표 미달 |
| `at RL-gait start (vs rl pose)` | RL 자세 | 작음 | RL 램프가 미완 |

---

## 3. 넣은 수정 ② — 복구 경로 넷

사용자가 로봇을 눕혀 복구를 시험했더니 네 번 다 **펌웨어 관절 보호(빨간불)** 로
DAMPING에 떨어졌다. ⚠️ **빨간 프로텍트는 우리 코드가 아니다** — deploy에 LED/protect
코드는 한 줄도 없다. 펌웨어의 관절 보호이고, **우리가 유발했다.**

로그가 사슬을 그대로 보여준다:

```
ModeMonitor disabled (cannot import name 'RobotStatesMsg')
 → cannot read /robot_states; entering CUSTOM without confirming the robot is standing
   → [recovery] standing again after 4.9 s      (실측 get-up 은 8.0 s)
     → joints moved 1.5934 / 1.6773 rad         (2 s 램프 = 0.8 rad/s)
       → 관절 보호 → DAMPING
         → [recovery] get-up never became available (state=IS_READY); stopping.
```

| | 무엇 | 근거 |
|---|---|---|
| **F1** | `ModeMonitor` 우회 import — 공개 경로 실패 시 `booster_interface.msg._robot_states_msg` 직접 | §5 |
| **F2** | get-up 완료 = **IS_READY + 다리 정지 + 직립**, 최소 대기 2→**8초** | `IS_READY`는 완료가 아니다. 다리를 오므려 서기가 남아 있고, 끝나면 `dq`가 0으로 간다 |
| **F3** | 진입 램프를 **이동량에 비례** `clamp(이동/0.3, 2.0, 8.0)` | 고정 2초가 0.05 rad도 1.68 rad도 같게 끌었다 |
| **F4** | `IS_READY`를 "get-up 불가"로 읽어 중단하던 것 | 이미 서 있으면 `is_recovery_available=False`가 **정상**이다 |

**F2에 직립을 같이 건 이유**: `/fall_down`은 ~1 Hz라 DAMPING 직후에도 낡은
`IS_READY`를 들고 있을 수 있고, **그때 다리는 힘이 빠져서 오히려 조용하다.**
"IS_READY + 조용함"만으로는 무너지는 중인 로봇을 서 있다고 읽는다. `tilt`는
LowState에서 500 Hz로 오므로 그 창이 없다. 임계 20°(정상 서기 실측이 4.9–13.8°).

**F3은 안 망가진 경로를 안 건드린다**: 최초 진입 0.4894 rad은 하한에 걸려 **2.0초
그대로**이고, 복구 재진입 1.59/1.68 rad만 **5.3/5.6초**가 된다.

---

## 4. 새 config 키 (`Goal_Pose_E0.yaml`, `safety.recovery`)

| 키 | 값 | 뜻 |
|---|---|---|
| `getup_min_wait_s` | 8.0 | get-up 최소 대기(실측값). 예전엔 하드코딩 2.0 |
| `getup_quiet_dof_vel_rps` | 0.15 | 다리 정지 판정 임계 |
| `getup_quiet_hold_s` | 0.5 | 그 아래로 지속돼야 하는 시간 |
| `getup_upright_tilt_rad` | 0.35 (20°) | 직립 조건 |
| `custom_entry_ramp_s` | 2.0 | 진입 램프의 **하한**(의미가 바뀌었다) |
| `custom_entry_max_rate_rps` | 0.3 | 진입 램프 속도 상한 |
| `custom_entry_ramp_max_s` | 8.0 | 진입 램프 시간 상한 |
| `gait_entry_ramp_s` | 1.0 | prepare → RL (`r` 뒤) |

---

## 5. ⛔ `HANDOFF_ROBOT_20260807.md` §4 정정

그 문서는 *"`booster_interface.msg`에 `RobotStatesMsg`가 없다"*고 적었다.
**절반만 맞다.**

```
$ ros2 topic info /robot_states
Type: booster_interface/msg/RobotStatesMsg      ← 이름이 정확히 이것이다
Publisher count: 1

$ cat .../booster_interface/msg/__init__.py
... FallDownState, ImuState, LowCmd, LowState, ...  ← 17개, RobotStatesMsg 없음
```

**메시지 타입은 존재하고 `__init__.py`만 export를 빠뜨렸다.** 그래서 공개 import가
실패하면 생성된 모듈(`_robot_states_msg`)을 직접 집도록 고쳤다. 다음 실행에서
`ModeMonitor disabled` 경고가 사라지면 **서 있는지 확인하는 게이트가 살아난 것**이다.

⚠️ 살아나도 **fail-closed로는 안 바꿨다.** 지금은 모드를 못 읽으면 경고 후 진입한다.
막아 버리면 게이트가 안 고쳐진 상태에서 로봇이 아예 안 도는데, 그건 사용자가
선택할 문제라 손대지 않았다. F2의 정지/직립 조건이 그 자리를 대신 지킨다.

---

## 6. 아직 안 고친 것 / 열린 질문

- **`r` 뒤 정책이 스스로 앞으로 넘어뜨리는지는 미확인.** 이번 로그의 낙상 넷은
  **사용자가 눕힌 것**이라 정책 탓으로 읽으면 안 된다. 자세 수정 뒤 `r`을 눌러
  `roll≈0 / pitch 45`가 다시 나오면 그때가 정책 문제다.
- **관절 영점 실측(R4)** — `hold-diag`의 tilt가 그대로면 이것이 다음이다.
  발을 평평하게 놓고 12관절 인코더 대 순기구학. `DEPLOY_REQUESTS_FROM_TRAINING.md` R4.
- **`/robot_state`(단수)와 `/motion_state`, `/enter_safe_mode`를 안 봤다.**
  특히 `/enter_safe_mode`(`booster_msgs/msg/BinaryData`)가 **빨간 프로텍트 발동을
  알려주는 토픽이면**, 지금은 못 보고 있는 그 순간을 감지할 수 있다.
- **PD 게인이 Booster 기본값의 절반이다.** Booster Gym `T1.yaml`은 Hip/Knee
  kp 200 kd 5인데 우리는 100/2다(감쇠 지표 0.354 대 0.200). deploy config 주석은
  *"Frozen E0@6200 config"*라고만 적어 K1용 튜닝 근거가 없다. **미검증.**

---

## 7. ⚠️ 커밋 추적 주의

이 세션의 deploy 변경은 **병렬 세션의 `git add -A`에 두 번 쓸려** 남의 커밋에
들어갔다. 코드는 정상이고 `HEAD == 작업트리`로 확인했지만, `git log -- deploy/`로
추적하면 엉뚱한 메시지가 나온다.

| 실제 내용 | 들어간 커밋 |
|---|---|
| 복구 경로 넷(F1–F4) | `544642a` "ND_dwell + eval_round 실패 전파" |
| `b`~`r` 자세 수정 | `b93a2d6` "⛔ 유휴 감지가 원리적으로 절대 발동하지 않는 상태였다" |
| 근거 기록 | `011973a` (ibatch §8-49), `e0a21d2` |

**교훈**: 이 저장소는 병렬 세션이 돈다. `git add -A` 대신 **경로를 명시**한다.

---

## 8. 같이 처리한 것 — N2/NA 채점 (학습 쪽)

CLAUDE.md의 선제 보고 항목이라 여기 남긴다. `logs/eval_rounds/n2na`.

**정확도 (공통 waypoint)** — 기준: N0_ctrl 2.82 cm, N1_path best 7.93 cm

| arm | 오차med | p90 | heading | 낙상 | strict |
|---|---|---|---|---|---|
| N2_pathgrid | 5.74 cm | 9.73 | 2.1° | 168 | 39.7% |
| NA_histzero | 5.47 cm | 9.68 | 2.1° | **12** | 20.3% |
| N8_pathdelay | 4.39 cm | 7.88 | 2.9° | 441 | 52.3% |

**지속 보행 (`forward_hold`)** — 요구속도 med 2.00–2.04로 프로브 정상 작동

| arm | 속도med | >1.0 m/s | 낙상 |
|---|---|---|---|
| N2_pathgrid | 1.05 | 67.0% | 1,296 |
| **N2_pathgrid_final** | **1.54** | 97.6% | **8** |
| NA_histzero | 1.22 | 94.9% | 49 |
| NA_histzero_final | 1.32 | 97.6% | 61 |
| N8_pathdelay_final | 1.28 | 94.2% | 65 |

- **N2가 N1의 오차를 절반으로** 줄였다(7.93 → 5.74). N0_ctrl 2.82에는 아직 두 배.
- **NA의 낙상 12**가 눈에 띈다(N2 168, N8 441). 같은 오차대에서. 다만 strict 20.3%로
  가장 낮아 **꼬리가 나쁘다**.
- **N1의 충돌(잘 걷는 체크포인트가 도착 못 함)이 N2에서 완화된다** — N2_final이
  1.54 m/s에 낙상 8. ⛔ **그런데 N2_final의 정확도가 이 표에 없다**(정확도 표는 best
  체크포인트만). 같은 함정을 반복하지 않으려면 그 숫자가 필요하고, **파레토 스캔이
  정확히 그것을 준다**.
- 발간격 14.5–17.4 cm — 전부 목표 10 cm 초과. **실기의 7.0 cm 문제는 N 배치에서 해소.**

🔔 **파레토 스캔은 미보고 상태다**(`005-pareto_N1_path`, `005-pareto_N2_pathgrid`).
`python tools/pareto_table.py logs/pareto/<run>` — ★ 표시가 붙은 iteration부터 본다.

---

## 확인 (ACK)

```
읽은 세션:
읽은 날짜:
hold-diag tilt 실측값:
```
