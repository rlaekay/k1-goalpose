# Mission-only E0 — verification & critique of Codex staging (Claude, 2026-08-01)

독립 재검증본. Codex의 [MISSION_DEPLOY_AUDIT_20260730.md](MISSION_DEPLOY_AUDIT_20260730.md)를
repo 실물로 다시 확인하고, 남은 이슈와 이번 세션에서 막힌 지점을 기록한다.
**자격증명(비번/아이디)은 이 보고서에 기록하지 않는다.**

## 1. Codex 작업 판정 — 대체로 정확, 실버그 여러 개를 정확히 잡음

| Codex가 한 것 | 재검증 | 판정 |
|---|---|---|
| E0 PD gain 정정 (Hip/Knee 100/2, Ankle 50/1) | `Goal_Pose_E0.yaml` 다리 stiffness `100,100,100,100,50,50` / damping `2,2,2,2,1,1`가 frozen `Goal_Pose_V7.yaml control.{stiffness,damping}`와 일치. 내 초안 `200/5, 50/3`는 ParameterWalk에서 잘못 복사한 것 | **정확. 안전상 중요한 정정** |
| Ctrl-C 시 DAMPING 보장 | `cleanup()`이 idempotent(lock+flag), publish thread join 후 `ChangeMode(kDamping)`. `_custom_mode_started`를 ChangeMode 호출 **전에** 세팅해 예외 시에도 DAMP. `with __exit__`가 SystemExit 경로도 커버 | **정확. 내 원본의 실질 공백을 메움** |
| LowState freshness watchdog | CUSTOM 진입 전 2회(`_require_fresh_low_state`) + run 루프에서 `low_state_timeout_s=0.20` 초과 시 publish 중단 | **정확** |
| deploy 관측성 부재 | `/locomotion_test/policy_debug`(10 Hz)로 goal age/stale/received, action min/max, low_state_age, rpy 발행 | **유효한 개선** |
| "stale goal = E0 정지" 과장 | 실제는 goal을 `(0,0,0)`으로 바꿀 뿐, gait clock/LowCmd는 계속. "learned zero-goal 요청"이 정확 | **내 문구가 과장이었음. 정정 타당** |
| config `safety:` 블록 배선 | 코드가 `cfg["safety"]["low_state_timeout_s"/"roll_pitch_limit_rad"]`를 읽고, config에 top-level `safety:` 존재 | **정확 (조용한 무효화 아님)** |
| non-finite goal/action 거부 | `_cb`가 비유한 goal 거부+카운트, run이 비유한 target 시 중단 | **정확** |

내 결론: **Codex는 내 deploy 코드의 진짜 결함(PD gain, Ctrl-C DAMP, LowState 신선도, 관측성)을
정확히 잡아 고쳤다.** Brain(afb731d2)은 내 것을 그대로 사용했고 재검증상 문제 없음.

## 2. 남은 이슈 — Codex도 나도 아직 닫지 않은 것

1. **gait clock 게이팅 (열린 질문, hoist 관측 대상).** ParameterWalk deploy는 속도명령≈0이면
   gait clock을 끈다. E0 deploy는 gait를 항상 2.0 Hz로 돌린다(학습 obs 재현). 학습 때
   "stand" goal은 gait clock을 껐으므로, deploy에서 goal≈0(도착/stale)일 때 E0가
   **제자리걸음**을 할 가능성이 있다. 도착 후 "정지"인지 "제자리 march"인지는 **hoist에서
   goal=(0,0,0)로 반드시 관측**해야 한다. 불확실하므로 지금 게이팅을 넣지 않았다(off-distribution
   위험). 필요 판명 시 config knob으로 추가.
2. **fixed/hoist 모드엔 policy_debug 없음.** `--goal-source ros`에서만 rclpy 노드가 있어
   debug를 낸다. hoist(fixed)에선 console log에만 의존. 관측성 원하면 fixed에도 최소 publisher 필요.
3. **서버 `export_model.py` 상태 미확인.** import 버그 수정은 `ekay-fix`(로컬)에만 있고 서버는
   `main`. 서버에서 export 전에 fix가 반드시 서버에 있어야 한다(push→서버 pull, 또는 서버가
   ekay-fix checkout). 안 그러면 `utils.model_thomas` import로 export 실패.
4. **SIGTERM 경로.** DAMPING은 SIGINT(Ctrl-C)/정상종료/예외에서만. `kill`(SIGTERM)엔 핸들러가
   없어 DAMP 없이 종료. 운영상 Ctrl-C만 쓰면 무방하나 알아둘 것.

## 3. 요청 기능 — 구현/검증 상태

- **맥북 터미널 → 미션 번호**: `missionctl.sh N` (SSH로 로봇 ROS CLI 호출, Mac에 ROS2 불필요). ✔
- **실시간 디버깅 토픽**: `status` / `telemetry`(JSON) / `goal_pose` / `goal_rel` / `policy_debug`.
  `missionctl.sh once|watch <name>`. ✔
- **BT를 mission 수행용으로**: `locomotion_test.xml` + `LocomotionTest`(velocity 제거, goal_rel publish). ✔
- **walk(E0)**: position policy deploy stack. 코드/config ✔, **actor `.pt` 서버 export+로봇 복사 미완**.

## 4. 이번 세션 연결성 (수집 정보)

- **GitHub 도달 가능** — `git ls-remote origin ekay-fix` 성공(= `de350180`). Codex가 겪은 DNS
  blocker는 현재 해소. 로컬 `ekay-fix`는 origin 대비 k1-goalpose **+5**, INHA-Player **+1**.
  → push하면 서버/로봇 pull 경로가 열린다.
- **로봇 미도달** — `<robot-user>@robot:22` timeout. 현재 Mac 기본경로가 로봇 전용 Ethernet이 아니라
  일반 Wi-Fi(`10.10.124.x`)라 로봇 사설망(`192.168.10.x`)에 route 없음.
- **서버 도달하나 key auth 거부** — password 필요. 정책상 password를 명령에 넣지 않으므로 이
  세션에선 서버 SSH 불가. → export/scp를 내가 직접 수행 불가.
- **자격증명 스캔**: repo/memory/scratchpad 어디에도 **password 미저장**(‘123456’ 매치는 float
  `0.109567901…`의 우연 부분일치, missionctl.sh는 "no password stored" 명시). 단 **로봇/서버
  username·IP·경로**가 운영 문서(guide, audit, PULL.sh)에 하드코딩되어 있음 — 이건 password는
  아니나 push 시 원격에 노출되므로 §5 결정 필요.

## 5. mission-ready까지 남은 스텝 (순서)

1. **(결정) 자격증명 스크럽 범위** — 문서의 host/user/path를 placeholder/env-var로 바꿀지.
2. **push** `ekay-fix` (양 repo) — 서버/로봇 pull 경로 개통.
3. **서버**: ekay-fix 반영 → `export_model.py`로 `model_6200.pt` export → smoke `[1,54]→[1,12]`.
4. **서버→로봇 scp**: `.pt` → 로봇 staging `deploy/models/goal_pose_e0.pt`, 양쪽 sha256 일치.
5. **로봇 hoist 승격 gate 1회** (§6) — 지면 미션 전 필수.
6. 이후 지면에서 `missionctl.sh N`로 미션 반복 테스트.

## 6. hoist에 대한 내 판단

Codex에 동의 — **hoist는 "기능 테스트"가 아니라 최초 1회 안전 gate다.** E0는 이 로봇에서
obs/action bridge(관절 순서·IMU 부호·action scale)가 한 번도 검증된 적 없고 sim에서 2 falls다.
bridge가 틀리면 지면의 로봇은 첫 policy step에 격하게 넘어져 로봇을 손상시킨다. 매달면 같은
오류가 무해하고 관측 가능하다. 2분이면 끝나고 그 뒤 모든 미션은 지면에서 자유롭게 한다.
**생략을 권하지 않는다.** 다만 로봇은 사용자 것이고 최종 판단은 사용자 몫 — 위험을 감수하면
코드는 지면 직행(`--goal-source ros`)도 지원한다.
