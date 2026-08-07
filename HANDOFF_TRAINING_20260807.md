# 학습 세션 인계 — 2026-08-07 (03:45 ~ 17:30 KST)

이 문서 하나로 이 세션이 한 일 전부를 읽을 수 있게 쓴다. 사용자가 "일단 한 번
정리한 뒤에 진행"하겠다고 해서 만든 문서다.

⚠️ **경계**: 같은 날 Codex가 병렬로 **배포/실기·MuJoCo** 쪽을 돌렸다
(`c699919`, `309adbe`, `44c5247`, `b71dc13`, `e0a21d2`, `011973a`, `12dcfe5`).
이 문서는 **학습 쪽**이 한 일이고, Codex 작업에 대한 내 감사는
[`FILTER_AUDIT-non-codex.md`](FILTER_AUDIT-non-codex.md)에 따로 있다.

---

## 0. 한 줄

**"왜 sim에서 3.9 cm인데 실기에서 세 걸음도 못 가나"의 답을 찾았고
(과제가 보행을 요구한 적이 없다), 고쳤더니 이번엔 속도와 정확도가 정면충돌한다.
아직 배포 가능한 체크포인트는 없다.**

---

## 1. 확정된 것

### 1-1. ⛔ 근본 원인 — 우리 과제가 보행을 요구한 적이 없다 (§8-44)

배포된 정책(`I3b_stance10`)과 M 배치 9개 arm 전부가
`goal_mode_mixture: {waypoint: 1.0, path: 0.0}` 위에서 돌았다. `_OFF_PATH`가
`_I1_BASE`부터 상속돼 한 번도 꺼진 적이 없다. 그 과제를 평가 출력에서 직접 재면:

| 항목 | 값 | 출처 |
|---|---|---|
| 요구 속도 median | **0.119 m/s** | `feasibility.required_speed_median` |
| 실제 몸통속도 median | **0.03 m/s** | `body_speed.median` |
| 순항 체류 | **0.59 %** | `cruise_share_of_valid` |
| 위치오차 median | 2.84 cm ✅ | `pos_err_m.median` |

**평가의 절반이 정지 상태인데 게이트는 통과였다.** 학습과 평가가 같은 과제라
지표가 보행을 잰 적이 없다. 베이스 config 주석이 이미 그렇게 경고하고 있었다.

### 1-2. ⛔ 배포 정책은 **실물 자산**에서 무너진다 (§8-47)

공통 프로토콜(`K1_robot_boxfoot.urdf`, 19.666 kg)로 채점하면:

| | 낙상률/구간 (waypoint) | 낙상률/구간 (보행) |
|---|---|---|
| `I3b_stance10` (배포된 정책) | **0.368** | **0.604** |
| 실물 자산으로 학습한 arm 전부 | 0.0009–0.0035 | 0.0005–0.0050 |

**세 자릿수 차이.** 이 정책은 지금까지 항상 **자기가 학습한(구) 자산**으로만
채점돼서 낙상 4건이 나왔다. 평가 동역학에서 다른 것은 **자산 하나뿐**이다.

⚠️ 그리고 이 정책의 보행 속도 median이 **1.12 m/s**로 높게 나온다. 같은 행의
낙상이 **26,042**다. **속도만 보면 고꾸라지는 것을 걷는다고 읽는다.**

### 1-3. path 모드는 걷게 만든다 — 그런데 도착을 잃는다

`N0_ctrl`↔`N1_path`는 `goal_mode_mixture` 하나만 다르고 둘 다 6000 iteration,
둘 다 배포 정책에서 warm start. 둘 다 `model_6000` 기준:

| | N0_ctrl (path off) | N1_path (path on) |
|---|---|---|
| 보행 속도 median | 1.10 m/s | **1.48 m/s** |
| 1.0 m/s 초과 체류 | 78.9 % | **98.3 %** |
| 보행 낙상률 | 0.0035 | **0.0007** |
| **정확도 median** | **2.82 cm** | **47.70 cm** ⛔ |

**더 빠르고 걷는 중엔 덜 넘어지는데, 목표에 도착하지 못한다.**

### 1-4. ⛔ 그 충돌은 무릎이 아니라 **절벽**이다 (파레토 스캔)

run 하나에 체크포인트가 60개인데 우리는 늘 2개(`best.pth`와 마지막)만 봤다.
8개로 훑으니:

| N1_path iter | 오차 | 보행속도 | 보행낙상률 |
|---:|---:|---:|---:|
| 100 | **7.84 cm** | 1.15 | 0.0363 |
| 900 | **39.89** | 1.38 | 0.0011 |
| 6000 | 47.69 | 1.48 | 0.0007 |

| N2_pathgrid iter | 오차 | 보행속도 | 보행낙상률 |
|---:|---:|---:|---:|
| 100 | **5.80 cm** | 1.05 | 0.0414 |
| 900 | 30.62 | 0.36 | 0.0000 |
| 5100 | 43.99 | **1.64** | 0.0018 |
| 6000 | 42.21 | 1.54 | 0.0005 |

**iteration 100→900 사이에 정확도가 통째로 무너지고 그 뒤 6000까지 평평하다.**
쓸 수 있는 파레토 점이 없다 — ★가 붙은 것이 전부 30~48 cm라 어떤 목표 반경으로도
도착이 아니다. **그리고 `N0_ctrl`이 정확도·속도·낙상 세 축 모두에서 그 전부를
지배한다.**

> ⛔⛔ **철회됨** (2026-08-07 야간, 5개 독립 감사). 위 문장은 **정확도 행만
> 체크포인트가 달랐던 비교**에서 나왔다 — N0 는 `best(2200)`, N1/N2 는 `model_6000`.
> 같은 `model_6000` 으로 맞추면 속도 1.10 대 1.48/1.54, 보행 낙상률 0.00347 대
> 0.00066/0.00052 로 **path 가 이긴다.** 지배는 정확도 한 축에서만 성립하고,
> `N0_ctrl model_6000` 의 정확도는 아직 측정된 적이 없다. ibatch §8-49·§8-50 참조.

> 이유(가설): path의 carrot은 dwell 중이 아니면 **절대 잡히지 않는다.** 정책이
> "따라붙되 도달하지 않는" 쪽으로 수렴하고, 그게 waypoint의 "도착해서 선다"와
> 정면충돌한다. dwell이 그 충돌을 없애려고 있는 장치인데 duty가 18 %뿐이다.

### 1-5. ⛔ 관절 영점 오차를 한 번도 시뮬레이션한 적이 없다 (§8-46)

`joint_encoder_bias`와 `joint_target_offset`이 **독립적으로** 뽑힌다
(`goal_pose.py:394`, `:398`). 실기의 엔코더 영점 오차 `b`는:

```
측정 q_meas = q + b  /  관측 q + b − default  /  PD tau = kp*(target_cmd − q − b)
→ 평형 q_eq = default + scale*a − b
sim(:793이 참값 dof_pos로 PD를 돈다):
  관측 dof_pos + encoder_bias − default (:866) / 평형 default + scale*a + target_offset (:628)
→ 같아지려면 encoder_bias = +b 이고 target_offset = −b.  정확한 등가다.
```

독립 draw에서 그 조합은 사실상 안 나온다. `I3a_jointcal3`·`M8_jointcal2`가 학습한
것은 **"무관한 두 고장이 동시에"**이고 훨씬 거칠다. MuJoCo가 본 ±3° 절벽도 그 거친
쪽 기준이라 **진짜 영점 오차의 허용 폭은 다시 재야 한다.**

### 1-6. ⛔ 학습에 배포 저역통과 필터가 **아예 없다** (§8-48)

`goal_pose.py:793`에서 PD가 **날것 목표**를 받는다. 액션 쪽에 있는 것은
`delay_steps`(제어주기 안 0–18 ms 순수지연)뿐이다. 배포는
`deploy_goal_pose.py:1430`에서 `filtered*0.8 + target*0.2`를 한다.
**`--rate-fixed-filter`를 켜도 로봇에는 시정수 10 ms 지연이 남고 sim에는 없다.**

### 1-7. 관측이 못 보는 것들 (§8-45)

actor 54채널: gravity(3)·ang_vel(3)·commands(10)·gait cos/sin(2)·dof_pos(12)·
dof_vel(12)·이전 action(12).

- **이력이 없다** — 단일 프레임. 자기 지연을 추정할 수단이 없다.
- **⛔ 발 상태가 하나도 없다** — 발에 관한 보상이 **여섯 개**인데 정책은 그중
  무엇도 직접 못 본다. `MA_feetcross`에서 `feet_cross`가 학습 중 −0.183 → −0.287로
  **나빠진** 것이 그 어려움의 증거다.
- **비평자가 랜덤화를 거의 못 본다** — 특히 관절 영점 12차원이 안 보인다.

---

## 2. 만든 것 (도구)

| 파일 | 하는 일 |
|---|---|
| `tools/idle_watch.sh` | 30초마다 세 신호(util 이력·python 프로세스·큐 깊이) 교차확인 → `queue/idle_state.json`. **정체(stall)** 신호 포함 |
| `tools/autopilot.sh` | 유휴 30분이면 `queue/plan/`에서 승격. 정체면 워커 재기동 + 고아 `.running` 복구 |
| `tools/round_status.sh` | ssh 한 번으로 읽는 요약. 상태 파일이 180초 낡으면 직접 샘플로 대체 |
| `tools/eval_round.sh` | **두 프로토콜** 채점: 정확도(공통 waypoint) + 지속 보행(`forward_hold`) |
| `tools/round_table.py` | 두 축을 한 표로. 없는 키는 0이 아니라 `-` |
| `tools/make_eval_cfg.py` | 공통 프로토콜에 그 arm의 **관측 인터페이스만** 이식 |
| `tools/pick_run.sh` | 체크포인트 N개 이상인 최신 run만 고른다 |
| `tools/expand_checkpoint.py` | 관측 확장 시 warm start 수술(가중치 + **옵티마이저 상태**). `--verify`가 float64 항등 + `optimizer.step()` 2회 실행 |
| `tools/pareto_scan.sh` / `pareto_table.py` | 체크포인트 여러 개를 두 축으로 재서 파레토 프론트 |
| `tools/clean_smoke_runs.sh` | 스모크 잔여만 지운다(0개 **AND** 20분 경과 **AND** 그 arm 미실행), 기본 dry-run |
| `tools/stage_obs_arms.sh` | 관측 arm을 plan/live 큐에 배치 |
| `tools/test_joint_zero.py` | 영점 분포 15개 검사 (Isaac 없이) |
| `tools/test_autopilot.sh` | autopilot 승격·복구 경로 10개 검사 (격리 사본에서 실제 실행) |

**env 쪽 추가** (전부 기본 off, 기존 arm 무관):
- `observation.{extra_foot_offset, extra_dof_tau, history_steps, privileged_extra}`
- `randomization.joint_zero.*` (구조화 고장 모드 5종 + per-env 커리큘럼)
- `control.action_filter_tau` (배포 필터)

---

## 3. 돌린 것과 결과

### 정확도 (공통 waypoint 프로토콜, `sweeps/N0_ctrl.yaml`)

| arm | 오차med | 낙상 | strict |
|---|---|---|---|
| **MA_feetcross** | **2.69 cm** | 4 | 83.6 % |
| **N0_ctrl** | 2.82 | 8 | **86.4 %** |
| I3b_stance10 (배포) | 3.10 | **2,488** ⛔ | 70.5 % |
| M3_scratch_phase | 3.30 | 16 | 79.3 % |
| M7_robotasset | 3.54 | 10 | 73.8 % |
| MB_obsdelay | 4.28 | 8 | 37.3 % |
| N8_pathdelay | 4.39 | 441 | 52.3 % |
| NA_histzero | 5.47 | 12 | 20.3 % |
| N2_pathgrid | 5.74 | 168 | 39.7 % |
| N1_path | 7.93 | 112 | 25.5 % |

### 지속 보행 (`--goal_pattern forward_hold`, 요구속도 med 2.04)

| arm | 속도med | >1.0 m/s | 낙상 |
|---|---|---|---|
| N2_pathgrid `final` | **1.54** | 97.6 % | **8** |
| N1_path `final` | 1.48 | 98.3 % | 10 |
| NA_histzero `final` | 1.32 | 97.6 % | 61 |
| N8_pathdelay `final` | 1.28 | 94.2 % | 65 |
| N0_ctrl `final` | 1.10 | 78.9 % | 53 |
| M3_scratch `final` | 0.99 | 48.0 % | 17 |
| M7_robotasset | 0.87 | 5.5 % | 26 |
| MA_feetcross | 0.82 | 1.5 % | 15 |
| MB_obsdelay | 0.67 | **0.0 %** | 16 |
| I3b_stance10 (배포) | 1.12 | 59.5 % | **26,042** ⛔ |

### 읽을 것 두 가지

- **`MB_obsdelay`가 1.0 m/s를 한 번도 못 넘는다(0.0 %).** 관측 지연만 주고 그것을
  식별할 수단(이력)을 안 주면 정책이 할 수 있는 것은 느려지는 것뿐이라는
  §8-45의 예측과 일치한다.
- **`NA_histzero`만 `best`↔`final` 격차가 없다.** 다른 arm은 정확도로 뽑은
  체크포인트가 보행에서 참사다(N1 118배, N2 162배, N8 24배). NA는 1.2배.
  이력+영점이 **학습 궤적 전체를 안정화**시킨 것으로 보인다 — 예상했던 "속도가
  오른다"가 아니라 "일관성이 생긴다" 쪽이다. ⚠️ 대조군(N4_hist, NZ_zeroiid)이
  아직 안 돌아서 상호작용 판정은 못 했다.

---

## 4. ⛔ 내가 낸 실수 (전부 수정했지만 다음 세션이 알아야 한다)

| # | 실수 | 대가 | 수정 |
|---|---|---|---|
| 1 | 큐 투입 실패 후 재시도 안 함 | **GPU 13.8시간 유휴** | `idle_watch` + `autopilot` |
| 2 | `eval_round`가 run을 시간순 최신으로 고름 → 체크포인트 0개 스모크를 집어 **엉뚱한 정책을 그 arm 이름으로 채점** | 잘못된 표 1회 | `pick_run.sh` + 2개 미만 skip |
| 3 | 수술이 **옵티마이저 상태**를 안 넓힘 → `NA_histzero`가 첫 `optimizer.step()`에서 사망 | 학습 1회 실패 | 모양 기준 확장 + `--verify`가 실제 `step()` 실행 |
| 4 | 공통 config로 관측 넓힌 arm 채점 시도 → `state_dict` 크기 불일치 | 채점 1회 실패 | `make_eval_cfg.py`(인터페이스만 이식) |
| 5 | "체크포인트 0개 = 스모크"로 판정하고 `rm -rf` → **막 시작한 학습 두 개 삭제** | GPU ~2분 | `clean_smoke_runs.sh`(3조건 AND, dry-run 기본) |
| 6 | `eval_round`가 전부 실패해도 rc0 → 큐가 "정상 완료"로 기록 | 실패 신호 유실 | 성공 0건이면 rc1 |
| 7 | ⛔ `pgrep -fc`가 **명령줄에 문자열만 있는 셸까지** 셈 → `nproc_gpu`가 0이 될 수 없어 **유휴 감지가 원리적으로 발동 불가** | autopilot 13시간 무동작 | `ps -eo comm=`로 실행파일 기준 |
| 8 | 자체 시험 하네스가 macOS에 없는 `timeout` 사용 → 8건 실패가 **대상 탓처럼 보임** | 오진 1회 | 배경 실행 + kill |
| 9 | "47.70 cm는 오염된 값"이라고 보고 → 실제로는 진짜 `N1 model_6000` 값이었음 | 잘못된 정정 1회 | 재채점 후 정정 보고 |

**공통 교훈**: 시간순 최신·프로세스 이름·종료코드 0 — **편의로 쓰는 대용(proxy)이
전부 한 번씩 틀렸다.** 그리고 순전파만 검증한 수술, `timeout` 없는 하네스처럼
**검증 도구 자체가 먼저 틀린 경우가 두 번** 있었다.

---

## 5. 지금 상태 (2026-08-07 17:30 KST)

**돌고 있는 것**
- `gpu0` `NZ_zeroiid` ~1,050/6,000 (관절 영점, 상관 수정 + iid)
- `gpu1` `NC_actfilter` ~1,060/6,000 (배포 필터 [0,30] ms)

**live 큐**
- `gpu0`: `006-pareto_fine_N1_path` → `820-ND_dwell`
- `gpu1`: `006-pareto_fine_N2_pathgrid` → `800-N4_hist`

**plan (autopilot 대체재)**
- `gpu0`: `N5_tau` · `N6_foot` · `N9_zerostruct` · `NB_zerocritic` · `N3_pathcross`
- `gpu1`: `N7_critic` · `N1_long`

**학습 완료·채점됨**: N0_ctrl, N1_path, N2_pathgrid, N8_pathdelay, NA_histzero
**미학습**: N3_pathcross, N4_hist, N5_tau, N6_foot, N7_critic, N9_zerostruct, NB_zerocritic
**미채점 run 21개** (M1/M2/M5/M8/M9, L1/L4, I3_rough 등)

**감시자**: `idle_watch`·`autopilot`·워커 2개 전부 정상. 자체 시험 10/10 통과.

---

## 6. 열려 있는 것

### 6-1. 가장 큰 것 — 배포 가능한 체크포인트가 없다

두 축을 동시에 만족하는 지점이 아직 없다. `N0_ctrl`(2.82 cm / 1.10 m/s)이 현재
최선인데 보행이 약하고, path arm들은 도착을 못 한다.

**돌고 있는 답**: `pareto_fine`(100~900 구간에 무릎이 있나) → 없으면 `ND_dwell`
(dwell duty 18.2 % → 40.0 %). 그것도 안 되면 **2단계 학습**(path로 걷기를 익힌 뒤
waypoint로 finetune)인데, 그건 레버가 아니라 절차라 더 비싸다.

### 6-2. 로봇이 필요한 측정 4건 (내가 못 한다)

`R2`(세 걸음 로그) · `R3`(실제 올라간 모델) · `R4`(영점 실측) · **`R6`(`pub_hz` 30초)**.
R6가 특히 중요하다 — Codex의 필터 서사 전체가 "루프가 실제로 느리다"에 매달리는데
그건 아직 측정된 적이 없고, `base_walk`가 **같은 필터·같은 루프·같은 dt로 걷는다**는
반증이 안 풀렸다([`FILTER_AUDIT-non-codex.md`](FILTER_AUDIT-non-codex.md) §4).

### 6-3. 안 한 것

- `E1_path` 평가 — 사용자가 처음 요청했는데 N 배치로 대체했고 그걸 명시하지 않았다
- `M4_scratch_phasefree` NaN(iteration 6,171) 원인 조사
- `HANDOFF_TO_TRAINING.md` 맨 아래 ACK 채우기
- 미채점 21개 run 일괄 채점

### 6-4. 학습에 반영할 후보 (감사에서 나온 것, 아직 안 건 것)

- `obs[42:54]`가 명령한 액션이지 **나간** 액션이 아니다 — `NC_actfilter`를 넣으면
  sim에도 그 차이가 생기므로 필터 통과 목표를 관측에 주는 것이 자연스럽다
- 우리 `gait_frequency` 2.0 vs `base_walk` 1.0 — 어떤 구동 지연이든 2배로 맞는다
- hip roll 포화 — 유일하게 상한에 붙는 관절

---

## 7. 다음 세션이 먼저 할 것

1. `ssh a6000 'bash /mnt/DATA/.../htwk-gym/tools/round_status.sh'`로 상태 확인
2. `logs/pareto/fine_*`가 있으면 `python tools/pareto_table.py`로 읽고 **보고**
   (사용자가 "파레토 결과 나오면 바로 보고해"라고 명시 요청)
3. `NC_actfilter`·`NZ_zeroiid` 완료 시 `tools/eval_round.sh`로 채점
4. **두 축을 항상 같이 본다.** 속도만 보면 고꾸라지는 것을 걷는다고 읽는다
