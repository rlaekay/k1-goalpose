# Sim-to-real 실기 세션 정량 감사 보고서

작성 시각: 2026-08-09 14:28 KST  
목표: 현재 학습 완료 정책으로 실물 보행을 재현하기 위해, 이번 세션의 모든 시험을
`의도 → 변경 변수 → 원시 데이터 → 정량 결과 → 해석 → 한계/반증 조건` 순서로 재구성한다.

이 문서는 Claude Code가 독립 재계산할 수 있도록 원시 CSV, 코드·모델 provenance,
계산 정의, 무효 처리한 실험까지 함께 기록한다. 결론보다 원시 데이터가 우선한다.

---

## 0. 결론부터

### 현재 가장 강한 결론

1. **실기 후보 순위는 현재 `I3b > NH > NP`다.** 동일한 1초 제자리 시험에서 몸 각속도
   p95는 I3b `0.254`, NH `0.595`, NP `1.245 rad/s`; 최대 관절 왕복폭은 각각
   `0.072`, `0.286`, `0.398 rad`였다. 다만 각 시험의 실제 정책률이
   `40.67/45.42/49.83 Hz`로 달라 완전한 단일변수 A/B는 아니다.
2. **`I3b + publish 100 Hz`에서 처음 얻은 51 Hz 결과 두 개는 성능시험으로 무효다.**
   50번 추론 중 실제로 달라진 q/dq/tau snapshot이 제자리 `18`, 보행 `22`개뿐이었고,
   다음 새 상태까지 각각 최대 `179`, `218 ms`가 걸렸다. 이 두 로그는 정책 성능이 아니라
   배포 scheduler race를 발견한 증거로만 쓴다.
3. **fresh-state 수정은 계측상 성공했다.** 수정 후 I3b 제자리 로그는 50행 모두 서로 다른
   `low_state_seq`, non-increasing sequence `0`, 정책 `50.44 Hz`, LowCmd `99.33 Hz`,
   callback 약 `495.9 Hz`였다. 소비 snapshot age는 median `8.44`, p95 `16.15`,
   max `27.36 ms`였다.
4. **수정 후 I3b는 1초 동안 발산하지 않았지만 역사 기준선보다 동적이었다.** 몸 각속도
   p95 `0.370 vs 0.272 rad/s`(+36%), 전체 관절속도 RMS `0.200 vs 0.123 rad/s`
   (+63%)였다. 반면 actor action RMS는 `0.0637 vs 0.0625`(+1.9%)로 거의 같았다.
   정책 출력 크기보다 plant 초기조건·필터·제어주기 차이가 더 유력하다.
5. **반복 가능한 좌우 비대칭은 확인됐지만 이를 곧바로 encoder zero라고 부를 수는 없다.**
   vendor PREPARE 3회에서 Hip Roll L−R `−2.167±0.006°`, Ankle Pitch `−1.223±0.006°`,
   sagittal pitch-chain `−2.307±0.257°`가 반복됐다. fresh RL 첫 snapshot의 chain은
   `−2.287°`로 probe 평균과 `0.019°` 차이였다. 그러나 바닥·하중·기구·유한강성 변형도
   같은 encoder-space signature를 만들 수 있고 common-mode zero는 이 측정으로 안 보인다.
6. **명목 Kp/Kd 오복사 가설은 약하다.** 정책 gain은 학습·배포 모두 Hip/Knee
   `100/2`, Ankle `50/1`; 학습 randomization은 **±5%**다. vendor PREPARE에서도 같은
   비대칭이 actor 실행 전에 나타나므로 RL gain이 단독 원인일 수 없다. 정적 Kd 효과의
   등가각은 최대 `0.141°`로 관측 chain `2.287°`보다 한 자릿수 작다. 다만 finite Kp는
   하중·기구 비대칭을 1–2° 추종오차로 드러내는 전달 경로다.
7. **가장 깨끗한 다음 단일변수 실험은 I3b fresh-state에서 필터 tau `10 → 24.2 ms`다.**
   역사 I3b가 실제 LowCmd 185.1 Hz에서 쓴 legacy EMA의 연속 등가값이 24.2 ms다.
   지금 10 ms는 훨씬 덜 smoothing하므로, 이를 먼저 복원해야 gain이나 actor를 바꾸지 않고
   현재의 추가 꿀렁임을 분리할 수 있다.

### 아직 말할 수 없는 것

- **아직 유효한 fresh-state 실물 보행 성공/실패 데이터가 없다.** stale 수정 뒤에는
  `0,0,0` 1초만 수행했다.
- NP의 실기 퇴화가 armature 하나 때문이라고 확정할 수 없다. NP actor가 부분 armature에
  co-adapt했을 가능성은 강하지만 single-run, 다른 policy rate, 다른 초기상태가 남는다.
- “오른발이 앞에 있음”의 원인이 특정 Hip Pitch zero라고 확정할 수 없다.
- old I3b가 `184초 hands-off`로 섰다고 인용할 수 없다. 첫 106초는 계속 손 외란을 준
  구간이며, 복구 뒤 48.2초도 손잡이 접촉 여부가 계측되지 않았다.

### 중요한 정정

앞선 분석에서 PD 랜덤화를 `±15%`라고 말한 것은 틀렸다. frozen V7 설정의
`dof_stiffness`와 `dof_damping`은 각각 `[0.95, 1.05]`, 즉 **±5%**다.
근거: `htwk-gym/envs/K1/Goal_Pose_V7.yaml:495-502`, `ibatch.md:473-480`.

---

## 1. 데이터와 증거 등급

### 1.1 보존한 원시 데이터

이번 세션 CSV 9개를 로봇 `/tmp`에서 회수한 바이트 그대로
`realdata/2026-08-09_codex_session/`에 보존했다. 파일별 SHA-256과 유효성은
`trial_manifest.csv`에 있다.

역사 기준선은 `realdata/2026-08-09_t_stand_i3b.csv`다.

주요 provenance:

| 대상 | 값 |
|---|---|
| I3b actor | `I3b_stance10/model_200`, MD5 `60006446e638566c8245bc6013c8780c` |
| 역사 배포 commit | `00ed5b80fcb741f79e2596aa10450c462914c5b6` |
| 역사 deploy SHA-256 | `1f1334808a0c78a9b048f86e961ee57b31b17feafbf6ac889d0dddc7008465b5` |
| 역사 run/config SHA-256 | `b8da26f9...a49aa166` / `4d31892d...51a5944` |
| 현재 fresh 수정 deploy SHA-256 | `d883ed172f48d79b6ce12f7d8b9c17d4a5898c3475ca193fa4fb4eadb9f8e64b` |
| 현재 run/config SHA-256 | `6d5beea47d754606107092e6c8f2ef4a38da84a6ae8f99f966912fa5dd68605f` / `86c690967f4b08fe738698c06d2ba13257c5475f344561b148c37f4205e93d78` |
| fresh CSV SHA-256 | `034ab557fc13b1d7c6fc0826300baf4f0dd8c8d232c6d146e52b0ae4623fee5e` |
| 역사 CSV SHA-256 | `401c640ffd354741addf3e35664a8e0580b519aff49dd01f7217eaf183ca0732` |

현재 수정은 **uncommitted dirty working tree**다. 따라서 hash가 commit보다 중요하다.
로봇 설치본과 로컬 deploy 파일의 SHA-256 `d883...e64b`가 일치함은 설치 직후 확인했다.

### 1.2 증거 등급

| 등급 | 뜻 | 이 문서에서의 예 |
|---|---|---|
| A | 보존된 원시 CSV에서 재계산 가능 | R1–R8, R10, 역사 H0 |
| B | 코드/config/hash로 직접 확인 | scheduler race, gain, filter, zero 분포 |
| C | 콘솔 원문·사용자 관찰 | 로봇이 꿀렁임, 뒤에서 잡음, 오른발이 앞섬 |
| D | 짧은 단일-seed MuJoCo smoke | 후보 screen, timing mechanism check |

등급 D 결과는 기전 확인과 다음 실험 선택에는 쓰되, 실물 성공 인증에는 쓰지 않는다.

### 1.3 통계 정의와 한계

- 정책률: 첫 행의 `tick_dt_s=0`을 버리고 `1 / mean(tick_dt_s)`.
- LowCmd rate: `pub_hz`의 median.
- 몸 각속도: `sqrt(gx²+gy²+gz²)`.
- 관절 왕복폭: 각 q 채널의 `max-min` 중 최댓값.
- 토크비: `abs(tau_j)/effort_j`, effort는 각 다리 `[45,30,30,45,20,20]`.
- sagittal pitch-chain: `(L HipPitch + L Knee + L AnklePitch) - (오른쪽)`.
- 백분위는 pandas/numpy linear interpolation을 사용했다.

모든 1초 실험은 **한 번의 궤적**이다. 40–50개 행은 독립 표본이 아니므로 행 수를
통계적 n으로 두거나 좁은 confidence interval을 만들면 안 된다. 아래 비율은 효과크기의
기술통계이지 모집단 추론이 아니다.

`policy_state_age_s`는 snapshot 시각부터 inference 후 로그 행을 쓴 시점까지다.
정책이 실제로 읽은 순간의 age에 inference/log overhead가 더해진 보수적 상한에 가깝다.

---

## 2. 실행되지 않은 시도

다음 두 시도는 argparse 또는 path 오류로 정책이 돌지 않았고 데이터가 없다. 실험 수에
포함하지 않는다.

| 시도 | 현상 | 판정 |
|---|---|---|
| `cd ~/Workspace/deploy/deploy && ./run_e0.sh ...` | `No such file or directory` | 미실행 |
| `--policy-path ... --max-pol`로 잘린 명령 | `unrecognized arguments` | 미실행 |

---

## 3. 모델 후보를 고른 이유와 학습 차이

### 3.1 I3b

- 원본: `I3b_stance10/model_200`.
- I2 계보에 `feet_offset_y=-10` 보상을 넣어 발 중심 간격을 약 `7 → 17.5–18 cm`로 넓힌
  모델이다 (`htwk-gym/tools/make_v7_arms.py:427-431`).
- structured `joint_zero` 도입 전 모델이라 persistent joint-zero coverage는 **정확히 0°**다.
- 구형 `K1_locomotion_armsdown.urdf`, armature 0을 사용했다. foot 질량·관성, hip 위치,
  velocity limit 등 실물과의 알려진 asset gap이 남는다.

### 3.2 NH

- 원본: `NH_zeroclock/model_6000`, I3b warm-start.
- box-foot 자산, base height 0.52, path task, dwell 중 gait clock 정지, correlated
  encoder/target joint-zero를 추가했다.
- armature는 0이다.
- clean waypoint 평가 `31.96 → 9.33 cm` 개선은 유효하지만, 그 평가는 joint-zero OFF였다.
  따라서 zero robustness 인증이 아니라 clean 성능/regularization 효과다.

### 3.3 NP

- 원본: `NP_clockarm/model_6000`, NH와 같은 레시피에서 `_ARM_ASSET`만 추가.
- 12개 다리 DOF에서는 Ankle Pitch/Roll armature `0.05`, Hip/Knee `0`이다.
- 벤더 분포는 Hip `0.028–0.048`, Knee 약 `0.0956`으로, NP는 **발목만 근사한 부분 모델**이다.
- 기대는 발목 반사관성으로 joint speed와 foot-crossing을 줄이는 것이었으나, 실기에서
  오히려 NP가 NH보다 동적이었다.

### 3.4 사전 MuJoCo 후보 screen S1

의도: 로봇 시간을 쓰기 전에 NH와 NP의 명백한 붕괴·포화 여부를 같은 seed에서 비교.

조건: 256 env × 8초, 같은 seed. 단일 smoke라 robustness 인증은 아니다.

| 지표 | NH | NP |
|---|---:|---:|
| falls | 2 | 0 |
| median speed | 1.39 m/s | 1.50 m/s |
| 최대 joint speed-limit 위반율 | 45.36% | 0.58% |
| torque saturation | 4.43% | 2.73% |
| foot crossing | 0 | 0 |
| 최소 발 간격 | 5.49 cm | 3.99 cm |

당시 해석: NP가 sim에서는 더 좋은 실기 후보로 보였다.  
현재 해석: 이 screen은 NP가 **자기 학습 물리**에서 좋아진다는 것만 보였다. 부분 armature
plant에 co-adapt한 actor가 armature가 전 관절에 존재하는 실물로 옮겨지는지는 검증하지 못했다.

---

## 4. 역사 기준선 H0 — “I3b가 0,0,0에서 잘 섰다”의 정확한 의미

### 의도

`b`~`r` prepare 자세 수정이 실제로 열린루프 서기를 가능하게 하는지 확인하고, I3b pure
balance stance의 timing·외란 복원 데이터를 얻는다.

### 정확한 조건

```bash
pkill -9 -f deploy_goal_pose.py 2>/dev/null
sleep 2
cd ~/Workspace/deploy
PYTHONUNBUFFERED=1 ./run_e0.sh fixed 0,0,0 \
  --hold-diag 15 \
  --log-timing /tmp/t_stand.csv
```

그 뒤 `b → 15초 hold diag → r`.

- actor: 현재와 같은 I3b MD5 `60006446...`.
- 배포 commit: `00ed5b8`.
- CUSTOM prepare pose, 각 다리: `[-0.1,0,0,+0.2,-0.1,0]`.
- r 뒤 RL pose: `[-0.2,0,0,+0.4,-0.25,0]`, 1초 ramp.
- policy gain: Hip/Knee `100/2`, Ankle `50/1`.
- prepare gain: Kp `[350,350,180,350,250,250]`, Kd `[7.5,7.5,3,5.5,5,5]`.
- position PD, torque feedforward off.
- `--rate-fixed-filter` 없음: legacy `y=0.8y+0.2u`.
- `--hold-prepare`는 당시 명령에도 없고 현재도 read되지 않는 no-op flag다.
- `fixed 0,0,0`은 모든 행 `walking=0`, `gait_freq=0`, phase 0인 balance stance다.

### timing

| 항목 | 값 |
|---|---:|
| 정책 tick median | 26.036 ms |
| 실제 정책률 | 38.41 Hz |
| LowCmd median | 185.12 Hz |
| legacy EMA 등가 tau | 24.21 ms |
| 등가 cutoff | 6.57 Hz |
| 2 Hz phase lag | 16.92° |
| callback age median | 1.051 ms |

옛 구현은 callback와 inference가 같은 counter deadline을 썼으므로 input이 fresh였을 가능성은
높지만, 당시 `low_state_seq/policy_state_age`가 없어 inference별 freshness는 사후 확정할 수 없다.
`low_state_age`만으로 정책이 latch한 q의 age를 증명하면 안 된다.

### 장기 로그의 올바른 분절

| 구간 | 행 | policy-row 시간 | tilt med/p90/max | 실제 조건 | 허용되는 해석 |
|---|---:|---:|---:|---|---|
| 외란 stance | 3,961 | 105.785 s | 1.428/6.918/16.761° | 사용자가 계속 외력을 줌 | perturbation rejection |
| 낙상 진행 | 543 | 14.490 s | 6.698/32.356/44.871° | 낙상 | 성공 구간 아님 |
| recovery gap | 0 | 15.66 s | — | get-up + re-entry | policy row 없음 |
| 복구 후 stance | 1,826 | 48.200 s | 0.658/1.411/5.187° | 의도적 push 없음, 손 접촉 미계측 | load-bearing policy stance |

`복구 후 63초`라는 옛 문구는 첫 row의 `tick_dt_s=15.668`에 recovery gap을 합친 값이다.
실제 policy-active 구간은 48.2초다. post 구간 발목 pitch torque median `3.45 N·m`는
발이 하중을 받았음을 보이지만, 손잡이가 일부 하중을 받지 않았다는 증명은 아니다.

### 역사 기준선 판정

I3b가 실제 하중을 받으며 균형을 닫고 외란 뒤 복구한 **유의미한 기준선**이다. 그러나
`hands-off 184초` 또는 `hands-off 106초`로 인용하면 안 된다. 현재 1초 시험과도 filter,
clock, policy/publish rate, gate, calibration, timeout이 모두 달라 직접 동등 비교가 아니다.

---

## 5. R1 — NP 기존 scheduler 제자리 1초

### 의도와 단일 변경

후속 학습 후보 NP가 실물에서 즉시 발진·붕괴하지 않는지 가장 짧게 확인한다. actor만 I3b에서
NP로 바꾸고 당시 기존 counter-clock/legacy filter를 유지했다.

```bash
cd ~/Workspace/deploy
PYTHONUNBUFFERED=1 ./run_e0.sh fixed 0,0,0 \
  --policy-path ./deploy/models/goal_pose_np.pt \
  --hold-prepare --hold-diag 1 \
  --max-policy-seconds 1.0 \
  --test-tilt-abort-deg 12 \
  --log-timing /tmp/np_stand.csv
```

### 데이터

| 지표 | 값 |
|---|---:|
| rows / span | 37 / 0.959 s |
| 정책률 / LowCmd | 37.54 / 185.12 Hz |
| tick median/p95/max | 25.95/37.30/58.74 ms |
| tilt start/median/p95/max/end | 0.823/0.823/2.137/2.392/0.846° |
| 몸 각속도 p95 | 0.726 rad/s |
| dq RMS / max | 0.417 / 1.821 rad/s |
| 최대 q 왕복폭 | 0.355 rad |
| 최대 torque/limit | 42.3% |
| action RMS | 0.151 |

hold 진단에서 평행기구 오차 drift는 거의 0, peak-to-peak 최대 `0.0004 rad`; RL 시작
`max|q_meas-q_cmd|=0.0276 rad`, RMS `0.0098 rad`였다.

### 결과와 해석

12° abort 없이 1초를 마쳤고 tilt도 작았다. 당시에는 “stand gate 통과”라고 봤다. 그러나
후속 비교에서 NP의 몸·관절 운동량이 I3b보다 훨씬 컸고 정책률도 37.5 Hz뿐이었다.
따라서 이 로그는 **즉시 낙상하지 않았다**는 제한된 통과이지 안정성 인증이 아니다.

---

## 6. R2 — NP 전방 0.2 m, 1.5초

### 의도

NP가 제자리 통과 뒤 실제 gait를 시작할 수 있는지, 최대 1.5초로 제한해 확인한다.
`0.2`는 속도 0.2 m/s가 아니라 전방 목표거리 0.2 m다.

```bash
cd ~/Workspace/deploy
PYTHONUNBUFFERED=1 ./run_e0.sh fixed 0.2,0,0 \
  --policy-path ./deploy/models/goal_pose_np.pt \
  --hold-prepare --hold-diag 1 \
  --max-policy-seconds 1.5 \
  --test-tilt-abort-deg 12 \
  --log-timing /tmp/np_walk_02.csv
```

### 정량 결과

| 지표 | 값 |
|---|---:|
| rows / span | 54 / 1.482 s |
| 정책률 | 35.77 Hz |
| 실제 gait clock | 약 1.46 Hz; 학습 nominal 2 Hz |
| tilt start/median/p95/max/end | 2.741/4.610/8.603/9.996/3.957° |
| 몸 각속도 p95/max | 3.034/3.930 rad/s |
| dq RMS/max | 3.407/21.893 rad/s |
| 최대 q 왕복폭 | 0.862 rad |
| 최대 torque/limit | 98.7% |
| phase end | 0.164 |

세부 관찰: 발목 pitch가 torque limit의 `91–99%`, 발목 roll 속도가 `20–22 rad/s`에
도달했다. 사용자는 로봇이 바닥으로 갈 것 같아 뒤에서 잡았다.

### 실패 원인 분해

- **정책/closed-loop 자체가 나빴다:** tilt max 10°, 발목 포화 근접, dq max 21.9는
  cleanup 이전 CSV에서 이미 관측됐다.
- **최종 바닥행은 cleanup과 교락됐다:** 마지막 policy row tilt는 3.96°였고 1.5초 normal
  timeout이었다. 당시 normal timeout도 DAMPING으로 보내 직립 로봇을 limp하게 만들었다.

따라서 “정책 때문에 그대로 바닥에 박았다”와 “전부 cleanup 탓이다” 둘 다 과장이다.
실제 정책 불안정과 잘못된 종료 모드가 함께 있었다. 이후 normal timeout은 PREPARE,
fault만 DAMPING으로 보내도록 수정했다 (`deploy_goal_pose.py:1892-1904`).

---

## 7. R3 — NP wall-clock만 적용한 제자리

### 의도

LowState counter를 시간으로 쓰던 오류를 wall monotonic으로 바꾸면 50 Hz가 복원되는지,
다른 actor/gain/filter는 유지한 채 확인한다.

### 데이터

| 지표 | R1 counter | R3 wall-clock + 1 ms polling |
|---|---:|---:|
| 정책률 | 37.54 | 38.45 Hz |
| tick max | 58.74 | 47.93 ms |
| tilt max | 2.392 | 2.245° |
| 몸 각속도 p95 | 0.726 | 1.301 rad/s |
| dq max | 1.821 | 2.508 rad/s |
| q 왕복폭 max | 0.355 | 0.423 rad |

### 판정

wall clock의 의미는 맞지만 1 ms polling이 Python main thread를 자주 깨워 SDK callback/publisher와
GIL 경쟁을 만들었다. 실제 정책률은 0.9 Hz만 늘었다. **wall-clock 전환만으로는 timing fix가
완료되지 않았다.** 이를 deadline까지 한 번 자는 방식으로 바꿨다.

---

## 8. R4–R6 — deadline scheduler에서 NP/NH/I3b 제자리 비교

### 의도

1 ms polling을 제거한 뒤 세 actor의 실물 제자리 closed-loop 동역학을 같은 1초 protocol로
비교한다. gain은 모두 동일하고, normal timeout은 PREPARE로 복귀한다.

### 원시 통계

| 지표 | NP R4 | NH R5 | I3b R6 |
|---|---:|---:|---:|
| rows / span | 50/0.984 s | 46/0.991 s | 41/0.984 s |
| 정책률 | 49.83 Hz | 45.42 Hz | 40.67 Hz |
| LowCmd median | 190.18 Hz | 196.26 Hz | 185.67 Hz |
| tick median/p95/max | 20.06/30.54/36.30 ms | 20.36/33.19/42.61 ms | 24.77/33.67/42.05 ms |
| tilt med/p95/max | 1.701/2.350/2.598° | 1.449/2.819/2.902° | 2.652/3.565/3.696° |
| 몸 각속도 med/p95/max | 0.384/1.245/1.370 | 0.175/0.595/0.657 | 0.162/0.254/0.297 rad/s |
| dq RMS/max | 0.459/2.267 | 0.310/1.553 | 0.125/0.856 rad/s |
| q 왕복폭 max | 0.398 | 0.286 | 0.072 rad |
| torque/limit max | 50.8% | 46.0% | 48.3% |
| action RMS | 0.151 | 0.125 | 0.0599 |

### 효과크기

I3b를 1로 놓으면:

| 지표 | NH/I3b | NP/I3b | NP/NH |
|---|---:|---:|---:|
| 몸 각속도 p95 | 2.34× | 4.90× | 2.09× |
| dq max | 1.82× | 2.65× | 1.46× |
| q 왕복폭 max | 4.00× | 5.56× | 1.39× |
| action RMS | 2.09× | 2.52× | 1.21× |

사용자 정성 관찰도 방향이 같았다: NP는 짧은 순간에도 전신이 크게 꿀렁였고, NH도
꿀렁였지만 NP보다 훨씬 덜했다.

### 해석

- NP는 49.83 Hz로 목표 rate에 가장 가까운데도 가장 동적이었다. 따라서 느린 scheduler만으로
  NP 퇴화를 설명할 수 없다.
- NP→NH에서 partial armature 하나를 제거하자 몸 각속도 p95가 52.2%, q 왕복폭이 28.1%
  감소했다. **NP의 partial-armature co-adaptation mismatch**와 일관된다.
- 그래도 NH는 I3b보다 몸 각속도 p95 2.34배, q 왕복폭 4배였다. joint-zero/box-foot/path
  fine-tune이 실기 standing을 자동으로 개선하지 않았다.
- tilt 절댓값만 보면 I3b가 가장 크지만, 해당 trial의 초기 pitch bias가 달랐고 tilt는
  자세 offset과 움직임을 섞는다. 동적 안정성 비교에는 angular speed와 q excursion이 더 직접적이다.

한계: actor별 한 번, 1초이고 실제 policy rate가 다르다. “armature가 유일 원인”의 인과확정이
아니라 **현재 실기 후보에서 NP를 제외하고 I3b를 기준으로 삼을 충분한 screen**이다.

---

## 9. S2 — timing/LowCmd rate MuJoCo mechanism check

### 의도

실기에서 발견한 40 Hz 정책률과 느린 gait clock이 I3b의 발 교차·낙상을 만들 수 있는지,
그리고 LowCmd 100 Hz가 plant 관점에서 가능한지 짧게 확인한다.

### 결과

| 조건 | 10초 falls | 최소 발 간격 |
|---|---:|---:|
| 학습형 50 Hz policy / wall-clock 2 Hz gait | 0 | +7.48 cm |
| 40.7 Hz policy / wall-clock 2 Hz gait | 1 | −2.99 cm |
| old counter gait 약 1.54 Hz | 5 | −6.81 cm |

rate-fixed tau 10 ms에서 LowCmd `100/150/185 Hz`는 모두 10초 falls 0; 100 Hz의 최소 발
간격은 `+6.31 cm`였다.

### 해석과 한계

정책·gait clock timing이 발 간격에 영향을 준다는 mechanism은 지지한다. 100 Hz LowCmd를
시도할 근거도 됐다. 그러나 짧은 단일-seed MuJoCo cell이며 real plant 인증이 아니다.
특히 EMA를 pure delay로 치환하면 안 된다. legacy EMA는 고주파 억제와 위상지연을 함께 만든다.

---

## 10. R7 — I3b publish 100 Hz 제자리: 성능시험 무효

### 의도

SDK Write/GIL 부하를 줄여 I3b policy 50 Hz와 LowCmd 100 Hz를 동시에 얻고, tau 10 ms로
제자리 안정성을 확인한다.

```bash
cd ~/Workspace/deploy
PYTHONUNBUFFERED=1 ./run_e0.sh fixed 0,0,0 \
  --policy-path ./models/goal_pose_i3b.pt \
  --publish-hz 100 \
  --rate-fixed-filter --filter-tau-ms 10 \
  --hold-prepare --hold-diag 1 \
  --max-policy-seconds 1.0 \
  --test-tilt-abort-deg 12 \
  --log-timing /tmp/i3b_pub100_stand.csv
```

### 겉으로 보인 결과

| 지표 | 값 |
|---|---:|
| 정책 / LowCmd | 51.03 / 99.97 Hz |
| tilt max/end | 3.333/0.974° |
| 몸 각속도 p95 | 0.306 rad/s |
| dq RMS/max | 0.194/1.192 rad/s |
| q 왕복폭 max | 0.097 rad |

### 숨겨진 실패

q/dq/tau 36개 값을 exact compare하면 50번 inference 중 새로운 snapshot은 `18`개뿐이다.
마지막 10행은 같은 snapshot이고 그 구간 span은 `179 ms`였다. `low_state_age` median
`4.53 ms`는 callback 도착시각만 재서 이 문제를 숨겼다.

원인은 callback가 main thread의 `next_inference_time`을 보고 상태 update를 건너뛴 race다.
main이 deadline을 먼저 앞으로 옮기므로 대부분 callback가 skip됐다.

**판정:** policy/plant 성능 비교에는 무효. scheduler race를 찾은 A/B로는 유효.

---

## 11. R8 — I3b publish 100 Hz 전방 0.2 m: 성능시험 무효

### 의도

R7이 겉으로 51/100 Hz를 만족했기 때문에 같은 설정으로 0.2 m 목표를 1초 실행했다.

### 원시 통계

| 지표 | 값 |
|---|---:|
| rows / span | 50 / 0.959 s |
| 정책 / LowCmd | 51.12 / 99.76 Hz |
| 서로 다른 q/dq/tau snapshot | 22/50 |
| 최대 next-fresh 간격 | 약 218 ms |
| tilt start/median/p95/max/end | 3.761/2.050/4.568/5.384/5.384° |
| 몸 각속도 p95 | 1.666 rad/s |
| dq RMS/max | 3.585/20.477 rad/s |
| q 왕복폭 max | 0.568 rad |
| torque/limit max | 67.1% |
| gait phase end | 0.9888 |

별도 kinematic reconstruction에서는 실제 발 간격 최소 `16.4 cm`, 명령 발 간격 최소
`18.6 cm`로 foot crossing은 없었다. 두 번째 gait cycle의 roll RMS는 `1.04→2.43°`,
몸 각속도 RMS는 `0.67→1.17 rad/s`로 증가했다.

### 해석

약 218 ms 동안 같은 관측으로 phase/action만 진행한 것은 2 Hz gait의 거의 반 주기
open-loop에 가깝다. 횡방향 표류가 먼저 커졌고 발 교차는 아직 없었다.

**판정:** I3b walking 능력을 판정할 수 없다. stale feedback failure signature만 남긴다.

---

## 12. stale-state 수정 내용

수정은 다음 세 가지다.

1. LowState callback는 inference deadline과 무관하게 매 callback마다 최신 q/dq/tau/IMU를 쓴다.
2. inference는 lock 아래서 하나의 원자 snapshot과 sequence를 복사한다.
3. late tick 뒤 같은 관측으로 catch-up inference를 몰아 실행하지 않는다.

근거 코드:

- 생산자 update와 race 설명: `htwk-gym/deploy/deploy_goal_pose.py:966-994`
- deadline sleep·catch-up 제거: `:1906-1922`
- snapshot 소비와 로그: `:1960-2017`
- final deploy SHA-256: `d883ed...e64b`

이 수정 뒤 CSV에 `policy_state_age_s`, `low_state_seq`가 추가됐다.

---

## 13. R9 — vendor PREPARE joint-zero probe 3회

### 의도

정책을 실행하지 않고 vendor PREPARE에서 좌우 encoder-space 비대칭이 반복되는지 확인한다.
이는 안전한 read-only 관측이며 특정 관절 zero를 곧바로 보정하는 시험은 아니다.

### 원자료

`realdata/2026-08-09_codex_session/joint_zero_probe.csv`.

### L−R 반복성

| 항목 | P1 | P2 | P3 | 3-run mean ± run-SD |
|---|---:|---:|---:|---:|
| Hip Pitch | −0.11 | −0.49 | −0.49 | −0.363±0.219° |
| Hip Roll | −2.16 | −2.17 | −2.17 | **−2.167±0.006°** |
| Hip Yaw | +0.33 | +0.25 | +0.25 | +0.277±0.046° |
| Knee | −0.68 | −0.74 | −0.74 | −0.720±0.035° |
| Ankle Pitch | −1.22 | −1.23 | −1.22 | **−1.223±0.006°** |
| Ankle Roll | −0.31 | −0.65 | −0.63 | −0.530±0.191° |
| sagittal chain | −2.01 | −2.46 | −2.45 | **−2.307±0.257°** |
| IMU pitch | −2.39 | −0.46 | −0.46 | — |

P1→P2에서 IMU pitch가 1.93° 변했지만 Hip Roll과 Ankle Pitch differential은 0.01°
이내였다. 반면 Hip Pitch와 Ankle Roll은 0.38/0.34° 움직여 자세·하중 민감성도 보인다.
P2와 P3는 사실상 재현됐다. n=3이고 각 run의 155/172/155 sample은 강하게 상관되어 있으므로
482개 독립표본으로 취급하지 않는다.

### fresh RL 첫 snapshot과 연결

fresh 첫 snapshot의 L−R:

| 항목 | probe mean | fresh t0 | 차이 |
|---|---:|---:|---:|
| sagittal chain | −2.307° | −2.287° | 0.019° |
| 6개 관절 differential RMSE | — | — | 0.290° |
| pitch 3관절 RMSE | — | — | 0.242° |

fresh t0는 symmetric RL ramp 직후 첫 actor action이 q에 반영되기 전이다. symmetric target 대비
12 q error RMS는 `0.540°`, max `1.082°`; chain은 L `−3.515°`, R `−1.227°`였다.
사용자의 “PREPARE에서 오른발이 왼발보다 앞” 관찰과 함께 독립적으로 반복된 좌우 비대칭
증거다. 다만 이 chain 부호만으로 어느 발이 앞에 놓이는지를 직접 예측하지는 않는다.

### 왜 아직 zero calibration 값이 아닌가

- vendor PREP 절대 pose는 대략 Hip 0°, Knee 6°, Ankle −5°이고 RL pose는
  −11.46°/+22.92°/−14.32°로 다르다.
- probe는 vendor target/LowCmd를 기록하지 않았고 실제 발·몸이 외부 좌표에서 대칭인지도 모른다.
- encoder zero, 바닥, 접촉, 좌우 질량, 기구 공차, finite-stiffness load deflection이 섞인다.
- 좌우 공통 zero error는 differential probe에 원리적으로 안 보인다.

따라서 확정된 것은 **반복 가능한 encoder-space 좌우 signature**이고, 특정 motor zero는 아니다.
이 값으로 YAML offset을 즉시 만드는 것은 금지한다.

---

## 14. R10 — freshness 수정 후 I3b 제자리 1초

### 의도

stale-state race가 실제로 제거됐는지 확인하고, 보행 전에 현재 hardware state에서 I3b
`0,0,0` 기준선을 다시 만든다.

```bash
cd ~/Workspace/deploy
PYTHONUNBUFFERED=1 ./run_e0.sh fixed 0,0,0 \
  --policy-path ./models/goal_pose_i3b.pt \
  --publish-hz 100 \
  --rate-fixed-filter --filter-tau-ms 10 \
  --hold-prepare --hold-diag 1 \
  --max-policy-seconds 1.0 \
  --test-tilt-abort-deg 12 \
  --log-timing /tmp/i3b_freshstate_stand.csv
```

### timing과 freshness

| 지표 | 값 |
|---|---:|
| rows / span | 50 / 0.972 s |
| 정책률 | 50.44 Hz |
| tick median/p95/max | 19.823/28.614/37.104 ms |
| tick >25 / >30 ms | 11/49, 2/49 |
| LowCmd median | 99.334 Hz |
| sequence unique / non-increasing | 50/50, 0 |
| sequence step median | 10 callbacks/policy tick |
| 추정 callback rate | 495.9 Hz |
| callback age median/max | 1.191/5.479 ms |
| consumed-state age median/p95/max | 8.443/16.146/27.357 ms |
| consumed-state age >20 ms | 1/50 |

50 Hz 평균은 복원됐지만 tick jitter는 남는다. 중요한 차이는 **모든 inference가 증가한 sequence를
소비했다**는 점이다. R7/R8의 179–218 ms hold는 사라졌다.

### motion

| 지표 | 값 |
|---|---:|
| tilt start/median/p95/max/end | 2.646/2.350/3.381/3.453/1.158° |
| roll range | −0.725…+1.010° |
| pitch range | +0.903…+3.436° |
| 몸 각속도 median/p95/max | 0.200/0.370/0.396 rad/s |
| dq RMS/p95(abs)/max | 0.200/0.458/1.083 rad/s |
| q 왕복폭 max | 0.131 rad |
| torque/limit max | 54.0% |
| action RMS | 0.06369 |
| action delta RMS | 0.01230 |
| pitch-chain L−R median/range | −2.920° / −5.204…−1.373° |

전반 0.5초와 후반 0.5초:

| 구간 | tilt median | omega RMS/p95 | dq RMS | chain median |
|---|---:|---:|---:|---:|
| 0–0.5 s | 2.787° | 0.268/0.373 | 0.230 | −2.159° |
| 0.5–1.0 s | 1.635° | 0.172/0.295 | 0.162 | −3.256° |

몸의 운동량과 tilt는 줄었지만 좌우 sagittal chain differential은 더 음수로 갔다. 이는
“지속 발산”보다는 **초기 rocking이 감쇠하면서 비대칭 자세로 수렴**한 모습이다.

### 역사 I3b 첫 1초와 기술 비교

| 지표 | old I3b 첫 1초 | fresh tau10 | 변화 |
|---|---:|---:|---:|
| policy / LowCmd | 38.38/약184 Hz | 50.44/99.33 Hz | 다른 system |
| filter tau | 약24.2 ms | 10 ms | 2.42× 덜 smoothing |
| tilt median/max/end | 1.316/3.402/1.361° | 2.350/3.453/1.158° | median +79%, max 유사 |
| omega p95 | 0.272 | 0.370 rad/s | +36% |
| dq RMS | 0.123 | 0.200 rad/s | +63% |
| q 왕복폭 max | 0.123 | 0.131 rad | +6.5% |
| torque/limit max | 44.9% | 54.0% | +9.1%p |
| action RMS | 0.06249 | 0.06369 | +1.9% |
| action delta RMS/sample | 0.00540 | 0.01230 | 2.28× |
| chain L−R median | −1.110° | −2.920° | −1.810° |

같은 I3b actor라도 같은 deploy system이 아니다. freshness, wall clock, policy rate, LowCmd rate,
filter, joint sample gate, timeout, calibration이 다르다. 따라서 위 표는 원인 분해가 아니라 다음
A/B를 선택하는 기술 비교다.

### 판정

- scheduler/freshness gate: **통과**.
- 1초 safety stand: **통과**; 12° abort/낙상 없음.
- 역사 안정성 재현: **미확정**; 움직임이 더 크고 초기 hardware signature가 다름.
- walking gate: **아직 열지 않음**. tau 단일변수 A/B 뒤 결정한다.

---

## 15. Kp/Kd를 어떻게 읽어야 하나

### 15.1 실제로 말하는 gain

정책 actor는 직접 torque를 내지 않고 joint position target을 낸다. motor 쪽 핵심 식은

`tau ≈ Kp * (q_cmd - q_meas) - Kd * dq`

이다.

| 구간 | Hip P/R | Hip Yaw | Knee | Ankle P/R |
|---|---:|---:|---:|---:|
| RL policy Kp/Kd | 100/2 | 100/2 | 100/2 | **50/1** |
| prepare Kp/Kd | 350/7.5 | 180/3 | 350/5.5 | **250/5** |

발목 gain이 다르다는 말은 맞다. RL에서 발목 Kp/Kd가 Hip/Knee의 절반이다. prepare에서는
발목 Kp 250, Kd 5이고 Hip/Knee와 같은 값이 아니다.

### 15.2 왜 지금 Kd를 범인으로 못 부르나

- vendor PREPARE는 우리 RL gain이 아닌데도 같은 L−R signature가 나온다.
- fresh t0의 `|Kd*dq/Kp|`를 등가각으로 환산하면 12관절 최대 `0.141°`; chain `2.287°`
  보다 한 자릿수 작다.
- 정지 로그에는 excitation이 부족해 Kd를 식별할 수 없다.

따라서 **Kd가 정적 비대칭의 주원인이라는 가설은 사실상 탈락**이다. 걷는 중의 damping과
외란 뒤 ringing에는 여전히 영향을 줄 수 있지만 그것은 별도 동적 A/B가 필요하다.

### 15.3 Kp는 root cause가 아니라 전달 경로

fresh t0 symmetric target 0°에서 Hip Roll은 L/R `−0.732/+1.082°`, tau_est
`+4.377/−6.320 N·m`였다. prepare PD 예측은 `+4.522/−6.510 N·m`로 잔차가
0.15/0.19 N·m뿐이다. 즉 이 순간의 1.814° 차이는 반대방향 하중과 finite Kp의
compliance로 거의 정량 설명된다.

이는 “zero 문제가 아니다”가 아니라, **관측 q 차이가 zero와 load deflection을 섞는다**는 뜻이다.
Kp를 올리면 tracking error가 줄 수 있지만 충격·ringing·안전 margin을 함께 바꾸므로 현재
데이터에는 올릴 근거가 없다.

### 15.4 명목 gain 복사와 randomization

- 학습·배포 nominal `100/2, 50/1`은 일치한다.
- 역사 조용한 구간 33,708 joint-sample의 Kp 항등식 residual MAE는 약 `0.556 N·m`로,
  단순 Kp 오복사 증거가 없다.
- 학습 DR은 ±5%뿐이라 실제 plant variation에 충분한지는 별도 문제다.

최종 판정: **“gain 숫자가 잘못 복사됐다”는 가설은 약하다. “현재 plant의 하중·관성·필터와
학습 gain 조합의 폐루프 감쇠가 다르다”는 가설은 살아 있다.** 두 문장은 다르다.

---

## 16. joint-zero 학습 coverage의 정확한 답

### I3b

persistent joint-zero randomization은 **0°**다. `init_dof_pos` Gaussian `0.05 rad`는 episode
초기 관절 자세 noise이며, 매 episode 동안 encoder와 target 좌표계가 어긋나는 persistent
zero fault와 다르다.

### NH/NP

env별 curriculum level을 `L`이라 하면 reset마다

`b_j | L ~ Uniform[-10° * L * w_j, +10° * L * w_j]`

이며 encoder bias `+b_j`, target offset `−b_j`로 같은 물리 zero fault를 상관되게 구현한다.
관절 draw는 같은 L 조건에서 독립이고, 모든 관절이 env의 같은 L을 공유한다.

| 관절 | weight | 초기 L=0.1 | L=1 ceiling |
|---|---:|---:|---:|
| Hip Pitch | 1.0 | ±1.0° | ±10° |
| Hip Roll | 0.5 | ±0.5° | ±5° |
| Hip Yaw | 1.0 | ±1.0° | ±10° |
| Knee | 1.0 | ±1.0° | ±10° |
| Ankle Pitch/Roll | 0.7 | ±0.7° | ±7° |

curriculum은 초기 `L=0.1`, fall이면 `−0.05`, 30초 timeout이면 `+0.05`다. goal segment
완료마다 오르는 것이 아니다. `target_fall_rate=0.05`는 코드에서 사용되지 않는다.

### 왜 “NH/NP는 ±10°를 학습했다”도 부정확한가

- 실제 `_zero_level` histogram을 로그하지 않았다.
- checkpoint env state에 `_zero_level`이 없다.
- resume하면 level이 다시 0.1에서 시작한다.
- 표준 clean eval은 `joint_zero.enabled=false`였다.

따라서 정확한 표현은:

> I3b는 zero를 한 번도 보지 않았다. NH/NP는 최대 ceiling이 관절별 ±5–10°인 미기록
> adaptive mixture를 보았지만, 실제 노출분포와 zero-on robustness는 복원·인증되지 않았다.

근거: `make_v7_arms.py:801-810`, `goal_pose.py:559-659`, `runner.py:218-250`,
`goal_pose_v7.py:514-532`.

---

## 17. 현재 sim-to-real 원인 우선순위

| 가설 | 현재 판정 | 직접 증거 | 반증/남은 측정 |
|---|---|---|---|
| scheduler가 stale state를 정책에 줌 | **확정, 수정됨** | R7 18/50, R8 22/50; code race | R10 50/50 seq 증가 |
| 정책/gait clock이 학습보다 느림 | **확정, 대부분 수정됨** | 35.8–40.7 vs 50 Hz; sim timing check | R10 50.44 Hz, jitter는 잔존 |
| 현재 좌우 plant/calibration 비대칭 | **강함** | PREP 3회 + fresh t0 재현 | multi-pose 외부기준 추정 필요 |
| I3b zero coverage 부족 | **확정된 coverage gap** | 학습 zero=0°, 오늘 signature≈2.3° | zero-on 실물 robustness는 미측정 |
| tau10이 역사 I3b보다 덜 감쇠 | **유력, 미검증** | old eq24.2ms vs current10ms | 동일 조건 tau A/B |
| NP partial armature mismatch | **시사적** | NP→NH omega p95 −52% | rate/seed 맞춘 반복 필요 |
| 명목 Kp 오복사 | **증거 약함** | config 일치, residual 0.556Nm | dynamic excitation 없이는 Kd 미식별 |
| Kd가 PREP skew의 원인 | **기각 쪽** | actor 전 재현, 등가각 max0.141° | 걷기 ringing 원인성은 별도 |
| foot crossing이 stale walk의 첫 실패 | **현재 로그에서는 아님** | 실제 gap min16.4cm | 유효 fresh walk 필요 |
| cleanup이 첫 NP floor contact의 전부 | **아님** | cleanup 전 tilt10°, torque99% | 다만 마지막 limp는 cleanup 교락 |

현재 실기 실패는 한 원인으로 환원되지 않는다. 확인된 software race는 고쳤고, 남은 가장 싼
분리는 **filter/initial asymmetry를 고정한 fresh I3b A/B**다.

---

## 18. 다음 실험 사전등록

### 18.1 다음 한 번: tau 24.2 ms 제자리 1초

목적: 같은 actor, hardware state, scheduler, policy 50 Hz, LowCmd 100 Hz를 유지하고
역사 I3b의 smoothing만 복원한다.

```bash
cd ~/Workspace/deploy && \
PYTHONUNBUFFERED=1 ./run_e0.sh fixed 0,0,0 \
  --policy-path ./models/goal_pose_i3b.pt \
  --publish-hz 100 \
  --rate-fixed-filter --filter-tau-ms 24.2 \
  --hold-prepare --hold-diag 1 \
  --max-policy-seconds 1.0 \
  --test-tilt-abort-deg 12 \
  --log-timing /tmp/i3b_fresh_tau24_stand.csv
```

바꾸는 것은 `filter-tau-ms` 하나뿐이다.

사전 판정:

1. 계측 유효성: `low_state_seq` 50 inference에서 모두 증가, state-age max <40 ms.
2. 안전: 12° abort 없음, torque/limit max <70%.
3. 개선: 현재 tau10 대비 `omega p95 0.370 → ≤0.30 rad/s` 또는 `dq RMS 0.200 → ≤0.16`
   중 하나를 만족하고 다른 하나가 악화하지 않음.
4. 자세 tradeoff: tilt max가 `4°`를 넘거나 chain bias가 계속 커지면 “조용해졌지만 잘못된
   자세에 고정”으로 별도 판정.

1초 single-run threshold이므로 통과하면 같은 조건을 한 번 반복해 방향 재현을 확인한다.

### 18.2 tau24가 낫다면

LowCmd만 `100 → 185 Hz`, tau24.2를 유지한다. 185 Hz에서
`alpha = 1-exp(-dt/tau) ≈ 0.20`이므로 역사 legacy filter를 거의 재현하면서 wall-clock과
fresh snapshot은 유지한다. old counter-clock 전체 rollback은 여러 변수를 한꺼번에 바꾸므로
하지 않는다.

### 18.3 유효한 첫 보행

제자리 A/B가 calmer일 때만 같은 actor/filter로 `fixed 0.2,0,0`, **0.5초**부터 시작한다.
30초 hands-off나 0.6 m 명령은 현재 단계의 목표가 아니다.

보행 판정은 낙상 여부만 보지 않는다.

- 모든 inference의 state sequence 증가
- tilt 추세와 roll RMS
- q/dq/tau saturation
- 실제/명령 발 간격과 crossing
- sagittal/hip differential 누적 drift
- 종료 PREPARE 전환 전후를 분리

0.5초 통과 뒤 0.8초, 1.0초로 늘린다. 사람 손 접촉 시점은 반드시 별도 기록한다.

### 18.4 zero 문제는 별도 lane

오늘의 3회 differential을 곧바로 calibration offset으로 쓰지 않는다. 이후 여러 서로 다른
안전 자세에서 vendor target, q, tau, IMU, 외부 발/몸 기준을 함께 수집해 common-mode와
load compliance를 분리해야 한다. 이 lane은 오늘 보행 재현을 막는 선행조건이 아니라,
재현성 없는 손캘리 상태를 장기적으로 없애기 위한 작업이다.

---

## 19. Claude Code 교차검증 체크리스트

Claude Code는 아래 결론을 믿지 말고 raw에서 다시 계산한다.

1. `trial_manifest.csv`의 모든 SHA-256을 실제 파일과 대조한다.
2. R1–R8, R10에서 rows/span, `1/mean(tick_dt[1:])`, pub median, tilt, omega,
   dq RMS/max, q p2p, torque ratio를 독립 재계산한다.
3. R7/R8의 q/dq/tau 36차원 exact-equality run을 찾아 fresh count `18/22`, 최대 next-fresh
   `179/218 ms`를 재현한다.
4. R10에서 `low_state_seq` unique 50, non-increasing 0, callback rate 약495.9 Hz,
   state age `8.44/16.15/27.36 ms`를 재현한다.
5. 역사 CSV를 44.181–149.970, 150.020–164.460, 180.120–228.320으로 분절하고
   105.785/14.490/48.200초 및 tilt 통계를 재현한다. post 첫 15.668초 gap을 제외한다.
6. `00ed5b8` 코드에서 역사 `t_s`가 `time.monotonic()`인지 확인한다. 기존
   `realdata/README.md:23-24` 설명과 충돌함을 표시한다.
7. I3b actor MD5, 역사 deploy commit/hash, 현재 deploy hash를 대조한다.
8. `Goal_Pose_E0.yaml`에서 policy/prepare gain과 torque limit를 확인한다.
9. frozen V7에서 gain DR가 ±5%임을 확인하고 ±15%라는 이전 주장을 기각한다.
10. I3b zero=0, NH/NP adaptive iid 분포, `_zero_level` checkpoint 미저장, standard eval
    zero-off를 코드에서 독립 추적한다.
11. probe 3회의 L−R와 fresh t0 L−R를 재계산한다. 일치가 encoder zero의 인과증명은
    아니라는 반론을 적극적으로 시도한다.
12. old vs fresh 비교에서 actor 외에 clock, rate, filter, gate, timeout, calibration이 함께
    달라졌음을 누락하지 않는다.

교차검증 결과는 각 항목을 `재현 / 숫자 불일치 / 해석 과장 / 원자료 부족` 중 하나로 표시하고,
불일치 시 사용한 코드와 정확한 분모를 함께 남긴다.

---

## 20. 근거 인덱스

- 원시 CSV와 manifest: `realdata/2026-08-09_codex_session/`
- 역사 CSV: `realdata/2026-08-09_t_stand_i3b.csv`
- 역사 실행·timing: `ibatch.md:5209-5296`, `MIDREPORT_DEPLOY_20260809.md:15-79`
- prepare pose와 진입: `HANDOFF_DEPLOY_ENTRY_20260807.md:46-106`
- 모델/MD5: `MIDREPORT_DEPLOY_20260809.md:15-28`
- current scheduler/freshness: `htwk-gym/deploy/deploy_goal_pose.py:966-994,1906-1922,1960-2017`
- current filter: `htwk-gym/deploy/deploy_goal_pose.py:2157-2175`
- deploy gain: `htwk-gym/deploy/configs/Goal_Pose_E0.yaml:46-76,124-163`
- I3b/NH/NP 정의: `htwk-gym/tools/make_v7_arms.py:427-431,668-676,801-810,1036-1042,1125-1135,1230-1255`
- joint-zero 구현: `htwk-gym/envs/K1/goal_pose.py:542-659`
- gain DR: `htwk-gym/envs/K1/Goal_Pose_V7.yaml:495-502`
- zero checkpoint state: `htwk-gym/utils/runner.py:218-250`, `htwk-gym/envs/K1/goal_pose_v7.py:514-532`
- current Codex 전체 transcript:
  `/Users/dmdrb/.codex/sessions/2026/08/09/rollout-2026-08-09T11-09-13-019fe448-0547-7f11-9973-2d9c7493f3e7.jsonl`
- Claude Deployer 원세션:
  `/Users/dmdrb/.claude/projects/-Users-dmdrb-RoboCup-k1-goalpose/6b2a2a27-b7e8-4400-be55-c45c83d03289.jsonl`

---

## 21. 최종 판단

현재 목표는 새 학습이 아니라 **기존 best actor의 실물 closed-loop를 학습 조건에 가깝게
복원해 첫 유효 보행을 얻는 것**이다.

지금까지 확인된 가장 큰 software 결함은 stale-state scheduler였고 이미 수정·검증됐다.
그 뒤 남은 데이터는 I3b가 세 actor 중 가장 낫지만 역사 I3b보다 초기 운동량이 크고,
현재 로봇에는 반복 가능한 약 2.3° 좌우 chain signature가 있다는 것을 보여준다.

이 상태에서 Kp/Kd를 바꾸면 filter, plant asymmetry, actor response를 다시 섞는다. 따라서
다음은 tau 24.2 ms 단일변수 stand A/B이며, 통과 시 0.2 m 목표 0.5초 fresh-state 보행으로
진행하는 것이 가장 빠르고 해석 가능한 경로다.
