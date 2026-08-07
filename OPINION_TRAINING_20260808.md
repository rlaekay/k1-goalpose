# 학습 담당 세션에게 — 의견서 (2026-08-08)

사용자 요청: **① mirror data augmentation ② joint encoder bias ③ pitch chain 모드
④ teacher–student** 넷에 대한 의견. "현재 하는 학습과 관련 문서도 열람해서 확인하고
그걸 반영한" 의견서.

이 문서는 **읽고 바로 arm 을 설계할 수 있게** 쓴다. 모든 숫자에 출처(파일:행 또는 명령)를
붙였고, **사실 / 추론 / 추측**을 표시로 갈랐다. 작업 원장은
`scratchpad/B4_opinion_notes.md`.

> ⚠️ **먼저 읽을 것**: [RETRACTIONS.md](RETRACTIONS.md), [AUDIT_20260807_2300.md](AUDIT_20260807_2300.md).
> 이 문서는 그 둘 위에 얹혀 있고, 그 둘과 충돌하는 주장을 하지 않는다.

---

## 0. 한 장 요약 — 네 질문에 대한 답 한 줄씩

| | 질문 | 답 |
|---|---|---|
> ⚠️ **작성 중 병렬 세션이 두 커밋을 냈다**(`8d763fb`, `e76e926`, 2026-08-08 01:27/01:28).
> 이 문서는 **그 뒤에 다시 맞췄다** — 겹치는 항목 셋은 §1-5 에 따로 모았다.
> 그리고 **배포 쪽이 지금 영점 추정기를 만들고 있다**(§C-5b). 그것이 B·C 의 권고를
> 바꾼다. 반드시 §C-5b 를 읽고 나서 §B-6 을 읽어라.

| **A** | mirror data augmentation 은? | **켜지 마라. 지금 켜져 있는 손실(LOSS)이 문헌상 더 안전하고 더 일관된 쪽이다.** 증강(DUP)은 우리 코드에서 켤 수조차 없고(§A-5), 켜려고 손대면 비평자 미러링이 조용히 빠진다. 대신 **손실 쪽의 결함 2개**(logstd 미결속, 지표 부재)를 고쳐라 |
| **B** | joint encoder bias 는 무조건 좋은 게 아니었어? | **아니다. 조건부다.** 조건 셋: ①`joint_zero` 블록으로만 켤 것(레거시 두 키는 다른 고장이다) ②**커리큘럼 수준을 로깅**할 것 ③**영점을 켠 평가 레인**을 만들 것. 지금 `NZ_zeroiid` 는 셋 중 둘이 빠져서 **arm 의 존재 이유가 측정되지 않았다** |
| **C** | pitch chain 이 뭐야, 하면 나아져? | **나아진다. 그리고 지금 우선순위가 뒤집혀 있다.** 사용자 절차가 만드는 것은 좌우 **공통모드**(pitch 사슬 + roll 사슬)인데 `mirror` 가중치가 0.10 으로 최저다. ⛔ 그리고 `anti_mirror 0.20` 의 유일한 근거 실험은 **`mirror` 모드를 잰 것**이었다(§C-3). 게다가 그 실험의 부호 규약이 실기와 다르다(§C-4) |
| **D** | teacher–student | **지금 하지 마라. 순서가 있다.** `N7_critic`(=asymmetric AC, 지금 5115/6000) 결과를 먼저 보고, `N4_hist` 재실행으로 "이력이 도움이 되는가"를 가른 뒤에만 의미가 있다. 증류는 **정책류를 넓히지 않는다** — 넓히는 것은 이력이고, 증류는 그것을 학습 가능하게 만드는 절차다(§D-2). 설계는 §D-4 에 사전고정해 뒀다 |

### 이 문서가 새로 찾아낸 것 (전부 원본 확인)

1. ⛔ **`NZ_zeroiid` 는 영점 오차를 꺼놓고 채점됐다.** `NC_actfilter` 에만 붙어 있던 경고가
   영점 arm 세 개(`NZ`·`N9`·`NB`)에도 똑같이 붙는다 → §B-3
2. ⛔ **`joint_zero` 커리큘럼 수준이 어디에도 기록되지 않는다.** NZ 가 실제로 몇 도에서
   학습했는지 **아무도 모른다** → §B-4
3. ⛔ **`anti_mirror = 0.20` 의 근거 실험은 물리적으로 `mirror` 모드였다** → §C-3
4. ⛔ **MuJoCo 영점 프로브의 부호 규약이 실기·Isaac 과 반대다.** 그 스윕의 임계값
   (±3° 절벽 등)을 `joint_zero.max_deg` 근거로 쓰면 안 된다 → §C-4
5. ⛔ **`probe_joint_zero.py`(R4)는 사용자 절차가 만드는 바로 그 모드에 눈이 멀었다.**
   같은 콜백이 이미 모으는 IMU 값으로 닫힌다 — 로봇 시간 0초 추가 → §C-5
6. ⛔ **2026-08-07 에 고친 대칭손실 맵 결함은 과거 mirror 실패(G10/H1)를 설명하지 않는다.**
   H1 은 obs 54 였다 → §A-2
7. ✅ **`claim_check.py` 가 `best.pth` 를 iteration 으로 풀지 않는다** — 나도 독립적으로
   찾았고, **병렬 세션이 같은 시각에 찾아 이미 고쳤다**(`e76e926`) → §1-3
8. ⚠️ **`K1_locomotion_armsdown.urdf` 의 `Hip_Pitch` z 가 실기보다 15 mm 높다.**
   N 배치는 안전하지만(`K1_robot_boxfoot.urdf`) **배포 계보 `I3b_stance10` 은 그 위에서
   학습됐다.** 배포 쪽 새 도구가 12개 URDF 를 파싱해 찾은 것을 교차확인했다 → §C-5b

---

## 1. 지금 학습 상태 (2026-08-08 01:37 KST)

`ssh a6000 'bash .../tools/round_status.sh'` 직접 실행. (첫 확인 01:11, 갱신 01:37)

### 1-1. 돌고 있는 것

| 카드 | 작업 | 진행 |
|---|---|---|
| gpu0 | **`025-NF_dwellclock`** (`pause_gait_during_dwell: true`) | 방금 시작 |
| gpu1 | `040-N7_critic` | 5115+ / 6000 |
| — | `011-eval_NE_ctrl100` | 채점 중 |
| gpu1 대기 | **`026-NA_histzero_v2`** (prime 수정 후 재실행) | — |

`030-ND_dwell` 은 **rc143(SIGTERM)로 중단**되고 `NF_dwellclock` 으로 교체됐다
— [AUDIT §2-4](AUDIT_20260807_2300.md) 의 "방향이 반대" 판정이 반영됐다. ✅
큐 워커 2, `idle_watch` 정상.

### 1-2. 방금 나온 백필 채점 (사용자가 보고를 요청한 두 축)

`python tools/round_table.py logs/eval_rounds/n2na` — **정확도**(공통 waypoint,
120 s, 256 env, `protocol_sha 852600f2`, `env_code_sha 2076c493`):

| arm | best iter | 오차 med | p90 | 낙상 | 표본탈락 | strict | 구간 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`NZ_zeroiid`** | **1700** | 10.43 cm | 13.86 | **0** | **0.00 %** | 8.5 % | 4613 |
| **`NC_actfilter`** | **100** | 19.92 cm | 26.00 | 31 | 0.67 % | 10.3 % | 4604 |
| `NA_histzero` ⛔무효 | 3300 | 5.47 | 9.68 | 12 | 0.30 % | 20.3 % | 4613 |

**지속 보행**(`forward_hold`):

| arm | 속도 med | >1.0 m/s | 낙상간격 | 낙상률/구간 |
|---|---:|---:|---:|---:|
| `NZ_zeroiid` best | 1.36 | 98.0 % | 1,997 s | 0.0005 |
| `NZ_zeroiid` final(6000) | 1.45 | 98.8 % | **낙상 0** | **0.0000** |
| `NC_actfilter` best | 1.18 | 94.5 % | 173 s | 0.0056 |
| `NC_actfilter` final(6000) | 1.49 | 98.2 % | 802 s | 0.0012 |

### 1-3. ⛔ 이 표를 읽는 규칙 — 그리고 `claim_check.py` 의 구멍

CLAUDE.md 가 요구한 대로 `claim_check.py` 를 먼저 돌렸다:

```
python tools/claim_check.py .../NZ_zeroiid.accuracy/report.json .../NC_actfilter.accuracy/report.json
→ ✅ 조건이 전부 같다. 이 리포트들은 나란히 비교해도 된다.
```

**그런데 그 판정이 틀렸다.** `selection.json` 을 직접 열면:

| arm | `best.pth` 가 가리키는 iteration |
|---|---:|
| `NZ_zeroiid` | **1700** |
| `NC_actfilter` | **100** |

원인은 [`tools/claim_check.py:37`](htwk-gym/tools/claim_check.py) 이
`os.path.basename(r["checkpoint"])` 를 찍기 때문이다. `best.pth` 는
**`--link_best` 가 매번 다시 만드는 심볼릭 링크**(RETRACTIONS C9)라, 두 arm 이 전혀
다른 iteration 이어도 문자열이 같다. **C3("N0 가 세 축 지배")를 만든 것이 정확히 이
불일치다.** 검증 도구가 자기가 막으려던 실패를 통과시켰다 — [AUDIT §6-3](AUDIT_20260807_2300.md)
의 "검증 도구가 먼저 틀린다" 다섯 번째 사례다.

### ✅ 이미 고쳐졌다 — 병렬 세션이 같은 시각에 같은 것을 찾았다

커밋 **`e76e926`**(2026-08-08 01:28, "claim_check 가 best vs final 을 못 잡고 있었다").
`_ckpt_label` 로 **심볼릭 링크를 해석**하고 `ckpt_iter` 를 **PROTOCOL 로 승격**해서
iteration 이 다르면 ⛔ 가 뜬다. 지표 9개에 `ckpt_iter` 의존성도 붙었다.
**맥 로컬처럼 링크를 못 푸는 환경에서는 fail-safe 로 ⛔** 를 찍는다.

> ⚠️ **그래서 §1-2 표를 다시 돌려 보면 이제 ⛔ 가 뜬다.** 이 문서의 §1-4 규칙은 그대로다.
> 그리고 **같은 라운드에서 나와 병렬 세션이 독립적으로 같은 결함에 도달했다** — 교차검증이
> 실제로 작동한 사례로 기록해 둔다.

### 1-4. 그래서 §1-2 표에서 지금 말할 수 있는 것 / 없는 것

**말할 수 있는 것 (사실)**
- `NZ_zeroiid` 는 **정확도 프로토콜 4,613 구간에서 낙상 0, 표본탈락 0.00 %** 다.
  이 저장소에서 측정된 어떤 arm 보다 낮다.
- `NZ_zeroiid` final(6000)은 **보행 프로토콜에서도 낙상 0** 이면서 속도 med 1.45 m/s,
  >1.0 m/s 체류 98.8 % 다. **속도를 지키면서 낙상이 0인 조합은 처음이다.**
- `NC_actfilter` 는 정확도 축에서 붕괴했다(19.92 cm). 그리고 선택기가 `model_100` 을
  골랐다 = **학습이 진행될수록 정확도가 나빠졌다**는 뜻이다.

**말할 수 없는 것 (규칙 위반이 되는 것)**
- ⛔ *"영점 랜덤화(레버)가 낙상을 없앴다"* — 대조군 `NE_ctrl100` 채점이 **지금 돌고 있다**.
  두 arm 모두 계보 B(`N1_path/model_100` warm start)라 `N1_path` 와 나란히 놓을 수 없다
  (RETRACTIONS C9). **`NE_ctrl100` 이 나오기 전에는 레버 효과를 말하지 마라.**
- ⛔ *"NZ 가 NC 보다 낫다"* — 두 arm 은 체크포인트가 it1700 대 it100 이다(§1-3).
  **같은 iteration 으로 맞춰 다시 재기 전에는 순위를 매기지 마라.**
- ⛔ *"NC_actfilter 는 실패한 레버다"* — 그 arm 은 **필터를 끈 config 로 채점됐다**
  (RETRACTIONS 미결항목). 존재 이유가 안 재졌다. **그리고 같은 결함이 NZ 에도 있다(§B-3).**

### 1-5. 이 문서와 병렬 세션 커밋의 겹침 (작성 중 들어왔다)

`8d763fb`(01:27) · `e76e926`(01:28). **서버는 이미 pull 했다.**

| 내가 §에 쓴 것 | 병렬 세션 | 이 문서의 처리 |
|---|---|---|
| §1-3 `claim_check` 가 `best.pth` 를 안 푼다 | ✅ **고쳤다**(`e76e926`) | 항목을 "확인+완료" 로 낮췄다 |
| §D-4 **P1** `_obs_prime_ids` 를 리셋 env 로 | ✅ **고쳤다**(`8d763fb`) — `_resample_goals` → `_reset_idx` 로 이동. **`obs_delay` 재추첨도 같은 버그였다**(6초마다 다시 뽑히고 있었다) | 선행조건에서 제거. `026-NA_histzero_v2` 가 이미 큐에 있다 |
| `ND_dwell` 방향 반대 | ✅ **`NF_dwellclock` 으로 교체**(`pause_gait_during_dwell: true`) | §1-1 갱신 |
| §C-3 `anti_mirror` 유령 인용 | ⚠️ **출처는 고쳤고**(`ibatch.md:3893-3920`) *"0.20 은 이 데이터로 정당화되지 않는다"* 까지 갔다. **그러나 그 실험이 어느 모드인지는 아직 못 봤다** | **§C-3 이 여전히 새 결론이다.** 아래 참조 |
| §B-3 `NZ` 가 영점 OFF 로 채점 | ❌ 미발견 | **새 결론** |
| §B-4 커리큘럼 수준 미로깅 | ❌ 미발견 | **새 결론** |
| §C-4 MuJoCo 프로브 부호 규약 | ❌ 미발견 | **새 결론** |
| §A 전체(대칭) | ❌ 다루지 않음 | **새 결론** |

⭐ 그리고 **배포 쪽이 지금 영점 추정기를 만들고 있다**(미커밋 5파일). §C-5b 에서 다룬다.

---

# A. 대칭 — 손실(LOSS)과 데이터 증강(DUP)

## A-1. 우리 구현이 정확히 무엇인가

### 두 스위치가 이미 **따로** 있다

| 개념 | 문헌 이름 | 우리 키 | 현재 값 | 코드 |
|---|---|---|---|---|
| 대칭 **손실** | LOSS (Yu+2018) | `algorithm.symmetry_coef` | **0.5, 전 arm 활성** | [`runner_v3.py:316-321`](htwk-gym/utils/runner_v3.py) |
| 대칭 **데이터 증강** | DUP (Abdolhosseini+2019) | `algorithm.mirror_augmentation_coef` | **키 자체가 V7 yaml 에 없다 → 0** | [`runner_v3.py:270-300`](htwk-gym/utils/runner_v3.py) |
| (죽은 키) | — | `algorithm.symmetric_coef: 10.` | **아무도 안 읽는다** | [`runner_v3.py:13`](htwk-gym/utils/runner_v3.py) 이 명시 |

확인: `NZ_zeroiid`·`NC_actfilter`·`NE_ctrl100`·`N7_critic` 네 run 의 `config.yaml` 전부
`symmetry_coef: 0.5` / `mirror_augmentation_coef` 없음.

### 손실의 정확한 형태

```python
# utils/runner_v3.py:317-321
if use_symmetry:
    mirrored_dist = self.model.act(self.env.mirror_obs(mb_obs))
    sym_loss = F.mse_loss(mirrored_dist.loc, self.env.mirror_actions(dist.loc))
    loss = loss + sym_coef * sym_loss
```

= Abdolhosseini 식 (2) 의 `L_sym = Σ_t ||π(s_t) − M_a(π(M_s(s_t)))||²` 와 **동일**하다
(합 대신 평균, 계수 0.5). 원논문은 Yu+2018 의 `w = 4` 를 쓰고 *"w 선택에 민감하지 않다"*
고 적는다.

### 미러 맵

[`envs/K1/goal_pose_v3.py:84-199`](htwk-gym/envs/K1/goal_pose_v3.py) `_build_mirror_maps`.

- 관절 부호: `Roll` 또는 `Yaw` 가 이름에 들어가면 −1, 나머지(Pitch/Knee) +1 (`:121`).
  URDF 로 교차확인했다 — 좌우 동명 관절이 **같은 월드축**을 쓰고(둘 다 `axis="1 0 0"` 등)
  한계만 반전돼 있다(`Left_Hip_Roll [-0.4, 1.57]` vs `Right_Hip_Roll [-1.57, 0.4]`).
  시상면 반사(y → −y)는 x·z 축 회전의 부호를 뒤집고 y 축 회전은 보존하므로 **맞다.**
- 관측 54채널 맵(`:152-167`): `gravity_y`, `angvel_x`, `angvel_z`, `goal_rel_y`,
  `heading_error`, `body_roll_target`, `feet_offset_y_target` 부호반전 /
  `foot_yaw_L ↔ foot_yaw_R` 교환+반전 / **보행시계 cos·sin 둘 다 부호반전**(=반주기 위상 이동) /
  `dof_pos`·`dof_vel`·`actions` 블록에 관절 순열·부호.
- 2026-08-07 수정: 확장 채널(`extra_foot_offset`/`extra_dof_tau`)과 이력 타일링(`:169-194`).
  [AUDIT §5](AUDIT_20260807_2300.md) 가 7개 폭에서 involution·전단사·부호·비혼합 **96/96 통과** 확인.

### ⭐ 우리는 문헌의 가장 큰 함정 둘을 이미 피하고 있다

**① 중립상태 병리 (Abdolhosseini §8).** 원논문:

> 대칭 정책은 중립상태(s = M_s(s))에서 대칭 액션밖에 못 내므로 **그 상태를 벗어날 수 없다.**
> 저자들의 대책은 "항상 비중립 자세에서 시작하라"이고, 그래도 가끔 hopping gait 로 수렴한다.

우리 관측에는 **보행시계 (cos φ, sin φ)** 가 있고 미러가 둘 다 부호반전한다.
`M_s(s) = s` 가 되려면 `cos φ = sin φ = 0` 이어야 하는데 **불가능**하다.
⇒ **우리 관측 공간에는 중립상태가 구조적으로 존재하지 않는다.** 이것은 사실상
PHASE 기법을 관측에 내장한 것과 같고, 원논문이 Cassie(위상 입력 있음)에서 PHASE 가
가장 좋다고 보고한 것과 방향이 같다.

**② 정규화 함정 (Abdolhosseini §4.5).** *"입력 정규화가 미러링 가정을 깰 수 있다"*.
우리 정규화는 [`Goal_Pose_V7.yaml` `normalization:`](htwk-gym/envs/K1/Goal_Pose_V7.yaml)
의 **config 상수**이고 running statistics 가 없다. 좌우 채널이 같은 상수를 쓰므로
미러 불변이다. ⇒ **해당 없음.**

## A-2. ⛔ 역사 — G10 의 "역효과" 는 대칭 때문이 아니었다

`docs-research/user_requirements.md:74`(G10)가 기록한 것:

> H1 mirror error p90 **0.080 → 0.150 (+87.5 %)**, touchdown 좌우 bias **2.9 → 9.4 cm**,
> 기준 통과 **0/31** = 직접 목표까지 악화

이 문장 하나가 이 저장소에서 "대칭은 역효과" 의 유일한 근거다. **원본을 열어 봤다.**

### 사실 ① H1 은 레버 5개를 동시에 바꿨다

`archive/sweeps/hbatch/H0-codex.yaml` ↔ `H1-codex.yaml` 의 실제 diff **전부**:

| 항목 | H0 | H1 |
|---|---|---|
| `symmetry_coef` | 0.0 | **0.5** |
| `mirror_augmentation_coef` | 0.0 | **0.5** |
| `init_dof_pos` σ | 0.05 | **0.075** |
| `joint_encoder_bias` | ±0.015 | **±0.025** |
| `joint_target_offset` | ±0.010 | **±0.020** |

**대칭 두 항 + 영점 DR 1.67~2배 + 초기자세 노이즈 1.5배.** `hbatch-codex.md:413` 자신이
*"어느 요소가 원인인지 이 결과만으로 분리할 수 없다"* 고 적는다.

### 사실 ② 2026-08-07 에 고친 결함은 H1 과 무관하다

`H1-codex.yaml:14` → `num_observations: 54`.
2026-08-07 수정은 **54 를 넘는 채널**이 항등순열로 남던 것이었다(`goal_pose_v3.py:126-133`).
H1 에는 54 초과 채널이 없다. ⇒ **그 결함은 과거 실패를 설명하지 않는다.**
(사용자 질문 "그 수정이 과거 실패를 설명하는가"에 대한 직접 답: **아니오.**)

### 사실 ③ 원인은 이미 특정돼 있었고, 그중 하나는 이미 고쳐졌다

`hbatch-codex.md:536-566` 의 2026-08-01 코드 감사가 **미러 맵 자체에는 부호·순열 오류가
없다**고 확인한 뒤 진짜 원인 넷을 짚는다:

1. **mirror PPO 의 importance-ratio 기준분포가 틀렸다** — 합성표본 `(Ms, Ma)` 의 행동밀도는
   `π_old(a|s)` 인데 `π_old(Ma|Ms)` 를 썼다. *"정책이 이미 완벽히 대칭일 때만 같고,
   대칭을 배우려는 바로 그 실험을 편향시킨다."*
   ✅ **이것은 현재 코드에서 고쳐져 있다** ([`runner_v3.py:285-300`](htwk-gym/utils/runner_v3.py)
   의 주석이 그 수정을 명시한다).
2. actor gradient 질량이 1.0 → **1.5배**가 되고 grad-norm clip 1.0 이 value/ordinary PPO 까지
   같이 재조정한다.
3. **critic 증강이 계수와 무관하게 항상 50:50** (`0.5 → 0.25` 로 낮춰도 압력이 안 준다).
   ⚠️ 현재 코드 [`runner_v3.py:256-262`](htwk-gym/utils/runner_v3.py) 에 **그대로 남아 있다.**
4. **KL 스케줄까지 바뀐다** — 증강 arm 만 `max(original_KL, mirror_KL)` 로 LR 을 제어한다
   ([`:364-365`](htwk-gym/utils/runner_v3.py)). 성능 차이에 **학습률 궤적 차이**가 섞인다.

> **판정 (추론)**: G10 의 역효과는 *"대칭 학습이 나쁘다"* 가 아니라
> ***"편향된 mirror PPO + 5레버 동시 + 다른 LR 궤적"* 의 결과다.**
> 그중 **가장 큰 것(1번)은 증강 경로에만 있었고, 손실 경로에는 없었다.**
> 즉 **지금 켜져 있는 `symmetry_coef` 는 그 실패의 원인 목록에 들어 있지 않다.**

## A-3. 문헌 — LOSS 대 DUP, 무엇이 표준인가

**Abdolhosseini, Ling, Xie, Peng, van de Panne. "On Learning Symmetric Locomotion."
MIG '19.** (PDF 원문 직접 확인)

네 방법: **DUP**(궤적 튜플을 미러해서 버퍼에 같이 넣는 데이터 증강) /
**LOSS**(보조 손실, = 우리 것) / **PHASE**(반주기만 학습하고 나머지는 미러) /
**NET**(망 구조로 등변성 강제).

| 원문 주장 | 절 |
|---|---|
| *"DUP is the **least effective** in enforcing symmetry, while **LOSS is the most consistent**."* | §7.1 |
| *"enforcing symmetry has **no consistent impact on learning speed**."* | §7.2 |
| 결론: *"can sometimes be **harmful to learning efficiency**, but in general it produces **higher quality motions**."* | §9 |
| DUP 은 **off-policy** 다 — 미러 튜플이 엄밀히 on-policy 가 아니라 PPO/TRPO 와 충돌 소지. 실무적으로는 치명적이지 않았다 | §4.1 |
| Yu+2018 의 LOSS 는 **커리큘럼이 있을 때만** 이득. 없으면 vanilla PPO 대비 무이득, **휴머노이드에서는 해로웠다** | §2 |
| Table 1·2: Walker2D/3D 에서 LOSS 가 대칭지표 최상위권, Cassie 에서는 PHASE 가 최상 | §7.3 |

**Mittal 외, "Leveraging Symmetry in RL-based Legged Locomotion Control"
(arXiv:2403.17320)**: 엄밀 equivariant 망 > 데이터 증강 > 나머지 (표본효율·성능 모두).
즉 **DUP 은 세 선택지 중 중간이고, 우리가 그것을 위해 감수해야 하는 구현 위험이 크다.**

> **문헌 요약 한 줄**: *증강이 손실보다 낫다는 근거는 없다. 손실이 가장 일관되고,
> 우리 구현 위험도 손실 쪽이 압도적으로 작다.*

## A-4. 영점 오차와의 상호작용 — 충돌하는가?

사용자가 두 주제를 묶어 물은 이유가 여기 있을 것 같아 따로 판정한다.

### 판정: **충돌하지 않는다.** 오히려 정합한다.

`anti_mirror`(좌우 반대 영점) 고장은 **정의상 비대칭**이다. 그래서 "대칭손실이 그것을
막지 않느냐" 는 자연스러운 걱정이다. **아니다.** 이유:

대칭손실이 요구하는 것은 `π(M s) = M π(s)` — **등변성(equivariance)** 이지
**불변성(invariance)** 이 아니다. env `i` 의 숨은 영점이 `b_i` 일 때, `M s` 는
*"영점이 `M b_i` 인 다른 로봇"* 의 유효한 관측이다. 손실이 요구하는 것은

> "좌우가 뒤집힌 영점 오차를 가진 로봇을 만나면, 좌우가 뒤집힌 액션을 내라"

이고, 이것은 **정확히 옳은 요구**다. 정책이 좌우 차이를 **무시하라**는 요구가 아니다.

그리고 `joint_zero` 의 다섯 모드 전부가 **좌우 대칭인 분포**를 만든다
(`iid` 균일대칭 / `single` 부호·관절 무작위 / `leg_common` 다리·부호 무작위 /
`anti_mirror`·`mirror` 부호 대칭). ⇒ **관측 분포 자체가 미러 불변**이므로
대칭손실이 요구하는 등변성은 데이터가 실제로 만족하는 성질이다.

> ⚠️ **단, 실제 제약은 다른 데 있다**: 정책이 등변적으로 행동하려면 `b_i` 를
> **관측에서 추정할 수 있어야** 한다. `history_steps: 1` 이고 토크 채널도 없으면
> `b` 는 거의 관측 불가능하고, 정책이 할 수 있는 최선은 "보수적으로 굴기"뿐이다.
> **이것은 대칭의 문제가 아니라 부분관측의 문제다** — 그리고 §D 의 논거로 이어진다.
> (`MB_obsdelay` 가 지연만 주고 식별 수단을 안 줘서 1.0 m/s 를 **한 번도** 못 넘은 것이
> 같은 모양이다. [HANDOFF_TRAINING §3](HANDOFF_TRAINING_20260807.md))

### ⚠️ 진짜 상호작용은 **비평자** 쪽이다 (증강을 켤 때만)

`hbatch-codex.md:552` 가 짚은 것: `joint_encoder_bias` 는 privileged obs 에 **없다**.
증강(DUP)은 미러 상태에 **원본의 advantage/return 을 그대로 복사**하는데, 숨은 영점이
실제로 미러되지 않으면 그 근사가 영점 범위가 커질수록 나빠진다.

✅ 이 구멍은 **이미 메워질 준비가 돼 있다**: `observation.privileged_extra` 가
비평자에 `joint_encoder_bias` 12채널을 넣는다([`goal_pose.py:1204-1214`](htwk-gym/envs/K1/goal_pose.py)).
`NB_zerocritic` 이 그 셀이다. ⇒ **DUP 을 언젠가 켠다면 `privileged_extra` 와 반드시 같이,
그리고 `mirror_privileged_obs` 를 먼저 쓰고 나서다**(§A-5 ③).

## A-5. ⛔ 현재 구현의 결함 셋 — 그리고 패치

### ① DUP 은 **켤 수가 없다** (설계상)

`mirror_augmentation_coef` 가 `Goal_Pose_V7.yaml` 에 없다. 그리고
[`make_v7_arms.py:984-991`](htwk-gym/tools/make_v7_arms.py) 의 `set_dotted` 가
**base config 에 없는 경로를 KeyError 로 거부**한다(오타가 죽은 config 를 만드는 것을
막는 장치다). ⇒ arm 정의만으로는 절대 못 켠다. **이건 결함이 아니라 안전장치이고,
이 의견서는 그것을 그대로 두기를 권한다.**

### ② ⛔ `mirror_privileged_obs` 가 **없다** — DUP 을 켜면 조용히 반쪽이 된다

```python
# utils/runner_v3.py:256
if use_mirror_aug and hasattr(self.env, "mirror_privileged_obs"):
```

전 저장소 grep: 이 메서드는 **`archive/envs/K1/goal_pose_hbatch.py:328` 에만 있다.**
현재 `GoalPoseV3`/`V7` 에는 없다. `hasattr` 가드라 **예외 없이 그냥 건너뛴다.**

⇒ 지금 누가 `mirror_augmentation_coef` 를 켜면 **actor 만 미러 PPO 를 받고 critic 은
원본 그대로** 학습된다. 조용한 반쪽 처치이고, 실패해도 원인을 못 찾는다.

**권고: DUP 을 켜지 않는다. 만약 켜야 한다면 `mirror_privileged_obs` 를 먼저 구현한다**
(privileged 14 = base_com/mass 4 + `base_lin_vel` 3 + height 1 + force 3 + torque 3;
필요한 부호: `com_y` latent `u → 1−u`, `lin_vel_y` −1, `force_y` −1, 토크는 축벡터라
`Tx`·`Tz` −1 / `Ty` +1. `privileged_extra` 를 켜면 접촉 2채널 교환 + 영점 12채널에
`mirror_act_perm`·`sign` 적용).

### ③ ⛔ `logstd` 의 좌우 쌍이 안 묶인다 (2026-08-01 지적, 미수정)

[`utils/model.py:27`](htwk-gym/utils/model.py):
```python
self.logstd = torch.nn.parameter.Parameter(torch.full((1, num_act), -2.0))
```
**상태무관 12차 파라미터**다. 대칭손실은 `.loc` 만 묶으므로 `π(M s)` 와 `M π(s)` 가
**평균은 같고 분산은 다른** 분포가 될 수 있다. 즉 지금 재는 "deterministic mirror error"
가 0이어도 **확률정책은 비대칭**일 수 있다.

**패치 (2줄, `runner_v3.py:320` 뒤)** — `logstd` 는 스케일이라 부호는 무의미, 순열만 적용:
```python
if use_symmetry:
    sym_loss = F.mse_loss(mirrored_dist.loc, self.env.mirror_actions(dist.loc))
    # 상태무관 logstd 의 L/R 쌍도 묶는다. 안 묶으면 평균만 대칭이고 분포는 비대칭이다.
    sym_loss = sym_loss + F.mse_loss(
        self.model.logstd, self.model.logstd[..., self.env.mirror_act_perm])
```
비용 0, 기존 arm 에 대한 영향은 **`|logσ_L − logσ_R|` 를 0으로 미는 것**뿐이다.
⚠️ 엄밀히는 no-op 이 아니므로 **새 arm 에서만** 켜라 (`symmetry_logstd: true` 플래그 권장).

### ④ 지표가 없다 — 손실은 켜져 있는데 **효과를 측정한 적이 없다**

`stats["symmetry_loss"]` 는 기록되지만, 우리가 실제로 원하는 것은
**정책 등변성 오차**(`||π(Ms) − Mπ(s)||` 의 p90)와 **좌우 touchdown bias** 다.
`eval_goal_pose.py:2687` 에 `symmetry_eval` 이 있고 `round_table.py` 는 그것을 안 찍는다.

**권고: `round_table.py` 에 `mirror p90` 열 한 개를 추가한다.** H1 의 기준(0.10)과
warm-start 값(0.080)이 이미 있으므로 **비교 가능한 축이 공짜로 생긴다.**

## A-6. A 권고 — 무엇을 하고 무엇을 하지 마라

| | 권고 | 근거 |
|---|---|---|
| ✅ | **`symmetry_coef: 0.5` 를 그대로 둔다** | 문헌상 LOSS 가 가장 일관됨(§A-3); 우리 맵은 96/96 검증됨; 중립상태·정규화 함정 둘 다 해당 없음(§A-1) |
| ⛔ | **`mirror_augmentation_coef` 를 켜지 않는다** | DUP 이 LOSS 보다 낫다는 근거 없음 + 우리 구현에 `mirror_privileged_obs` 부재·critic 50:50·KL 스케줄 오염 세 결함(§A-5) |
| ✅ | **`logstd` 대칭항을 새 arm 에서 켠다**(`symmetry_logstd`) | §A-5 ③, 비용 0 |
| ✅ | **`round_table.py` 에 mirror p90 열 추가** | 손실을 켜 놓고 효과를 한 번도 안 쟀다(§A-5 ④) |
| ⛔ | **G10 을 "대칭은 역효과" 로 인용하지 않는다** | H1 은 5레버 묶음이고 주원인(편향 mirror PPO)은 이미 수정됨(§A-2). 재인용하려면 RETRACTIONS 규칙대로 "왜 이번엔 다른지"를 먼저 써야 한다 |
| ⚠️ | 계수를 바꿔야 한다면 **0.5 → 0.25 가 아니라 0.5 유지** | 원논문 *"w 선택에 민감하지 않다"*; 그리고 계수를 낮춰도 (증강을 켠 경우) critic 압력은 안 준다(§A-2 ③) |

**한 문장**: *대칭은 지금 형태(LOSS 0.5)가 맞다. 손댈 곳은 계수가 아니라 **지표와 logstd** 다.*

---

# B. `joint_encoder_bias` — "무조건 하는 게 좋은 게 아니었어?"

## B-1. 정면 답: **아니다. 조건부다.**

정확히 말하면 **세 갈래로 나눠야** 한다.

| 무엇 | 판정 |
|---|---|
| **① 실기의 영점 드리프트를 모델링하는 것** | **거의 무조건 좋다.** 사용자가 "꽤나 자주" 발생한다고 보고했고, 실측 위험 순위 1위다(`ibatch.md:2730`). 정책은 지금 **1도도 본 적이 없다** |
| **② 그것을 `joint_zero` 블록으로 켜는 것** | **맞는 방법.** 상관구조가 실기와 정확히 등가다(§B-2) |
| **③ 레거시 `joint_encoder_bias`/`joint_target_offset` 두 키로 켜는 것** | ⛔ **하지 마라.** 그건 영점 드리프트가 아니라 **다른 고장**이다 |

즉 사용자의 직관("무조건 좋다")은 **①에 대해서는 맞고**, 지금까지 우리가 실제로 한 것
(③)에 대해서는 **틀렸다**. 그 구분이 §8-46 의 발견이다.

### 왜 ③이 다른 고장인가 (코드로 확인)

```
실기:   측정 q_meas = q + b
        온보드 PD  tau = kp*(target − q_meas) = kp*(target − q − b)
        ⇒ 평형    q_eq = target − b
Isaac:  관측      dof_pos + joint_encoder_bias − default      (goal_pose.py:1083)
        목표      default + scale*a + joint_target_offset     (goal_pose.py:847-848)
        PD        kp*(applied − dof_pos)                      (goal_pose.py:864)
        ⇒ 평형    q_eq = default + scale*a + target_offset
```
**둘이 같아지려면 `encoder_bias = +b` **이면서** `target_offset = −b`.** 근사가 아니라 정확한 등가.
그런데 레거시 두 키는 [`goal_pose.py:399-406`](htwk-gym/envs/K1/goal_pose.py) 에서
**독립적으로** 뽑힌다. ⇒ `I3a_jointcal3`·`M8_jointcal2` 가 학습한 것은
*"엔코더가 X, 구동기가 무관한 Y 만큼 틀어진 두 개의 고장"* 이다.

`joint_zero` 는 [`:561-562`](htwk-gym/envs/K1/goal_pose.py) 에서 `+b / −b` 를 지킨다. ✅

**질적 차이가 실기에서 확인된다.** 상관형(옳은)에서는 **관측 좌표계에서 관절 루프가
완전히 정상**이고 깨지는 것은 참 기구학뿐이다. `--hold-diag 15` 실측
([HANDOFF_DEPLOY_ENTRY §1](HANDOFF_DEPLOY_ENTRY_20260807.md)):

| 측정 | 값 |
|---|---|
| 전 관절 추종오차 | max **0.6°**, rms 0.27° — **매우 정상** |
| 몸통 tilt | **4.9 ~ 13.8°, 15초 내내** |

**"관절은 완벽한데 몸이 기울어 있다"** = 상관형 영점 오차의 signature 그 자체다.
비상관형이면 관절 추종오차 자체가 커야 한다.

## B-2. 켰을 때의 **대가** — 데이터

### ① `I3a_jointcal3` (±3°, 레거시 비상관 방식) — `ibatch.md:2706-2716`

| | 과제오차 | strict | 낙상 | 2 cm 미만 스윙 |
|---|---:|---:|---:|---:|
| `I2a_dr` (drift **OFF**) | 4.15 cm | 94.9 % | 2 / 4,651 | **0.2 %** |
| `jointcal3` (drift **ON ±3°**) | 5.59 cm | 83.9 % | **12 / 4,641** | **6.4 %** |

**낙상 0.43 → 2.59/1000 = 6.0배 (단측 p = 0.0064). 클리어런스 2 cm 미만 32배.**

⚠️ **원본 자신이 단서를 단다**: 이 비교에는 *정책 차이*와 *시험 조건 차이*가 섞여 있다
(2×2 의 나머지 두 칸이 없다). 그리고 이 arm 은 §B-1 ③의 **비상관** 방식이라
같은 각도라도 상관형보다 훨씬 거칠다.

### ② `NZ_zeroiid` (상관형 + 커리큘럼, 방금 채점됨)

**이것이 §① 2×2 의 빠진 칸 하나를 실제로 채운다** — "영점을 넣고 학습, 영점 없이 시험":

| | 정확도 | 낙상 | 보행 속도(final) | 보행 낙상(final) |
|---|---:|---:|---:|---:|
| `NZ_zeroiid` | 10.43 cm (it1700) | **0 / 4,613** | 1.45 m/s | **0** |

⇒ **낙상 축에서 대가가 0이다.** 오히려 이 저장소 최저다.
⇒ **정확도 축에서는 대가가 있어 보인다**(10.43 cm) — 그러나 계보 B 라
**`NE_ctrl100` 이 나와야 확정된다**(§1-4).

### ③ 그리고 DR 은 공짜가 아니다 — 일반 원리

넓히면 보수화된다. 우리 저장소의 실증:
- `MB_obsdelay`: 지연만 주고 식별 수단을 안 줬더니 **1.0 m/s 를 0.0 % 넘겼다**
  ([HANDOFF_TRAINING §3](HANDOFF_TRAINING_20260807.md))
- `_ZERO_ON` 주석 자신이 인용하는 Margolis+ RSS2022: *"커리큘럼 없이는 느리게도 못 걷는다"*

**⇒ 정량화 시도**: 지금 있는 세 점으로 "폭 대 대가" 를 그리면

| 폭 | 방식 | 낙상률 변화 | 정확도 변화 |
|---|---|---|---|
| 0 | — | 기준 | 기준 |
| ±3° 고정 | 비상관 | **×6.0** | +1.44 cm |
| ±10° 상한 + per-env 커리큘럼 | 상관 | **×0 (낙상 0)** | +?(대조군 대기) |

**해석 (추론)**: 낙상 폭증을 만든 것은 *폭* 이 아니라 **①비상관 모델 + ②고정 폭(커리큘럼 없음)**
쪽일 가능성이 크다. 상관형+커리큘럼은 같은 상한(더 넓은 ±10°)에서도 낙상 0이다.
⚠️ 단 §B-4 때문에 **NZ 가 실제로 몇 도에서 학습했는지 우리는 모른다.** 그 값 없이는
이 표의 세 번째 행이 "±10°" 라고 말할 수 없다.

## B-3. ⛔ 결함 1 — `NZ_zeroiid` 는 **영점을 꺼놓고** 채점됐다

```
학습 config:  logs/K1/K1/Goal_Pose_V7/2026-08-07-16-48-06_NZ_zeroiid/config.yaml:466
              joint_zero: { enabled: true, ... }
평가 config:  logs/eval_rounds/n2na/2026-08-07-16-48-06_NZ_zeroiid.cfg.yaml:509
              joint_zero: { enabled: false, ... }
              joint_encoder_bias.range: [0.0, 0.0]              (:496-499)
```

원인은 [`tools/make_eval_cfg.py`](htwk-gym/tools/make_eval_cfg.py) 의 설계다 — 공통
프로토콜에 **관측 인터페이스만** 이식하고 `randomization.*` 는 공통값을 쓴다. 그 설계는
옳다(arm 마다 다른 시험지를 주면 비교가 아니다). **문제는 그것이 전부라는 것이다.**

⇒ 지금 `NZ`(그리고 앞으로 `N9_zerostruct`·`NB_zerocritic`)의 채점이 답하는 질문은
**"영점 랜덤화를 하고 나면 깨끗한 로봇에서 얼마나 손해인가"** 하나뿐이다.
**"영점이 틀어진 로봇에서 얼마나 이득인가"** 는 안 재진다.
이것은 `NC_actfilter` 에 붙어 있는 경고와 **정확히 같은 결함**이고,
`hbatch-codex.md:481` 이 2026-08-01 에 *"held-out joint-offset severity grid 를 공통 eval 에
둔다"* 라고 처방한 바로 그 항목이다. **여섯 달 전 처방이 아직 미착지다.**

### 수정 — **강건성 레인** (설계 완료, 실행만 남음)

`eval_round.sh` 에 3단계로 추가한다. 공통 프로토콜을 **한 벌 더** 만들되
`joint_zero` 만 켜고 **커리큘럼은 끈다**(고정 심각도):

```yaml
# sweeps/N0_ctrl_zero{1,3,5}.yaml — N0_ctrl.yaml 복사 + 아래 블록만 교체
randomization:
  joint_zero:
    enabled: true
    max_deg: {1.0 | 3.0 | 5.0}
    curriculum: false          # ⛔ 반드시 끈다. 켜면 arm 마다 다른 난이도로 채점된다
    init_level: 1.0            # level 고정 = max_deg 그대로
    min_level: 1.0
    step: 0.0
    modes: {iid: 0.2, single: 0.3, leg_common: 0.2, anti_mirror: 0.2, mirror: 0.1}
    joint_weight: [1,1,1,1,1,1, 1,1,1,1,1,1]   # ⛔ 평가에서는 깎지 않는다(§C-7 ③)
```

**사전 고정 판정 기준** (숫자를 보기 전에 못 박는다):

| 관측 | 결론 |
|---|---|
| `NZ` 가 `NE_ctrl100` 대비 **±3° 레인에서 낙상률 절반 이하** | 영점 DR 이 이득이다. 유지 |
| `NZ` 가 ±3° 에서 `NE` 와 **낙상률 차이 없음** | 커리큘럼이 낮은 수준에 머물렀거나 레버가 무효. §B-4 로그를 먼저 본다 |
| `NZ` 가 ±0° 레인에서 정확도 손해 > 5 cm **이면서** ±3° 이득 없음 | **끈다** |

비용: arm 당 3레인 × 120 s × 256 env ≈ 기존 정확도 레인 3배. GPU 소.

## B-4. ⛔ 결함 2 — **커리큘럼 수준이 어디에도 기록되지 않는다**

[`goal_pose.py:473-483`](htwk-gym/envs/K1/goal_pose.py) 의 `_zero_level` 은 env 마다
낙상하면 −0.05, 살아남으면 +0.05 로 움직인다. **전 저장소 grep 결과 이 값을 읽는 곳은
`tools/test_joint_zero.py`(단위시험) 하나뿐**이다. 학습 로그·리포트·리코더 어디에도 없다.

⇒ **`NZ_zeroiid` 가 6000 iteration 동안 실제로 몇 도의 영점 오차를 겪었는지 아무도 모른다.**
"영점에 강건하다" 가 **반증 불가능한 문장**이 된다.

**산술로 추정 (추론, 검증 필요)**: 에피소드 30 s, step 0.05, init 0.1
→ 성공 18 에피소드면 1.0 포화. 학습 총량은 env 당 약 2,880 s = 96 에피소드.
NZ 는 낙상이 거의 없었으므로 **초반에 1.0 으로 포화했을 가능성이 높다**
(= `max_deg 10° × joint_weight`). 그렇다면 §B-2 표의 세 번째 행이 진짜로 "±10°" 이고,
**±10° 상관형 + 커리큘럼에서 낙상 0** 이라는 매우 강한 결과가 된다.
**그러나 지금은 증거가 없다.**

**패치 (4줄)** — `_resample_joint_zero` 끝에:
```python
# 커리큘럼 수준을 리포트로 내보낸다. 이 값이 없으면 "영점에 강건하다"가 반증 불가능해진다.
self.extras.setdefault("zero_level", {})
lv = self._zero_level
self.extras["zero_level"] = {
    "mean": float(lv.mean()), "p10": float(lv.quantile(0.1)),
    "p90": float(lv.quantile(0.9)), "max_deg": float(z.get("max_deg", 10.0))}
```
그리고 `runner_v3.py` 의 `ep_info` 에 얹으면 recorder 가 자동으로 찍는다.
⚠️ **완전 no-op** — `joint_zero.enabled: false` 인 arm 은 이 함수에 들어오지도 않는다.

## B-5. 문헌의 폭 대 우리 폭

| 출처 | 랜덤화 대상 | 범위 |
|---|---|---|
| Gear-Driven Humanoid (arXiv:2504.00614) | joint position **encoder offset** | ±0.01 rad = **±0.57°** |
| Duke Humanoid (arXiv:2409.19795) | joint offset | ±0.02 rad = **±1.15°** |
| ECHO (arXiv:2603.16188) | joint offset error | ±0.01 rad = **±0.57°** |
| BeamDojo (arXiv:2502.10363) | actuator offset | ±0.05 rad = **±2.9°** |
| **우리 `joint_zero.max_deg`** | 상관형 영점 | **±10° = ±0.175 rad** |

**우리 상한이 문헌 최대의 3.5배, 최소의 17배다.**

**그런데 이것이 곧 "너무 넓다" 는 아니다.** 세 가지 차이가 있다:

1. **우리는 per-env 커리큘럼이 있다.** 문헌 값들은 전부 **고정 폭**이다.
   `max_deg` 는 상한이지 실제 학습 폭이 아니다 — 그래서 §B-4 로그가 필요하다.
2. **우리 로봇의 캘리브레이션 절차가 문헌 로봇들보다 거칠다.** 사용자 절차는
   핀·지그 없이 **엉덩이 받쳐놓고 눈대중**이다(§C-1). 문헌 값들은 공장 캘리브레이션
   잔차를 상정한 값이다.
3. **실기가 준 브래킷이 문헌보다 크다.** `--hold-diag` 잔차 tilt **2.0 ~ 10.9°**(§C-1).

> **권고 (§B-6 로 이어짐)**: `max_deg 10` 은 **사슬 모드에는 적절하고 `iid` 에는 너무 넓다.**
> 모드별로 상한을 나눠야 한다 — 사슬 오차는 *사슬 합*이 10° 여도 관절당 3.3° 이지만,
> `iid` 10° 는 관절당 10° 다.

## B-6. B 권고

| | 권고 | 근거 |
|---|---|---|
| ✅ | **켜라. 단 `joint_zero` 블록으로만.** 레거시 두 키는 영구히 `[0,0]` 으로 둔다 | §B-1 |
| ✅ | **커리큘럼은 필수다**(레버가 아니라 전제). `curriculum: true` 유지 | ±3° 고정에서 낙상 6배(§B-2 ①), Margolis 인용 |
| 🔴 | **§B-4 로깅 패치를 다음 arm 전에 넣어라.** 이게 없으면 이 축 전체가 측정 불가다 | §B-4 |
| 🔴 | **§B-3 강건성 레인을 만들어라.** 없으면 arm 의 존재 이유가 안 재진다 | §B-3 |
| ✅ | 모드별 상한 분리: `iid`/`single` 은 ±3°, 사슬 모드는 ±10° | §B-5, §C-7 |
| ✅ | **같이 켤 것: `privileged_extra`(=`NB_zerocritic`).** 영점 12차원이 비평자에게 가장 큰 구멍이다 | `goal_pose.py:1187-1192` |
| ⚠️ | **같이 켜면 안 되는 것: 다른 새 레버.** H1 이 5레버를 동시에 켜서 아무것도 못 배웠다 | §A-2 |
| ⛔ | MuJoCo 스윕의 ±3° 절벽을 `max_deg` 근거로 쓰지 마라 | §C-4 (부호 규약이 다르다) |

**한 문장**: *"무조건 좋다" 는 ①실기 고장을 모델링한다 ②커리큘럼을 쓴다
③그 효과를 재는 레인이 있다 — 셋이 갖춰졌을 때만 참이다. 지금은 ②만 있다.*

---

# C. `pitch_chain` 모드 — 이게 뭐고, 하면 나아지는가

## C-1. 사용자 절차가 만드는 δ 의 상관구조 (물리 유도)

**사용자 절차**: 핀 없이 **엉덩이를 받쳐놓고**, **상체를 눈대중으로 세우고**, **중력으로**
다리를 늘어뜨린 자세를 영점으로 정의. 합격 기준은 stand 모드에서 안 넘어짐.

영점을 그 자세에서 정의하면, 관절 `j` 의 오차는
`b_j = −(그 자세의 참 관절각 − URDF 영점 자세의 각)` 이다. 이것을 원인별로 분해한다.

### (1) pitch 사슬 — 상체를 눈대중으로 세운 오차 δ_p

평지에서 발이 바닥에 평평하고 몸통이 수직이려면
([HANDOFF_DEPLOY_ENTRY §1](HANDOFF_DEPLOY_ENTRY_20260807.md) 가 이미 유도했다):

```
hip_pitch + knee_pitch + ankle_pitch = 0
```

상체를 δ_p 만큼 잘못 세운 채 영점을 잡으면, **정책이 "0" 이라고 믿는 자세의
사슬 합이 실제로는 δ_p** 다. 그리고 세 pitch 관절축은 **좌우 모두 `axis="0 1 0"`**
(URDF·MJCF 직접 확인)이므로 **좌우가 같은 관절좌표 부호**로 들어간다.

⇒ **모양: `mirror`(공통모드) × pitch 3관절 × "사슬 합이 δ_p"**

**실기 브래킷 (사실)**: `--hold-diag 15` 에서 tilt **4.9 ~ 13.8°** 인데
명령 자세의 사슬 합이 **−0.05 rad = 2.9°** 다. ⇒ **미설명 잔차 2.0 ~ 10.9°.**
그 잔차의 가장 유력한 후보가 δ_p 다(다른 후보: IMU 장착 bias).

### (2) roll 사슬 — 사용자 정정 ②가 옳다

*"상체를 눈대중으로 세우는 것은 **roll 도** 영향이 있다. 두 다리로 정해지는 body roll 도 있다"*

맞다. 그리고 pitch 와 **같은 구조**다:

```
hip_roll + ankle_roll = 0        (다리마다)
```

`Hip_Roll`·`Ankle_Roll` 축이 **좌우 모두 `axis="1 0 0"`** 이므로, 몸통 roll 오차 δ_r 도
**좌우 같은 관절좌표 부호**로 들어간다.

⇒ **모양: `mirror`(공통모드) × roll 2관절 × "사슬 합이 δ_r"**

> ⚠️ 헷갈리기 쉬운 점: **몸통 roll 은 물리적으로 반대칭 양**(미러하면 부호가 뒤집힌다)인데,
> **관절좌표로는 좌우 같은 부호**다. Roll 관절의 미러 부호가 −1 이기 때문이다.
> 즉 *"물리적 대칭성"* 과 *"관절좌표 부호 일치"* 는 다른 말이고, 지금 `joint_zero` 의
> 모드 이름은 **관절좌표 기준**이다. §C-3 의 오해가 여기서 나왔다.

### (3) 좌우 비대칭 — 사용자 정정 ①이 옳다

*"중력으로 늘어지는 관절은 재현된다는 것은 틀렸다 — 앉힌 엉덩이 각도, 모터 마찰,
backdrivability 때문에 항시 재현되지 않는다"*

맞다. 늘어지는 각도는 각 관절의 **마찰 데드밴드** 안 어디든 될 수 있고,
좌우 관절의 마찰은 독립이다. ⇒ **`iid`(관절 독립) + `leg_common`(다리 전체)** 성분.
그리고 "앉힌 엉덩이 각도"는 좌우 공통이므로 **Hip_Pitch 에 몰린 `mirror` 성분**을
하나 더 만든다.

### 정리 — 이 절차가 만드는 분포

| 성분 | 원인 | joint_zero 모드 | 걸리는 관절 |
|---|---|---|---|
| δ_p (pitch 사슬 합) | 상체 눈대중 + 앉힌 엉덩이 각도 | **`mirror`** | Hip_P / Knee / Ankle_P (좌우) |
| δ_r (roll 사슬 합) | 상체 눈대중(roll) | **`mirror`** | Hip_R / Ankle_R (좌우) |
| 관절별 산포 | 마찰·backdrivability | `iid` | 전부 |
| 다리 단위 산포 | 한쪽 프롭 기하 차이 | `leg_common` | 다리 6관절 |
| 스탠스 폭 오차 | 좌우가 반대로 틀어짐 | `anti_mirror` | 주로 Hip_R |

## C-2. 현재 5모드가 덮는 것과 빈 곳

[`goal_pose.py:443`](htwk-gym/envs/K1/goal_pose.py) `_ZERO_MODES = (iid, single, leg_common, anti_mirror, mirror)`.
`_ZERO_STRUCT` 가중치는 `iid 0.20 / single 0.30 / leg_common 0.20 / anti_mirror 0.20 / mirror 0.10`.

| 성분 | 덮는가 | 어떻게/왜 아닌가 |
|---|---|---|
| 관절별 산포 | ✅ | `iid` |
| 다리 단위 | ✅ | `leg_common` |
| 스탠스 폭 | ✅ | `anti_mirror` |
| 단일 관절 손상 | ✅ | `single` |
| **δ_p 사슬** | ⚠️ **부분** | `mirror` 가 좌우는 묶지만 **한 다리 안의 3 pitch 관절은 독립**이다. 사슬 합이 큰데 관절값은 작은 조합(δ/3 씩)이 거의 안 뽑힌다. 그리고 가중치가 **0.10 으로 최저** |
| **δ_r 사슬** | ⚠️ **부분** | 같은 이유. 게다가 `joint_weight` 가 Hip_Roll 을 **0.5 로 깎는다** |

### 왜 "사슬 합" 이 따로 중요한가 (핵심 논거)

- `iid` 로 관절당 진폭 `a` 를 뽑으면 사슬 합의 표준편차는 약 `a` 다 — 즉 사슬 합이
  **큰 경우도 나오긴 한다.** 그러나 그때는 **관절값도 같이 크다.**
- 실기의 절차가 만드는 것은 **관절값은 작은데 사슬 합만 큰** 조합이다
  (눈대중 3° 오차가 세 관절에 1°씩 나뉜다).
- 그리고 **open-loop 로 설 수 있느냐를 정하는 것은 사슬 합**이다:
  tilt 13.8° 에서 CoM 이 앞코까지 여유의 **96 %** 를 먹는다(같은 문서 §1).
- ⇒ 지금 분포는 **실기에서 가장 잘 일어나면서 가장 결정적인 조합을 거의 안 뽑는다.**

## C-3. ⛔ `anti_mirror = 0.20` 판정 — **근거 실험은 `mirror` 모드였다**

`Goal_Pose_V7.yaml` 과 `goal_pose.py:453-456` 둘 다 이렇게 쓴다:

> `anti_mirror 0.20  # 좌우 반대 ← 실기 증상을 재현한 모양(ibatch 8-43)`
> *"Hip_Roll 에 −5° 를 넣으면 실기의 '다리가 모이며 부딪힘' 이 재현됐고 +5° 는 낙상 0"*

[AUDIT §6-6](AUDIT_20260807_2300.md) 이 **§8-43 에 그 실험이 없다**(유령 인용)고 했고,
[AUDIT §7-1](AUDIT_20260807_2300.md) 이 실체를 `logs/mujoco/sig/hiprbias±5` 에서 찾았다.

**2026-08-08 01:27(`8d763fb`)에 병렬 세션이 `ibatch.md:3893-3920` 을 정정했다.** 원 숫자
전부(base 0 / **−3° 0** / **+3° 0** / **−5° 6** / +5° 0, 각 20 s, **n=1**, `--goal-hold` 는
결정론적이라 seed 를 늘려도 표본이 안 는다)와 *"⇒ `anti_mirror` 비중 0.20 은 이 데이터로
정당화되지 않는다"* 까지 이미 적혀 있다. **여기까지는 이 문서의 새 결론이 아니다.**

**새 결론은 그 다음이다 — 그 실험이 어느 모드인지.** 코드를 직접 열었다:

```python
# htwk-gym/play_mujoco_goalpose.py:400-403
if args.hiproll_bias_deg != 0.0:
    b = math.radians(args.hiproll_bias_deg)
    joint_bias[11] = b           # Left_Hip_Roll
    joint_bias[17] = b           # Right_Hip_Roll
```

**좌우에 같은 관절좌표 부호로 같은 값을 넣는다.**
[`goal_pose.py:551-558`](htwk-gym/envs/K1/goal_pose.py) 의 정의로는:

```python
for mode_id, sign in ((3, -1.0), (4, 1.0)):   # 3 = anti_mirror, 4 = mirror
    pair = torch.cat((s, sign * s), dim=-1)
```
- `anti_mirror`(3): `b_R = −b_L`
- `mirror`(4): `b_R = +b_L`  ← **`--hiproll-bias-deg` 가 하는 일이 정확히 이것**

> ## ⛔ 판정
> **`anti_mirror = 0.20` 의 유일한 근거로 인용된 실험은 `mirror` 모드를 잰 것이다.**
> 그리고 `_ZERO_STRUCT` 는 `mirror` 에 **다섯 모드 중 최저인 0.10** 을 준다.
> **인용은 실체가 있었지만 가리키는 모드가 반대였다.**
>
> `8d763fb` 의 정정은 *"한 관절만 흔들었으니 12관절 좌우반대로의 외삽이다"* 까지 갔다.
> **한 걸음 더 있다 — 외삽의 방향 자체가 틀렸다.** 그 한 관절은 좌우 **같은 부호**였고,
> 12관절로 늘리면 그것은 `anti_mirror` 가 아니라 `mirror` 가 된다.

그리고 §C-1 이 독립적으로 같은 결론에 도달한다 — 사용자 절차가 만드는 것도 `mirror` 다.
**두 경로가 만나므로 이것은 우연이 아니다.**

⚠️ **`anti_mirror` 를 버리라는 뜻은 아니다.** 실기 증언 *"다리가 모이면서 발끼리 부딪혀서
넘어져"* 는 **좌우가 반대로 모이는** 모양이고 그건 `anti_mirror` 다. 즉
**두 모드 모두 근거가 있는데, 근거가 서로 바뀌어 인용돼 있었다**:

| 모드 | 물리적 의미 | 실제 근거 |
|---|---|---|
| `mirror` | 몸통 pitch/roll 영점 오차 (공통모드) | ⭐ `sig/hiprbias±5`(−5° 6낙상 / +5° 0낙상) + 사용자 캘리브레이션 절차 + `--hold-diag` 잔차 tilt |
| `anti_mirror` | 스탠스 폭 / 다리 모임 | 실기 증언(발끼리 부딪힘) + 실기 로그 좌우 발 겹침 9.9 % |

## C-4. ⛔ 그 MuJoCo 프로브의 **부호 규약이 실기와 반대다**

같은 파일에서 `joint_bias` 가 쓰이는 곳 **두 군데 전부**:

```python
# play_mujoco_goalpose.py:477   관측
targets = policy.inference(t, (q + joint_bias).astype(np.float32), ...)
# play_mujoco_goalpose.py:506   PD
tau = kp * ((cmd_q + joint_bias) - q) - kd * dq
```

`q` 는 MuJoCo 의 **참** 관절각이다. 그러면

| | 관측 | PD 평형 | 관측 좌표계에서의 정상상태 오차 |
|---|---|---|---|
| **실기 / Isaac** | `q + b` | `q = cmd − b` | **0** (관절 루프는 완전 정상) |
| **MuJoCo 프로브** | `q + b` | `q = cmd + b` | **2b** (5°면 10° 추종오차) |

⇒ **MuJoCo 프로브는 `encoder = +b` 와 `target = +b` 를 같이 넣는다.**
그것은 §8-46 이 *"영점 드리프트가 아니다"* 라고 판정한 바로 그 **비상관(부호가 어긋난)**
고장이다. 실기·Isaac 의 상관형과 **질적으로 다른 고장**이다.

**교차검증 (실기 데이터)**: 만약 실기 고장이 MuJoCo 프로브형이라면 `--hold-diag` 에서
**관절 추종오차가 2b** 로 보여야 한다. 실측은 **max 0.6°** 였다.
⇒ **실기는 상관형(Isaac 형)이지 MuJoCo 프로브형이 아니다.**

> ### 파생 결론 셋
> 1. ⛔ **`I3a_jointcal3` 의 "±3° 절벽" 과 `hiprbias±5` 의 임계값을
>    `joint_zero.max_deg` 근거로 쓰면 안 된다.** 다른 고장의 임계값이다.
>    (`make_v7_arms.py:601-612` 의 `M8_jointcal2` 설계 주석이 그렇게 쓰고 있다.)
> 2. ✅ **Isaac 의 `joint_zero` 구현은 맞다.** 고칠 것 없다.
> 3. ⚠️ **MuJoCo 프로브는 고쳐야 한다** (1줄): `:506` 을
>    `tau = kp * ((cmd_q - joint_bias) - q) - kd * dq`.
>    ⛔ **다만 이것은 기존 `logs/mujoco/sig/*` 전부를 무효화하는 변경이다.**
>    Codex 쪽 자산이므로 **학습 세션이 단독으로 고치지 말고 조율할 것.**
>    당장은 **그 로그를 영점 근거로 인용하지 않는 것**으로 충분하다.

## C-5. ⛔ 그리고 R4(실기 영점 실측)는 **이 모드에 눈이 멀었다**

[`tools/probe_joint_zero.py`](htwk-gym/tools/probe_joint_zero.py) 헤더가 스스로 적는다:

> *"⚠️ 한계: 좌우가 같은 방향으로 같이 틀어진 공통(common-mode) 오차는 안 잡힌다.
> 이 측정은 **하한**이다."*

**§C-1 이 유도한 것이 정확히 공통모드다.** ⇒ 지금 상태로 R4 를 돌리면
**사용자 절차가 가장 잘 만드는 오차를 구조적으로 못 잰다.**
이것은 [AUDIT §6-4](AUDIT_20260807_2300.md)("한 번도 안 돈 진단 도구") 의 항목인데,
돌려도 안 닫힌다는 사실은 아무도 안 적었다.

### 닫는 법 — **로봇 시간 추가 0초, 코드 ~15줄**

그 프로브의 콜백은 이미 `imu_state.rpy` 를 같이 모은다(`:29-30`).
**발이 바닥에 평평한 정지 자세**에서:

```
참 기구학:   body_pitch = −(hip_P + knee + ankle_P)_true
엔코더:      q_meas = q_true + b
⇒  Σb_pitch = (hip_P + knee + ankle_P)_meas + imu_pitch        ← 다리마다
⇒  Σb_roll,L = (hip_R + ankle_R)_meas,L    + imu_roll          ← 다리마다
```

**우변이 전부 이미 수집되는 값이다.** 좌우 평균이 공통모드, 좌우 차가 기존 대칭 지표.

```python
# probe_joint_zero.py 끝에 추가 (읽기 전용, 로봇 시간 0초)
LEG = {"L": (0, 1, 2, 3, 4, 5), "R": (6, 7, 8, 9, 10, 11)}   # 인덱스는 실제 배열에 맞춰 확인할 것
for side, (hp, hr, hy, kn, ap, ar) in LEG.items():
    chain_p = (q[hp] - DEFAULT[0]) + (q[kn] - DEFAULT[3]) + (q[ap] - DEFAULT[4])
    chain_r = (q[hr] - DEFAULT[1]) + (q[ar] - DEFAULT[5])
    print("%s: pitch 사슬 영점합 = %+.2f deg   roll 사슬 영점합 = %+.2f deg"
          % (side, math.degrees(chain_p) + rpy[1], math.degrees(chain_r) + rpy[0]))
print("공통모드(좌우 평균)가 probe 의 기존 대칭 지표가 못 보는 성분이다.")
```

⚠️ **정직한 한계**: 이 값은 `Σb + (IMU 장착 bias)` 의 **합**이다. 둘을 못 가른다.
**그러나 그것이 오히려 옳은 표적이다** — `b`~`r` 구간에서 로봇을 넘어뜨리는 것은
*"명령한 자세가 참 수직에서 얼마나 벗어나 있는가"* 이고, 그것이 정확히 이 합이다.
[HANDOFF_DEPLOY_ENTRY](HANDOFF_DEPLOY_ENTRY_20260807.md) 의 판정표(2° 안쪽 / 그대로)에
바로 꽂을 수 있는 수다.

## C-5b. ⭐ 배포 쪽이 **지금** 영점 추정기를 만들고 있다 — 이게 B·C 의 권고를 바꾼다

이 의견서를 쓰는 중에 저장소에 미커밋 파일 다섯 개가 나타났다(2026-08-08):

```
htwk-gym/tools/kinematics_k1.py           K1 다리 FK + 선형대수 (stdlib only)
htwk-gym/tools/estimate_joint_zero.py     12-자세 delta 추정기 (관측성 분석 포함)
htwk-gym/tools/collect_joint_zero.py      로봇에서 자세 수집 (안전 절차 포함)
htwk-gym/deploy/utils/joint_zero.py       배포 경계에서 delta 보정
htwk-gym/deploy/tests/test_joint_zero_apply.py
```

**전문을 읽었다.** 요약과, 그것이 학습 쪽 판단을 어떻게 바꾸는지:

### 무엇을 하는가

바닥 평면을 기준으로 쓴다(Yamane US8805584 / iCub 계열의 double-support self-calibration).
자세 12개를 잡고 자세마다 잔차 9개(양 발바닥이 중력에 수직 / 두 발이 같은 평면 /
스탠스 폭·각 불변). 미지수는 `delta` 12 + nuisance 2. **부호 규약이 학습과 정확히 같다**
(`q_meas = q_true + delta`, 그리고 보정은 **읽기 `−delta` / 쓰기 `+delta` 둘 다**).

### ⭐ 그 문서가 §C-1 과 §C-5 를 독립적으로 확증한다

| 이 의견서 | 추정기 문서 |
|---|---|
| §C-1 "pitch 사슬은 `hip_P + knee + ankle_P` 합으로만 나타난다" | *"the sole's pitch is exactly (q_hip_pitch + q_knee + q_ankle_pitch) and **only that sum is visible**"* — 발 자세만 쓰면 σ_min = **0**(특이) |
| §C-5 "IMU bias 와 공통모드가 안 갈린다" | *"IMU roll/pitch mounting bias is **EXACTLY degenerate** with a common-mode leg tilt"* → `--imu-bias` 를 **거부**한다. 그리고 결론도 같다 — 정책이 같은 IMU 를 obs[0:3] 로 먹으므로 *"arguably the frame you want"* |

**두 세션이 서로 모른 채 같은 물리에 도달했다.** 이 축의 유도는 신뢰해도 된다.

### ⇒ §C-5 의 내 제안은 **부분적으로 대체된다**

추정기는 사슬 **합**을 넘어 개별 관절까지 푼다(높이/공면성 + 자세 다양성으로 분해).
내 IMU 한 줄짜리 계산은 그것의 **가장 약한 부분집합**이다.

**그래도 남기는 이유 둘** (권고 유지):
1. 추정기는 **로봇을 CUSTOM 모드로 움직인다**(스쿼트 1.0 rad 포함, 2인 필요, 안전 절차 있음).
   내 계산은 **`b` 자세 정지 상태에서 읽기만** 하므로 위험이 0이고, `--hold-diag` 로그로
   **사후에도** 계산할 수 있다.
2. **교차검증용**: 추정기가 낸 `delta` 의 사슬 합이 내 한 줄 계산과 안 맞으면
   둘 중 하나가 틀린 것이다. 공짜 검산이다.

### ⇒ 그리고 **학습 쪽 권고에 분기가 생긴다**

| 배포 상태 | 학습이 랜덤화해야 하는 폭 |
|---|---|
| **지금** (추정기 미검증, 로봇 미실행 — R7 대기) | **교정 전 전체 오차.** §B-6 대로 |
| 추정기가 로봇에서 검증되고 배포가 `delta` 를 보정하면 | **추정기 잔차만.** 훨씬 좁아진다 |

⛔ **지금 폭을 좁히지 마라.** 추정기는 아직 로봇에서 한 번도 안 돌았고
(`--self-test`/`--observability` 는 로봇 없이 돈다), 검증되기 전에 폭을 좁히면
**닫히지 않은 약속 위에서 학습**하게 된다. 이 저장소가 반복한 실패다.

⭐ **다만 사전등록은 지금 해 두는 게 맞다**: 추정기가 로봇에서 잔차 σ 를 내면
`joint_zero.max_deg` 를 **3σ** 로 내리고 §B-3 강건성 레인으로 재확인한다.

### ⚠️ 그리고 학습 자산 관련 사실 하나 (`kinematics_k1.py` 가 12개 URDF 를 파싱해 확인)

> `Hip_Pitch` 관절 원점 z 가 **실기는 −0.077**, 그런데 **`K1_locomotion_armsdown.urdf`
> (E/F/I/L 배치가 학습한 자산)는 −0.062** 다. **15 mm, 양다리, 순수 수직.**
> 다리 사슬에서 **유일한 차이**이고 나머지는 바이트 동일.

✅ **N 배치는 안전하다** — `_ROBOT_ASSET = K1_robot_boxfoot.urdf`(직접 확인: −0.077).
⚠️ **그러나 배포된 `I3b_stance10` 계보와 M 배치 이전 arm 전부는 15 mm 틀린 다리로 학습했다.**
[HANDOFF_TO_TRAINING](HANDOFF_TO_TRAINING.md) 이 기록한 "틀린 발 리비전" 과 **다른 항목**이다
(그건 질량, 이건 기하). 새로 발견된 것이므로 여기 적어 둔다.
(부수: 발바닥 평면도 mesh −0.026896 대 충돌상자 −0.024 = **2.90 mm** 차이. 양발 공통이라
base height 방향으로만 들어가고, `base_height_target 0.52` 논의에 3 mm 로 붙는다.)

## C-6. **하면 나아지는가** — 판정

### ✅ 나아진다. 근거 넷이 독립적으로 같은 곳을 가리킨다.

1. **물리 유도** — 사용자 절차는 공통모드 사슬 오차를 만든다(§C-1)
2. **실기 실측** — `--hold-diag` 잔차 tilt **2.0–10.9°** 가 그 크기의 브래킷(§C-1)
3. **MuJoCo 실측** — 유일하게 실기 증상을 재현한 스윕이 **공통모드**였다(§C-3)
4. **분포 논거** — 지금 5모드는 "관절값 작고 사슬 합 큰" 조합을 거의 안 뽑는다(§C-2)

### ⚠️ 얼마나 나아지는지는 아직 모른다 — 그래서 판정 측정을 사전 고정한다

이 축은 **`NZ_zeroiid`(iid 단독) → `N9_zerostruct`(구조 혼합)** 의 사다리로 이미 설계돼
있다. 사슬 모드는 그 사다리의 **세 번째 칸**이다. 판정은 §B-3 강건성 레인에서:

| 관측 | 결론 |
|---|---|
| 사슬 arm 이 **±3° 사슬 레인**에서 `N9` 대비 낙상률 절반 이하 | 사슬 모드가 이득. 채택 |
| 차이 없음 | `mirror` 가중치만 올리는 것으로 충분. 모드 추가 불필요 |
| 깨끗한 레인(±0°) 정확도가 5 cm 이상 나빠짐 | 폭 과다. `max_deg` 를 5° 로 |

## C-7. C 권고 — 그리고 구현

### ① `mirror` 가중치를 올려라 (가장 값싼 한 줄)

```python
# tools/make_v7_arms.py, _ZERO_STRUCT
"modes": {"iid": 0.15, "single": 0.20, "leg_common": 0.15,
          "anti_mirror": 0.20, "mirror": 0.30}   # mirror 0.10 -> 0.30
```
근거: §C-3(원본 실험이 mirror 였다) + §C-1(사용자 절차가 mirror 다).
**`anti_mirror` 는 그대로 둔다** — 실기 증언이 독립 근거다.

### ② 사슬 모드 두 개를 추가하라 (`goal_pose.py`, ~20줄)

```python
_ZERO_MODES = ("iid", "single", "leg_common", "anti_mirror", "mirror",
               "pitch_chain", "roll_chain")

# --- pitch_chain (mode 5): 몸통 pitch 영점 오차 delta_p ---------------------
# 캘리브레이션 때 상체를 눈대중으로 세운 오차는 좌우 공통이고, 한 다리의
# (Hip_P + Knee + Ankle_P) 합으로만 나타난다. 관절값은 작은데 사슬 합이 큰
# 조합을 iid 는 거의 안 뽑는다. 그리고 open-loop 로 설 수 있느냐를 정하는
# 것은 관절값이 아니라 이 합이다(HANDOFF_DEPLOY_ENTRY §1: hip+knee+ankle=0).
m_pc = mode == 5
if bool(m_pc.any()):
    k = int(m_pc.sum())
    # delta = 사슬 합. amp 가 아니라 level 을 직접 곱한다(관절당이 아니라 합에 거는 값).
    delta = (torch.rand(k, 1, device=dev) * 2 - 1) * max_rad * level[m_pc]
    w = torch.rand(k, 3, device=dev) + 0.3         # 사슬 배분(어느 관절이 얼마나)
    w = w / w.sum(dim=1, keepdim=True)             # 합 = 1 -> 세 관절 합이 정확히 delta
    row = torch.zeros(k, self.num_dofs, device=dev)
    for leg_off in (0, half):                      # 좌우 같은 부호 = 공통모드
        for c, j in enumerate(self._pitch_idx):    # 한 다리 안의 Hip_P, Knee, Ankle_P
            row[:, leg_off + j] = delta.squeeze(-1) * w[:, c]
    b[m_pc] = row                                  # 다른 모드들과 같은 대입 형태

# --- roll_chain (mode 6): 몸통 roll 영점 오차 delta_r -----------------------
# 사용자 지적: 상체를 눈대중으로 세우는 것은 roll 에도 영향이 있고, 두 다리로
# 정해지는 body roll 도 있다. 축이 좌우 모두 +x 이므로 이것도 공통모드다.
# 사슬은 (Hip_R + Ankle_R).
```
`self._pitch_idx` / `self._roll_idx` 는 `dof_names` 에서 이름으로 찾아 `__init__` 에
한 번만 만든다(하드코딩 금지 — Isaac 의 DOF 순서가 URDF 와 같다는 보장이 없다.
`_resample_joint_zero` 가 `:512-525` 에서 이미 같은 이유로 검사한다).

⚠️ **기본값은 0.0 이어야 한다.** `_ZERO_STRUCT` 를 쓰는 arm 에서만 켠다 →
기존 arm 완전 무영향. `test_joint_zero.py` 에 검사 추가:
`pitch_chain` 표본의 `(hip+knee+ankle)` 합이 `[-max_rad*level, +max_rad*level]` 안이고
**좌우가 같은지**, `roll_chain` 도 마찬가지인지.

### ③ `joint_weight` 의 Hip_Roll 0.5 를 재검토하라

```yaml
joint_weight: [1.0, 0.5, 1.0, 1.0, 0.7, 0.7,  1.0, 0.5, 1.0, 1.0, 0.7, 0.7]
#                    ^^^ Hip_Roll                    ^^^
# 근거 주석: "±2°에서 이미 토크 20 % 포화였다"
```

**문제**: 이 가중치는 **피해 크기**로 정해졌다. 그런데 가중치가 곱해지는 것은
**우리가 시뮬레이션하는 고장의 확률·크기**다. *"가장 위험한 축을 가장 적게 학습한다"* 가 된다.

그리고 §C-1 이 말하듯 **몸통 roll 눈대중 오차가 정확히 Hip_Roll 에 떨어진다** —
가장 깎아 놓은 축이 실기에서 가장 크게 틀어질 축이다.
게다가 per-env 커리큘럼이 이미 "너무 어려움" 을 처리하므로 **중복 안전장치**다.

**권고**: 학습에서는 `1.0` 으로 올리고 커리큘럼에 맡긴다.
**평가 레인에서는 반드시 전부 1.0** (§B-3) — 평가에서 깎으면 시험이 쉬워질 뿐이다.
⚠️ 이것은 단독 레버로 재야 한다(`N9` 대비 `joint_weight` 하나 차이).

### ④ 모드 이름이 오해를 부른다 (문서화만)

`mirror`/`anti_mirror` 는 **관절좌표 기준** 이름인데, `goal_pose_v3.py` 의 미러 맵은
**물리 기준**이다. 두 체계에서 같은 단어가 반대를 뜻한다(§C-1 ⚠️).
**코드는 그대로 두고**(바꾸면 기존 config 가 깨진다) config 주석에 한 줄 적어 둘 것:

> `mirror` = 좌우 관절좌표 **같은 부호** = 몸통 pitch/roll 공통모드
> `anti_mirror` = 좌우 관절좌표 **반대 부호** = 스탠스 폭 / 다리 모임

---

# D. teacher–student — 이론과 우리 적용

## D-1. 이론 — 무엇을 teacher 에게 주고 student 가 무엇으로 대체하는가

| 방법 | teacher 가 받는 것 | student 가 받는 것 | 무엇을 최소화하나 |
|---|---|---|---|
| **RMA** (Kumar+ 2021, arXiv:2107.04034) | 특권 환경인자 `e_t` → 인코더 `μ` → **extrinsics `z_t`** (저차원). base policy 는 `(s_t, a_{t−1}, z_t)` | 상태·액션 **이력** → `φ` | 2단계: base 와 `μ` 를 **얼리고** `‖φ(hist) − μ(e_t)‖²`. 배포는 두 모듈 **비동기** |
| **Lee+ 2020** (Science Robotics, arXiv:2010.11251) | 특권 상태(지형·접촉·마찰) | 고유수용 **이력**(TCN) | 잠재 일치 + 행동 복제 |
| **CTS** (arXiv:2405.10830) | 특권 상태 → `E_θt` | 이력 `o_{t−H:t}` → `E_θs` | **동시**: `L_ppo,t + L_ppo,s + L_rec`, `L_rec = ‖E_θs(o) − E_θt(s)‖²`. env 를 두 그룹으로 나눠 각각 `z^t`/`z^s` 로 롤아웃, policy·critic 공유 |
| **L2T** (arXiv:2402.06783) | 특권 | 고유수용(+노이즈) | 공동학습 + student 액션 혼합. **행동분포 L2** 가 asymmetric critic 손실을 섞는 것보다 근소 우위 |
| **DAgger 계열** | — | — | student 자신의 분포에서 teacher 라벨을 받아 **분포 이동**을 없앤다 |

**보고된 이득**: CTS 는 2단계 대비 속도추종 오차 **약 5 %** 개선(0.103 → 0.098 등),
push 생존율은 사실상 동률. L2T 는 표본 **50 % 절감**.
⇒ **2단계 대 동시학습의 차이는 크지 않다. 큰 차이는 "특권 신호를 쓰느냐 마느냐" 다**
(CTS 논문의 proprioceptive-only 기준선 0.119 대 0.098).

## D-2. ⭐ 왜 asymmetric actor-critic 만으로는 부족한가 — 정확한 구분

우리는 이미 asymmetric AC 를 쓴다: `est_value(obs, privileged_obs)`
([`model.py:33-35`](htwk-gym/utils/model.py)). `N7_critic` 이 그 privilege 를 14 → 29 로
넓히는 arm 이고 **지금 5115/6000 이다.**

**두 개를 혼동하면 안 된다.**

| | 비평자 특권 (asymmetric AC) | 증류 (teacher–student) |
|---|---|---|
| 무엇을 고치나 | **가치추정의 분산·편향**. 같은 관측인데 조건이 다른 env 들을 하나로 평균하지 않게 된다 | **최적화 문제 자체**. 어려운 POMDP RL → 쉬운 MDP RL + 지도학습 |
| 정책의 가설류 π | **안 바뀐다.** 여전히 `π(a|o)` | **안 바뀐다.** student 도 `π(a|o_hist)` |
| 배포 정책 입력 | 한 글자도 안 바뀜 | 한 글자도 안 바뀜(이력을 쓰면 그만큼 넓어짐) |

> **핵심**: **정책류를 넓히는 것은 증류가 아니라 관측 이력이다.**
> 증류가 하는 일은 *"그 넓은 정책류를 실제로 학습 가능하게 만드는 것"* 이다.
> 비평자 특권은 **advantage 의 잡음을 줄일 뿐 정책이 표현할 수 있는 함수를 안 늘린다.**

**우리 경우 이 구분이 결정적인 이유** — [AUDIT §2-3](AUDIT_20260807_2300.md):

> `history_steps: 1`, RNN 없음, **관측에 모드 표시 채널 없음**.
> path 의 carrot 간격 median 1.30 m ↔ waypoint 시작거리의 55.8 % 가 0.5 m 초과.
> ⇒ **겹치는 대역에서 정답이 정반대인데 기억 없는 정책은 하나만 고를 수 있다.**

이것은 **가치추정 문제가 아니라 정책 표현력 문제**다. 비평자를 아무리 잘 만들어도
`π(a|o_single_frame)` 가 두 모드를 구분할 수 없다는 사실은 안 바뀐다.
⇒ **`N7_critic` 이 성공해도 이 결손은 안 닫힌다.** 그건 `N4_hist` 가 닫을 문제다.

⚠️ 반대 방향의 근거도 있다: 최근 문헌은 teacher-student 가 **teacher 의 occupancy measure
와 준최적성을 물려받는** 단점을 지적하고, asymmetric AC 가 그것을 피한다고 본다
(KiVi arXiv:2509.23650 등). **그래서 순서가 중요하다** — §D-4.

## D-3. 우리에게 있는 조각 / 없는 조각

| 조각 | 상태 |
|---|---|
| 비평자 특권 확장 | ✅ `observation.privileged_extra` (14 → 29: 접촉 2 + 관측지연 1 + **영점 12**) |
| 관측 이력 | ✅ `observation.history_steps` (프레임 폭 × k, 미러 맵도 타일링 지원) |
| warm-start 수술 | ✅ `tools/expand_checkpoint.py` (가중치 + **옵티마이저 상태**, `--verify` 가 float64 항등 + 실제 `step()` 2회) |
| 미니배치 PPO 러너 | ✅ `utils/runner_v3.py` |
| **저차원 잠재 인코더 `μ`** | ❌ 없다. 비평자는 특권을 **직접** 먹는다(`cat(obs, priv)`) — 압축된 `z` 가 없다 |
| **student 인코더 `φ`** | ❌ 없다 |
| **증류 손실 / 두 그룹 롤아웃** | ❌ 없다 |
| **얼린 teacher 로 롤아웃하는 경로** | ❌ 없다 |

### 코드가 얼마나 필요한가 (견적)

| 방식 | 필요한 것 | 규모 (추정) |
|---|---|---|
| **RMA 형 2단계** | `μ`(특권→z), actor 를 `(o, z)` 입력으로, 2단계 러너(teacher 얼리고 `φ` 회귀) | `model.py` +40줄, 새 `runner_distill.py` ~150줄, `make_v7_arms` 항목 |
| **CTS 형 동시** | 위 + env 를 두 그룹으로 나눠 롤아웃, `L_rec` 추가 | 위 + `runner_v3.py` 침습적 수정 ~80줄 |

⛔ **`runner_v3.py` 는 지금 도는 arm 전부가 쓰는 파일이다.** 침습적 수정은 **별도 러너
파일**로 분기해야 한다(`train_v7.py` 가 러너를 고르게). CTS 형은 그래서 더 비싸다.

## D-4. 구체 arm 설계 — 사전 고정

### ⛔ 먼저: **지금 하지 마라.** 순서가 있다.

증류는 *"이력이 있으면 더 잘한다"* 가 참일 때만 값어치가 있다. 그 명제가 아직 **미검증**이다:

- `NA_histzero`(이력 5 + 영점)는 **[AUDIT §5-2](AUDIT_20260807_2300.md) 로 무효**
  (깨진 대칭손실 + 6.6초마다 평탄화된 이력 위에서 완주)
- ✅ [AUDIT §5-1](AUDIT_20260807_2300.md) 의 `_obs_prime_ids` 결함은
  **`8d763fb`(2026-08-08 01:27)에서 고쳐졌다** — `_resample_goals` → `_reset_idx` 로 이동.
  ⭐ 같은 커밋이 **`obs_delay` 재추첨도 같은 버그였다**고 적는다 — *"그 로봇의 성질이라
  에피소드 안에서는 고정"* 이라고 주석에 적어 둔 관측 지연이 **6초마다 다시 뽑히고 있었다.**
  ⇒ `N8_pathdelay`·`MB_obsdelay`·`N4_hist` 의 지연 축 결과가 전부 그 위에 있었다

**⇒ 남은 선행조건 하나:**

| # | 선행 | 상태 |
|---|---|---|
| ~~P1~~ | `_obs_prime_ids` 수정 | ✅ **완료** (`8d763fb`) |
| **P2** | `N4_hist` 재실행(수정 후) + `N7_critic` 채점 | ⏳ `026-NA_histzero_v2` 가 gpu1 큐에 있다. `N7_critic` 은 5115/6000 |

**P2 가 둘 다 이득 없음이면 증류는 하지 않는다** — 증류의 전제가 무너진다.

⚠️ 그리고 `NA_histzero_v2` 는 **이력 + 영점** 묶음이다(레버 2개). "이력이 이득인가" 를
단독으로 가르려면 `N4_hist`(이력만) 도 필요하다. 지금 큐에 있는 것은 `NA` 쪽이다.

### 그리고 나서: `NT_distill` (RMA 형 2단계)

**왜 CTS 형이 아니라 RMA 형인가**: ①문헌 이득 차가 5 % 수준(§D-1) ②CTS 형은
`runner_v3.py` 를 침습적으로 고쳐야 한다(§D-3) ③RMA 2단계는 **1단계가 그냥 지금
`NB_zerocritic` 이다** — 새로 학습할 것이 사실상 없다.

| | 설계 |
|---|---|
| **teacher(1단계)** | `NB_zerocritic` 그대로. actor 입력 54 + `z_t`(8차), critic 입력 54 + 29 |
| **`μ` 의 입력** | privileged 29 중 **랜덤화 인자만**: base com/mass 4 + push force/torque 6 + 관측지연 1 + **영점 12** = 23 → MLP(23→64→8) → `z_t` |
| **`z_t` 차원** | **8** (RMA 와 동일) |
| **student(2단계)** | `φ`: 최근 **20 프레임**(=400 ms)의 `(o_t, a_{t−1})` → 1D conv 3층 → `z̃_t`. teacher·`μ`·actor 는 **전부 얼린다** |
| **손실** | `‖z̃_t − z_t‖²` (MSE). ⚠️ **행동 손실은 안 쓴다** — RMA 대로. 액션은 얼린 actor 가 `z̃` 를 먹어서 나온다 |
| **롤아웃 분포** | **student 정책으로 롤아웃**(DAgger 형). teacher 분포에서만 학습하면 배포 시 분포 이동을 그대로 맞는다. L2T 가 지적한 그 실패다 |
| **예산** | 1단계 0 (NB 재사용) / 2단계 **500 iteration** 이면 충분(지도학습이다) ≈ **0.3 GPU-시간** |
| **warm start** | `expand_checkpoint.py` 로 actor 첫 층을 54 → 62 로 넓히고 새 8열을 **0으로**. 수술 직후 `z` 무시 = NB 와 출력 동일 → 비교가 "z 를 더하니 좋아지는가" 가 된다 |

### 사전 고정 판정 기준 (숫자 보기 전)

`NT_distill` 을 §B-3 의 **강건성 레인(±0/±1/±3/±5°)** 에서 `NB_zerocritic` 과 대조한다.
두 arm 은 **같은 iteration** 으로 맞춘다(§1-3).

| 관측 | 결론 |
|---|---|
| ±3°·±5° 레인에서 **낙상률이 절반 이하** **이면서** ±0° 정확도 열화 ≤ 2 cm | ✅ 증류 채택. 다음은 CTS 형 검토 |
| ±3° 이득이 있으나 ±0° 정확도가 5 cm 이상 나빠짐 | ⚠️ `z` 차원을 4로 줄이고 재시도 (1회만) |
| 어느 레인에서도 차이 없음 | ⛔ **폐기.** `φ` 회귀 오차 `‖z̃−z‖` 를 같이 보고, 그것이 작은데도 이득이 없으면 **`z` 가 애초에 정책에 쓸모없는 정보**라는 뜻 → 특권 채널 선택을 다시 본다 |
| `φ` 회귀 오차가 크다(설명분산 < 0.5) | ⛔ 이력 20프레임으로 영점이 **식별 불가**하다는 뜻. `extra_dof_tau`(=`N5_tau`)를 student 입력에 넣고 재시도 — 토크가 영점의 가장 직접적인 관측 가능 흔적이다 |

> ⭐ 마지막 줄이 이 설계의 진짜 값어치다: **증류가 실패해도 "영점은 고유수용으로
> 식별 불가능하다" 는 판정을 남긴다.** 그러면 처방이 DR(보수화)로 확정되고,
> `z` 추정에 더 이상 GPU 를 쓰지 않는다.

---

# 2. 우선순위 — 무엇부터

정렬 기준: **(측정을 가능하게 하는가) > (값싼가) > (근거가 강한가)**.

| # | 항목 | 비용 | 왜 여기인가 |
|---|---|---|---|
| **1** | **§B-4 `_zero_level` 로깅** | 코드 4줄, GPU 0 | 이게 없으면 영점 축 **전체가 반증 불가능**하다 |
| **2** | **§B-3 강건성 레인** | GPU 소 | `NZ`·`N9`·`NB`·(장차 `NT`) 넷의 **존재 이유가 안 재지고 있다** |
| **3** | **§C-7 ① `mirror` 가중치 0.10 → 0.30** | config 한 줄 | 근거 둘이 독립적으로 같은 곳을 가리킨다(§C-1, §C-3) |
| **4** | **§A-5 ③ logstd 대칭항 + ④ mirror p90 열** | 코드 ~5줄 | 대칭을 켜 놓고 효과를 한 번도 안 쟀다 |
| **5** | **§C-7 ② 사슬 모드 2개** | 코드 ~20줄 + arm 1개 3.6 h | 3번이 부분적으로 대체하므로 3번 결과를 보고 |
| **6** | **§D-4 `NT_distill`** | 0.3 GPU-시간 + 코드 ~190줄 | **P2 가 통과한 뒤에만** |
| ✅ | ~~`claim_check` 체크포인트 해석~~ | — | **완료** `e76e926` |
| ✅ | ~~`_obs_prime_ids` 수정~~ | — | **완료** `8d763fb` |
| — | ⛔ `mirror_augmentation_coef` | — | **하지 않는다**(§A-6) |
| — | ⛔ `joint_zero` 폭을 지금 좁히는 것 | — | 추정기가 로봇에서 검증되기 전에는 안 된다(§C-5b) |

**GPU 예산 관점**: 1·4 는 GPU 를 안 쓴다. 2·3 은 소량. 5·6 은 arm 하나씩.
⇒ **방금 회수한 `ND_dwell` 의 3.6 GPU-시간 하나면 이 목록의 GPU 부분 전체가 들어간다.**

---

# 3. 내가 못 닫은 것 / 누가 닫나

| 열린 것 | 왜 못 닫았나 | 누가 |
|---|---|---|
| `NZ` 대 `NE_ctrl100` 레버 판정 | `NE` 채점이 **지금 돌고 있다** | 학습 세션, 곧 |
| `NZ` 대 `NC` 순위 | best 가 it1700 대 it100 (§1-3) | 학습 세션 — 같은 iteration 재채점 |
| `NZ` 커리큘럼이 실제 도달한 각도 | 로깅이 없다(§B-4). **사후 복원 불가** | 다음 arm부터 |
| 사슬 모드의 실제 이득 | arm 이 아직 없다 | 학습 세션 (§C-7 ②) |
| 실기 영점의 **공통모드** 크기 | `probe_joint_zero.py` 가 구조적으로 못 잰다(§C-5). **배포 쪽 추정기가 이미 이것을 겨냥한다(§C-5b)** | **사용자/배포 세션** — R7(자세 수집)이 로봇에서 아직 안 돌았다 |
| 추정기 잔차 σ (→ `max_deg` 를 3σ 로) | 로봇에서 한 번도 안 돌았다 | 배포 세션 → 그 뒤 학습 세션 |
| MuJoCo 프로브 부호 수정 | 기존 `logs/mujoco/sig/*` 전부 무효화. **Codex 자산** | 배포/Codex 세션과 조율 |
| `symmetry_eval` 이 실제로 무엇을 재는지 | `eval_goal_pose.py:2687` 을 읽지 않았다 | 학습 세션 (§A-5 ④ 전에) |
| 15 mm Hip_Pitch 자산 오차의 영향 범위 | N 배치는 안전하지만 **배포 계보(`I3b_stance10`)는 그 위에서 학습됐다**(§C-5b) | 학습 세션 — 재학습 대상인지 판단 |
| `N4_hist`(이력 **단독**) | 큐에 있는 것은 `NA_histzero_v2`(이력+영점, 레버 2개)다 | 학습 세션 (§D-4) |

---

# 부록. 확인한 파일과 명령 (재현용)

```
htwk-gym/envs/K1/goal_pose_v3.py:84-205      미러 맵 / mirror_obs / mirror_actions
htwk-gym/utils/runner_v3.py:109-112,256-321  symmetry_coef / mirror_augmentation_coef / 손실
htwk-gym/utils/model.py:27,33-35             logstd 12차 / est_value(cat(obs, priv))
htwk-gym/envs/K1/Goal_Pose_V7.yaml:63-64     symmetric_coef(죽음) / symmetry_coef 0.5
htwk-gym/envs/K1/Goal_Pose_V7.yaml:470-525   joint_encoder_bias / joint_target_offset / joint_zero
htwk-gym/envs/K1/goal_pose.py:418-562        영점 유도 + _resample_joint_zero
htwk-gym/envs/K1/goal_pose.py:847-848,864    dof target + PD
htwk-gym/envs/K1/goal_pose.py:1083,1180-1214 obs 에 encoder_bias / privileged_extra
htwk-gym/tools/make_v7_arms.py:682-960,984   N arm 정의 / set_dotted
htwk-gym/tools/probe_joint_zero.py:1-45      공통모드 한계 명시 + imu rpy 수집
htwk-gym/play_mujoco_goalpose.py:396-403,477,506   MuJoCo 영점 프로브
htwk-gym/resources/K1/K1_robot_boxfoot.urdf  관절 축·한계 (좌우 같은 축 확인)
htwk-gym/resources/K1/K1_serial.xml:104-147  MJCF 축 교차확인
archive/sweeps/hbatch/H0-codex.yaml ↔ H1-codex.yaml   H1 의 실제 diff(레버 5개)
hbatch-codex.md:391-413,536-566              H1 실패 원인 감사
ibatch.md:2706-2740                          jointcal3 대가 실측
docs-research/user_requirements.md:74        G10

ssh a6000 'bash .../tools/round_status.sh'
ssh a6000 '... python tools/round_table.py logs/eval_rounds/n2na'
ssh a6000 '... python tools/claim_check.py <NZ>.accuracy/report.json <NC>.accuracy/report.json'

문헌: Abdolhosseini+ MIG'19 (PDF 원문) / Yu+ 2018 / arXiv:2403.17320 /
      arXiv:2107.04034 (RMA) / arXiv:2010.11251 (Lee+2020) /
      arXiv:2405.10830 (CTS) / arXiv:2402.06783 (L2T) /
      DR 범위: arXiv:2504.00614, 2409.19795, 2603.16188, 2502.10363
```
