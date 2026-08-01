# H batch — 진단과 M-cell 단축 실험 (non-codex)

> 2026-08-01. `hbatch-codex.md` 하단 사용자 코멘트에 대한 수행 결과와,
> 외란·joint DR·mirror loss를 **빠르게** 가르기 위한 2-GPU 실험 설계.
> Codex 작업 감사는 [non-codex.md](non-codex.md)에 있다.

---

## 1. 왜 H 배치가 실패했나 — selection 표가 답을 준다

네 arm의 `selection.md` 후보 표를 나란히 놓으면 궤적이 **하나**다.

| iteration | H0 | H1 | H2 | H3 | 낙상 |
|---:|---:|---:|---:|---:|---|
| 0 | 7.3 | 7.3 | 7.3 | 7.3 | 29 |
| 100 | **10.7** | **12.3** | **13.2** | **10.5** | 3–18 |
| 200 | 12.4 | 14.7 | 13.5 | 13.9 | 3–26 |
| 2600+ | ~38 | ~38 | ~38 | ~39 | 0–15 |

**위치는 단조 악화, 낙상은 감소.** E2(−44.8% 속도, 91.1% 언더슈트)와 G2(>1 m/s 체류
0.5%)에서 두 번 본 collapse가 세 번째로 재현됐다. 정책이 **덜 움직여서 안전을 산다.**

결정적인 것은 **H0에서도 일어났다**는 점이다. H0는 대조군 취지였다. 따라서 원인은
각 arm의 처치가 아니라 **네 arm이 공유하는 층**에 있다.

그리고 iteration 100에 이미 **용량-반응**이 보인다: H0 10.7 < H1 12.3 ≈ H2 13.2.
DR/외란 dose가 클수록 빨리 나빠진다. H3(=H0+heel)는 10.5로 H0와 같다 — heel 보상은
초기에 중립이다.

### 여기서 나오는 두 가지 결론

1. **12000 iteration은 낭비였다.** 판정은 100에 이미 났다. arm당 ~10시간 × 4 = 40 GPU시간이
   "warm start가 이긴다"를 확인하는 데 쓰였다. → **200 iteration으로 자른다 (60배 단축).**
2. **"아무것도 안 바꾸고 fine-tune만" 한 대조군이 없다.** H0조차 10레버 묶음이다
   (새 URDF, 외란, jitter, bias, hold, flicker, encoder bias, target offset, LR 재시작,
   path 의미 변경). 그게 없으면 **레버의 비용과 fine-tune 자체의 비용을 구분할 수 없다.**

---

## 2. M-cell — 2 GPU 단일 wave, 약 25–40분

각 셀 = **M0 + 레버 정확히 1개.** 따라서 `셀 − M0`가 그 레버의 효과다.

| 셀 | 레버 | 질문 | GPU |
|---|---|---|---|
| **M0_control** | 없음 | **fine-tune 자체가 망가뜨리는가** ← 빠져 있던 셀 | cuda:0 |
| **M1_force** | 시나리오 외란 | 외란의 비용은 얼마인가 | cuda:1 |
| **M2_jointdr** | encoder bias + target offset | joint DR의 비용은 얼마인가 | cuda:0 |
| **M3_mirror** | symmetry loss | mirror loss의 비용은 얼마인가 | cuda:1 |

공통: G1@10700 warm start(동일 SHA), fresh Adam `5e-6`, 4096 env, 200 iteration,
체크포인트 0/25/50/100/200, goal 관측 노이즈 전부 OFF(G1이 본 적 없다), heel 보상 제거,
cruise-stability 보상 제거.

**GPU 배치 근거**: 카드당 2셀 동시는 G 배치에서 이미 검증됐다(GPU0=G1+G2, GPU1=G3+G4,
같은 4096 env). 4개를 한 wave로 돌리면 공유 서버의 부하 조건도 동일해져
**느린 이웃이 처치 효과로 위장하지 못한다.** M0와 M1은 가장 갈릴 쌍이라 서로 다른 카드에 뒀다.

### 실행

```bash
cd ~/RoboCup/k1-goalpose/htwk-gym && conda activate k1goalpose && git pull && nohup bash tools/run_mcells.sh > mcells.log 2>&1 & sleep 2 && tail -f mcells.log
```

끝나면:

```bash
python tools/compare_mcells.py
```

### 설계에 박아둔 함정 방지

| 함정 | 조치 |
|---|---|
| **`ramp_steps: 12000`이 4800 step 실험 안에서 40%까지만 올라감** → 처치가 실제로 안 들어감 | M1은 `ramp_steps: 1`. 생성기가 `ramp > 실행 step/4`면 **INVALID로 거부** |
| G1의 저장 Adam LR `1.71e-4`(선언값의 34배) 복원 → NaN | `load_optimizer_state: false` 강제, 생성기가 검사 |
| 셀마다 다른 시험지 | 평가는 **네 셀 모두 `M0_control.yaml` 하나로** 실행 |
| 전체 eval 스위트 7종 × 20 체크포인트 = 140회 | clean + combined 2종만, 체크포인트 0/50/100/200, **공유 `model_0`은 1회만** → 26회 |
| 한 셀 실패가 나머지를 막음 | 셀별 스모크, 실패 셀만 `logs/mcells/smoke_failures/`로 |
| 직렬 평가로 GPU 1 유휴 | 평가도 2카드 라운드로빈 |

### 읽는 순서 (중요)

**M0의 절대 열화를 먼저 본다.**
- M0가 200에서 +3 cm 이상 열화 → **범인은 레버가 아니라 fine-tune 설정**(LR, KL, warm start
  정합성)이다. 레버별 차이 해석은 그 다음이다.
- M0가 안정적 → 레버별 차이를 그대로 해석한다.

`compare_mcells.py`가 이 판정을 자동으로 출력한다.

### mirror augmentation을 이번에 넣지 않는 이유

transition augmentation은 표본을 `π_old(a|s)`에서 뽑으면서 PPO 분모로
`log π_old(Ma|Ms)`를 쓴다(`runner_v3.py:183–195, 295–298`). 두 값은 **old policy가 이미
완전 대칭일 때만** 같다. 즉 ratio가 구조적으로 편향돼 있다. 이 상태로 돌리면 augmentation이
아니라 **버그를 측정**하게 된다. M3는 `symmetry_coef`만 켜서 그 경로를 건드리지 않는다.
분모를 고친 뒤 wave 2에서 augmentation을 묻는다.

---

## 3. 사용자 코멘트 수행 결과

| 코멘트 | 수행 |
|---|---|
| heel-ahead 보상 **탈락시켜** | ✅ 전 셀에서 `heel_strike_ahead: 0.0`. H3 계보 폐기 |
| joint position DR **진행시켜** | ✅ M2로 격리 측정. mild dose(±0.015 / ±0.010) 채택 — H1의 강한 dose는 mirror와 교란돼 있었다 |
| 외력을 **상체·팔 위주로 몰기** | ✅ 이미 구현돼 있었다(`HEIGHT_TIER_DEFAULTS`): arm_proxy 0.60 + chest 0.30 = **상체·팔 90%**, shank/hip 각 0.05 |
| **z축 높이 랜덤화 + 각 레벨 전방향** | ✅ tier별 `offset_z_m` 랜덤, 3 시나리오 중 2개가 `direction_mode: uniform` |
| **충격량을 실제 시나리오로** (정면 전속력 충돌 없음 / 뒤에서 0.3–0.7 m/s / 팔 걸림) | ✅ 구현돼 있었다: `omni_shove` 0.50, `rear_push` 0.30(rear_cone ±22.5°), `arm_entanglement` 0.20(twist 1–4 N·m). **40–150 N 정면 충돌 클래스 삭제**, 15–40 N으로 하향 |
| — **다만 어떤 config에도 켜져 있지 않았다** | ⛔ `scenario_aware.enabled`가 H0–H3 어디에도 없다. **M1이 이걸 처음 켠다** |
| cruise-stability의 **scheduling/curriculum 제거** | ✅ 전 셀 `high_speed_stability: 0.0`. 속도×가속 sigmoid 게이트는 곧 스케줄이므로 제거 |
| 팔은 **`k1/K1_locomotion.urdf` 자세로** | ✅ `K1_locomotion_hbatch-codex.urdf`가 이미 그 값이다(shoulder roll ∓1.35, elbow 0). 검증 완료 |

### 팔에 대한 보류 사항 (결정은 사용자 것)

지시대로 참조 각도를 쓴다. 다만 [non-codex.md](non-codex.md) §6-3의 메시 실측 결과를
기록으로 남긴다 — 참조 각도는 겹침을 없애는 대신 **ego 반폭 0.156 → 0.250 m,
yaw 관성 +32.4%**를 되돌린다. 겹침 0이면서 손끝이 힙 뒤 11.3 cm, ego 반폭 0.145 m,
관성이 armsdown보다도 3.7% 낮은 자세가 존재한다(shoulder roll ∓93.4°, elbow pitch
+124.9°, elbow yaw ∓51.4°, 전 관절 한계 5° 이상 여유). **지금은 적용하지 않는다.**
M-cell이 공통층을 안정화한 뒤 단독 arm으로 물을 값어치가 있다.

---

## 4. 남은 질문에 대한 답

### Q. "E0 평가가 잘못돼서 종합 1위인지 판단 못하는 것 아니었나?"

**아니다. 그건 내 오류였고 철회했다.** E0에는 **유효한 평가가 있다**:

```
K1_walk/select_results/E0_armB_armsdown/report.json
  model_6200 · 자기 config(오염 없음) · waypoint 4633구간(path 0)
  2.72 / 5.01 cm · 2.52° · strict 89.29% · 낙상 2 · 게이트 3/4 통과
  정상종료 99.3% · 미도달 0.0%
```

내가 `cd K1_walk/v7` 상태에서 `ls K1_walk/`를 돌려 실패한 것을 "파일 없음"으로 결론냈다.
**E0@6200은 이 프로젝트 최고 정책이 맞다.** 상세는 [non-codex.md](non-codex.md) §0.

### Q. "같은 목표인데 빠를 때와 느릴 때 편차가 심하다. 빠른 쪽에 맞추는 고상한 방법은? 랜덤화로 되나?"

**랜덤화만으로는 안 된다. 원인이 랜덤성이 아니라 구조적 제약이기 때문이다.**

편차의 정체는 **목표 발생 시점의 gait phase 정렬 운**이다. `gait_process`는
`gait_frequency`로 **고정 속도로만** 진행하고(`goal_pose.py:570`), 목표는 그와 무관하게
재샘플된다. 좌/우 swing이 phase 0.25 / 0.75에 있으므로, 옆·뒤 목표가 불리한 phase에
도착하면 로봇은 **최대 반 주기를 기다려야** 방향을 틀 수 있다. 1.8 Hz에서 반 주기는
0.28 s이고, 그 사이 목표는 이미 가까워져 있다.

정책은 gait clock 2채널을 **보지만 바꿀 수는 없다.** 이게 핵심이다.

> **고상한 해법: 케이던스를 정책이 쓸 수 있는 자유도로 만든다.**
> `gait_frequency`를 명령 속도에 커플링하면(`f = clip(v_cmd / (2·L_max), 1.6, 3.2)`,
> `L_max ≈ 0.28 m`) 로봇은 가속·방향전환 시 케이던스를 올려 **불리한 phase를 빨리 빠져나간다.**
> 하드코딩이 아니라 **명령 분포의 수정**이고, 관측 54차원을 건드리지 않는다.

이건 [non-codex.md](non-codex.md) §5-D의 고속 lean 처방과 **같은 수정**이다.
1.8 Hz에서 1.5 m/s는 0.417 m 보폭을 요구하는데 실현 가능은 ~0.28 m라, 정책은 상체를
던지는 것 말고 방법이 없다. 케이던스 커플링 하나가 **8번(고속 기울기)과 22번(속도 편차)을
동시에** 겨냥한다.

**단, M-cell에는 넣지 않는다.** 지금 물어야 할 것은 "공통층이 왜 망가지는가"이고,
여기에 다섯 번째 레버를 더하면 또 묶음이 된다. **M-cell이 공통층을 정리한 직후 첫 후보다.**

### Q. "이제 fine-tuning 같은 것만 하면 되나? 학습시간 단축 방법은?"

**fine-tuning만으로 충분하지 않지만, 지금 단계에서는 fine-tuning이 맞다.**

- **맞는 이유**: 바꾸는 것이 보상·DR·관측 노이즈처럼 **기존 정책 위에서 미세 조정 가능한
  것**이면 warm start에서 수백~수천 iteration이면 된다. G1도 E0@6200에서 fine-tune으로
  나왔다.
- **안 되는 경우**: 관측 차원, 액션 차원, URDF 형상(팔 자세 포함), 과제 정의(path 비중)처럼
  **정책이 본 적 없는 입력 분포**를 만들면 fine-tune이 오히려 해롭다. H 배치가 정확히
  그랬다 — 새 URDF + 새 노이즈 + 새 path 의미를 한꺼번에 얹고 warm start에서 출발했다.

**시간 단축 3단계** (판단 근거와 함께):

| 방법 | 절감 | 근거 |
|---|---|---|
| **판정 iteration을 200으로** | **60배** | H 4 arm 전부 100에서 판정 가능했다 |
| **평가를 2종·40 s로** | **5배** | 7종 × 120 s는 진단에 과하다. 공유 `model_0`은 1회만 |
| **카드당 2셀 동시** | **2배** | G 배치에서 4096 env로 검증됨 |

합쳐서 **H 배치 40 GPU시간 → M-cell 약 0.7 GPU시간.** 통과한 셀만 길게 학습한다.

---

## 5. 다음 단계 (M-cell 결과에 따라)

| M0 결과 | 다음 |
|---|---|
| M0 안정 (+3 cm 미만) | 레버별 Δ를 그대로 읽고, 비용 없는 레버만 채택해 본 학습 |
| M0 열화 | **fine-tune 설정 자체를 수리한다** — LR 궤적, KL 스케줄, warm start의 obs 정합성 순으로 이분 탐색 |

두 경우 모두 그다음 후보는 **케이던스-속도 커플링**(8번 + 22번 동시)이고,
그 뒤가 mirror augmentation(분모 수정 후)과 팔 자세 재검토다.
