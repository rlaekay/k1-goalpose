# H 배치 진단과 A/M 2단 스크린 — non-codex

> 2026-08-03. `hbatch-results-codex/`의 H0–H3 결과를 전수 분석하고, Codex의 M-cell
> 스크린을 검토해 **실행 가능한 형태로 고친** 문서. `hbatch-codex.md`는 Codex 소유,
> 이 문서는 내 소유다.

---

## 1. H 배치는 무엇이 잘못됐나 — 4개 arm이 같은 곳으로 갔다

`selection.json`의 stage-1(32후보 × 20 s) 전수:

| iteration | H0 | H1 | H2 | H3 |
|---:|---:|---:|---:|---:|
| **0** (warm start) | **7.25 cm** | 7.25 | 7.25 | 7.25 |
| 100 | 10.73 | 12.31 | 13.18 | 10.45 |
| 200 | 12.35 | 14.75 | 13.54 | 13.90 |
| 400 | 17.16 | 22.80 | 29.03 | 19.83 |
| 4200 | 37.49 | 39.71 | 39.16 | 38.11 |
| 12000 | **39.43** | **40.22** | **39.58** | **40.56** |

**네 arm 모두 iteration 0부터 단조 열화해 같은 값(≈0.40 m)으로 수렴한다.**
121개 checkpoint 중 `model_0`(=G1 warm start)이 **4/4에서 우승**했고, 그래서
`hbatch-comparison-codex.md`의 네 행이 비트 단위로 동일하다.

> **처치가 무엇이든 결과가 같다 = 공통 부분이 레버 효과를 완전히 덮었다.**
> 이 상태에서 외란·joint DR·mirror loss를 다시 재는 것은 의미가 없다.
> 사용자 지시 "공통적인 타스크만 우선 해결"이 정확히 이 지점이다.

**0.40 m라는 값도 낯설지 않다.** `gbatch_results.md` §2에서 E1/V7/G3/G4가 보인
"전 카테고리 동일한 상수 오차"와 같은 서명이다 — **멈추지 못하고 일정 거리를
표류하는 정책**. 즉 H 학습은 정책을 그 attractor로 밀어 넣었다.

**손상은 iteration 100 안에 이미 1.4~1.8배**다. 따라서 진단 실험은 **150~200
iteration이면 충분**하고, 12,000까지 갈 이유가 전혀 없었다. (Codex도 같은 결론에
도달했다 — 이 판단은 옳다.)

---

## 2. 공통 부분에서 무엇이 바뀌었나

G1(warm start) → H 4종이 **공유**하는 변경:

| # | 변경 | 크기 | 감사 상태 |
|---|---|---|---|
| **A** | **팔 asset 교체** `armsdown` → `hbatch-codex` | **양팔 yaw I_zz +32.4%, ego 반폭 +60%** | ⛔ **미감사** |
| B | task class `Goal_Pose_V7` → `Goal_Pose_HBatch` | path controller/dwell/보상 의미 변경 | 부분 |
| C | fresh Adam, LR 1.71e-4 → 5e-6 (adaptive 1e-6~1e-5) | 34배 감소 | Codex 감사함 |
| D | `stand_posture` 0 → −1, `stop_ang_speed_threshold` 0 → 0.3 | 보상 항 2개 | 미감사 |
| E | low-dose 외란 + goal jitter + joint DR | 레버 (M1/M2가 시험) | 설계됨 |

### 2-1. A가 유력한 이유 — 그리고 이 교체는 물리적 이득이 0이다

메시 정확 감사(`Left_Arm_4.STL` 전 정점 7050개, Trunk 충돌 프리미티브
박스 `0.12×0.18×0.2 @ (0,0,0.1)` + 골반 실린더 `r0.05 l0.1 @ (0,0,−0.06)`):

| asset | Trunk 관통 | 손끝 x | ego 반폭 | 양팔 yaw I_zz | G1 대비 |
|---|---|---|---|---|---|
| **armsdown** (G1이 학습) | 22.0 mm | −0.121 | 0.160 | **0.07624** | — |
| **hbatch-codex** (H 4종 전부) | 0 | −0.028 | **0.250** | **0.10090** | **+32.4%** |
| **hbatch** (신규) | 0 | −0.029 | 0.155 | **0.06981** | −8.4% |

**그런데 armsdown의 22 mm 관통은 물리적으로 존재하지 않는다.**

```yaml
asset.collapse_fixed_joints: true      # Goal_Pose_V7.yaml:73, H0-codex.yaml:84
# 그리고 armsdown의 팔 관절 8개는 전부 type="fixed"
```

→ 팔 링크가 **Trunk와 한 강체로 병합**되므로 PhysX는 그 관통을 **평가하지 않는다.**
사용자가 본 겹침은 **렌더링에만 존재하는 시각 결함**이다.

> `gbatch.md` §8-5④의 444,180 N은 **armSWING**이었다 — 거기서는 팔꿈치가 `revolute`라
> 손이 **독립 강체**가 되어 진짜 자기충돌이 일어났다. armsdown은 그 경우가 아니다.
>
> **결론: asset 교체는 렌더링 결함을 고치려고 실제 동역학을 32.4% 바꿨다.**
> warm start 정책은 I_zz 0.0762에 맞춰진 요(yaw) 응답 모델을 갖고 있는데
> 0.1009로 뛰면 자기 회전 반응을 체계적으로 잘못 예측한다. 이것이 iteration 100
> 안에 나타나는 즉각적 열화와 부합한다.
>
> **확정은 아니다. 그래서 실험으로 분리한다.**

---

## 3. Codex M-cell 스크린의 결함

`hbatch-codex.md` §"하단 코멘트 실행안"의 4셀(M0 control / M1 force / M2 jointdr /
M3 mirror_off)은 설계 의도는 옳지만 **실행 불가능한 결함**이 있다:

```python
# tools/make_mcell_configs.py:147  (수정 전)
put(cfg, "asset.file", "resources/K1/K1_locomotion_hbatch-codex.urdf")
```

**유력 용의자(A)가 4개 셀 전부에 하드코딩돼 있다.** M0이 "minimum-allowed G1
continuation"이라고 선언돼 있지만 실제로는 +32.4% 관성 변경을 포함한다.
따라서 A가 원인이면 **M0~M3 네 셀이 모두 똑같이 무너지고 H 배치를 그대로 재현**한다.
0.7 GPU-시간을 써서 "네 레버 다 실패"라는 이미 아는 답을 다시 얻는다.

> `hbatch-codex.md` §M-cell은 "필수 팔 asset"을 **전제**로 못박았는데, §2-1이
> 보이듯 그 전제 자체가 검증된 적이 없다. **전제를 가설로 내려야 한다.**

---

## 4. 고친 설계 — A 스크린 → M 스크린 2단

### Wave A — 공통 실패 원인 (4셀, 150 iter, 2 GPU)

레버는 전부 OFF. **셀 간 차이가 정확히 하나씩**이다.

| cell | asset | task | 나머지 | 답하는 질문 |
|---|---|---|---|---|
| **A0_asset_g1** | `armsdown` (G1 그대로) | HBatch | fresh Adam 5e-6 | task+optimizer만으로 무너지나 |
| **A1_asset_fixed** | `hbatch` (신규, I_zz −8.4%) | HBatch | 〃 | 겹침만 고치고 관성 유지하면 보존되나 |
| **A2_asset_codex** | `hbatch-codex` (+32.4%) | HBatch | 〃 | **관성 32% 증가가 범인인가** |
| **A3_task_v7** | `armsdown` | **`Goal_Pose_V7`** | 〃 | **task class 변경이 범인인가** |

**읽는 법**

| 관측 | 결론 |
|---|---|
| A0 보존, A2 열화 | **asset이 원인** → A1로 확정, 이후 전부 A1 또는 armsdown |
| A0·A2 둘 다 열화, A3 보존 | **task class가 원인** → HBatch path/dwell/보상 재감사 |
| A0·A2·A3 전부 열화 | 남은 것은 **C(optimizer/LR) 또는 D(보상 2항)** → Wave A2로 분기 |
| A0 보존, A3도 보존 | 공통부는 무해 → 바로 Wave M |

**150 iteration인 이유**: 손상이 iteration 100에 이미 1.4~1.8배로 나타난다(§1).
150이면 방향 판정에 충분하고, 12,000 대비 **1.25%** 비용이다.

### Wave M — 레버 3종 (4셀, 200 iter, 2 GPU)

Wave A가 확정한 **무해한 공통 기반** 위에서만 실행한다.

| cell | 처치 | 사용자 질문 |
|---|---|---|
| M0_control | 없음 | paired control |
| **M1_force** | scenario-aware 외란 | **외란** |
| **M2_jointdr** | encoder ±0.015 / target ±0.010 rad | **joint DR** |
| **M3_mirror_off** | `symmetry_coef` 0.5 → 0 | **mirror loss** |

Codex의 M 셀 정의·평가 프로토콜(held-out 공통 시험지, paired seed, 조기중단)은
그대로 쓴다. **바뀌는 것은 asset 하나뿐이다.**

### 2-GPU 배치 — 왜 4셀 동시인가

과거 실측(같은 서버): 카드당 2 프로세스에서 GPU 96–98%, VRAM ≈8 GB / 49 GB.

| 방식 | 벽시계 | 판정 |
|---|---|---|
| 4셀 동시 (2/카드) | 150 iter × t × (2 / 1.5배 처리량) ≈ **200 t** | ✅ 채택 |
| 2셀씩 2 wave (1/카드) | 2 × 150 × t = **300 t** | ❌ 1.5배 느림 |

카드당 2 프로세스는 처리량이 2배가 아니라 **약 1.5배**다(시분할). 그래도 순차보다
빠르고, **4셀이 동시에 끝나 조기중단 판정을 같은 시점에 내릴 수 있다**는 이점이 크다.

```
GPU 0 : A0_asset_g1  +  A2_asset_codex      ← asset 대조가 같은 카드 (열/클럭 조건 동일)
GPU 1 : A1_asset_fixed + A3_task_v7
```

**asset 3종을 같은 카드에 몰지 않고 A0/A2를 짝지은 이유**: 핵심 대조가
`armsdown vs codex`라 이 둘의 하드웨어 조건을 동일하게 묶는다.

**예상 비용**: G1 실측 ≈7.2 s/iter(2 프로세스/카드) → 150 iter ≈ **18분** + smoke·eval
포함 **40분 내외**. Wave M까지 합쳐 **1.5 GPU-시간 미만**.

---

## 5. 실행

```bash
cd ~/RoboCup/k1-goalpose && git pull && cd htwk-gym && conda activate k1goalpose && CELLS="A0_asset_g1 A1_asset_fixed A2_asset_codex A3_task_v7" ITERS=150 bash tools/run_mcells.sh
```

통과 후 Wave M:

```bash
cd ~/RoboCup/k1-goalpose/htwk-gym && CELLS="M0_control-codex M1_force-codex M2_jointdr-codex M3_mirror_off-codex" ITERS=200 bash tools/run_mcells.sh
```

`run_mcells.sh`가 이미 하는 것(그대로 유지): 셀별 inference/mechanics smoke → 통과
셀만 GPU 투입 → 2 iteration health marker(240 s) → 실패 셀만 별도 로그 →
GPU당 persistent worker queue로 병렬 eval.

---

## 6. 이 문서가 `hbatch-codex.md`와 다른 점

| 항목 | Codex | 이 문서 |
|---|---|---|
| 공통 실패 진단 | "H0도 pure control이 아니다"까지 | **열화 곡선 전수 + 0.40 m 수렴 + asset 정량** |
| asset | **전제**(필수 팔 asset) | **가설**(A 스크린의 축) |
| armsdown 겹침 | 물리 문제로 취급 | **`collapse_fixed_joints`로 병합 → 시각 전용** |
| 1단계 | M 4셀 (레버) | **A 4셀 (공통 원인) → M 4셀 (레버)** |
| iteration | 200 | **150 → 200** |
| GPU 배치 | M0+M2 / M1+M3 | **대조쌍을 같은 카드에** |

**유지하는 것**: G1 config SHA 강제, fresh Adam `2e-6`~`5e-6`, `load_optimizer_state=false`,
model_0 byte-identical 검사, held-out 공통 시험지, paired seed, 조기중단 규칙,
5단계 smoke, persistent eval queue. 이 하네스 설계는 좋고 그대로 쓴다.

---

## 7. 미해결 / 다음

- **`stand_posture −1`, `stop_ang_speed_threshold 0.3`(변경 D)은 아직 어느 셀도
  분리하지 않는다.** A0~A3가 전부 열화하면 이게 다음 축이다.
- Wave A가 asset을 범인으로 지목하면, **`hbatch.urdf`(신규)로 갈지 `armsdown`으로
  남을지**는 A1 결과가 정한다. 시각 결함은 물리적 비용이 0이므로 급하지 않다.
- `non-codex.md` §6-3의 팔 자세 권고(−93.4°/+124.9°/−51.4°)는 **190점 부분표본으로
  구해 7.9 mm 관통이 남아 있었다.** 전 정점 7050개 + 3 mm 여유로 다시 푼 값이
  **−98.55° / +80.21° / −12.51°**이고 `K1_locomotion_hbatch.urdf`에 반영했다.
  단 shoulder roll 한계 여유가 **1.2°**로 빠듯하다 — 실기 적용 전 재검토 필요.
