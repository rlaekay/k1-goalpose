# 사용자(인간) 요구사항 전수 추출 — k1-goalpose 저장소 루트 전체 문서

대상 20개 문서 (총 8,905줄):
`MASTERPLAN.md`(1117) · `masterplan2.md`(846) · `masterplan3.md`(241) · `MASTERPLAN_feedback.md`(877) ·
`gbatch.md`(617) · `ebatch.md`(370) · `ebatch-codex.md`(104) · `gbatch_results.md`(449) ·
`gbatch_results-codex.md`(95) · `hbatch.md`(213) · `hbatch-codex.md`(675) · `non-codex.md`(432) ·
`missions.md`(270) · `MISSION_READINESS_REVIEW.md`(76) · `MISSION_DEPLOY_AUDIT_20260730.md`(401) ·
`ROBOT_DEPLOY_E0_GUIDE.md`(598) · `STATE_ESTIMATION.md`(665) · `K1_LEARNING_HISTORY_KO.md`(743) ·
`README.md`(84) · `UPSTREAM.md`(32)

---

## 0. 사용자 원문이 물리적으로 존재하는 곳 (verbatim 인용 가능 범위)

| 형태 | 위치 | 개수 |
|---|---|---|
| HTML 주석 `<!-- -->` | **`masterplan2.md`** 159·183·256·300·339·372·413·471·528·538·631·687·701·705·737·741·810 | **17** |
| HTML 주석 `<!-- -->` | **`gbatch.md`** 26·64·139·178·209·252·286·302·321 | **9** |
| 최상위 불릿 `- ...` | **`MASTERPLAN_feedback.md`** 2~852 (`> **[답변]**`은 전부 어시스턴트) | **50** |
| `#### video 감상` 절 | **`masterplan2.md`** 832–847 (E0/E1/E2/V7 영상 관찰 원문) | **13줄** |
| `#### 질문` 절 | **`gbatch.md`** 476–478 | **2** |
| 어시스턴트가 인용한 사용자 발화 | `gbatch_results.md` 241·289–290·382, `hbatch.md` 28, `hbatch-codex.md` 98, `ebatch.md` 319, `gbatch.md` 618 | **7** |
| 어시스턴트 요약(“사용자 결정/선택/지시 N”) — **verbatim 아님** | `MASTERPLAN.md` 31·335–342·440–441·559·630, `non-codex.md` 66–91(지시 1–22), `missions.md` 91·190–205, `MISSION_READINESS_REVIEW.md` 38–45 | — |

**`masterplan3.md` · `ebatch.md`(주석) · `ebatch-codex.md` · `gbatch_results-codex.md` · `hbatch-codex.md` ·
`STATE_ESTIMATION.md` · `K1_LEARNING_HISTORY_KO.md` · `README.md` · `UPSTREAM.md` 에는 사용자 원문이 0줄이다.**
→ **문서가 최신일수록 사용자 목소리가 사라진다.** 이것이 이 추출의 가장 중요한 구조적 발견이다.

### 0-1. `non-codex.md` §1-2 (66–91행)이 보존한 **H 배치 사용자 지시 1–22 번호 대응표**
원문은 채팅에만 있고 저장소에는 제목만 남아 있다. 아래 표 전체에서 `[지시 N]`으로 참조한다.

| N | 제목 (어시스턴트 요약) | N | 제목 |
|---|---|---|---|
| 1 | masterplan 흐름 파악 | 12 | hbatch.md 작성 |
| 2 | 버전별 수정의도 대비 결과 | 13 | heel 접지 보상 |
| 3 | 데이터 유효성 분류 | 14 | joint DR 점검·강화 |
| 4 | ebatch 보고 | 15 | y축 mirror 증강 + loss |
| 5 | gbatch 보고 | 16 | E2 감속 원인 |
| 6 | H0 = current bests | 17 | 시뮬 시점 영상 + 외력 화살표 |
| 7 | H1/H2 보수적 실험 | 18 | sim2real 서치 |
| 8 | 고속 상체 기울기 | 19 | URDF 팔 겹침 |
| 9 | 외란·jitter 의무화 + “괜찮았나” | 20 | 모든 질문 응답 |
| 10 | train+eval 단일 하네스 | 21 | G1 압승 / E1·E0 인상 검증 |
| 11 | 모델/하네스 적절 활용 | 22 | 정옆·정뒤 감속 |

### 0-2. 배치 계보 (「어느 배치에서 시도됐나」 열의 값)

```
v0/v1 → armA/B/C/D → v3 → v7(=E배치: E0·E1·E2·V7_full) → G배치(G1·G2·G3·G4)
      → H배치(H0·H1·H2·H3, 전부 model_0 선택 = 처치 미평가)
      → A스크린(A0~A3, 계획) / M셀(M0~M3, 계획) → 실기(E0 mission 1–5)
```
값: **E** · **G** · **H** · **A/M(계획)** · **실기** · **미시도**

---

## A. GOALS — 최종 정책이 해야 하는 것

| # | 원문 인용 (사용자 원문) | 파일:줄 | 무엇을 요구했나 | 반복(전체) | **어느 배치에서 시도됐나** | 현재 상태 |
|---|---|---|---|---|---|---|
| **G1** | `\| 최종 위치 오차 (median / p90) \| ≤ 5 cm / ≤ 10 cm \|` · `\| 최종 heading 오차 \| ≤ 10° \|` · `\| 넘어짐률 \| 0% \|` | MASTERPLAN.md:15,16,17 | 5cm/10cm/10°/낙상 0으로 도달·정지 | **22회+** — MP:15-17,210,285,316,780,865,954,1038 / mp2:26-32,624 / mp3:29-33 / gbatch:17 / gbatch_results:65 / ebatch:289 / non-codex:32,144 / hbatch-codex:296,337-343 / hbatch:15 / K1_HIST:11,44 / DEPLOY_GUIDE:18-21 | **E·G·H 전부** | **위치·heading은 E0가 통과, 낙상은 전 배치 미통과.** E0@6200 `2.72/5.01 cm, 2.52°, strict 89.29%, 낙상 2`(non-codex:29) = 3/4. G1 5.52 cm(2/4). H 전 arm FAIL — warm-start조차 `6.899/11.761 cm, 낙상 24.473/1000`(hbatch-codex:337-341) |
| **G2** | `모든 버전에서 body속도를 표기하지 않음. body속도는 1.3m/s정도 나와야함.` | FB:347 | 몸통 속도 1.3 m/s + 전 리포트/영상 속도 표기 | **9회** — FB:109,233,347,852 / mp2:56-77,300 / mp3:48 / gbatch:189-192 / [지시 21] | **E1(path) → G1(grid) → H(전 arm)** | **표기 완료, 속도 목표 미달·미확정.** E0 p99 1.20(mp3:48), G1 body p90 **1.50 m/s**·>1m/s 17.7%(gbatch_results:145-148). 그러나 “구간최고 p90”은 **리셋 텔레포트로 최대 203 m/s 오염**(ebatch:259-268) → 수정됨. H baseline path speed **0.875 m/s**, 1 m/s 도달률 69.28%(hbatch-codex:344-345). **지속가능 상한 여전히 미측정** |
| **G2b** | `나는 가속과 각가속을 빠르게 하고싶은데 이 속도 값이 현재 뭐지? 가능하다면 1.4, 1.5m/s 정도까지로 올리고 싶은데` | FB:109 | 1.4~1.5 m/s + 가속/각가속 프로파일 | 2회 (FB:109,347) | **G1(속도×곡률 그리드)** | **미달.** 어시스턴트 판정 “1.4 m/s가 물리적으로 가능한지는 아직 **미검증**”(FB:148). 새 근거: **케이던스가 물리적 상한** — `gait_frequency [1.8,2.4]`가 명령속도와 독립이라 1.5 m/s는 0.417 m 보폭 요구(K1 실현가능 0.26~0.31 m) → **불가능**(gbatch_results:251-269, non-codex:251-283). **처방(케이던스-속도 커플링)은 H에 미반영** |
| **G3** | `속도가 0일때랑 0으로 수렴할 때랑 자세가 비슷함←이건 좋은 점. 왜 이러는지 찾아서 살려` | FB:664 | 도착 종착 자세 = 정지 자세 (RLKick 핸드오프) | 3회 (FB:664 / mp2:19-45 / gbatch:61) | **E·G·H 전부 (goal_reached 유지)** | **구현·보존.** `goal_reached` 흡수상태(FB:666-679). armB 안 멈춤 0.5%, E0 도달후이탈 0.1%(non-codex:33) |
| **G4** | `시작 속도를 randomize하거나, dist to goal에 노이즈를 추가하는 정도가 아니라 goal이 마구잡이로 변할 때(ball 관측이나 bt분기 때문에 shaking 하는 경우)에도 robust하게 걷기를 학습해야 함` | FB:246 | BT 분기급 큰 점프에 강건한 보행 | **7회** — FB:200,246 / mp2:631,844 / gbatch:199-203 / [지시 9] / hbatch-codex:74 | **E2 → G2 → H 전 arm(의무화)** | **구현, 효과 분리 실패.** `goal_bt_flicker` + `--stress jitter`(mp2:632-646). G1~G4 stress **낙상 0·직립 ~100%**(gbatch_results:376). 그러나 “필터링을 배웠다 vs 그냥 느리다”가 **분리 안 됨**(gbatch:201-202). H jitter fall 0.4219/env·min PASS이나 **angular p90 3.2 > 3.0 FAIL**(hbatch-codex:352-354) |
| **G5** | `그리고 외란은 무조건 더 줘야해. 단순히 팔만 걸리는게 아니라 로봇 두 대랑 부딪히는 경우도 많아. urdf있으니까 충격량 계산해서 학습에 반영해.` | FB:366 | URDF 충격량 계산해 외란 대폭 강화 | **9회** — FB:366,510,572 / mp2:441-457 / gbatch:194-197 / gbatch_results:287-300,368-384 / ebatch:308-315 / [지시 9,13] / hbatch-codex:125-145 | **E2 → G2/G3 → H 전 arm → M1(계획)** | **구현했으나 실제 전달량이 1/10이었다 — 최대 발견.** collision 40~150 N 설정(mp2:449)이 **Isaac Gym decimation 10 중 1번만 submit** → 실충격량 명목의 **≈1/10 ≈ 0.08 g**(non-codex:106,130 / hbatch-codex:127). H에서 substep 재submit으로 수정. H force **5 s survival 97.19% < 98% FAIL**(hbatch-codex:349) |
| **G5b** | `push 현실화: duration 1→3초로 늘리되 강도는 낮춤(15N→5N) <- 힘 키우자.` | FB:510 | armD의 약화를 되돌려 힘 증대 | 1회 | **E2/G2/G3/H** | 구현 “최대 힘 10배”(FB:517) — 단 위 1/10 버그로 실효는 미달 |
| **G6** | `팔 제발 내려. urdf수정하라고 했잖아. 팔은 그냥 사람처럼 중력방향으로 늘어트리게 만들고…` / `일단 팔뚝은 straight down, always 90deg로 하고,` | FB:303 · mp2:538 | T자 팔을 90° 수직으로 내릴 것 (**재지시**) | **8회, 최고 강도** — FB:215,303 / mp2:528,538 / gbatch:139,178,477 / [지시 19] | **E0(armsdown) → G3(armswing) → H(hbatch-codex urdf) → A스크린(A0/A1/A2)** | **구현·검증 후 H에서 역전됨.** E0 armsdown이 재학습 견딤, turn 2.7 cm 기여(mp3:43). **그런데 H가 팔 asset을 `hbatch-codex.urdf`로 교체 → yaw I_zz +32.4%, ego 반폭 +60%**(hbatch:46,60). **그 관통은 `collapse_fixed_joints:true`라 물리적으로 존재하지 않는 렌더링 결함**(hbatch:63-76) ⇒ **“물리적 이득 0에 동역학 +32.3% 지불”**(non-codex:389). **H 공통 붕괴의 유력 용의자** |
| **G7** | `팔뚝을 바꾸라는게 아니라 팔뚝을 shoulder yaw로 돌려서 팔꿈치를 살짝 돌리면 팔 끝(손)이 엉덩이 뒤로 가도록 하라고.` | **gbatch.md:139** | 팔꿈치+yaw로 손을 힙 뒤로 (**3번째 재지시**) | 3회 (FB:303 / mp2:538 / gbatch:139) | **G3(armswing URDF, 스크립트 팔)** | ⚠️ **두 번 “불가능” 오판 후 사용자가 옳았음이 확인됨.** “**말씀이 맞았고 제가 틀렸다. 됩니다. 구현했다**”(gbatch:140). T포즈 기준으로만 따진 실수 — 팔이 수직이면 축 역할이 바뀐다(gbatch:145-150). 적용: Elbow_Pitch +119.2°, Elbow_Yaw ∓90.5°, 손 COM x −0.031, I_zz −69%(gbatch:156-164). **G3는 성공률 0%로 실패**(gbatch_results:186) — 팔 스크립트가 용의자 3순위(gbatch_results:204-206) |
| **G7b** | `제발 그냥 하라고. 내가 하라고 목에서 피를 토하면서 말 하잖아.` | **gbatch.md:178** | 동적 후방 스윙을 더 미루지 말 것 | 1회, **문서 전체 최고 감정 강도** | **G3에서 최초 실행** | 동적 스윙 구현(`K1_locomotion_armswing.urdf`, 비학습 DOF 스크립트, obs 54 유지)(gbatch:172-177). 판정 신호는 **0.5 m/s 하드컷 대신 `(goal_dist<0.1)&(|v|<0.1)&(|ω|<0.3)`**(gbatch:185). **G3 실패 후 H/A/M 어디에도 없음 → 소멸** |
| **G8** | `속도의 커플링을 의도적으로 줄 수는 없어도 커플링이 잘 나타나는 path를 만들거나 해서 병진 회전운동이 커플링되어있어도 정확하게 따라가는 dist도 training해야한다고 강력하게 생각함.` | mp2:300 | 병진-회전 커플링 상황에서도 정확 추종 | 2회 (mp2:300 / gbatch:252) | **G1(속도×곡률 그리드 + serpentine)** | ❌ **그리드가 구조적으로 무효였다.** 30/30 칸이 **약 1분 만에 전부 활성, 성공률 98%** — 승급 기준 `path_lag < keepup_gap 2.0 m`인데 **leash가 `path_lag ≤ 1.44 m`를 이미 보장**해 실패 불가(gbatch_results:107-132). 추가로 `path_lag=max(gap−lookahead,0)`가 **단측**이라 캐럿 추월(=floor 붕괴)이 “완벽”으로 기록(non-codex:109). ⇒ **G1의 실제 조건 = 폐기했던 E3 그 자체**(gbatch_results:132) |
| **G9** | `lookaheadpoint가 계속 이어지는듯한(trajectory를 following하고있는 듯하게)goal을 줄 수는 없나? … 커다란 8자나 spiral, star, random trajectory 위를 lookahead point(0.5-3.0m)를 두고 빨리 따라가도록 … 별로면 말고.` | FB:176 | 연속 공급 waypoint = path 모드 | **5회** — FB:176 / mp2:183 / gbatch:26,478 / [지시 6] | **E1 → G1 → H(path 0.35 고정)** | ⚠️ **E1 측정 전량 무효 → G1이 dwell로 수리 → H에서 다시 최대 실패원.** `lookahead_m`이 데드코드(mp2:186), floor 도입 후 재평가 오염(mp3:55-67). G1 dwell로 turn 이탈 100%→0%, 52.1→5.5 cm(gbatch_results:96-103) = **G 배치 최대 성과**. 그러나 H clean **낙상의 89.6%가 path**(hbatch-codex:356), path fall 62.69/1000 = 게이트의 12.5배 |
| **G9b** | `path 어쩌구를 자세히 설명해봐. 이게 bt(check sim2real branch)에서 잘 활용될 수 있는 구조야?` | **gbatch.md:478** | path 구조가 실제 BT에서 쓸 수 있나 | 1회 | **실기(missions.md carrot)** | **미답변 (gbatch.md에 이 질문에 대한 `[답변]`이 없다).** 실기에서는 별도로 “2 m radial carrot”으로 구현됨(missions.md:207-227) — path 모드와 다른 메커니즘 |
| **G10** | `e0좋은거 인정. 근데 symmetry loss에 대해서 다시 평가해봐. 그냥 right hip yaw가 몸 중심으로 치우쳐있어. 발 끝의 앞뒤는 잘 맞는데 중심 기준으로 대칭이 아님.` | **gbatch.md:64** | 좌우(중심 기준) 비대칭 해결 | **10회** — FB:215,581 / mp2:834,838,841,845 / gbatch:64,209 / [지시 15] / hbatch:139 | **E0(최초 활성) → G(계측) → H1/H2(mirror 증강) → M3(mirror off, 계획)** | ⚠️ **지표가 3번 틀렸고, 처치는 역효과.** ① `symmetric_coef`는 죽은 키(mp2:131) ② 지표가 **전후(x)축만 재서 좌우를 구조적으로 못 봄**(gbatch:68-72) ③ **`extras["v7"]`가 리포트에 배관 안 됨 = 값이 아무 데도 없었음**(gbatch:74-76, 210-217). 수정 후 실측: `feet_lat_offset` G1 **0.0002**(해결) 그러나 `bias_Ankle_Pitch` **7.8°**(악화)(gbatch_results:392-401). H1 mirror error p90 **0.080→0.150 (+87.5%)**, touchdown 좌우 bias **2.9→9.4 cm**, 기준 통과 **0/31**(hbatch-codex:397-398) = **직접 목표까지 악화** |
| **G11** | `어떻게 계산하는지 알려주고, 그리고지금 값을 보면 되잖아? 로그 다 줬잖아 제발 봐` | **gbatch.md:209** | 로그를 실제로 열어보고 계산식 공개 | 1회, 강한 질책 | **G(배관 수정)** | **“봤고, 그래서 없다는 걸 확인했다. … 로그를 안 본 게 아니라 로그에 없었다.”**(gbatch:210-221). 4개 report.json 전부 `v7extras=없음`. 계산식 공개(gbatch:223-234) |
| **G12** | (`[지시 8]`) `빠른 가속을 위해 상체가 기울어지는 것은 좋으나, 고속 안정성이 떨어진다.` | gbatch_results.md:241 (인용) · hbatch-codex.md:98 | 가속 lean은 허용, 고속 안정성만 확보 | 4회 — [지시 8] / gbatch_results:239-283 / non-codex:251-283 / hbatch-codex:96-109 | **H2(속도·가속 sigmoid gate)** | ❌ **H2 처치가 최종 정책에 들어가지 않았다(model_0 선택).** cruise pitch/roll/ωxy 7.5°/5.5°/1.30은 **baseline 값**이고 **cruise coverage 2.756% < 5%로 인증 불가**(hbatch-codex:419). 그리고 **케이던스 산술이 H2 실패를 예측한다** — 물리적으로 불가능한 보폭 요구에 대한 유일한 순응이 감속(non-codex:276-278). **케이던스-속도 커플링은 H에 미반영** |
| **G13** | (`[지시 13]`) `기존 외란을 조사했는데 충분히 큰 힘이었고 그래도 안정적이었다면 이런 하드코딩은 필요 없다. 애매하면 gait만 만진 hbatch3를 따로.` | gbatch_results.md:289-290 (인용) | 조건부 지시 — 애매하면 H3 단독 arm | 3회 — [지시 13] / gbatch_results:287-311 / hbatch-codex:111-124 | **H3(단독 격리)** | **조건 판정: 애매 → H3 생성(설계 모범, 평가 A등급 non-codex:82). 결과 FAIL.** heel target share가 **iteration 0의 2.508%가 최고**, 학습할수록 하락(hbatch-codex:440-442). “**직접 touchdown target도 악화**”(hbatch-codex:492). 전제인 “충분히 컸나”는 1/10 버그로 **성립 안 함** |
| **G14** | (`[지시 22]`) 정옆/정뒤 목표에서 속도가 떨어진다 / “같은 방향인데도 가끔 빠름” | gbatch_results.md:315 · hbatch-codex.md:529 | 측면·후진 감속 원인 규명 | 4회 — [지시 22] / gbatch_results:315-364 / non-codex:188-227 / hbatch-codex:362-367 | **H(`--goal_pattern lateral/reverse`)** | ⚠️ **Codex 설명이 반증됐고, 진짜 원인의 반증 축이 H에 없다.** Codex의 constellation 설명은 **E0가 lateral을 straight보다 28% 빠르게** 해서 반증(non-codex:195-205). 진짜 원인 = **path 학습의 전방 prior**(path 노출과 단조: E0 +35% → E1 −4% → V7 −22%). **그런데 H 4 arm 전부 `path: 0.35` 고정 → 가설 반증 축 없음**(non-codex:220-223). 실측: lateral 0.5 m/s 99.87% 도달하나 1.0 m/s는 **9.57%**(hbatch-codex:364) |
| **G15** | `battery가 low하면 protect하는것도 들어가있어?` | FB:286 | 배터리 저전압 보호 | 2회 (FB:286 / mp2:684 / gbatch:239-241) | **미시도** | ❌ **미구현 확정, 4개 문서 연속 유예.** “아니, 전혀 안 들어가 있다”(FB:288) → mp2:684 우선도 “중” → mp3 없음 → gbatch:239 “**배터리 랜덤화는 여전히 미구현**” → H/A/M 없음 |
| **G16** | `관절이 일정 각도를 넘어가면 soccer mode 등에선 protect mode로 진입 했었는데 custom mode에서도 그러는지 조사가 필요` | mp2:339 | CUSTOM에서 PROTECT 걸리는지 실기 확인 | 3회 (mp2:339 / gbatch:50-51 / DEPLOY_GUIDE:255-265) | **미시도(실기 검증) / 실기 코드에 부분 반영** | **조사 완료, 실기 검증 미착수.** “CUSTOM에서 PROTECT가 안 걸린다고 가정하는 게 안전”(mp2:355), “**우리 페널티 + 우리 런타임 guard가 유일한 방어선일 수 있다**”(gbatch:50-51). 실기 deploy에 watchdog은 들어감(roll/pitch 1 rad, LowState 0.2 s, NaN — DEPLOY_GUIDE:255-265). **학습측 `torque_limits` 랜덤화는 여전히 미구현** |
| **G17** | (`[지시 14]`) joint DR 점검·강화 | non-codex.md:83 | encoder/target zero offset 등 관절 DR 강화 | 2회 — [지시 14] / hbatch-codex:147-163 | **H0/H1(encoder ±0.015~0.025, target ±0.010~0.020) → M2(계획)** | **구현.** 빠져 있던 것 3종 식별(episode-constant encoder bias, motor target offset, backlash)(hbatch-codex:155-160). H1에서 mirror와 **함께** 켜서 **교란 확정**(hbatch-codex:579) → M2가 단독 분리 예정 |
| **G18** | (`[지시 17]`) 시뮬레이터 시점 영상 + 외력 화살표 | non-codex.md:86 | top-view 대신 시뮬 시점, 외력 시각화 | 3회 — [지시 17] / gbatch:26 / hbatch-codex:223-234 | **H(전 arm 영상)** | **구현, 평가 A.** 신규 로거 없이 카메라 행렬로 3D 재투영(non-codex:86,419). 400 frames, force arrow 75, path carrot 400(hbatch-codex:457) |
| **G19** | `smooth turn을 학습히키는데 trajectory와 stack of waypoints가 사용되나? … 확인해서 시각화 작업 해둬 video로 보게(isaac gym rgba? rgb문제로 터졌었는데 이번엔 관련 조항 확인해서 제대로 해)` | **gbatch.md:26** | SmoothTurn 메커니즘 확인 + RGBA 안전한 영상 | 2회 (gbatch:26 / [지시 17]) | **G4** | **답변·구현.** waypoint stack `(num_envs,4,3)`, RGBA 버그는 커밋 `5493840`에서 이미 수정(gbatch:27-46). `draw_goal_sequence` 추가. **G4는 34%만 학습, 낙상 1016회로 판정 보류**(gbatch_results:213-235) |
| **G20** | `video log에서 constellation error말고 body velocity(xyw)도 표기해주고, position error도 표기해줘` | FB:605 | HUD에 v/vx/vy/ω, 거리, heading, 외란 | 3회 (FB:347,366,605) | **E·G·H 전부** | **구현 완료**(mp2:604-612) |
| **G21** | `-x성분속도가 현저히 느림` | FB:618 | 후진 속도 개선 | 2회 (FB:618,742) → [지시 22]로 승계 | **H(`--goal_pattern reverse`)** | **측정됨, 미해결.** reverse 0.5 m/s 100% 도달(p90 0.62 s)이나 **1.0 m/s는 23.41%**(hbatch-codex:365) |
| **G22** | `연속적인 goto를 보기 힘듬(data dist 문제)` | FB:633 | 연속 goto가 기본 동작이 되게 | 2회 (FB:633 / mp2:183) | **E1 → G1 → H** | 구조적 원인 확인(29.5% 이동거리 0, FB:639). path로 해결 시도했으나 **path가 낙상의 89.6%를 만듦** |
| **G23** | (`[지시 6]`) `H0 = current bests` | non-codex.md:75 | H0를 현재 최고들의 통합 기준선으로 | 1회 | **H0** | ❌ **평가 C.** “H0가 통합 기준선이 아니다 — **10레버 묶음**이다”(non-codex:165-181). H0 clean 위치 median **7.25→17.16 cm(+136.5%)**, never-arrived **12.91→59.54%**(hbatch-codex:377-380) = task collapse |
| **G24** | (`[지시 7]`) H1/H2 보수적 실험 | non-codex.md:76 | 레버를 보수적으로 하나씩 | 1회 | **H1/H2** | ❌ **평가 B−.** H2는 stability+force schedule+flicker **3레버 묶음**(hbatch-codex:90). H1은 mirror 2항+joint DR 동시(hbatch-codex:579) |
| **G25** | (실기) mission 1–5 정의 — 제자리 회전 3회 / ±3 m 전후진 3회 / ±2 m 좌우 3회 / 반지름 6 m 원 위 random 4점 / 1 m 간격 ㄹ자 9 m | missions.md:190-205 (사용자 지정 순서, missions.md:91) | 실기에서 돌릴 5개 미션 | 1회 | **실기(코드 완료, 실행 미완)** | **BT/FSM/토픽 코드 완료, 실행 차단.** “`.pt` 미export·robot 미복사이므로 export/copy/hash/E0 smoke 전에는 실행 금지”(missions.md:46). hoist 안전 gate 미통과 |

---

## B. PROCESS — 어시스턴트가 일하는 방식에 대한 요구

| # | 원문 인용 | 파일:줄 | 무엇을 요구했나 | 반복(전체) | **배치** | 현재 상태 |
|---|---|---|---|---|---|---|
| **P1** | `이 파일의 모든 피드백과 지시사항을 읽고 답변과 수정사항을 하나도 빠짐없이 아래에 댓글로 달아` | FB:2 | **하나도 빠뜨리지 말고** 인라인 답변 | 3회 (FB:2 / [지시 20] / [지시 12]) | E·G·H 전부 | **이행.** FB 50개 불릿 전부 답변. [지시 20] 평가 A “누락 없음”(non-codex:89). **예외 1건: `gbatch.md:478`(path/BT 질문)에 답변 없음** |
| **P2** | `변인통제 후 실험을 할 수 있는 버전을 만들어. v7 4가지 학습 모드 중 가장 평이한거에서 변화시키면 되겠지?` | mp2:741 | 한 번에 한 변수만 | **9회** — FB:432,448,473 / mp2:701,741 / gbatch:106 / ebatch:180 / non-codex:64,165 / hbatch:109 | E·G·H·A/M | ⚠️ **약속했으나 5번 위반, 매번 사후에 발견.** armD 12레버 → E2 7레버 → G2 7레버(재발) → **H0 10레버**(non-codex:165-179) → H1 mirror+DR 동시. “E2·G2 두 번 같은 실패”(gbatch_results:436), “**세 번째 반복 위험**”(non-codex:429). A/M 스크린이 처음으로 셀당 정확히 1레버 |
| **P3** | `이거 언제 구현할거야? 미구현이면 일단 스케쥴은 하라니까. 언제 어떤 학습에서 **가 확인이 되면 그 때 진행하겠다. 이렇게 돼야지.` | mp2:538 | 미구현 항목에 조건부 일정 명시 | 2회 (mp2:538 / gbatch:128-130) | G(§3 “사용자 강조 순서” 정렬) | ⚠️ **형식 이행 → 조건 충족 → 실행 안 됨.** 조건(E0 낙상≤37 & 위치≤5cm, mp2:592)이 **mp3:30-33에서 충족**됐으나 mp3의 G 배치에 팔 항목 없음. gbatch.md §3이 “**사용자 강조 순서**”로 8항목을 다시 정렬한 것이 이 지시의 직접 산물 |
| **P4** | `모든 의도에 대한 디버깅이 가능한 로거/eval을 만들어뒀는지 점검해보고 … 이건 앞으로 어떤 학습을 하든 꼭 병행할수박에 없도록 만들어` | mp2:810 | 계측 감사 + **잊을 수 없는 강제 구조** | **7회** — mp2:687,810 / gbatch:209,372-390 / [지시 3,10,17] | G·H | **가장 잘 이행된 요구.** 8개 의도 중 3개 계측 불가 → 보강(mp2:800). `smoke_v7.py`(27검사) 학습 전 게이트 + `train_and_eval.sh` 자동 eval(mp2:806). H는 **5단계 smoke(STATIC/PATH_MECHANICS/TRAIN_UPDATE/DISTURBANCE/VIDEO) + atomic marker + SHA 대조**(hbatch-codex:240-246). [지시 10] 평가 **A+**. **80개 artifact SHA 재계산 mismatch 0**(hbatch-codex:304) |
| **P5** | `왜 설계만 한거야. 테스트 해볼 계획도 함께 세워야지` | mp2:687 | 구현했으면 판정 기준까지 | 2회 (mp2:687 / [지시 13]) | G·H | **이행.** stress jitter 판정 4단계(mp2:688-698). H는 arm별 채택 기준표(hbatch-codex:286-296) |
| **P6** | `이 문단을 12살 아이에게 설명하듯이 분해해줘` / `이거 왜그런지 12살 아이에게 설명하듯 elaborate please` / `이 그리드에 대해 더 설명해줄 수 있나?` / `이렇게 안 해도 되는 이유가 뭐더라?` / `이게 뭐야. damping이던가?` / `뭐가 학습이 안 됐다는건지 잘 모르겠어` | mp2:471,705 · gbatch:252,286,302,321 | 설명 눈높이 / 개념 재설명 | **6회** | E·G | **전부 이행.** 보물찾기(mp2:473), 시계바늘(mp2:706), 그리드(gbatch:253-278), heading sin/cos(gbatch:287-299), armature vs damping(gbatch:303-318), E3 실패의미(gbatch:322-334) |
| **P7** | `빨리 반영 안하고 뭐하냐 병신아` / `제발 그냥 하라고. 내가 하라고 목에서 피를 토하면서 말 하잖아.` / (인용) `제발 한번에 제대로 해` | mp2:413 · gbatch:178 · gbatch:618 | 즉시 실행 / 한 번에 제대로 | **3회, 최고 감정 강도** | G·H | **1건은 전달 실패(이미 구현돼 있었음, mp2:422), 2건은 실제 미이행이었고 이후 실행.** “**늦어서 죄송하다. 두 번이나 ‘불가능’이라고 했는데 검증이 부족했다**”(gbatch:170). gbatch §8-8이 “제발 한번에 제대로 해”에 대한 응답으로 **패치 대신 재작성 3건**을 선언(gbatch:617-618) |
| **P8** | (`[지시 12]`) `hbatch.md 작성` / `그리고 새로운 버전들의 설명서는 masterplan2.md에 새로 작성한다.` | non-codex.md:81 · FB:3 | 배치별 설계 문서 작성 | 2회 | 전부 | **이행.** masterplan2/3 → gbatch.md → hbatch.md/hbatch-codex.md |
| **P9** | `논문과 사례들을 뒤져서 이 문제를 잘 해결한 사례를 가져오고, 편법 말고 legit하게 해결한 사례를 꼭 찾아와서 적용해. 없으면 스스로 고민해보고 댓글 달아.` | mp2:256 | 선행연구 근거, 편법 금지 | **6회** — mp2:183,256 / FB:151,807 / [지시 18] / hbatch-codex:496-509 | E·G·H | **이행 + 자기반증.** Margolis RSS2022로 자기 제안 E3 철회(mp2:269, gbatch:121-123). Rudin IROS2022, ETH pure-pursuit, SmoothTurn(mp3:126), sim2real 10항목(hbatch-codex:496-509, 평가 B+) |
| **P10** | `real log 없이 시뮬에서만 real에서도 낭낭하게 적용가능한 제약법에 관한 논문 리서치해서 수정하진 말고 나에게 보고만 해.` | mp2:372 | 조사만, **코드 건드리지 마라** | 1회 (명시적 no-op) | 미시도(의도적) | **이행.** “지시대로 코드는 건드리지 않고 조사 결과만 보고한다”(mp2:373) |
| **P11** | (인용) `공통적인 타스크만 우선 해결` | **hbatch.md:28** | H의 개별 레버보다 공통 붕괴를 먼저 | 1회 | **A 스크린(계획)** | **이행 중.** “처치가 무엇이든 결과가 같다 = 공통 부분이 레버 효과를 완전히 덮었다. 이 상태에서 외란·joint DR·mirror loss를 다시 재는 것은 의미가 없다. **사용자 지시 ‘공통적인 타스크만 우선 해결’이 정확히 이 지점이다**”(hbatch:26-28). Codex의 M셀은 **용의자 asset이 4셀 전부에 하드코딩**돼 있어 실행 불가(hbatch:90-98) → A 스크린 4셀로 재설계 |
| **P12** | `그러면 버전별 의도에 따라서 버전을 통폐합한다.` | FB:420 | 버전 통폐합 | **7회** — FB:420,432,448,473,519 / [지시 2] / mp2:83-89 | E·G·H | **이행.** armA 폐기/armB 채택/armC 삭제/armD 해체/v3 흡수(mp2:83-89) |
| **P13** | `각 버전의 의도, 테스트하고자 하는 것, 리워드함수의 특징, 알고리즘의 특징을 먼저 조사하고 … 댓글로 간략히 정리한 후에 내 video feedback을 확인한다` | FB:400 | 순서 지정(의도 정리 → 영상 확인) | 2회 (FB:400 / [지시 2]) | E·G | **이행.** 버전별 레버 대조표(FB:402-417), G 배치는 **의도-config 불일치 발견**(G3 grid OFF, G4 강외란 ON)(gbatch_results-codex:15-20) |
| **P14** | (`[지시 3]`) 데이터 유효성 분류 | ebatch.md:40 · non-codex.md:72 | 어느 숫자가 유효한지 먼저 분류 | 3회 — [지시 3] / ebatch:40-77 / gbatch_results:36-50 | E·G·H | **최고 평가(A+).** 106행 CSV `v7-data-validity-codex.csv`. **E 배치 절반이 무효**(base config로 평가), 속도 지표 15.7~17.6% 오염(ebatch:259) |
| **P15** | (`[지시 19]`) URDF 팔 겹침 — **이중검사** 요구 | non-codex.md:322 | 겹침 검사를 두 번 하라 | 1회 | H·A | **이행했고 그 이중검사가 자기 오류를 잡았다.** “1차 시도는 내 근사가 틀려 실패했고, 그 실패를 이중검사가 잡았다”(non-codex:322). 190점 부분표본 → 7050 전정점 재검증에서 **권고안이 7.9 mm 관통 잔존**으로 무효화(non-codex:369-373) |
| **P16** | (`[지시 11]`) 모델/하네스 적절 활용 · `[지시 10]` train+eval 단일 하네스 | non-codex.md:79-80 | 서버 GPU까지 실제로 완주 | 2회 | G·H | **이행(A).** H는 서버 실행·결과 반입·hash 검증까지 완주 |
| **P17** | `[요약] GPU: 학습 전 항상 nvidia-smi / 두 GPU 동시 점유 금지 / 충돌 시 양보 / $HOME 안 씀 / 남의 파일 안 지움` (사용자 상시 규칙, verbatim 아님) | MASTERPLAN.md:26-41 | 공유 서버 4대 원칙 **매번 반복 적용** | 4회 (MP:26-41,440 / gbatch:48-49 / hbatch-codex:7-10) | 전부 | **이행.** Codex도 “다른 사용자의 workspace는 열람하지 않는다”(hbatch-codex:7). ⚠️ **GPU 2장 사용은 mp3:107 이후 관행화** — 사용자 재허용 기록 없음 |
| **P18** | `이런 분석 좋다` / `의도한대로 한거 맞음` / `e0좋은거 인정` | mp2:701 · mp2:528 · gbatch:64 | (긍정 확인) | 3회 | — | 승인 반영 |
| **P19** | `이건 별로임 버려.` | mp2:737 | §6 표의 한 항목 폐기 | 1회 | — | ⚠️ **폐기 내용이 소실.** “§6 표에서 삭제했다”(mp2:738) → 표 번호가 2,3,4,5,11,7,8,10으로 **9번 결번**. 무엇을 버렸는지 추적 불가 |
| **P20** | (실기) `맥북 터미널 → 미션 번호` · `실시간 디버깅 토픽` · `BT를 mission 수행용으로` · `walk(E0)` | MISSION_READINESS_REVIEW.md:40-44 | 실기 운용 4대 요청 기능 | 4회 (+missions.md:91) | **실기** | **3/4 코드 완료, 1/4 차단.** `missionctl.sh N`(Mac에 ROS2 불필요) ✔ / status·telemetry·goal_pose·goal_rel·policy_debug ✔ / `locomotion_test.xml` + velocity 제어기 완전 제거 ✔ / **walk(E0): `.pt` 서버 export + 로봇 복사 미완**(REVIEW:44) |

---

## C. 철회 / 폐기 / 소멸 항목

### C-1. 사용자가 **스스로 철회하거나 수정한** 요구

| 항목 | 원래 | 철회·수정 | 파일:줄 |
|---|---|---|---|
| 팔 스윙 트리거 0.5 m/s 하드컷 | `body속도가 0.5m/s이상이 되면`(FB:303) | `이건 0.5로 자르지 말고, goal reached랑 같이 판단하는건 어때?` | FB:303 → **mp2:538** → 최종 채택 gbatch:185 |
| 팔 splay 80° | (어시스턴트 제안) | `일단 팔뚝은 straight down, always 90deg로 하고` | mp2:538 |
| `shoulder pitch나 roll은 건들지 말 것` | FB:303 | `의도한대로 한거 맞음` — 고정 마운트 각도 변경은 **사용자가 직접 허용** | FB:303 → mp2:528 |
| 팔꿈치 “불가능” 판정 | (어시스턴트가 2회 “기구학적으로 불가능” 선언, mp2:560) | `팔뚝을 바꾸라는게 아니라 …` → **사용자가 옳았고 구현됨** | mp2:560 → **gbatch:139 → gbatch:140** |
| `armature: 0.02` 단독 검증 | mp2 보류 항목 | **사용자 판단으로 폐기** | gbatch.md:301 |
| §6 표의 1개 항목 | (내용 미상) | `이건 별로임 버려.` → **원 내용 소실** | mp2:737 |
| armA / armC / armD / v3 단독 | 각 버전 유지 | `b만 살리고` / `c는 삭제` / `버전 D는 삭제하고` / `모든 자원을 armA/B에` | FB:432,448,473,519 |

### C-2. 현실·물리적 제약으로 폐기 (어시스턴트 판단)

| 항목 | 사유 | 파일:줄 |
|---|---|---|
| heading sin/cos 2채널 | obs 54→55 → warm start 전멸 | FB:42-53 / mp2:704-735 / gbatch:285-299 |
| Wh/m 에너지 제약 | 실기 전류·전압 로그 부재 | FB:278-284 / gbatch:336 |
| 거리 커리큘럼(v3) | 속도 목표와 정반대 | mp2:125-127 |
| PD 게인 200/5 | armD 붕괴의 “주범”. 실기 deploy에서도 **frozen 100/2로 정정** | FB:491 / MISSION_READINESS_REVIEW:11 |
| 200 Hz 물리(armC) | 주 게이트 전패 | FB:450-471 / gbatch:337 |
| E3 무커리큘럼 | Margolis ablation이 “학습 자체가 안 됨”으로 반증 | mp2:269 / gbatch:320-334 |
| 진행방향 정렬 보상(③) | “**게걸음은 RoboCup에서 실제로 유용한 기동**이고 금지하면 안 된다” | gbatch_results.md:362-364 |
| M00/M10/M01/M11 mirror 4셀 | 실행 전 폐기 — G1이 이미 `symmetry_coef=0.5`였고 H0가 pure control이 아님 | hbatch-codex.md:620-624 |
| heel-ahead reward | “탈락시켰다. 모든 M-cell에서 scale 0” | hbatch-codex.md:527 |

### C-3. 🔴 앞선 문서에 있다가 뒤 문서에서 조용히 사라진 것 — 망각 위험

| # | 항목 | 마지막 등장 | 이후 상태 | 위험 |
|---|---|---|---|---|
| 1 | **팔 동적 후방 스윙** | gbatch:172-186 (**G3에서 최초 구현·실행**) | G3 실패 후 **hbatch/hbatch-codex/non-codex/A·M 전부 언급 없음.** 실기 가이드는 오히려 “**팔을 흔드는 script를 켜지 않는다**”(DEPLOY_GUIDE:278) | 🔴 “목에서 피를 토하면서” 요구한 항목이 실패 1회 후 소멸 |
| 2 | **케이던스-속도 커플링** | non-codex:280-283, gbatch_results:275 (“✅ **H0/H1/H2 전부**”로 채택 선언) | **H0–H3 config 전수에 `gait_frequency: [1.8, 2.4]` 그대로**(non-codex:238). **채택 선언이 실행되지 않음** | 🔴 최상 — G2b/G12 두 요구의 유일한 처방 |
| 3 | **정확도 레버** (`goal_reached` 등급화 / `heading_near_goal` / `constellation_radius` 0.5) | non-codex:228-249, gbatch_results:353-364 | **H0–H3 4/4 동일, `heading_near_goal` 4/4 꺼짐**(non-codex:237). “**정확도 레버를 하나도 건드리지 않았다**” | 🔴 최상 — 5 cm 게이트 복구 수단 0개 |
| 4 | **배터리 보호 / torque_limits 랜덤화** | gbatch:239-241 | H/A/M 없음 | 🟠 사용자 직접 질문 |
| 5 | **런타임 guard 자체 구현(학습측)** | mp2:685(우선도 높음), gbatch:50-51 | 실기 deploy watchdog으로 일부 대체, 학습측 없음 | 🟠 안전 |
| 6 | **MaxSpeed 상한 단독 탐색** | FB:94-97,148 / mp2:681(높음) | mp3:84 “미측정”으로만 잔존, 이후 소멸. **1.3 m/s 목표의 합리성을 판정할 유일한 수단** | 🔴 최상 |
| 7 | **`path` dose를 축으로 (0.35→0.15)** | non-codex:225-226, 409 (요구 델타 #5) | **H0–H3 4/4가 0.35 고정** → [지시 22] 반증 축 없음 | 🟠 상 |
| 8 | **H0 분할(H0a/H0b)** | non-codex:183-186, 405 (요구 델타 #1) | H는 그대로 10레버로 실행 → **네 arm 모두 붕괴** | 🔴 예측이 적중했는데 반영 안 됨 |
| 9 | **milestone 0** (K1 ParameterWalk 베이스라인 재현) | MASTERPLAN:53,81 / UPSTREAM:30 | 영구 미착수. `parameter_walk.py` 디버그 코드 그대로 | 🟡 |
| 10 | **milestone 4** (export + MuJoCo 검증) | MASTERPLAN:57,293 | MuJoCo sim-to-sim은 **끝까지 안 함**. export는 실기 경로로 대체되었으나 **`.pt` export조차 미완**(missions.md:46) | 🔴 최종 목표 경로 |
| 11 | **v4/v5/v6 (기상·킥·낙법)** | MASTERPLAN:720-770 (전체 설계·검증 상태) | **mp2 이후 전 문서에서 0회 언급.** K1_HIST:14 “구현 또는 계획 단계” | 🟠 완전 실종 |
| 12 | **`arrival_hold`** | MASTERPLAN:516-519 (평가 지표까지 설계) | goal_reached+각속도로 사실상 대체, **명시적 폐기 기록 없음** | 🟡 |
| 13 | **에피소드 30초 유지(사용자 선택)** | MASTERPLAN:342 | 재확인 없음 | 🟡 |
| 14 | **`goal_reach_radius` 0.1 m ↔ 게이트 5 cm 충돌** | MASTERPLAN:520-521, ebatch:289-306 | 각속도 조건만 추가. **dwell/heading 조건 미반영, 공 내부 기울기 미구현** | 🟠 게이트 신뢰성 |
| 15 | **사용자 영상 피드백 미응답 3건** | mp2:834(E0 대칭)·837(E1 steady-state error “**그냥 offset to goal 이 학습된 것 같은데**”)·844(V7 “**jitter에 robust한 특성만 살려서 다음 version에 어떻게 반영할 수 잇을까?**”) | mp2:834는 gbatch:64에서 재지시 후 응답. **mp2:837·844는 끝까지 미응답** | 🔴 최상 |
| 16 | **`report.json`에 env 코드 git SHA** | mp3:231 “**미구현, 다음 작업 1순위**” | ✅ **완료** — `ENV_CODE_SHA` 기록·비교(gbatch:386), H는 effective test-config SHA까지(hbatch-codex:91) | ✅ 해결 |
| 17 | **`segments.csv` 항상 저장** | gbatch_results:50 “**H 배치에서 반드시 저장할 것**” | ✅ 완료 (hbatch-codex:320) | ✅ 해결 |
| 18 | **`gbatch.md:478`** path/BT 구조 질문 | gbatch:478 | 답변 없음 | 🟠 |

---

## D. 사용자가 말한 **모든 수치 기준·게이트** 총람

| # | 수치 | 원문/맥락 | 파일:줄 | 배치 |
|---|---|---|---|---|
| N1 | 위치 median ≤5 cm / p90 ≤10 cm | `\| 최종 위치 오차 (median / p90) \| ≤ 5 cm / ≤ 10 cm \|` | MASTERPLAN:15 | E·G·H |
| N2 | heading ≤10° | `\| 최종 heading 오차 \| ≤ 10° \|` | MASTERPLAN:16 | E·G·H |
| N3 | 낙상 0% | `\| 넘어짐률 \| 0% \|` | MASTERPLAN:17 | E·G·H |
| N4 | Δx∈[-2,2] m, Δy∈[-1.5,1.5] m, Δθ∈[-π,π] | `목표 샘플링: Δx ∈ [-2, 2] m …` | MASTERPLAN:8 | 전부 (실기 clamp까지 — DEPLOY_GUIDE:432-433) |
| N5 | 목표 재샘플 4~8초 | `에피소드: 4~8초마다 목표 재샘플링` | MASTERPLAN:10 | E·G·H |
| N6 | **1.3 m/s** | `body속도는 1.3m/s정도 나와야함.` | FB:347 | E1·G1·H |
| N7 | **1.4~1.5 m/s** | `가능하다면 1.4, 1.5m/s 정도까지로 올리고 싶은데` | FB:109 | G1 |
| N8 | lookahead 0.5~3.0 m | `lookahead point(0.5-3.0m)를 두고 빨리 따라가도록` | FB:176 | E1·G1 |
| N9 | 반경 3 m 원, 50 Hz 랜덤 | `반경3미터 원 안에서 50hz로 random sample되는 항목` | FB:200 | E·G·H (`--stress jitter`) |
| N10 | 0.5 m/s (팔 스윙 트리거) | `body속도가 0.5m/s이상이 되면` | FB:303 | **철회**(mp2:538) |
| N11 | 팔뚝 90° 수직 | `팔뚝은 straight down, always 90deg` | mp2:538 | E0·G3 |
| N12 | push 힘 증대 | `duration 1→3초로 늘리되 강도는 낮춤(15N→5N) <- 힘 키우자.` | FB:510 | E2·G2·H |
| N13 | maxvel 10초 지속 | `maxvel을 10초정도 뽑아냈을 때 안정성` | FB:852 | 미시도 |
| N14 | 실기 미션 거리 — ±3 m 전후진 / ±2 m 좌우 / 반지름 6 m 원 / 1 m 간격 ㄹ자 9 m | missions.md 표 | missions.md:194-205 | 실기(미실행) |
| N15 | 실기 도달 threshold 10 cm / 6° | `goal_reached_xy_m: 0.10`, `goal_reached_theta_deg: 6.0` | missions.md:138-139 | 실기 |

### D-2. 어시스턴트가 파생시킨 주요 수치 (사용자 게이트 종속)

| # | 수치 | 파일:줄 |
|---|---|---|
| N16 | 정지 판정 `\|v_xy\|<0.1` (+`\|ω\|<0.3` 추가) | mp2:38,431 |
| N17 | `goal_reach_radius = 0.1 m` — ⚠️ N1과 충돌 | mp2:39 / non-codex:233 |
| N18 | 낙상 게이트 프로토콜 = 256 env × 120 s에서 0회 | MASTERPLAN:315-318 |
| N19 | 속도 커리큘럼 grid: speeds [0.3…1.8], curvatures [0.25…1.0], 시드 (0.3,0.25) | mp2:286 / gbatch:269-273 |
| N20 | leash 3.5 m, keepup_gap 2.0 m — ⚠️ **구조적으로 실패 불가** | mp2:237,250 / gbatch_results:120-128 |
| N21 | dwell `duration 3–5 s / interval 12–24 s` (물리에서 역산) | gbatch:519-525 |
| N22 | collision 40~150 N / 0.05~0.15 s; support 3~15 N / 1.5~3 s → H는 40~100 N, **실적용 2.4~10 N·s** | mp2:449 / hbatch-codex:140 |
| N23 | stress 판정: 낙상률 >0.5/env·분 실패, \|ω\| p90 >3 rad/s 경련 | mp2:639 |
| N24 | 관절보호 임계 = 한계의 85% | mp2:329 |
| N25 | H 채택 기준: G1 `5.52/7.42 cm, 2.54°` 악화 금지, never-arrived ≤1.5%, path speed ≥0.95 m/s, fall ≤5/1000 | hbatch-codex:288-292 |
| N26 | A 스크린 150 iter / M 셀 200 iter, fresh Adam 2e-6~5e-6 | hbatch:127-128, 198 |
| N27 | 실기 hoist 단계 A~F (goal 0 → 0.2 m → 0.5 m → 지면) | DEPLOY_GUIDE:296-302 |

### D-3. 🔴 수치 기준 **상호 충돌**

| # | 충돌 | 상세 |
|---|---|---|
| **X1** | **1.3~1.5 m/s (N6/N7) ↔ 과제 구조상 최대 요구속도 0.574 m/s** | N4+N5의 이론 상한 = 2.5 m/4 s = **0.625 m/s**(mp2:66-76). “**학습도 측정도 불가능했다**”. path 모드가 해소책이나 E1 무효·G1 그리드 무효·H 붕괴로 **끝까지 미확정** |
| **X2** | **1.3 (N6) ↔ 1.4~1.5 (N7) ↔ 커리큘럼 상단 1.6/1.8 (N19)** | 세 개의 서로 다른 목표치가 확정 없이 공존 |
| **X3** | **1.5 m/s ↔ 케이던스 물리 상한** | `gait_frequency [1.8,2.4]`에서 1.5 m/s는 **0.417 m 보폭** 요구, K1 실현가능 0.26~0.31 m ⇒ **명령 자체가 물리적으로 불가능**(gbatch_results:263 / non-codex:265-266). 속도 목표와 gait 설정이 **정면 모순**이며 처방이 미반영(C-3 #2) |
| **X4** | **낙상 0% (N3) ↔ 전 배치 실적** | 최선 E0 2회 / G2 0회(그러나 body p90 0.19 = “안 움직여서 살아남음”, gbatch_results-codex:52). H는 24.473/1000. mp3:230 “**외란을 켜는 G2/G3는 일시적 악화가 정상**”으로 사실상 유예 |
| **X5** | **`goal_reach_radius` 0.1 m (N17) ↔ 위치 게이트 5 cm (N1)** | “5–10 cm에서 보상 파밍”(MASTERPLAN:521). 정량: 10→5 cm는 **+0.005/step**, 10 cm 문턱 통과는 **+1.0/step = 200배**(ebatch:298-301). G1이 정확히 5.5~5.7 cm에 수렴. **처방(공 내부 기울기) 끝까지 미구현** |
| **X6** | **`constellation_radius` 1.0 ↔ 위치/heading 게이트** | `2r²=2`라 180° 회전 = 2 m 이동과 등가(FB:811). 처방 0.5~0.7 — **G·H 전부 1.0 유지** |
| **X7** | **속도(G1 계보) ↔ 정확도·안전(E0 계보)** | E0→G1: 위치 **2.03배 악화**, 낙상 **19배**, strict **−55.1%p**, 대가로 >1 m/s 체류 **5.9배**. “**속도를 6배 사서 정확도를 2배, 안전을 19배 팔았다**”(non-codex:155-159). H가 G1을 채택선으로 삼자 non-codex:431이 “**채택선은 G1이 아니라 E0여야 한다**”고 반대 |
| **X8** | **강건성 ↔ 속도 (3회 재현)** | E2 body p90 0.58→0.32(−44.8%), never-arrived 0→53.34% / G2 p90 0.19, >1m/s 0.5% / H0 never-arrived 12.91→59.54%. “**강건성 항만 켜면 정책은 항상 ‘덜 움직이기’로 도망친다**”(ebatch:233-234) |
| **X9** | **[지시 8] 가속 lean 허용 ↔ [지시 13] heel-ahead** | “가속하려면 접지점이 CoM보다 **뒤**여야 한다. 항상 앞을 요구하면 가속 자체를 금지”(gbatch_results:305-306) → 속도 조건부로만 해소 |
| **X10** | **GPU 1장 원칙 (P17) ↔ 2장 사용** | MASTERPLAN:31,440은 사용자 명시 허용 조건. mp3:107 “A6000 2장”, hbatch:144-158 “4셀 동시” — 재허용 기록 없음 |
| **X11** | **훈련 envelope ±2/±1.5 m ↔ 실기 BT 3 m lookahead** | MASTERPLAN:474-475 “3 m lookahead는 **OOD**”. 실기는 **2 m radial cap + per-axis clamp**로 타협(missions.md:216-224) — 순수 lateral carrot은 y=±2 m까지 나올 수 있어 **방향이 변형됨**(missions.md:223-224) |
| **X12** | **legacy `all_gates_pass`(5 cm, fall 0) ↔ H 전용 gate(5.52 cm)** | 동시 요구 시 H0가 구조적으로 FAIL — Codex가 명시적으로 분리(hbatch-codex:296) = **원래 게이트의 사실상 하향** |

---

## E. 종합 판정

**달성**: G1의 위치·heading(E0 2.72/5.01/2.52°) · G3 정지자세 · G6 팔 내리기 · G7 팔꿈치 후방(구현) · G18 영상 · G20 HUD · P1·P4·P5·P6·P8·P9·P10·P11·P12·P13·P14·P15·P16

**미달성/미측정**: G1 낙상 0 · G2/G2b 속도 1.3~1.5 · G4/G5 강건성 효과 분리 · G8 커플링(그리드 무효) · G10 대칭(역효과) · G12 고속 lean(미평가) · G13 heel(FAIL) · G14 측면감속(반증 축 부재) · G15 배터리 · G16 실기 PROTECT · G25 실기 미션(차단)

**구조적 문제 5가지**

1. **사용자 목소리의 소멸.** mp2 17개 + gbatch 9개 HTML 주석 → hbatch/hbatch-codex/non-codex/mp3 **0개**. 최신 문서일수록 사용자 원문을 상속하지 않고, `non-codex.md`의 「지시 1–22」 제목표만이 유일한 흔적이다.
2. **“한 번에 하나씩”이 5번 위반됐고 매번 사후에 발견됐다** (armD 12 → E2 7 → G2 7 → H0 10 → H1 2). A/M 스크린이 처음으로 셀당 1레버.
3. **사후분석이 정확히 예측한 처방이 다음 배치에 반영되지 않았다.** 케이던스 커플링(“H0/H1/H2 전부 채택” 선언 후 미반영), 정확도 레버 0개, path dose 축 부재, H0 분할 — 넷 다 `non-codex.md`가 요구했고 넷 다 H에 없으며, H는 네 arm 모두 붕괴했다.
4. **최고 정책이 배치를 넘을수록 후퇴했다.** E0(2.72 cm/낙상 2) → G1(5.52/38) → H(6.899/24.5‰, 전 arm이 warm-start를 못 이김). H 4개 arm 결과가 **byte-identical**(hbatch-codex:319-322) = 처치 효과 측정 자체가 성립 안 함.
5. **최종 목표 경로(실기 배포)가 여전히 `.pt` export 한 단계에서 막혀 있다** (missions.md:46, MISSION_READINESS_REVIEW:44). 이 단계는 사용자 본인이 수행해야 하는 항목으로 표시돼 있다(DEPLOY_GUIDE:417 `[사용자]`).
