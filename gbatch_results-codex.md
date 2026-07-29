# G-batch 결과 보고서 — Codex 전수 감사

범위: G1–G4 각 6개, 총 24개 파일. 모든 clean core report는 seed 0, 256 env × 120 s, env SHA `a7782bb…`의 own config 평가다.

## 결론

- **G군 내부의 배포 가능한 균형에서는 G1이 수치상 압승**이다. 위치, heading, strict success와 commanded-speed 추종을 함께 유지한 유일한 arm이다. raw path mean-speed median은 G3/G4 `1.212/1.190 m/s`가 G1 `1.038 m/s`보다 높고 body p90도 G4 `1.58`이 G1 `1.50 m/s`보다 높지만, 이들은 overspeed·정확도 붕괴·낙상을 동반하므로 “더 좋은 속도”가 아니다.
- 그러나 G1은 E0보다 위치오차와 낙상이 악화됐다. G1은 Hbatch의 speed warm-start이지 곧바로 배포 winner는 아니다.
- G2는 “안 움직여서 살아남는” 저속 collapse다. 외력 ON 평가가 없어 robust 성공으로 볼 수 없다.
- G3와 G4는 문서의 ablation과 실제 config가 다르다. G3는 G1+G2가 아니며, G4는 순수 SmoothTurn 실험이 아니다.
- 고속/path와 낙상의 연관은 강하지만, 기존 로그에는 pitch-vs-speed/acceleration이 없어 상체 lean이 직접 원인이라고 확정할 수 없다.

## 의도와 실제 modification 불일치

| 버전 | 문서 의도 | 실제 생성 config | 해석 |
|---|---|---|---|
| G1 | path floor+dwell + speed×curvature grid | 의도대로 grid ON, V7 robust/noise/protection OFF. 단 legacy 약 kick/push는 남음 | speed 가설의 유일한 usable arm |
| G2 | 강외란 + flicker 0.01 | path OFF, 강외란/full perception noise/flicker 0.01 | 여러 robust 레버가 묶여 원인 분리 불가 |
| G3 | G1+G2+protection+scripted arms | path 35%지만 **grid OFF**, flicker **0.004**; 강외란+protection+arms | 문서의 G1+G2 가설이 실제로 실행되지 않음 |
| G4 | E0+SmoothTurn | V8 base의 path 35%, 강외란, flicker 0.004, protection까지 동시 ON | 순수 SmoothTurn ablation 아님 |

## 권위 있는 clean core 수치

| 버전 | ckpt | 위치 med/p90 | heading | strict/loose | falls | body med/p90/p99 | 핵심 판정 |
|---|---:|---:|---:|---:|---:|---:|---|
| **G1** | 10700 | **5.52/7.42 cm** | **2.54°** | **34.22/97.78%** | 38 | 0.21/1.50/2.08 | 빠르지만 path 낙상 많음 |
| G2 | 7500 | 25.93/85.78 cm | 5.12° | 10.27/22.10% | **0** | 0.07/0.19/0.39 | 저속 collapse |
| G3 | 5500 | 49.00/63.20 cm | 5.88° | 0/0% | 17 | 0.19/1.51/1.89 | 통합 실패 |
| G4 | 3400 | 42.04/63.61 cm | 4.82° | 7.29/9.42% | **1016** | 0.17/1.58/3.02 | sequential catastrophic fail |

G1은 G2/G3/G4 대비 위치 median이 각각 78.7/88.7/86.9% 낮고, p90도 91.3/88.3/88.3% 낮다. heading도 최저이고 strict도 최고다. 따라서 “G1 압승”이라는 사용자의 인상은 G군 내부 비교에서는 맞다.

단 E0와 비교하면 G1은 median `2.72→5.52 cm`(2.03배), p90 `5.01→7.42 cm`, strict `89.29→34.22%`(−55.07%p), falls `2→38`이다. speed를 얻었지만 정확도·안전 비용이 크다.

## 버전별 상세 평가

### G1 speed

- waypoint 3019개, path 1613개.
- path segment mean-speed median `1.038 m/s`, commanded median `1.034 m/s`, final root-speed median `0.878 m/s`, lag p90 `0.469 m`.
- falls 38회 중 path 34회(89.5%). 완료+fall을 단순 attempt로 근사하면 path fall `34/(1613+34)=2.06%`, waypoint `4/(3019+4)=0.132%`, 약 **15.6배**다.
- position gate 5 cm는 `5.52 cm`로 FAIL이다. speed 증가는 명백하지만 지속가능 1.3–1.5 m/s 상한은 현 데이터로 증명하지 못했다.

위 `1.038 m/s`와 body p90 `1.50 m/s`는 G1이 실제로 빠르게 움직였다는 증거다. 그러나 이것을 “carrot을 올바른 간격으로 추종했다”는 증거와 혼동하면 안 된다.

grid `30/30 active`, success `0.9796`, 기존 `path_lag/keepup`은 숙련 증거로 사용할 수 없다. 당시 `path_lag=max(gap-lookahead,0)`였으므로 robot이 carrot을 추월해 `gap<lookahead`가 된 floor 붕괴는 오히려 `lag=0`의 완벽한 성공으로 기록됐다. promotion은 이 one-sided 값에 `2 m`라는 느슨한 기준을 적용했고 dwell도 섞였다. raw gap median은 `0.622 m`지만 각 step의 sampled lookahead와 짝지어진 값이 없어 이 수치만으로 floor 충족을 복원할 수도 없다. 또 checkpoint에는 `grid_active/trials/success`가 저장되지 않아 `30/30`은 report 프로세스의 종료 snapshot이지 resume 가능한 학습 상태가 아니다. 정확한 표현은 **“G1은 30개 cell에 노출되며 빠르게 움직였지만, 기존 report는 양방향 path 추종 품질을 측정하지 못했다”**이다.

### G2 robust

- falls 0이지만 never-arrived `63.27%`, combined never-arrived `97.45%`, along residual `−0.431 m`다.
- body p90 `0.19 m/s`, `>1 m/s` 체류 `0.50%`다.
- clean/goal-churn 평가에서 외력이 OFF였으므로 “충돌에 견뎠다”는 결론은 불가능하다. 정확한 표현은 “clean 환경에서 거의 움직이지 않아 넘어지지 않았다”다.

### G3 full

- 위치 `49.0/63.2 cm`, strict/loose 모두 0, never-arrived `67.67%`, falls 17.
- path falls 12회. 단순 attempt 근사 path fall 0.781%, waypoint 0.161%로 약 4.9배다.
- 실제 config가 G1 grid와 G2 flicker 0.01을 포함하지 않아 “G1+G2 통합 실패”라고 부르면 부정확하다. 실행된 다중 레버 묶음이 실패한 것이다.

### G4 SmoothTurn

- 총 4100 iteration까지만 학습했고 best는 3400이다.
- falls 1016 중 seq 1009(99.3%). sequential completion/banking 성공 row가 0이라 본래 completion-time/banking 가설을 측정하지 못했지만 catastrophic fail은 확실하다.
- generic waypoint selector는 성공한 sequential 정책을 올바르게 채점할 구조도 아니었다. 재시도하려면 seq completion/banking 전용 selector가 필요하다.

## 고속에서 상체가 기울고 불안정한 문제

현재 로그에는 pitch/roll, acceleration phase, steady cruise phase, capture point가 없다. “가속 때문에 lean한다”는 영상 가설을 직접 수치 검증할 수 없다. 다만 G1에서 path 낙상률이 waypoint보다 약 15.6배이고 median fall time이 segment 시작 후 1.2초라 speed demand/초기 가속과 불안정의 연관은 강하다.

이미 global orientation penalty가 `-20`, `ang_vel_xy=-0.2`다. global pitch penalty를 더 세게 하면 유용한 가속 lean까지 죽여 E2식 저속화를 반복할 수 있다. H2는 `v≥0.8 m/s AND |a_xy|≤0.3 m/s²`의 steady cruise에만 pitch/roll/ωxy/vz를 벌하고, 가속 phase에서는 이 항을 꺼 둔다.

## 옆/뒤 goal이 갑자기 생기면 느려지는 이유

이 goal은 velocity command가 아니라 로봇 좌표의 `(dx,dy,dtheta)` pose다.

- lateral은 `dx=0,dtheta=0`이라 몸 방향을 그대로 유지한 순수 sidestep을 요구한다.
- straight는 `dy=dtheta=0`인데 dx가 `[-2,2]`라 forward와 backward가 한 category에 섞인다.
- combined는 이동방향과 final heading을 독립 표본화하므로 서로 반대일 수 있다.
- constellation은 위치+heading을 한 kernel로 묶는다. 진행방향으로 몸을 돌려 빠르게 전진하면 `dtheta=0` heading 보상을 잃으므로 느린 side/back-step이 과제 정의상 합리적인 해다.
- `action_rate=-1.5`, root-acc penalty, 기존 momentum/gait phase가 급반전을 더 늦춘다.

같은 화면 방향으로 빨리 갈 때는 path heading과 진행방향이 정렬됐거나, combined dtheta가 우연히 정렬됐거나, 현재 yaw/momentum/gait phase가 이미 맞은 경우다. viewer의 world 방향과 robot-local 방향이 다르다는 점도 체감 차이를 만든다.

기존 report의 category speed는 구간 종료 순간이라 이 과도응답을 수치 판정하지 못한다. H 하네스는 `--goal_pattern lateral|reverse`를 별도 실행한다. 다음 추가 지표는 switch 후 0–2초의 min speed, time-to-0.5/1.0, bearing(front/side/back), desired-heading/travel alignment, gait phase다.

## 24개 G 파일 판정

행별 판정은 `v7-data-validity-codex.csv`에 있다.

- 각 `report.json`/`report.md` 8개: own-task pose/heading/falls/category core는 유효. 단 speed-tracking은 waypoint 혼입, segment peak는 reset teleport 오염, `v7_extras`는 마지막 256-env snapshot이라 해당 하위 지표는 제외한다.
- `selection.md`/`BEST_CHECKPOINT` 8개: own-config paired selection이지만 후반 60%의 최대 12개 후보만 봤다. warm-start fine-tune에서 유용할 수 있는 초기 checkpoint를 배제했으므로 “전체 run best”가 아니라 “샘플된 tail best”다.
- mp4 4개: deterministic env0 정성 근거만 가능.
- stress 4개: run config를 전달하지 않았다. G1/G2/G4는 common-base goal-churn 참고만, G3는 16-DOF checkpoint를 12-DOF plant에 적용해 embodiment가 달라 완전 무효다. 네 stress 모두 외력 OFF다.

추가로 `per_start_distance`는 path를 섞었고, G4 generic waypoint metric은 sequential 성공을 채점하지 못한다. 이 표들은 채택 근거에서 제외한다.
