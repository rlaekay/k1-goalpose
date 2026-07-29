# E-batch 전수 분석 — Codex 재판정

작성 기준: `MASTERPLAN.md`, `masterplan2.md`, `masterplan3.md`, `MASTERPLAN_feedback.md`, `gbatch.md`, `K1_walk/v7`의 E 계열 82개 파일, 그리고 올바른 E0 기준선 `K1_walk/select_results/E0_armB_armsdown/report.json`.

## 결론

- **goal-pose 종합 1위는 E0@6200**이다. 위치 median/p90 `2.72/5.01 cm`, heading median `2.52°`, strict success `89.29%`, 낙상 `2`회다.
- **구형 path 과제에서 속도가 가장 높았던 것은 E1**이다. 다만 학습 뒤 path 의미가 바뀌었으므로 현재 재평가로 E1의 원래 path 능력이나 배포 가능성을 판정할 수 없다.
- E2는 실제로 매우 느려졌다. 유효한 own-task 재평가에서 위치 `12.95/33.78 cm`, strict `15.29%`, never-arrived `53.34%`다. E0 대비 body-speed p90이 `0.58→0.32 m/s`로 44.8% 감소했다.
- V7은 E1+E2+protection 통합 효과를 입증하지 못했다. 현재 own-task 결과는 path code drift의 영향을 받으므로 compatibility 진단일 뿐이다.
- 충돌 외력을 학습한 E2가 외력에도 괜찮았는지는 **미검증**이다. clean과 jitter stress에서 모두 외력이 꺼져 있었다.

## 실험 흐름과 각 modification의 질문

| 버전 | 실제 modification | 원래 묻고자 한 것 | 현재 판정 |
|---|---|---|---|
| E0 | armB + arms-down URDF + RunnerV3 minibatch + mirror loss 0.5 + gait frequency 1.8–2.4 Hz | 팔 관성/학습 스택을 바꾸면서 armB 정확도를 보존하는가 | 성공. 단 여러 변경을 한 번에 넣어 각 효과는 분리 불가 |
| E1 | E0 + 구형 path 50% + scalar speed curriculum | moving goal이 지속 속도를 실제로 높이는가 | 구형 공통 평가에서 속도 상승은 관찰됨. 현 코드로 원 가설 재검증 불가 |
| E2 | E0 + collision/support + goal jitter/bias/hold/flicker | 강건성의 clean cost와 jitter 내성을 확인 | clean 정확도·속도 붕괴. 강건성 이득은 외력 ON 평가 부재로 미확인 |
| V7 | E1 + E2 + joint/power protection + settled-stop | 모든 개선의 통합 winner인가 | 실패/미검증. path drift와 다중 레버 때문에 원인 분리 불가 |

중요한 문서 교정:

- E0는 실제 생성기에서 goal jitter, bias, hold, flicker를 모두 껐다. 따라서 “E0가 4 cm jitter/5 cm bias/2–3 step hold도 견뎠다”는 과거 문구는 틀렸다. E0가 견딘 것은 일반 IMU/joint observation noise다.
- E0는 arms-down과 symmetry뿐 아니라 minibatch와 cadence 범위도 바뀌었다. 성능 향상을 팔 또는 symmetry 하나에 귀속할 수 없다.
- `masterplan3.md`와 `gbatch.md`의 “E2 own-task 재평가 미완”은 현재 파일보다 오래된 기록이다. E2 재평가는 이제 두 코드 SHA에서 사실상 동일하게 재현됐다.

## 올바른 수치 비교

| 버전 | 증거 등급 | checkpoint | 위치 median/p90 | heading median | strict | falls |
|---|---|---:|---:|---:|---:|---:|
| **E0** | 유효 기준선, SHA 미기록 | 6200 | **2.72 / 5.01 cm** | **2.52°** | **89.29%** | 2 |
| E1 | 현 코드 compatibility만 | 11400 | 52.10 / 57.94 cm | 3.30° | 8.25% | 94 |
| **E2** | 유효 own-task | 11400 | **12.95 / 33.78 cm** | 3.71° | 15.29% | 2 |
| V7 | 현 코드 compatibility만 | 7500 | 40.38 / 57.79 cm | 4.05° | 4.19% | 38 |

E2 재현성:

| 재평가 | env SHA | 위치 median/p90 | strict | falls |
|---|---|---:|---:|---:|
| `reeval_20260727-225001` | `5bafed7` | 13.08 / 34.88 cm | 15.28% | 0 |
| `reeval_e_batch_gpu1_20260728-201804` | `a7782b` | 12.95 / 33.78 cm | 15.29% | 2 |

0↔2회 낙상 차이는 PhysX 반복 편차 범위지만 위치오차와 실패 모드는 동일하다.

## E2 robust가 느려진 이유

E0와 E2의 waypoint 요구속도는 사실상 같다. 요구속도 median/p90은 E0 `0.119/0.312`, E2 `0.121/0.310 m/s`다. 과제가 덜 요구해서 느린 것이 아니다.

| 지표 | E0 | E2 | 변화 |
|---|---:|---:|---:|
| 접근속도 median / p95 | 0.115 / 0.354 | 0.097 / 0.287 m/s | −15.9% / −18.8% |
| body speed p90 / p99 | 0.58 / 1.20 | 0.32 / 0.63 m/s | −44.8% / −47.5% |
| `>0.5 m/s` 시간 | 11.67% | 2.84% | −75.6% |
| never-arrived | 0% | 53.34% | +53.34%p |
| along residual median | −0.002 m | **−0.150 m** | 목표 앞 15 cm에서 조기 정지 |
| overshoot share | 45.9% | 8.9% | E2의 91.1%가 undershoot |

E2는 curriculum/ramp 없이 첫 시점부터 `40–150 N`, `3–20 N·m`, `0.05–0.15 s` 충돌과 `3–15 N`, `1.5–3 s` support를 3–8초마다 받았다. 동시에 goal jitter/bias/hold/flicker를 묶었다. feed-forward actor가 진짜 goal 전환과 observation flicker를 시간적으로 구분하기 어렵고, `action_rate=-1.5`, `goal_progress=0`인 조건에서 가장 쉬운 해는 goal 입력에 덜 반응하는 저이득·저속 정책이다.

인과적으로 말할 수 있는 범위는 “robust bundle 전체가 cost를 만들었다”까지다. 네 요소를 한 arm에 묶었으므로 어느 하나가 단독 주범인지는 데이터로 분리할 수 없다.

## “E1, E0가 좋아 보인다”는 인상과 숫자

구형 동일 공통 protocol의 속도 지표는 다음과 같다.

| 버전 | body median | p90 | p99 | `>1 m/s` |
|---|---:|---:|---:|---:|
| E0 | 0.33 | 1.10 | 1.49 | 14.4% |
| **E1** | **0.38** | **1.31** | **1.73** | **22.5%** |
| E2 | 0.19 | 0.70 | 1.28 | 3.6% |
| V7 | 0.32 | 1.18 | 1.57 | 16.9% |

따라서 체감은 절반이 아니라 정확히 두 축으로 맞다.

- 목표 도달·정지의 종합 1위: **E0**.
- 구형 path semantics에서 속도 1위: **E1**. E0 대비 p90 +19%, `>1 m/s` 체류 +56%.
- E1의 구형 waypoint 위치오차는 약 `6.3→37.9 cm`로 악화됐고, 현재 path 재평가는 drift 때문에 무효다. 즉 “E1이 배포 종합 우승”이라는 근거는 없다.

## jitter와 외력 판정

구형 jitter stress는 네 버전 모두 낙상 0회다. E2의 angular-rate p90은 E0보다 10.9% 낮지만 speed p90도 `0.61→0.34 m/s`로 44.3% 낮다. filtering 개선과 단순 저속화를 분리할 수 없다.

모든 clean report와 `stress:jitter`에서 external perturbation은 OFF다. 확인된 것은 “그 외력을 넣어 학습하니 clean 성능이 무너졌다”뿐이며, collision recovery 향상은 확인되지 않았다.

## 82개 E 파일 판정

전체 106개 파일의 행별 판정은 `v7-data-validity-codex.csv`에 있다. E 관련 82개의 규칙은 다음과 같다.

- legacy E0/E1/E2/V7 16개: `report.md`, `selection.md`, `BEST_CHECKPOINT` 12개는 wrong base config/path-gate 때문에 무효. 네 `report_stress_jitter.md`는 goal-churn 생존 참고만 가능.
- `reeval_20260727-225001/E2_robust` 11개: `own_task/report.json`, `report.md`, `segments.csv`는 유효. BEST/selection/video는 선택 provenance 또는 정성 근거만. common/stress는 OOD 또는 goal-churn 참고만.
- `reeval_e_batch_gpu1_20260728-201804` 33개: E2 own-task 핵심 3개만 권위 있음. E1/V7은 path drift 때문에 current-code compatibility만. 모든 common/stress는 제한적.
- `reeval_20260729-144525` 22개: 20260728 산출물과 byte-identical/core-identical인 같은-seed 중복이다. 독립 표본으로 세지 않는다.

과거 report의 `segment_peak_p90/max`는 reset teleport 오염으로 무효이며, waypoint의 stale `path_speed`가 섞인 speed-tracking 표도 무효다. Hbatch 하네스에서는 reset guard와 `category==PATH` gate를 코드로 강제했다.

## 다음 실험에 적용한 교훈

- 외란, jitter, bias, hold, flicker를 분리하고 낮은 dose부터 ramp한다.
- nominal clean/path demand를 항상 보존한다.
- 외력 ON collision-only/support-only/combined 평가가 없으면 “robust”로 채택하지 않는다.
- warm-start step 0과 초기 checkpoint를 selection 후보에 반드시 포함한다.
- never-arrived, undershoot, closing speed, `>0.5 m/s` 체류를 early reject gate로 쓴다.
- 학습/평가 env SHA가 다르면 task semantics 성능을 authoritative로 표시하지 않는다.
