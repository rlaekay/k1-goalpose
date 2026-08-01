# H-batch 설계·구현·검증 계획 — Codex

이 문서는 E0/E1/E2/V7과 G1–G4의 모든 결론, 사용자의 하위 질문, H0–H3 정의, 학습/평가 하네스, sim-to-real 조사 결과를 한 곳에 집대성한다.

## Codex 서버 작업 경계

- 서버에서는 사용자 소유 경로 `<SERVER_WS>/k1-goalpose` 안만 조회한다. 다른 사용자의 workspace는 열람하지 않는다.
- 진단 단계의 서버 작업은 검색·읽기 전용이다. 코드와 문서 수정, 검증, commit/push는 로컬 저장소에서 수행한다.
- 로컬 수정이 모두 끝난 뒤에만 서버의 사용자 저장소를 동기화하고 최종 smoke/train harness를 실행한다.
- 서버 비밀번호 같은 인증정보는 문서, 로그, commit 또는 영구 메모리에 저장하지 않는다.

## 한 줄 결정

**G1@10700의 speed/path 계보를 warm-start로 쓰되, E2/G2처럼 robust 레버를 한꺼번에 세게 넣지 않는다.** 모든 H 버전에 낮은 dose의 외란과 goal jitter를 의무화하고, H1/H2에는 y-axis mirror augmentation+loss, H2에는 가속 lean을 허용하는 순항 전용 안정화, H3에는 gait touchdown 하나만 격리한다.

## 데이터가 말하는 현재 best

- 정확도 best: E0@6200, `2.72/5.01 cm`, heading `2.52°`, strict `89.29%`, falls 2.
- 사용 가능한 speed/path best: G1@10700, path-segment mean-speed median `1.038 m/s`, body p90 `1.50 m/s`. G3/G4의 raw speed 일부는 더 높지만 overspeed·정확도 붕괴·낙상이 함께 나타나므로 winner가 아니다. 기존 one-sided `path_lag/grid success`도 추종 품질 증거에서 제외한다.
- G1의 비용: waypoint `5.52/7.42 cm`, strict `34.22%`, falls 38 중 path 34.
- robust 실패 경고: E2 body p90 `0.32 m/s`, never-arrived `53.34%`; G2 body p90 `0.19 m/s`, never-arrived `63.27%`.

따라서 H의 출발점은 G1이다. E0 수치는 되찾아야 할 accuracy 목표/reference이고, 1차 reject gate는 G1의 waypoint median/p90/heading을 악화시키지 않는 것이다. E0와 G1을 weight-level에서 동시에 “합치는” 것은 불가능하므로, G1이 잃은 waypoint 정확도를 dwell과 checkpoint selection으로 단계적으로 되찾는다.

### E2 robust가 느려진 이유와 재발 방지

E0와 E2의 waypoint 요구속도 median/p90은 `0.119/0.312` 대 `0.121/0.310 m/s`로 사실상 같았다. 과제가 느려진 것이 아닌데 body-speed p90은 `0.58→0.32 m/s`(−44.8%), never-arrived는 `0→53.34%`, overshoot share는 `45.9→8.9%`가 되어 E2 시도의 **91.1%가 undershoot**였다. 따라서 “조금 더 안정적”이 아니라 목표 입력에 약하게 반응하는 저속 collapse다.

원인 해석은 단일 ablation이 없어 인과 확정은 못 하지만, E2가 처음부터 no-ramp collision/support와 goal jitter+bias+hold+flicker를 한꺼번에 켰고, `action_rate=-1.5`, `goal_progress=0`인 feed-forward policy에서 입력 변화에 둔감한 저이득 행동이 쉬운 해였다는 설명이 데이터와 가장 잘 맞는다. 더구나 당시 외력은 physics substep 적용 오류로 명목 impulse의 약 1/10이었고 force-ON eval도 없었으므로, 느려진 대가로 실제 충돌 강건성을 얻었다는 증거도 없다.

H에서는 이 실수를 다음처럼 막는다: low-dose H0에서 시작하고 force는 ramp하며 새 변경을 사전 정의한 arm별 bundle로 분리한다. warm-start `model_0`과 100/200/... 초기 checkpoint를 모두 후보에 넣고, selector가 G1 waypoint `5.52/7.42 cm, 2.54°`, never-arrived≤`1.5%`, path speed≥`0.95 m/s`, from-rest 도달률/시간, 전체·waypoint·path fall rate를 직접 reject한다. 최종 평가는 arm별 학습 외란이 아니라 공통 held-out force-ON+jitter 시험을 사용한다. 즉 reward가 좋아 보여도 E2형 저속화는 selection 단계에서 탈락한다.

### 2026-07-30 smoke 재감사: 42–45% 실패의 실제 의미

외란 body-name과 event-overlap을 고친 뒤에도 서버 smoke는 네 arm 모두 31개 중 30개를 통과하고 running lookahead occupancy 하나만 실패했다. H0/H3는 floor 안쪽 sample이 45%, H1은 43%와 2 falls, H2는 42%와 2 falls였다. 외란은 300 step 중 118–151 step에서 실제 활성화됐고 force/torque class·크기·body 검사는 통과했다. 따라서 이 실패를 다시 외란 cadence 탓으로 돌리거나 `30%→50%`로 완화하는 것은 틀리다.

공통 V7 path 구현을 다시 추적해 다음 네 원인을 확인했다.

- `path_lag=max(gap-lookahead,0)`는 carrot을 추월해 `gap<lookahead`가 된 실패를 **0 오차**로 만들었다. grid promotion도 이 one-sided 값만 사용해 floor 붕괴를 성공으로 학습했다.
- G1 checkpoint는 `grid_active/trials/success`를 저장하지 않았다. report의 `30/30 active`는 eval 프로세스 끝 snapshot일 뿐 resume state가 아니어서, H는 매 프로세스마다 0.3 m/s 한 cell에서 재시작했다.
- `min(robot_speed,path_speed)` drag와 leader-heading 방향 scalar 보정은 빠른 warm-start가 느린 carrot을 추월했을 때 floor를 충분히 복구하지 못했다. 특히 tangent 방향에서는 `step += shortfall`이 2-D 거리 floor를 수학적으로 만족하지 않는다.
- dwell timer를 4–8 s path reroll마다 다시 끊었고, dwell 종료 복구를 즉시 running 실패로 셌으며, running path에서도 `goal_reached/stand_posture`가 catch-and-stop을 보상했다.

수정은 threshold가 아니라 의미 자체를 바로잡는다. H는 exact 2-D radial annulus projection을 3.2 m/s의 world-step 상한 아래에서 사용하고, 정상 robot drag는 2.1 m/s에서 clip한다. dwell-resume은 2.1 m/s 이상의 제한된 recovery rate를 써 최대 sampled floor 2.4 m를 1.15초 안에 복구하며 teleport하지 않는다. curriculum 성공은 signed `gap-lookahead`의 양쪽을 검사하고 dwell 및 명시적 recovery를 제외하며, fall은 강제 실패로 기록한다. zero-step 초기 segment는 trial로 세지 않는다. dwell 중 world carrot pose와 gait clock을 멈추고, arrival/stand reward는 waypoint 또는 dwell에서만 켠다. active dwell은 path reroll에도 timer와 parked pose를 보존한다.

단, recovery 제외가 실패를 영구히 숨기면 안 된다. 그래서 dwell 종료 recovery 표시는 최대 2.0초 뒤 강제로 끝나며, 그때도 floor가 회복되지 않았다면 이후 step은 일반 floor 실패와 grid trial로 채점된다. post-train gate는 recovery 제외 steady sample이 실제로 1개 이상 있어야 하므로 “전 구간 recovery 처리”도 통과할 수 없다.

또 하나의 공통 오류는 timeout mask였다. 이전 코드는 `time_out_buf` tensor를 매 step 새로 만들면서 `extras["time_outs"]`는 reset 때만 갱신했고, physical fall과 goal-resample이 같은 step이면 timeout이 fall을 가렸다. 이제 episode timeout/segment boundary/physical failure를 분리하고, physical failure가 항상 우선하며, PPO와 eval에 매 step 같은 최신 mask를 전달한다.

H의 `initial_active: all`은 G1이 30개 cell을 실제로 **노출받았던 학습 분포**를 복원하는 장치이지 30개를 숙련했다는 주장이 아니다. 이후 checkpoint는 grid state와 scalar speed/EMA를 함께 저장·resume한다. 기존 G1 checkpoint에는 이 state가 없으므로 명시적 `all`이 필요하다. 일반 selection/eval은 후보마다 동일한 config protocol을 보장하기 위해 task state를 의도적으로 복원하지 않는다. checkpoint 고유 curriculum을 재현하는 진단만 `eval_goal_pose.py --restore_task_state`를 명시한다.

### 2026-07-30 실제 launch에서 발견한 H1/H2 NaN과 재발 방지

staged smoke가 네 arm을 모두 통과시킨 첫 실제 launch에서 H0/H3는 정상적으로 epoch를 진행했지만 H1/H2는 첫 PPO iteration 안에서 actor mean 전체가 NaN이 되어 종료됐다. 서버에서 G1@10700 checkpoint를 직접 읽은 결과 저장된 Adam LR는 `1.70859375e-4`였고, H YAML의 `5e-6`보다 **34.17배** 컸다. 기존 loader는 model뿐 아니라 이 optimizer state/LR를 복원하면서 controller scalar만 `5e-6`으로 남겨 첫 update를 숨은 고LR로 실행했다. H1/H2의 신규 reflected PPO score-gradient와 결합된 뒤 unclamped `exp(new_logprob-old_logprob)`가 overflow할 수 있었고, nonfinite gradient clipping도 fail-open이었다.

기존 64-env 1-iteration smoke는 마지막 20번째 optimizer step 뒤 새 policy forward를 하지 않고 바로 정상 종료했으므로, 마지막 step이 NaN을 만들더라도 구조적으로 탐지하지 못하는 blind spot이 있었다. 수정 후에는 다음을 모두 hard gate로 둔다.

- G1→H0–H3는 model과 task state만 같은 조건으로 warm-start하고 모두 fresh Adam `5e-6`에서 시작한다. 동일 H run의 진짜 resume에서만 optimizer 복원을 허용하며, 그때는 controller LR도 param-group LR와 동기화한다.
- adaptive KL은 optimizer **후** original/mirrored policy를 다시 forward하고 둘 중 큰 KL로 제어한다. `KL=0`인 첫 pre-update 비교로 LR을 올리지 않으며 범위는 `1e-6–1e-5`다.
- ordinary PPO는 전 arm에서, reflected PPO는 H1/H2에서 log-ratio exponent 입력을 ±10으로 제한한다. reflected action은 old mirrored Gaussian의 5σ support 안에서만 transition PPO에 쓰고 symmetry mean loss는 전체 표본에 유지한다.
- rollout/return/logprob/loss/gradient/parameter/Adam state/post-update policy에 finite gate를 두고 `clip_grad_norm_(error_if_nonfinite=True)`로 최초 오염 지점에서 즉시 실패한다.
- mirrored critic의 privileged COM-y는 실제 offset이 아니라 raw uniform latent `u∈[0,1]`였으므로 잘못된 `u→-u`를 물리적으로 맞는 `u→1-u`로 고쳤다. smoke는 original/mirrored latent가 모두 `[0,1]` support에 남는지 검사한다.
- TRAIN_UPDATE는 production과 같은 `4096 env × horizon 24 × 5 epochs × 4 minibatches × 2 iterations`, 총 40 update와 마지막 post-update forward를 atomic health marker로 증명한다.

## H0–H3 frozen 정의

공통:

- task `K1/Goal_Pose_HBatch`, observation/action `54/12` 유지.
- warm start G1@10700.
- waypoint/path `0.65/0.35`, G1 speed×curvature 분포를 유지하되 위에서 감사한 floor/dwell/curriculum 의미는 수정.
- legacy synchronized velocity kick는 0으로 끈다.
- 새 팔 asset `K1_locomotion_hbatch-codex.urdf` 사용, arm script와 16-DOF armswing은 사용하지 않는다.
- 모든 버전에 goal observation jitter, segment bias, 2–3 step hold, rare flicker와 multi-body force를 nonzero로 넣는다.
- 모든 버전에 reset pose DR + episode-constant encoder bias + motor-target offset을 넣는다.
- 모든 버전은 G1 model weight/task state만 warm-start하고 Adam state는 새로 시작한다. H 선언 LR은 `5e-6`, adaptive 범위는 `1e-6–1e-5`이며 네 arm이 같은 optimizer 출발점을 쓴다. 이 수정 이후 결과 protocol은 `2026-07-30-codex-v3`로 올려 예전 completed v2 suite가 새 arm과 비교되는 것을 금지한다.
- H0/H3는 mirror loss와 mirror transition augmentation가 모두 0인 control이고, H1/H2만 두 항을 함께 켠다.

| 버전 | modification | 가설 | 다른 버전과의 차이 |
|---|---|---|---|
| **H0** | G1 + 정확한 팔 + low-dose 외란/jitter + mild joint offsets | 현재 best speed를 최소 필수 robustness와 함께 보존 가능한가 | 통합 기준선 |
| **H1** | H0 + y mirror transition augmentation + mirror loss + stronger joint offset DR | 좌우 편향과 calibration gap을 nominal speed 손실 없이 줄이는가 | stability/gait reward 없음 |
| **H2** | H1 + steady-high-speed stability reward + 더 느린 ramp이지만 최종적으로 더 잦고 high-speed-biased인 disturbance + flicker 2배 | 가속 lean/가속시간은 유지하면서 강화된 고속 안정화 bundle이 순항 pitch/roll/ωxy와 외력회복을 낮추는가 | reward 단일 ablation이 아니라 명시적인 stability/robustness bundle |
| **H3** | H0 + forward-path touchdown placement reward 하나 | heel/capture-point형 foot placement가 고속 낙상을 줄이는가 | H1/H2를 상속하지 않는 gait-only ablation |

정확한 config:

- H0 force: interval 8–14 s, event probability 0.25, collision share 0.25, collision `40–100 N`, `3–12 N·m`, `0.05–0.10 s`, support `3–8 N`, `0.2–1 N·m`, `0.5–1.5 s`.
- H1: H0 force 그대로, encoder bias ±0.025 rad, target offset ±0.020 rad, init q σ 0.075 rad, mirror augmentation 0.5, mirror loss 0.5. reflected action이 frozen mirrored Gaussian에서 5σ 밖이면 tail PPO augmentation에서는 제외하되 mean-equivariance loss에는 남긴다. 매 update의 log-ratio는 `[-10,10]` 안에서 계산하고 유효 augmentation 표본이 10% 미만이면 실패한다.
- H2: interval 6–12 s, base probability 0.35, collision share 0.35, 72,000 control-step ramp; path `v≥0.8 m/s`에서는 event probability를 2배(상한 1.0)로 높이고, goal flicker를 0.001→0.002/step, high-speed stability scale을 −0.5로 바꾼다. 따라서 H2가 이겨도 reward 하나의 인과효과가 아니라 이 세 요소의 bundle 효과로 해석한다.
- 위 차이는 **학습 분포**다. 최종 cross-arm 평가는 arm별 설정을 그대로 시험하지 않는다. 네 arm 모두 같은 held-out profile(`interval 6–12 s`, probability 0.50, collision share 0.35, high-speed probability boost 2.0, ramp 1; encoder ±0.025 rad, target ±0.020 rad, init q σ 0.075 rad; goal flicker 0.001/step)을 강제한다. 그렇지 않으면 H2의 정책 효과와 더 어려운 시험지가 섞인다. 실제 적용된 commands/noise/randomization 전체를 SHA-256으로 report에 기록하고 report별 hash가 H0와 다르면 비교를 거부한다.
- H3: H0와 같고 `heel_strike_ahead=+0.10`만 추가.

생성물은 `htwk-gym/sweeps/hbatch/H0-codex.yaml`부터 `H3-codex.yaml`까지다.

## 고속 lean을 죽이지 않고 안정화하는 방법

사용자의 관찰은 “빠른 가속 때 상체가 기울고, 고속에서 안정성이 떨어진다”이다. lean 자체를 항상 벌하면 빠른 가속도 같이 죽는다. H2의 reward gate는 다음과 같다.

```text
speed_gate = sigmoid((|vxy| - 0.8) / 0.1)
steady_gate = sigmoid((0.3 - |axy|) / 0.08)
penalty = speed_gate * steady_gate *
          (pitch² + roll² + 0.10|ωxy|² + 0.02 vz²)
```

가속 `|axy|>0.3 m/s²`에서는 steady gate가 0으로 가므로 전방 lean을 허용한다. `v≥0.8`이고 가속이 잦아든 순항에서만 upright/low angular-rate를 학습한다. H2 reject 기준은 H1 대비 time-to-1.0 m/s 10% 초과 퇴화, path fall 악화, 순항 pitch/roll/ωxy 미개선이다.

여기서 속도는 한 step의 trunk-link 속도가 아니라 기존 약 0.2초 저역통과 속도이고, 가속도도 `alpha=0.1`로 한 번 더 저역통과한다. 그렇지 않으면 보행 주기 안의 torso sway가 매 step `|a|>0.3`으로 오인되어 steady gate가 영원히 꺼지는 문제가 생긴다.

## “첫 접지 heel이 몸보다 앞” 보상이 해결하는가?

**직접 해결책으로 단정할 수 없다.** heel-ahead 이진 보상은 overstride, braking impulse, 무릎 충격을 키워 오히려 고속 안정성을 해칠 수 있다. 현재 simulator는 foot link의 net force만 주며 실제 sole contact point는 주지 않는다.

외력 ON 검증이 전혀 없어 “기존 외란이 충분했으니 gait hardcoding이 불필요하다”는 전제도 아직 판정할 수 없다. 그래서 애매한 경우에 해당하며 H3를 만들었다.

H3는 다음처럼 보수적으로 제한한다.

- contact onset에서만 계산.
- forward path이며 body forward speed `>0.6 m/s`일 때만 켬. side/back/turn/stand에는 0.
- 기존 foot-corner kinematics의 heel `x=-0.1015 m`를 사용.
- 이진 “앞/뒤” 대신 target `clip(0.08*vx, 0.02, 0.12 m)`에 Gaussian으로 맞춤.
- H0+이 reward 하나만 비교해 인과를 보존.

## 외란: 크기는 충돌급이었나, 실제로 괜찮았나, 한 곳에만 주나?

기존 V7의 collision 설정은 `40–150 N × 0.05–0.15 s`, 즉 명목 충격량 `2–22.5 N·s`였다. 그러나 Isaac Gym의 rigid-body force는 immediate physics timestep에만 유효한데, 기존 코드는 decimation 10개 중 한 번만 submit했다. 따라서 실제 적분 충격량은 명목값의 약 1/10인 `0.2–2.25 N·s`였고, 당시 eval도 control `dt=0.02 s`로 적분해 이를 10배 과대보고했다. **기존 외란이 충분히 컸다는 결론은 낼 수 없다.**

- 기존 구현은 `Trunk/base_indice` 한 rigid-body COM에만 force와 독립 random torque를 `LOCAL_SPACE`로 가했다.
- 두 번째 로봇 actor, 형상 접촉, 팔/다리 타격, 접촉점 moment arm, 상대 로봇 dynamics가 없다.
- 긴 support force도 로봇이 돌면 방향이 같이 돌았다.
- 외란 중 fall/reset 뒤 active force가 새 episode에 남을 수 있었다.

그리고 clean 및 jitter stress에서 모두 외력이 OFF였으므로 **괜찮았는지는 미검증**이다. G2/E2의 0-fall clean은 force recovery 증거가 아니다.

HBatch는 다음을 수정했다.

- Trunk, 양 hip-roll, 양 shank body 중 하나를 event마다 선택한다. Isaac Gym은 knee joint 이름이 아니라 그 child rigid-body인 `Left_Shank`/`Right_Shank`를 노출한다. fixed 팔 link는 `collapse_fixed_joints=true`에서 Trunk로 합쳐지므로 upper-arm을 별도 body로 거짓 계수하지 않았다. 로드 후 5개 이름 중 하나라도 없으면 env 생성을 즉시 실패시킨다.
- ENV_SPACE force로 world 방향을 유지.
- event 생성·만료는 50 Hz control step에서 한 번만 처리하되 held wrench를 500 Hz physics의 decimation 10개 직전에 모두 다시 submit한다. duration은 control step 단위로 올림 양자화되므로 H collision의 실제 범위는 `40–100 N × 0.06–0.10 s = 2.4–10 N·s`, support는 `3–8 N × 0.5–1.5 s = 1.5–12 N·s`다. report의 expected impulse도 요청 duration이 아니라 이 실제 적용 duration으로 계산한다.
- reset 시 해당 env의 모든 force/torque/timer를 clear.
- 외란을 ramp하며 event probability로 clean sample 비중을 보존.
- eval report에 force event 수, active share, force 중 fall을 기록. event 0이면 보고서가 자동으로 “robust 근거 아님”을 표시.

여전히 실제 두 로봇 collision은 아니며 multi-body wrench proxy다. 최종 검증에는 두 K1 actor의 실제 contact scenario가 별도 필요하다.

## joint-position domain randomization 감사와 강화

기존 값:

- reset `q_init ~ N(0,0.05 rad)`.
- 매 step observation `q_noise ~ N(0,0.01 rad)`.
- stiffness/damping ±5%, friction 0–2.

빠져 있던 것:

- episode-constant encoder zero bias.
- motor/PD target zero offset.
- encoder scale/nonlinearity, backlash/deadzone.

새 구현은 actor observation에 `q_meas=q_true+encoder_bias+iid_noise`, PD target에 `target_offset`을 넣되 reward/limit은 true q를 사용한다. H0/H3 bias ±0.015, target ±0.010 rad; H1/H2 bias ±0.025, target ±0.020 rad다. 무근거 ±0.05 rad 상시 target offset은 Kp=100에서 5 N·m을 만들 수 있어 사용하지 않았다.

다음 real-log 기반 확대 순서는 joint group별 ±0.01/0.02/0.03 rad grid, encoder scale, static friction/backlash다. 모든 range는 실기 로그/system ID로 다시 좁혀야 한다.

## y-axis mirror augmentation와 mirror loss

기존 RunnerV3에는 `MSE(π(Ms), Mπ(s))` mirror loss만 있고 transition data augmentation는 없었다. H1/H2에는 둘 다 넣었다.

- mirrored obs와 action으로 PPO log-prob를 다시 계산한다. 원 sample의 old log-prob를 재사용하지 않는다.
- reward/done/advantage/return은 reflection에서 보존한다.
- asymmetric critic의 14 privileged channels도 mirror한다: COM-y raw latent는 `u→1-u`, linear-vy/force-y는 sign flip; torque는 axial vector라 Tx/Tz sign flip, Ty 유지.
- H0/H3는 mirror loss와 transition augmentation를 모두 0으로 되돌린 비-mirror control이다. H1/H2만 loss `0.5`와 augmentation `0.5`를 함께 켜 사용자 요청의 전체 mirror intervention을 검증한다.

NVIDIA Isaac Lab도 data augmentation와 mirror loss를 별도 switch로 정의한다: [Isaac Lab symmetry configuration](https://isaac-sim.github.io/IsaacLab/main/_modules/isaaclab_rl/rsl_rl/symmetry_cfg.html).

## 팔 겹침과 URDF 이중검사

사용자가 지목한 참조는 루트의 `k1/K1_locomotion.urdf`다. `htwk-gym/resources/K1/K1_locomotion.urdf`와 이름은 같지만 팔 각도가 다르므로 경로를 구분해 검사했다.

| joint | 참조 rpy |
|---|---|
| ALeft/ARight Shoulder Pitch | 0 0 0 |
| Left Shoulder Roll | **−1.35 0 0** |
| Right Shoulder Roll | **+1.35 0 0** |
| Left/Right Elbow Pitch/Yaw | 0 0 0 |

기존 armsdown은 shoulder ±1.5708, elbow pitch 2.08, elbow yaw ±1.58이라 참조와 다르다. H asset은 기존 htwk dynamics/mass/collision을 보존하고 팔 fixed rpy만 참조와 같게 했다. generator와 smoke가 8개 joint를 XML numeric tolerance `1e-6`, fixed type으로 각각 비교한다. 동적 smoke는 DOF/obs/action 12/54/12, finite reward/obs도 검사한다.

## 새 평가 지표

공통 core:

- waypoint 위치 median/p90, heading, strict/loose, never-arrived, undershoot/overshoot, falls.
- path-only commanded/achieved speed, falls per attempt.
- 매 control step의 per-env signed `gap-lookahead`: `gap/lookahead` p2/p50, floor deficit p50/p90, behind lag p50/p90, 실제 per-env leash 이탈률. 의도된 dwell-resume recovery 비율을 따로 기록하고, 이를 포함한 floor-collapse와 제외한 floor-collapse를 모두 낸다. 기존 one-sided segment `path_lag`는 backward-compatible 참고값일 뿐 채택 근거가 아니다.
- reset 뒤 0.25 s를 제외한 pose-difference speed. path category가 아닌 sample은 speed-tracking에서 강제 제외.

고속 phase:

- acceleration: `v>0.3` 및 forward acceleration `>0.3`; pitch median/p90, time-to-0.8/1.0.
- cruise: `v≥0.8` 및 `|axy|≤0.3`; pitch/roll abs p90, `|ωxy|` p90, `|vz|` p90, cruise exposure.
- H2 채택 시 acceleration pitch 자체는 reject 지표가 아니고 가속시간 퇴화 여부를 본다.

외란:

- event count/active duty, force body/direction/magnitude, force-active falls.
- 실제 physics substep에 적용된 impulse/torque-impulse, max tilt, speed loss, 90% speed recovery time·≤5 s recovery share, 2/5 s survival을 전체/high-speed/collision/support별로 분리해 report에 저장한다. episode timeout과 rollout 종료는 각 2 s/5 s horizon의 관측 가능성에 맞게 right-censor하고, event와 outcome record 수가 다르면 overlap로 명시한다.

방향전환:

- `--goal_pattern lateral`, `--goal_pattern reverse`를 별도 실행.
- lateral은 근접 0 m 목표가 섞이지 않게 좌/우 부호를 무작위로 고른 `|dy|=1–2 m`, reverse는 `dx=−1–−2 m`로 고정한다. 둘 다 `dtheta=0`이다.
- switch 후 0–2 s min speed, time-to-0.5/0.8/1.0, initial bearing, gait-phase quarter별 응답을 `segments.csv`/JSON에 저장한다. 임계 도달은 raw speed가 아니라 **새 goal의 최초 world direction으로 투영한 filtered velocity**로 계산한다. 옛 방향으로 빨리 달리는 것은 lateral/reverse 성공으로 세지 않는다.

symmetry/DR:

- mirror involution `M(Mx)=x`, action/obs permutation bijection, policy equivariance error.
- fixed encoder bias grid와 joint-group별 성능 저하.
- forward path의 첫 접지 transition을 rollout 전체에서 streaming histogram으로 수집한다. heel body-x p10/median/p90, trunk 앞 접지 비율, 동적 target±1σ 비율, overstride, 접지 직전 하강속도 p90, contact-force p90, 좌/우 heel median 차이를 낸다. H1은 policy equivariance와 좌우 접지 bias를 함께 보고, H3는 heel target 개선과 impact 비열세를 직접 검증한다.

외력 outcome의 시간 정렬도 다시 검사했다. HBatch의 새 wrench는 `simulate` 뒤에 설치되어 **다음 control step의 모든 physics substep**부터 작용한다. 따라서 설치와 같은 step에 이미 reset된 event는 `cancelled_before_application`으로 빼고, force 낙상과 impulse는 pre-step에 실제 active였던 wrench만 센다. recovery baseline은 설치 직후/적용 직전의 goal-progress speed다. 미회복 상태에서 segment가 바뀌거나 dwell/floor-recovery가 시작되면 recovery만 right-censor하고 survival은 계속 추적한다. episode timeout과 rollout 종료는 2 s/5 s survival denominator에서 관측 가능한 horizon만 남기며, physical fall은 censor가 아니라 회복 실패로 남긴다.

## 영상: top view 대신 simulator 시점에 표시

새 RGB logger나 debug actor를 추가하지 않았다. 기존 Isaac Gym `IMAGE_COLOR` RGBA sensor와 frame 수집을 그대로 쓰고, 매 frame follow-camera pose/FOV 숫자만 저장한다. 후처리에서 world 3D를 perspective image로 투영한다.

- 최근 moving-carrot trace: 녹색 path.
- 현재 path lookahead/carrot: amber `PATH CARROT`.
- waypoint: 녹색 `WAYPOINT GOAL`과 heading arrow.
- 외력: 선택된 body COM에서 시작하는 빨간 화살표. 길이는 force 크기의 제곱근으로 scale.
- path mode에는 별도 “final goal”이 없으므로 존재하지 않는 goal을 거짓으로 그리지 않는다.
- H config에서 top-down constellation inset은 끄고 실제 simulator perspective overlay를 사용한다.

6초 RGBA video smoke는 학습 전 실행된다. 첫 외란이 3–4초에 오므로 2초 영상은 빨간 화살표를 구조적으로 담을 수 없었다. 이제 mp4/report 존재와 전체 force event뿐 아니라 **실제로 기록된 env0 force-active frame>0**을 검사한다. 이는 과거 RGB/RGBA, graphics device −1, “report에는 event가 있지만 영상에는 화살표가 없는” 실패를 함께 막는다.

## train + eval 단일 하네스

`tools/run_hbatch_suite.sh`:

1. committed H0–H3 config가 generator와 semantic하게 같은지 `--check`한다. **실행 중 tracked YAML을 다시 쓰지 않는다.** 과거 launcher가 YAML을 재생성해 서버 worktree를 dirty하게 만들고 다음 `git pull`을 막았던 문제를 제거했다.
2. `[STATIC]`: task/interface/URDF와 새 path controller, full-grid restore, arrival/dwell, post-train reject gate를 exact 값으로 고정한다.
3. `[PATH_MECHANICS]`: H event와 legacy push/kick를 모두 0으로 한 뒤 300-step rollout을 실행한다. per-env 초기 floor, deterministic radial floor/leash fixture, 모든 비-reroll step의 analytic rate budget, dwell world pose 정지·최소 지속시간·gait clock pause/resume, finite obs/reward, checkpoint-state round-trip을 hard gate로 검사한다.
4. `[TRAIN_UPDATE]`: production과 같은 4096 env × horizon 24 × 5 epoch × 4 minibatch로 PPO 2 iteration, 총 40 update를 disposable하게 수행한다. H 전용 entrypoint도 다른 v3/v7 trainer와 같이 `isaacgym`을 `torch`보다 먼저 import한다. H1/H2 mirror transition augmentation, mirrored critic, backward/autograd와 각 arm reward가 본 학습 전에 실행되며, 고유 token의 atomic health marker가 exact shape·40 update·fresh `5e-6` optimizer·parameter 변화·finite gradient/Adam state·마지막 post-update forward를 증명해야 한다. 고유 tag로 생긴 smoke log directory만 종료 후 삭제한다.
5. `[DISTURBANCE]`: path/grid/goal noise와 joint encoder/target offset을 `[0,0]`으로 끈 별도 rollout에서 interval `3–4 s`, probability 1, ramp 1로 collision/support 두 class, 다섯 body, 크기 상한, force/torque 동일 body, env당 동시 active body≤1, 만료 clear를 검사한다. range가 `[0,0]`인 DR은 “nonzero sample”을 요구하지 않고 runtime buffer가 실제 0인지 검사하므로 외력 격리 자체를 실패로 오판하지 않는다. 추가로 force-active control step이 실제로 존재하고 wrench submit 호출 수가 정확히 `control steps × decimation`인지 검사해 1/10 impulse 회귀를 막는다. production hook에는 substep별 GPU→CPU 동기화가 없다.
6. `[VIDEO]`: support-only 외란으로 10초 평가·앞 6초 녹화를 하고 실제 env0 force-active frame, renderer-confirmed 빨간 화살표, path carrot과 움직인 trace를 모두 요구한다. 부모가 매 실행 고유 completion token을 발급하며, eval의 마지막 filesystem 작업인 `eval-complete-codex.json` atomic marker에 그 token과 이번 실행에서 실제 생성한 report/mp4의 byte 수·SHA-256을 기록한다. 별도 verifier가 token·marker·JSON counter·hash와 MP4 전 frame decode 수를 함께 검사해 재사용 경로의 stale artifact도 배제한다. Isaac Gym camera native teardown이 그 뒤 nonzero status를 내더라도 이 증거가 전부 일치할 때만 경고로 허용하고, 하나라도 빠지면 실패한다.
7. 다섯 stage를 가능한 끝까지 실행해 실패 원인을 한 로그에 모으고, 모두 통과한 arm만 GPU에 올린다. 성공 smoke log는 삭제하고 실패 arm만 `logs/hbatch/smoke_failures/Hx-codex.log`에 남긴다. launch 뒤에도 각 tmux arm을 즉시 성공으로 선언하지 않고 production runner가 고유 token으로 2회 finite iteration을 attestation한 뒤 10초 grace와 최종 status/`pane_dead` 재검사까지 통과할 때까지 최대 300초 poll한다. 모든 arm pane을 만들 때까지 별도 anchor가 session을 유지해 첫 arm의 즉시 사망이 뒤 arm launch를 연쇄 실패시키지 않게 한다. arm wrapper 자체를 tmux의 첫 프로세스로 두고 conda activation·`cd`·train/eval 전부를 알려진 pending log 안에서 실행하며 child와 `tee`의 exit status를 둘 다 판정한다. bootstrap/nonzero/signal/log-I/O/timeout/marker 직후 사망은 arm별 `logs/hbatch/training_failures/Hx-codex.log`로 승격하고, 정상 arm의 pending copy는 최종 성공 시 삭제한다. `exec bash`로 죽은 pane을 살아 있는 것처럼 보이게 하던 방식은 제거했다.

정책 성능과 코드 불변식을 섞지 않는다. frozen G1의 running floor occupancy와 dwell 도착률은 숫자를 그대로 `NOTE`로 남기지만 pre-training launch gate가 아니다. 반대로 floor controller의 2-D projection/rate limit/dwell 정지는 policy·외란과 무관한 hard gate다. 학습 뒤에는 `gap/lookahead<0.75`가 dwell-resume recovery를 제외하고 10% 이하, per-env leash 이탈이 1% 이하인지 reject gate로 판정한다. 이는 30% 기준을 50%로 완화한 것이 아니라 잘못된 pre-training 질문을 deterministic mechanics 검사와 post-training 성능 검사로 분리한 것이다.

외란 stage의 6초×256 env에서 env당 첫 이벤트 약 1회면 coverage에 충분하고 최대 1.5초 support보다 주기가 길어 겹치지 않는다. 최초 구현의 `0.2–0.6 s`는 아직 외란을 학습하지 않은 G1 warm-start에 실제 H 학습 설정보다 약 64–110배 높은 event rate를 가했다. runtime도 새 event 직전에 해당 env의 기존 force/torque 전체를 지워 한 번에 한 body에만 외란이 남도록 이중 방어한다.

`tools/train_and_eval_hbatch.sh`:

1. train.
2. warm-start를 `model_0.pth` 후보로 넣고 100/200/… 초기 checkpoint까지 selection에 강제 포함한다. 기존 tail 60% 편향을 제거한다. 후보마다 seed뿐 아니라 task-grid/counter/optimizer와 무관한 env state를 같은 protocol로 되돌린다. H1/H2는 cruise sample coverage와 touchdown 좌우 bias, H2는 순항 안정성, H3는 touchdown target/overstride를 selection ratio에 직접 포함한다. 후반 두 stage에는 **공통 held-out 외력 profile**의 paired combined force+jitter screen도 붙인다. 이 판정은 `selection_gates_pass`이며 direction/force-recovery/video/cross-arm까지 통과했다는 뜻은 아니다.
3. clean, force ON, goal-jitter, jitter+force, lateral, reverse, perspective force-video를 평가한다. 각 run config는 정책/구조를 재현하는 입력일 뿐이고, 시험 난이도를 정하는 noise/joint DR/disturbance는 모든 arm에 동일한 `hbatch_common_eval`로 교체된다.
4. selection/report/segments/video를 먼저 `*-partial-<pid>`에 복사하고 모든 report와 nonempty mp4 검증 뒤 `COMPLETE` marker를 쓰고 원자적으로 최종 이름으로 바꾼다. 중간 실패 폴더는 비교 대상이 아니다.
5. 완료된 최신 H0–H3만 lock 하에 다시 비교해 `shared_eval_videos/hbatch/hbatch-comparison-codex.md`/`.json`을 갱신한다. 각 report의 HBatch protocol version, env/eval code SHA, **effective test-config SHA**, seed, duration, env 수, task-state protocol이 H0와 정확히 같지 않으면 cross-arm PASS를 금지한다. H1은 H0 대비 mirror error와 좌우 touchdown bias, H2는 H1 대비 95% speed/cruise coverage 비열세·10% 가속시간 회귀 제한·세 순항 안정성 지표의 절대 상한과 non-worse/최소 하나 strict improvement, high-speed force recovery 비열세/최소 하나 strict improvement를 판정한다. H3는 H0 대비 heel target 개선·overstride/impact 비열세를 판정하고, path fall은 strict 감소 또는 둘 다 0을 요구한다.

서버 실행:

```bash
cd htwk-gym
conda activate k1goalpose
bash tools/run_hbatch_suite.sh
```

이전 launcher가 바꾼 네 YAML 때문에 `git pull`이 막힌 서버는 먼저 그 네 파일만 보존적으로 stash한다. 이 stash는 옛 generator 부산물이므로 새 config 위에 pop하지 않는다.

```bash
cd <SERVER_WS>/k1-goalpose
git stash push -m "server-hbatch-yaml-before-codex-fix" -- \
  htwk-gym/sweeps/hbatch/H0-codex.yaml \
  htwk-gym/sweeps/hbatch/H1-codex.yaml \
  htwk-gym/sweeps/hbatch/H2-codex.yaml \
  htwk-gym/sweeps/hbatch/H3-codex.yaml
git pull --ff-only
cd htwk-gym
bash tools/run_hbatch_suite.sh
```

현재 로컬에는 Isaac Gym/CUDA/PyYAML runtime이 없어 GPU dynamic/video smoke와 학습은 실행하지 않았다. Python syntax, shell syntax, config/URDF static invariant는 로컬에서 검사한다. 실제 GPU launch는 위 하네스가 smoke 통과 arm에만 수행한다.

## H 버전별 채택 기준

| 판정 축 | H0 | H1 | H2 | H3 |
|---|---|---|---|---|
| waypoint | G1 median/p90/heading `5.52/7.42 cm, 2.54°`보다 악화 금지, never-arrived≤1.5%; E0는 회복 목표 | H0의 세 지표 5% 이내, never-arrived≤1.5% | H1의 세 지표 5% 이내, never-arrived≤1.5% | H0의 세 지표 5% 이내, never-arrived≤1.5% |
| speed | path mean median ≥0.95 m/s | H0의 95% 이상 | H1의 95% 이상; time-to-1.0 퇴화≤10% | H0의 95% 이상 |
| path floor/leash | recovery 제외 `gap/lookahead<0.75` ≤10%, leash 밖 ≤1% | 동일 | 동일 | 동일 |
| falls | fall-context 전수 분류, 전체≤5/1000·waypoint≤2/1000 공통; G1 path 20.6/1000에서 path≤5/1000 | H0 path 비열세 | H1 path 비열세 | H0 path보다 감소; 둘 다 0이면 별도 필수 gate인 heel-target strict 개선과 impact 비열세로 gait 가설 판정 |
| force | event>0, 5 s survival≥98%, recovery≤5 s share≥90%·p90≤2 s | 동일하며 high-speed reference 확보 | high-speed recovery share/p90 모두 H1 비열세, 하나 이상 strict 개선 | H0와 동일 |
| stability/gait | baseline first-contact 분포 확보 | mirror p90≤0.10, H0 대비 policy equivariance strict 개선, 좌우 heel median 차이 개선 또는 이미 ≤5 mm | cruise coverage≥5% 및 H1의 95% 이상; pitch≤20°/roll≤15°/ωxy≤3 rad/s, 세 지표 모두 H1 비열세이고 하나 이상 strict 개선; accel pitch 허용 | touchdown 표본≥100, target±1σ≥40%이면서 H0보다 strict 개선, overstride≤10%·impact 비열세·path fall 감소 |
| symmetry | 기록 | 좌우 touchdown bias 절대값≤2 cm | H1의 mirror/bias 보존 | 가설 아님 |

일반 `report.json`의 legacy `all_gates_pass`(waypoint median 5 cm, 전체 fall 0)는 참고로 계속 남긴다. 그러나 H selector/비교기가 이것을 위 H 전용 gate와 **동시에** 요구하지는 않는다. 그렇게 하면 H0의 명시적 G1 보존선 5.52 cm와 nonzero rate budget을 만족해도 legacy 5 cm/0 fall 때문에 구조적으로 FAIL이 되어 표의 기준과 모순된다. H checkpoint selection은 G1 waypoint `5.52/7.42 cm, 2.54°`와 H 절대/arm별 ratio만 사용하고, cross-arm 채택은 authoritative report 여부와 위 절대·상대 H gate를 직접 판정한다. 낙상은 완료구간만 세는 survivor bias를 피하려고 `(falls)/(completed+falls)×1000`으로 전체/waypoint/path를 각각 계산하며, 모든 fall context가 분류되지 않으면 통과시키지 않는다.

H0가 G1 speed/accuracy를 보존하지 못하면 H1–H3의 해석 전에 dose를 더 낮춘다. H1이 speed를 해치면 mirror coefficient를 0.5→0.25로 낮춘다. H2가 가속을 10% 이상 늦추면 global pitch를 만지지 않고 steady gate/scale만 낮춘다. H3는 H0 대비 fall 감소가 없을 때 두 arm 모두 0 fall이 아니라면 폐기한다. 둘 다 0 fall이면 heel-target share가 strict 개선되고 impact가 비열세인 경우에만 gait 가설을 지지한다.

## 2026-08-01 HBatch 완료 결과와 변경사항별 판정

### 데이터 반입·무결성·비교 가능성

서버에서는 사용자의 프로젝트 경로 안에서 결과 이름만 확인하고, `shared_eval_videos/hbatch` 전체를 로컬 `hbatch-results-codex/`로 복사한 뒤 내용을 읽었다. 복사본에는 H0–H3 완료 suite 4개, Markdown 33개, JSON 61개, CSV 20개, MP4 4개가 있다. 28개 `eval-complete-codex.json`이 지시한 80개 artifact의 byte 수와 SHA-256을 다시 계산한 결과 mismatch는 0개였다. 네 top-level `COMPLETE`도 모두 존재한다.

평가 protocol 자체도 cross-arm 비교 조건을 지켰다. 모든 arm은 protocol `2026-07-30-codex-v3`, seed 0, env/eval code SHA `3d34d274ef2d644ae763e90405dd80ba07f28949`를 쓰며, clean/force/jitter/combined/lateral/reverse/video-force 각각의 effective protocol SHA가 arm 사이에서 같다. 따라서 아래의 문제는 서로 다른 시험지를 비교해서 생긴 것이 아니다.

하지만 **최종 policy 비교에는 결정적인 한계가 있다. H0·H1·H2·H3 selector가 모두 iteration 0인 `model_0.pth`를 골랐다.** launcher는 같은 G1@10700 warm-start를 각 run의 `model_0.pth`로 복사하므로, 최종 full suite는 네 개의 H intervention policy가 아니라 같은 학습 전 G1 weight를 네 번 평가한 것이다.

| arm | 최종 선택 | 학습된 H 변경이 최종 policy에 들어갔나 | full-suite 용도 |
|---|---|---|---|
| H0 | `model_0.pth` | 아니오 | H 공통 시험에서 warm-start 절대성능 측정 |
| H1 | `model_0.pth` | 아니오 | mirror/강화 joint DR 효과 판정 불가 |
| H2 | `model_0.pth` | 아니오 | high-speed stability/강화 force/flicker 효과 판정 불가 |
| H3 | `model_0.pth` | 아니오 | heel touchdown reward 효과 판정 불가 |

이 해석은 단순히 checkpoint 이름만 본 추측이 아니다.

- `hbatch-comparison-codex.json`의 H0–H3 `metrics` 객체가 field-by-field 완전히 같다.
- clean, force, lateral, reverse, video-force의 `segments.csv`는 각 mode 안에서 네 arm의 SHA-256이 같다.
- 네 MP4도 모두 1,618,931 bytes이고 SHA-256 `d5a3a3d442e0072e50186857b0e146cef2f3e29986fdcce44f3ce1ea8a6730a7`로 byte-identical이다.
- jitter/combined를 포함한 report의 측정값도 같다. 다른 것은 checkpoint/config 경로, 생성 시각, wall-clock metadata뿐이다.

따라서 comparison의 H1/H2/H3 `nonworse` PASS는 개선 증거가 아니라 동일 policy의 equality다. 올바른 결론은 두 층으로 나뉜다.

1. **채택 결과는 유효하다:** selector가 arm당 평가한 32개 후보 중 31개 nonzero 학습 checkpoint가 warm-start보다 종합적으로 나빠서 배포 후보를 만들지 못했다. selector가 `model_0`을 보호한 것은 정상 동작이다. run마다 생성된 121개 전체를 정밀평가한 것은 아니므로, 평가되지 않은 checkpoint까지 모두 나쁘다고 확대 해석하지 않는다.
2. **처치 효과의 정밀 full-suite 비교는 성립하지 않는다:** 동일 policy 결과에서 H1/H2/H3 modification의 효과가 0%였다고 주장하면 안 된다. 학습된 처치 policy가 full suite에 한 번도 노출되지 않았다.

현재 비교기의 총 verdict는 H0/H1/H2/H3 모두 **FAIL**이다. `model_0`이 네 arm 중 현재 best로 남았다는 것과 H gate를 통과했다는 것은 전혀 다른 말이다.

### 선택된 공통 warm-start의 절대 성능

아래 값은 네 arm에서 동일하다. H 수정의 효과가 아니라, 공통 held-out H 시험에서 G1 warm-start가 보인 baseline이다.

| 축 | 측정값 | 사전 기준 | 판정·댓글 |
|---|---:|---:|---|
| waypoint 위치 median | 6.899 cm | ≤5.52 cm | 25.0% 초과, FAIL |
| waypoint 위치 p90 | 11.761 cm | ≤7.42 cm | 58.5% 초과, FAIL |
| waypoint heading median | 3.593° | ≤2.54° | 41.5% 초과, FAIL |
| never-arrived | 12.713% | ≤1.5% | 8.48배, anti-collapse FAIL |
| 전체 fall | 24.473/1000 | ≤5/1000 | 4.89배, FAIL |
| waypoint fall | 3.927/1000 | ≤2/1000 | 1.96배, FAIL |
| path fall | 62.690/1000 | ≤5/1000 | 12.54배, 가장 큰 안전 실패 |
| path speed median | 0.8748 m/s | ≥0.95 m/s | 7.92% 부족 |
| 1 m/s 도달률 | 69.28% | ≥80% | 10.72%p 부족 |
| 1 m/s 도달 p90 | 3.398 s | ≤3.0 s | 13.27% 느림 |
| cruise pitch/roll/ωxy p90 | 7.5°/5.5°/1.30 rad/s | ≤20°/15°/3 | 값 자체는 PASS |
| cruise coverage | 2.756% | ≥5% | 기준의 55.1%뿐, 고속 안정성 인증 불가 |
| force 5 s survival | 97.193%, n=1,318 | ≥98% | 0.807%p 부족 |
| force 90% speed recovery | 97.436%, p90 0.10 s, n=234 | ≥90%, ≤2 s | 회복은 PASS |
| high-speed force recovery | 97.927%, p90 0.00 s, n=193 | H1/H2 상대 기준 | baseline 참고값 |
| jitter fall | 0.4219/env·min | ≤0.5 | PASS |
| combined fall | 0.4844/env·min | ≤0.5 | PASS이나 한계의 96.9%로 여유가 작음 |
| jitter/combined angular p90 | 3.2/3.2 rad/s | ≤3.0 | 둘 다 6.67% 초과, FAIL |

clean은 256 env×120 s에서 완료 4,584 segment와 fall 115회를 기록했다. 전체 시도 기준 4,699개 중 waypoint fall은 12회, path fall은 103회다. 즉 위치 오차만 약간 다듬으면 되는 상태가 아니라 **낙상의 89.6%가 path에서 난다.** path 명령속도 median 1.03 m/s에 실제 median 0.875 m/s이고 tracking ratio median은 0.88이다. speed bin이 높아질수록 ratio가 떨어져 현재 지속 가능한 ceiling은 대략 1.2–1.4 m/s 부근으로 보인다.

외력 1,654 events에서는 살아남은 event의 90% 속도회복이 빠르지만, 5 s survival은 전체 97.19%, 고속 subset 94.57%다. 따라서 “로봇 간 충돌 외력에도 이미 충분히 괜찮았다”고 결론 내릴 수 없다. 특히 **회복시간 통계는 살아남고 recovery eligibility가 생긴 사건에 조건부**이므로, 빠른 p90 0.10 s가 survival 실패를 상쇄하지 않는다. collision 5 s survival은 97.26%, support는 97.16%로 두 class 모두 98% gate 아래다.

goal jitter 단독은 fall budget을 통과하지만 angular p90이 3.2 rad/s라 흔들림 gate를 실패한다. force를 같이 넣으면 fall이 0.4219→0.4844/env·min으로 14.8% 증가하고 5 s survival은 95.27%가 된다. combined의 high-speed force record는 2개뿐이고 recovery eligible은 0개라, combined report로 고속 회복을 말하면 안 된다.

급격한 방향전환은 “방향을 잡는 것”과 “높은 속도로 계속 가는 것”이 분리된다.

- lateral 0.5 m/s: 99.87% 도달, p90 0.98 s; 0.8 m/s 58.2%, 1.0 m/s 9.57%.
- reverse 0.5 m/s: 100% 도달, p90 0.62 s; 0.8 m/s 73.5%, 1.0 m/s 23.41%.

따라서 옆/뒤 goal에서 속도가 급감한다는 관찰은 수치로도 맞다. 새 goal 방향으로의 0.5 m/s 응답은 빠르지만, 대부분이 0.8–1.0 m/s로 이어지지 않는다.

### H0 — 공통 low-dose robustness bundle

**의도:** G1의 속도/정확도를 보존하면서 작은 외란, goal jitter/flicker, encoder/target offset, 정확한 팔 asset을 추가한다.

**결과 댓글:** H0는 강건성 방향의 신호는 만들었지만, 그 대가로 목표 수행을 포기하는 policy를 학습했다. 아래 clean 값은 20 s×32 후보 stage, combined 값은 같은 길이의 후속 paired robust screen에서 가져온 iteration 0→400 변화다.

| 지표 | model 0 | model 400 | 변화 |
|---|---:|---:|---:|
| 위치 median | 7.25 cm | 17.16 cm | +136.5% 악화 |
| 위치 p90 | 11.57 cm | 24.48 cm | +111.6% 악화 |
| strict success | 17.42% | 7.76% | −55.5% |
| never-arrived | 12.91% | 59.54% | +46.63%p |
| raw fall | 29 | 3 | −89.7% |
| path fall | 84.29/1000 | 11.41/1000 | −86.5% |
| combined fall | 0.469/env·min | 0.328/env·min | −30.0% |
| combined force 5 s survival | 94.16% | 97.74% | +3.59%p |
| path speed | 0.911 m/s | 0.897 m/s | −1.5% |

iteration 12000은 짧은 screen에서 fall 0이지만 위치 median/p90이 39.4/45.9 cm, never-arrived 59.9%, strict success 5.49%다. 이것은 “안전하게 목표를 수행”한 것이 아니라 **가만히 있거나 목표 약 0.4 m 전에서 멈춰 낙상 노출을 피하는 robustness–task-collapse**다. 최종 120 s에서도 model 0이 model 100보다 위치 6.67/11.20 cm vs 10.49/16.34 cm, strict success 26.49% vs 8.99%로 명확히 낫기 때문에 selector가 model 0을 고른 것은 타당하다.

**판정:** H0 FAIL. H1–H3의 세부 레버보다 먼저 공통 H fine-tuning objective/distribution을 고쳐야 한다.

### H1 — mirror augmentation/loss + 강화 joint-position DR

**의도:** robot-local y축 대칭 augmentation와 mirror loss로 좌우 equivariance를 높이고, encoder ±0.025 rad/target ±0.020 rad/init-q σ0.075 rad로 calibration gap을 줄인다.

**결과 댓글:** H1 full suite는 model 0이므로 H1 처치의 최종 효과를 측정하지 못했다. 그러나 selection screen의 모든 학습 checkpoint는 조합이 역방향으로 갔다는 강한 진단 근거를 준다.

- warm-start mirror error p90은 0.080이다. 비zero checkpoint 31개의 범위는 0.135–0.165, 중앙값 0.150이고 기준 0.10 통과는 0/31이다. iteration 12000의 0.150은 초기보다 87.5% 크다.
- touchdown L/R median bias는 warm-start 2.9 cm에서 학습 checkpoint 최소 6.5 cm, 중앙값 8.7 cm, iteration 12000 9.4 cm로 악화했다. 기준 2 cm 통과는 0/31이다.
- H0/H1의 공통 비zero checkpoint 31개를 맞춰 비교하면 H1은 31/31에서 raw fall이 더 많고 strict success가 더 낮다. 31개 중 strict success가 정확히 0인 checkpoint도 20개다.

| 공통 비zero checkpoint의 중앙값 | H0 | H1 | H1 변화 |
|---|---:|---:|---:|
| raw fall | 3 | 12 | 4배 |
| strict success | 4.18% | 0% | 붕괴 |
| path speed | 0.851 | 0.772 m/s | −9.2% |
| path fall | 11.03 | 38.17/1000 | +246% |
| 전체 fall | 4.04 | 16.22/1000 | +301% |
| 1 m/s 도달률 | 82.69% | 73.47% | −9.22%p |
| 1 m/s 도달 p90 | 3.20 | 3.40 s | +6.2% |

120 s 후보 model 0→100에서도 위치 median 7.15→12.84 cm, p90 11.88→21.63 cm, strict success 23.56→8.06%, never-arrived 13.07→46.21%로 악화했다. fall은 96→89로 7.3%밖에 줄지 않았고 path fall은 51.78/1000으로 그대로다. combined screen도 fall 0.492→0.598/env·min, angular p90 3.20→3.46 rad/s, force survival 95.77→94.85%로 나빠졌다.

**판정:** H1 FAIL. 현재 묶음은 speed/accuracy/fall뿐 아니라 직접 목표였던 mirror error와 touchdown 좌우 bias까지 악화했다. 다만 mirror와 강화 joint DR를 한 arm에 함께 넣었으므로 어느 요소가 원인인지 이 결과만으로 분리할 수 없다. 다음에는 `mirror-only`, `joint-DR-only`, `둘의 interaction`을 따로 둬야 한다. mirror coefficient를 바로 낮춰 재실행하기 전에 mirror map/gradient와 joint-offset sensitivity sweep을 각각 검증한다.

### H2 — 가속 보존형 고속 안정화 + 강화 force/flicker bundle

**의도:** 가속 중 lean은 허용하고 steady high-speed에서만 pitch/roll/ωxy를 줄이며, 더 잦은 고속 외란과 flicker로 회복을 학습한다. H2는 reward 하나가 아니라 stability+disturbance schedule+flicker의 bundle이다.

**결과 댓글:** 선택된 H1/H2가 같은 model 0이므로 full suite의 차이는 전부 0이다. pitch 7.5°, roll 5.5°, ωxy 1.30 rad/s, 1 m/s p90 3.398 s, 고속 force recovery 97.927%/0 s는 H2의 성과가 아니라 공통 baseline이다. cruise coverage도 2.756%로 5% gate를 못 채웠다. comparison의 `nonworse`는 equality이고 `strictly improves`는 모두 FAIL이다.

selection의 초기 paired combined screen에는 유망하지만 확정할 수 없는 신호가 있다.

| checkpoint | H2 vs H1 combined fall | H2 vs H1 angular p90 | H2 vs H1 force survival |
|---|---:|---:|---:|
| iteration 100 | 0.398 vs 0.504, −20.9% | 3.36 vs 3.46, −2.9% | 96.95 vs 92.91%, +4.04%p |
| iteration 200 | 0.410 vs 0.750, −45.3% | 3.52 vs 3.60, −2.2% | 96.60 vs 91.61%, +4.99%p |

반면 clean은 iteration 100에서 H2 위치 median이 H1보다 7.0% 나쁘고, iteration 200에서는 8.2% 좋지만 raw fall이 44.4% 많아 일관되지 않는다. H2 iteration 12000도 model 0 대비 위치 median 7.25→39.58 cm(+445.7%), p90 11.57→45.61 cm(+294.3%), strict success 17.42→1.67%(−90.4%)로 공통 task collapse를 피하지 못했다. 같은 model 0의 짧은 robust screen도 arm/stage에 따라 survival이 약 2–3%p 흔들리므로 단일 seed·20 s 신호를 효과 확정으로 올리면 안 된다.

**판정:** H2는 채택 FAIL, 개별 bundle 효과는 **미식별**이다. 초기 robustness 신호 때문에 영구 폐기할 근거도 없지만, common objective를 고치기 전에 12k 재학습할 근거도 없다. 이후 동일 nonzero iteration 100/200을 최소 3 seeds의 paired full suite에 강제로 올리고, signal이 재현될 때 stability-only/force-only/flicker-only/interaction을 분해한다.

### H3 — forward heel touchdown gait-only ablation

**의도:** H0에 first-contact heel placement reward 하나만 더해 고속 path fall을 줄이는지 본다.

**선택 baseline의 gait 기술통계:** clean touchdown 20,520개로 표본 수는 충분하다. heel-ahead 26.657%, target±1σ 3.202%, overstride 0.0146%, heel x p10/median/p90 −15.8/−4.3/+3.6 cm, L/R bias 2.9 cm, precontact down-speed p90 1.98 m/s, contact-force p90 770 N이다. target share는 40% 기준의 8.0%뿐이고, 접지의 73.3%가 trunk 앞이 아니다. 하지만 H0/H3가 동일 model 0이므로 이것은 H3 결과가 아니라 시작점이다.

**결과 댓글:** H3 reward가 들어간 checkpoint는 의도한 target share도 개선하지 못했다.

- 20 s×32 checkpoint에서 target share 최고는 iteration 0의 2.508%다.
- iteration 100/200/300/12000은 각각 1.734/1.159/0.816/0.990%다.
- late checkpoint(≥3000) 중앙값은 0.949%이고, 어느 checkpoint도 iteration 0을 넘지 못했다.
- 120 s model 0→100은 target 3.554→2.341%(−34.1%), 위치 median 6.765→9.975 cm(+47.5%), p90 11.464→15.672 cm(+36.7%)다. fall은 96→39(−59.4%), path speed는 0.830→0.865 m/s(+4.2%)지만 직접 gait 목표와 정확도를 잃었다.

H0의 trained checkpoint에는 동일 raw touchdown metric이 모두 저장되지 않아 같은 iteration의 직접 H0/H3 gait 차이는 계산할 수 없다. 그럼에도 H3 내부에서 reward target이 일관되게 내려갔으므로 현재 구현/scale의 긍정적 증거는 없다.

**판정:** H3 FAIL, 현재 heel reward는 보류한다. “기존 외란이 충분했으니 hardcoding이 불필요하다”는 주장도 force 5 s survival 97.19%, high-speed 94.57%, path fall 62.69/1000 때문에 성립하지 않는다. gait reward를 다시 만지기 전에 reward activation 횟수·평균 크기·전체 reward 대비 비중·접지 proxy의 정확성을 계측해야 한다.

### 무엇은 실제로 성공했나

정책 성능과 하네스/기구 검증을 분리하면 성공한 부분도 분명하다.

- path controller는 426,921 step 중 steady 418,328 step을 확보했다. dwell-resume recovery를 제외한 `gap/lookahead<0.75`는 0%, leash 밖은 0.00234%, recovery share는 2.013%로 세 gate를 통과했다. 과거 lookahead-floor 실패는 controller 관점에서는 해결됐다.
- fall context 분류는 완료됐고 survivor-bias 보정 분모가 쓰였다.
- force는 5개 loadable body, collision/support class와 실제 event count를 남겼다. 외란은 한 군데에만 주지 않는다.
- lateral/reverse는 새 goal 방향으로 투영한 속도로 측정되어 옛 방향 momentum을 성공으로 잘못 세지 않는다.
- simulator-view 영상은 400 frames, force arrow 75, path carrot 400, trace 398을 report/manifest에 남겼다. 네 video가 동일한 것은 같은 model 0·seed·protocol의 추가 증거다.
- 모든 artifact hash는 복사 뒤 다시 맞았다. 단 로컬 runtime에 MP4 decoder가 없어 400-frame decode는 독립 재실행하지 못했고, server completion attestation과 MP4 hash를 검증했다.

### 공통 학습 붕괴의 가장 유력한 설명과 한계

H0에도 동일한 붕괴가 있고 H3는 H0와 거의 같은 초기 trajectory를 보이므로 mirror/stability/heel 같은 arm-specific 레버가 공통 원인은 아니다. 네 arm에서 학습이 진행될수록 위치가 약 0.4 m로 모이고 fall은 줄어드는 패턴은 **공통 objective/distribution이 움직임 위험보다 목표 수행 포기를 더 유리하게 만든 local optimum**과 일치한다.

코드상 특히 점검할 조합은 다음이다.

- `goal_progress=0`: 먼 거리에서 닫히는 속도에 대한 직접 dense 보상이 없다.
- `constellation_weight=0.2`: heading이 맞을 때 0.4 m 거리의 constellation 값은 `exp(-0.2×0.4²)=0.9685`로 목표점의 96.85%다. 즉 40 cm 앞에서 안전하게 멈춰도 dense goal reward 손실이 약 3.15%뿐이다.
- `goal_reached=1.0`: 0.1 m 안에서 정지해야만 받는 sparse bonus다.
- `only_positive_rewards=true`: motion cost가 큰 상태의 total reward가 0에 clip되면 과감히 접근하는 행동 사이의 차등 신호가 사라질 수 있다.
- 35% moving path, all-active speed×curvature grid, mandatory goal noise/joint offsets/disturbance, fresh optimizer, 새 arm asset/path semantics가 네 arm의 공통 변경이다.

이 조합은 데이터와 맞는 **원인 가설**이지 아직 단독 원인 증명은 아니다. 특히 학습 로그에 reward-term occupancy/크기와 gradient contribution이 없으므로 “오직 constellation 때문”이라고 단정하지 않는다. H0에서 robustness가 실제로 좋아진 만큼, 단순 코드 고장보다는 reward가 허용한 과도한 robustness–task trade-off가 더 유력하다.

### 다음 실행의 객관적 순서

1. **12k 재실행 금지:** 먼저 iteration 0/25/50/100/200만 저장한다. 100에서 waypoint median/p90, never-arrived, strict success, path speed/fall, combined survival을 model 0과 paired 비교하고 accuracy-preservation을 못 지키면 자동 중단한다.
2. **공통 변경 ladder:** exact G1 → H arm asset만 → 새 path semantics만 → all-active grid → goal jitter → joint offsets → disturbance 순으로 하나씩 추가해 최초 붕괴 지점을 찾는다. 각 단계는 같은 checkpoint/seed/eval protocol을 쓴다.
3. **reward 감사:** env-step별 constellation, goal-reached, survival, action-rate, orientation, collision 등 각 term의 mean/nonzero share와 clipped-to-zero share를 저장한다. 0.3–0.5 m에 정체된 env와 정상 도착 env를 분리한다. 이 증거 뒤에만 low-dose progress/sharper distance shaping 같은 reward arm을 정의한다.
4. **selector 상태를 명시:** 모든 arm이 iteration 0을 선택하면 comparison을 숫자 equality로 PASS/FAIL하지 말고 `NO_TRAINED_ARM_SELECTED`; 개별 ablation은 `ABLATION_NOT_EVALUATED`로 표시한다. 배포 후보 보호는 유지하되 과학적 결론을 분리한다.
5. **provenance 강화:** report에 policy checkpoint SHA-256, source warm-start SHA-256, effective training-config SHA를 넣고 모든 mode의 completion token을 non-null로 만든다. 현재 video 4개만 고유 token이 있고 non-video 24개는 token이 null이라 stale/misassociation 방어가 약하다.
6. **H1 분해:** 1차는 200-iteration fine-tune으로 mirror-loss ablation과 joint-DR-only를 분리한다. 같은 방향의 survivor만 paired seeds로 재확인하고, 그 뒤 interaction을 추가한다. held-out joint-offset severity grid를 공통 eval에 둔다.
7. **H2 분해:** iteration 100/200의 robustness 신호를 paired full suite로 재현한 뒤 stability-only, force schedule-only, flicker-only를 분리한다. cruise coverage≥5%가 아니면 고속 stability 결론을 내리지 않는다.
8. **H3 보류:** 모든 arm에 같은 raw touchdown logger를 적용하고 H0/H3의 fixed nonzero iteration을 맞춘다. activation/magnitude 검증 전에는 heel reward를 키우지 않는다.

최종 채택표:

| arm | 배포 채택 | modification 효과 판정 | 다음 행동 |
|---|---|---|---|
| H0 | FAIL | robustness는 늘었으나 task collapse가 훨씬 큼 | 공통 objective/distribution 원인 격리 |
| H1 | FAIL | 조합은 명백히 해로움; mirror와 DR 개별 원인은 미분리 | mirror-only/DR-only로 분해 |
| H2 | FAIL | 짧은 robustness 신호는 있으나 full-suite 미식별 | common 수정 후 3-seed fixed-checkpoint 재시험 |
| H3 | FAIL | 직접 touchdown target도 악화 | reward 보류, activation/proxy 감사 |

결론은 **HBatch가 새 winner를 만들지 못했다**이다. 동시에 selector가 warm-start를 지켜 잘못된 학습 policy의 채택을 막았고, path/force/direction/video 평가 하네스는 원인 규명에 쓸 수 있는 수준의 자료를 남겼다. 다음 실험의 첫 질문은 “H1/H2/H3 중 누가 이겼나”가 아니라 “왜 H0부터 iteration 100 안에 정확도와 도달률을 버리는가”여야 한다.

## humanoid locomotion sim-to-real에서 가장 자주 부딪히는 문제

우선순위는 다음과 같다.

1. **actuator dynamics와 control interface**: 실제 torque-speed/current limit, PD gain, motor lag, deadzone/backlash, battery voltage sag가 simulator의 이상적인 position target과 다르다. actuator model과 latency가 빠지면 real transfer가 실패할 수 있다. [Tan et al., RSS 2018](https://www.roboticsproceedings.org/rss14/p10.html).
2. **지연과 clock jitter**: 50 Hz policy, sensor sampling, 20 Hz goal pipeline, inference/transport 지연이 고정값이 아니다. timing variation은 단순 DR만으로 충분하지 않을 수 있다. [Sandha et al., CoRL/PMLR 2021](https://proceedings.mlr.press/v155/sandha21a.html).
3. **contact reality gap**: sole geometry, friction, compliance, floor unevenness, contact solver가 humanoid의 좁은 support polygon에서 큰 차이를 만든다. torso push만으로 link-level collision/contact를 대체할 수 없다.
4. **mass/CoM/inertia와 URDF mismatch**: 팔 자세, 케이블, 카메라, 배터리, fastener가 CoM과 yaw inertia를 바꾼다. 이번 arm overlap은 단순 visual 문제가 아니라 dynamics 문제다.
5. **encoder/IMU calibration**: joint zero, IMU mounting quaternion, bias/filter, observation order/normalization/default-q/action-unit 불일치가 policy 입력 전체를 어긋나게 한다. 현재 deploy 코드는 SDK의 index와 raw joint position을 그대로 쓰며 joint-name/order/zero 검증표가 없어 대비가 충분하지 않다. 실기에서는 무부하 기준자세에서 `/joint_states`의 name/position과 SDK serial index를 읽기 전용으로 수집하고, URDF zero/default-q와 signed offset table을 만들어 observation 직전에 적용해야 한다. 이 table의 SHA와 한-frame sim/real observation diff가 일치하기 전에는 motor command를 허용하지 않는다.
6. **robustness–optimality trade-off**: 넓은 randomization은 정책을 보수적으로 만들 수 있다. 프로젝트 내부의 E2/G2가 같은 경고이고, 고전 실험도 randomization의 robustness와 peak performance trade-off를 보고한다. [Tan et al., RSS 2018 PDF](https://roboticsproceedings.org/rss14/p10.pdf).
7. **외력 randomization의 한계와 효용**: random force와 episodic actuation offset은 transfer에 도움이 될 수 있지만 event 크기/위치/nominal 비중과 실제 평가가 필요하다. [Campanaro et al., L4DC/PMLR 2024](https://proceedings.mlr.press/v242/campanaro24a.html). M1은 무조건 균일 body sampling을 쓰지 않고, `arm_proxy 0.60 + chest 0.30`으로 상체·팔에 90% 이상을 집중한다. 세기와 duration은 bounded uniform, 로봇 기준 수평방향과 tier 내 접촉 높이는 uniform으로 뽑아 특정 외란 하나를 외우지 못하게 한다.
8. **reward/task/interface 반복설계**: sim-to-real은 one-shot이 아니며 state/action/reward 및 실제 interface를 반복 검증해야 한다. [Xie et al., Cassie CoRL/PMLR 2020](https://proceedings.mlr.press/v100/xie20a.html).
9. **export parity와 safety**: PyTorch→TFLite/ONNX의 normalization/action clipping/order, saturation, thermal/current limit, E-stop, tether test가 동일해야 한다.
10. **K1 공식 stack 차이**: Booster의 현재 training/deploy stack은 Isaac Lab/Isaac Sim, ONNX/TorchScript export, MuJoCo/real deployment 흐름을 제공한다. 현재 legacy Isaac Gym task와 asset/interface 차이를 diff해야 한다. [Booster Robotics `booster_train`](https://github.com/BoosterRobotics/booster_train).

실기 전 체크리스트:

- URDF mass/inertia/CoM/joint axis/limit/collision과 실제 payload 측정.
- 실제 PD gain, control period, torque/current/voltage 로그로 actuator fit.
- joint zero와 IMU mount/bias를 timestamp와 함께 기록.
- real observation vector를 한 frame dump해 sim vector와 element-wise 비교.
- TFLite/ONNX output을 동일 observation batch에서 PyTorch와 비교.
- stand→tether walk→slow goal→side/back→high speed→외력 순의 단계적 시험.
- target q, measured q, dq, current, IMU, goal age/confidence를 같은 clock으로 저장.

## 최종 질문별 짧은 답

- 고속 lean은 나쁜가? **가속 중 lean은 허용한다.** 이번 공통 원인 screen에서는 속도×가속 sigmoid reward와 그 curriculum을 모두 0으로 빼고, 다방향 force 학습 전후의 고속 생존·회복으로 먼저 검증한다.
- 기존 collision force는 충분히 컸나? **아니다. 명목 설정과 실제 전달 impulse가 달랐으므로 기존 결과로 충분성을 주장할 수 없다.** 새 모델은 `omni_shove`, 뒤에서 미는 `rear_push`, 팔/그물 걸림 `arm_entanglement`만 두고 정면 전속력 collision class를 제거했다. eval은 설정 impulse와 physics substep에 실제 제출된 impulse의 median/p90/max 상대오차를 기록한다.
- 그 force에도 괜찮았나? **외력 ON 평가가 없어 모른다.**
- 외력은 한 군데였나? **기존은 Trunk 한 곳이었고 새 모델은 Trunk/양 hip/양 shank의 loadable body와 네 높이 tier를 쓴다.** 수평 8방향 count, tier/body count와 z-offset을 report에 남겨 편향을 검출한다.
- heel-ahead reward로 8번이 해결되나? **탈락시켰다.** 모든 M-cell에서 scale 0이며 이후 production 후보에도 넣지 않는다.
- joint position DR은 충분했나? **아니어서 진행한다.** M2는 persistent encoder `±0.015 rad`, motor target `±0.010 rad`만 M0에 추가하고, clean과 같은 held-out offset probe로 비용과 이득을 분리한다.
- 옆/뒤 goal에서 왜 느리고 같은 방향인데도 가끔 빠른가? **현재 yaw·momentum·gait phase와 목표 발생 시점의 정렬 차이가 크고, 고정 cadence clock을 정책이 관측만 할 뿐 바꿀 수 없는 것이 편차를 키운다.** 단순 랜덤화는 phase coverage를 늘리지만 빠른 해로 수렴시킨다는 보장은 없다. 공통 M-cell 뒤에는 cadence를 policy action으로 늘리지 않고도 command speed/turn demand에 연속적으로 결합하는 phase-rate conditioning을 단독 ablation으로 시험한다. 이는 특정 방향 if문이 아니라 모든 명령에 같은 equivariant 관계를 적용하는 방법이다.
- G1 압승인가? **G군 내부에서 정확도와 속도를 함께 유지한 usable arm으로는 yes. raw speed만 보면 G3/G4 일부 수치가 더 높지만 overspeed·정확도 붕괴·낙상 때문에 winner가 아니다. E0 대비 accuracy/falls는 열세이고 기존 path_lag/grid로 숙련까지 주장할 수 없다.**
- E1/E0가 좋은가? **E0의 자기-config waypoint 평가는 실제로 존재하고 core 수치는 유효하다:** model 6200, 4,633구간, position 2.72/5.01 cm, heading 2.52°, strict 89.29%, 낙상 2회다. 다만 provenance가 새 H protocol보다 약하고 path 표본이 0이므로 “모든 과제를 합친 종합 1위”가 아니라 **waypoint 정확도 1위**라고만 말한다. E1의 구형 path 수치는 현재 path 의미가 달라 deploy winner 근거가 아니다.
- E2 robust는 왜 느려졌고 H에서 어떻게 막나? **요구속도는 같았는데 body p90이 44.8% 줄고 53.34%가 never-arrived한 저이득 collapse다. bundled no-ramp robustness와 action-rate/progress 구조가 가장 유력하며, H는 low-dose/ramp, early-checkpoint selection, 1.5% never-arrived·speed/acceleration gate와 공통 force-ON eval로 재발을 막는다.**
- 팔: 모든 M-cell은 `K1_locomotion.urdf`의 고정 팔 각도를 이중검사해 복제한 `K1_locomotion_hbatch-codex.urdf`를 사용한다. 이 필수 dynamics 차이 때문에 M0는 byte-identical G1이 아니라 **minimum-allowed G1 continuation**으로 표기한다.
- 학습: 구조/관측/액션을 바꾸지 않는 외란·offset·mirror-loss 평가는 짧은 fine-tune이 맞다. 200 iteration, checkpoint 0/25/50/100/200, fresh Adam `2e-6`(범위 `5e-7–2e-6`)으로 먼저 screen하고, 통과 레버만 paired seeds와 긴 학습으로 올린다. URDF DOF나 관측차원을 바꾸는 경우에는 이 결론을 재사용하지 않는다.

## 2026-08-01 mirror augmentation/loss 코드 감사 및 최소 fine-tune 설계

### 결론 댓글

H1의 실패를 단순히 “mirror coefficient가 너무 컸다”로 결론 내리면 안 된다. 현재 **obs/action/privileged mirror map 자체에는 HBatch 설정에서 명백한 부호·순열 오류가 없지만, transition PPO augmentation의 importance-ratio 기준분포가 잘못되어 있다.** 또한 H1은 mirror 두 항과 강화 joint-position DR를 동시에 바꿔 처치 효과가 섞였고, augmentation을 켜는 순간 critic·KL controller까지 함께 달라진다. 따라서 기존 H1은 mirror의 유효성 시험이 아니라 `mirror loss + 통계적으로 편향된 mirror PPO + 강화 DR + 다른 KL 제어`의 bundle 시험이다.

수치도 “좌우 대칭이 좋아졌지만 다른 성능을 희생했다”는 해석을 지지하지 않는다. warm-start의 mirror p90 `0.080`이 H1 비zero checkpoint 31개에서 `0.135–0.165`(중앙 `0.150`, 통과 `0/31`)로 오히려 커졌고, touchdown 좌우 bias도 `2.9 cm`에서 최소 `6.5`, 중앙 `8.7`, 마지막 `9.4 cm`로 악화했다. 같은 31개 checkpoint에서 H1은 H0보다 fall이 `31/31` 모두 많고 strict success가 `31/31` 모두 낮았다. 중앙값도 strict `4.18%→0%`, path speed `0.851→0.772 m/s`(−9.2%), path fall `11.03→38.17/1000`(+246%), 전체 fall `4.04→16.22/1000`(+301%)이다. 즉 현재 구현/조합에는 유효한 mirror 학습 신호가 관측되지 않았다.

### mirror map 감사 댓글

| 대상 | 현재 mapping | 감사 판정과 조건 |
|---|---|---|
| policy obs 54차원 | gravity-y, angular velocity x/z, goal-y, heading, body-roll target, feet-y target를 반전하고, L/R foot-yaw를 부호 반전하여 교환한다. gait cos/sin은 모두 반전하고, q/dq/last-action 세 12차원 block은 L/R 교환 뒤 Roll/Yaw만 부호를 바꾼다 (`goal_pose_v3.py:83–151`). | **현재 asset에서는 맞다.** obs 실제 layout은 `gravity 0:3 / ω 3:6 / command 6:16 / clock 16:18 / q 18:30 / dq 30:42 / action 42:54`와 일치한다 (`goal_pose.py:769–808`). cos/sin 동시 반전은 half-cycle 이동이고 L/R swing 중심이 0.25/0.75라 맞다 (`goal_pose.py:1043–1046`). |
| action 12차원 | 이름의 `Left_↔Right_`; Pitch/Knee는 `+`, Roll/Yaw는 `−` (`goal_pose_v3.py:96–145`). | **현재 K1 URDF의 좌우 joint axis/기본 자세와 일치한다.** 다만 이름 문자열로 축을 추론하므로 asset이 바뀌면 joint axis·limit 기반 정적 검사를 반드시 다시 해야 한다. |
| privileged obs 14차원 | raw COM-y latent `u→1-u`; body linear-vy와 force-y 반전; axial torque Tx/Tz 반전 (`goal_pose_hbatch.py:62–76`). | **현재 `base_com=U[-0.1,0.1]`에서 맞다.** `return_noise=True`가 실제 offset이 아니라 U[0,1] 원표본을 저장하기 때문이다 (`goal_pose.py:143–157`, `utils/utils.py:5–28`). Gaussian 또는 비대칭 범위로 바꾸면 `1-u`는 틀리므로 distribution/range를 static assert해야 한다. HBatch는 다섯 body 외력의 합을 robot frame으로 바꾼 뒤 critic 8:14에 넣으므로 vector 부호도 맞다 (`goal_pose_hbatch.py:263–273`). |

joint encoder bias와 motor target offset은 privileged obs에 없다. 분포가 좌우대칭이면 기대값 수준의 MDP 대칭은 유지되지만, 각 rollout 표본의 숨은 offset을 실제로 mirror하지 않은 채 같은 advantage/return을 복사하는 것은 exact per-sample symmetry가 아니다. H1처럼 offset 범위를 키울수록 이 근사가 나빠질 수 있다. 이것은 map의 부호 오류는 아니지만 mirrored critic/PPO target의 추가 오차원이다.

### loss·scale·gradient 감사 댓글

1. **가장 큰 오류는 mirrored PPO denominator다.** rollout action은 `a~π_old(.|s)`이고 합성 표본은 `(Ms, Ma)`다. permutation/sign mirror의 Jacobian 절댓값은 1이므로 이 합성 action을 실제로 낸 behavior density는

   `q(Ma|Ms) = π_old(a|s)`이다.

   따라서 PPO denominator는 원 `old_logprob(s,a)` 또는 `M#π_old(.|s)`의 log-prob여야 한다. 현재 코드는 `π_old(.|Ms)`를 만들고 `log π_old(Ma|Ms)`를 denominator로 쓴다 (`runner_v3.py:183–195`, `295–298`). old policy가 이미 완전 대칭일 때만 두 값이 같다. update 시작 시 numerator와 이 잘못된 denominator가 같은 network라 ratio가 인위적으로 1이 되는 것은 on-policy 증명이 아니다. 5σ support filter (`runner_v3.py:197–210`)도 표본분포와 denominator 불일치를 고치지 못한다. 따라서 앞의 “원 sample old log-prob를 재사용하지 않는 것이 맞다”는 설명은 이번 감사로 **폐기**한다.
2. **mirror actor scale은 평균이 아니라 추가 질량이다.** 총 loss는 ordinary actor `1.0`에 mirror actor `0.5`, symmetry MSE `0.5`를 별도로 더한다 (`runner_v3.py:306–319`, H1 YAML `45–46`). 즉 actor 쪽 score-gradient 질량만 nominal `1.5배`가 되고 consistency gradient까지 추가된다. 전체 gradient는 norm 1.0으로 한 번에 clip되므로 (`runner_v3.py:328–333`) mirror arm에서 clipping이 잦으면 value/ordinary PPO까지 함께 재조정된다. 항별 gradient norm·cosine·clip 빈도가 없어 실제 scale은 현재 로그로 감사할 수 없다.
3. **valid sample 비율로 mirror loss를 낮추지 않는다.** valid subset 내부 평균을 그대로 `0.5`배 하므로 (`runner_v3.py:278–298`) valid share가 작아져도 선택된 tail-safe 표본의 전체 minibatch 가중치는 줄지 않는다. support threshold `0.1`은 학습을 멈추는 안전장치일 뿐 unbiased weighting이 아니다.
4. **critic augmentation은 coefficient와 무관하게 항상 50:50이다.** augmentation coefficient가 양수이면 original/mirrored value MSE를 무조건 반씩 평균한다 (`runner_v3.py:257–265`). 그러므로 `mirror_augmentation_coef 0.5→0.25`로 낮춰도 critic symmetry 압력은 전혀 줄지 않는다. 같은 original return을 mirrored privileged state에 주는 근사와 hidden joint offsets가 함께 critic/advantage에 다시 피드백된다.
5. **symmetry MSE는 mean만 묶는다.** `MSE(μ(Ms),Mμ(s))` 양쪽 branch 모두로 gradient가 흐르는 식 자체는 올바른 equivariance objective지만 (`runner_v3.py:314–319`), 항별 gradient 충돌을 측정하지 않았다. actor의 `logstd`는 action마다 하나씩 학습되는 12차원 parameter (`model.py:27–32`)인데 L/R pair equality가 강제되지 않는다. 현재 deterministic mirror-error가 좋아져도 stochastic policy 분포는 비대칭일 수 있으므로 pairwise `|logσ_L-logσ_R|`를 별도 지표로 둬야 한다.
6. **augmentation이 KL schedule도 바꾼다.** H0/loss-only는 original KL만 보지만 augmentation arm은 optimizer step 뒤 original/mirror KL 중 큰 값으로 LR을 조절한다 (`runner_v3.py:343–380`). 그러므로 성능 차이는 data augmentation뿐 아니라 학습률 궤적 차이도 포함한다. original KL, mirror KL, LR을 모두 checkpoint마다 저장해야 한다.
7. bound penalty와 entropy는 original-state distribution에만 계산된다 (`runner_v3.py:300–311`). 주원인으로 볼 근거는 없지만 mirror state의 action-bound/entropy가 관리되지 않는 작은 비대칭이다.

### optimizer resume 감사 댓글

최종 H1 재실행은 G1의 잘못된 Adam LR `1.7086e-4`를 이어받지 않았다. H1은 `load_optimizer_state: false`, configured LR `5e-6`, clamp `1e-6–1e-5`다 (H1 YAML `24`, `38`, `48–49`; `runner_v3.py:116–128`). loader도 model/env state만 읽고 fresh Adam을 유지한다 (`runner.py:190–216`). 따라서 과거의 34.17배 LR/첫 launch NaN은 **최종 H1 악화의 설명이 아니다.**

반대로 짧은 실험을 중간 checkpoint에서 재개하면서 이 flag를 그대로 두면 매번 Adam moment를 버려 동일 run의 연속 학습이 아니게 된다. 아래 실험은 iteration 0부터 200까지 uninterrupted로 돌리고, 조기중단은 해당 run directory에 정확히 `STOP` 파일을 두어 checkpoint를 저장한 뒤 정상 종료한다 (`runner_v3.py:449–462`). 장애 후 같은 run을 진짜 resume할 때만 optimizer state를 복원하고 controller/param-group LR 일치를 확인한다.

### H1에서 악화한 원인의 우선순위

| 우선순위 | 가능한 원인 | 근거 | 현재 판정 |
|---:|---|---|---|
| 1 | mirrored PPO behavior-density mismatch | `π_old(a|s)` 표본에 `π_old(Ma|Ms)` denominator 사용 | **코드 수준 확정 오류**, augmentation 효과 판정 전에 수정 필요 |
| 2 | mirror+DR 동시 변경 | H1은 mirror 0.5/0.5와 encoder ±0.025, target ±0.020, init-q σ0.075를 동시에 변경 (`make_hbatch_configs.py:202–211`) | **실험 설계상 확정 confound** |
| 3 | 과도하고 불투명한 gradient scale | base actor +0.5 mirror actor +0.5 MSE, critic 50:50, global clip 1.0 | 유력; 항별 norm/cosine/clip 로그로 확인 필요 |
| 4 | mirrored critic/advantage의 hidden-latent 근사 | stronger target/encoder offsets는 명시적으로 mirror되지 않음; 같은 return 재사용 | 가능한 interaction; DR 고정 factorial로 분리 |
| 5 | logstd 대칭 누락 | mean equivariance만 직접 최적화 | deterministic p90과 stochastic 대칭의 괴리 가능 |
| 6 | optimizer/LR 34배 resume bug | 최종 재실행은 fresh Adam `5e-6` | **최종 H1 원인에서 제외** |
| 7 | obs/action/privileged sign/permutation 오류 | 현재 54/12/14 layout·URDF·vector 변환과 모두 일치 | **현재 asset/config에서는 제외**, config 변경 회귀검사만 유지 |

### 2-GPU 최소 mirror-only fine-tune 실험

**목적:** H0의 DR/objective를 한 글자도 바꾸지 않고 mirror mean loss와 transition augmentation의 주효과·상호작용만 분리한다. H1처럼 강화 joint DR를 섞지 않는다.

| cell | symmetry coef | transition augmentation coef | 질문 |
|---|---:|---:|---|
| M00 | 0 | 0 | 동일 fine-tune 자체의 paired control |
| M10 | 0.5 | 0 | mean-equivariance loss만 유효한가 |
| M01 | 0 | 0.5 | corrected transition augmentation만 유효한가 |
| M11 | 0.5 | 0.5 | 두 항이 보완/충돌하는가 |

공통 조건은 같은 G1 checkpoint SHA, H0의 low-dose 외란/jitter와 mild joint DR, 4096 env, horizon 24, 5 mini-epochs, 4 minibatches, fresh Adam `5e-6`, checkpoint `0/25/50/100/200`이다. model 0은 네 cell·seed에서 byte-identical이어야 한다. train seed는 `42, 31415, 27182` 세 개를 cell 간 paired로 쓰고, 각 checkpoint는 고정 eval seed `0`의 같은 observation bank/trajectory protocol로 비교한다. iteration 50과 100 생존 cell은 held-out eval seed `1`도 추가한다. 두 GPU는 seed마다 wave A `GPU0=M00, GPU1=M10`, wave B `GPU0=M01, GPU1=M11`로 돌려 동시에 다른 seed를 섞지 않는다. 총 6 wave이며 12k 전체학습은 하지 않는다.

다만 **현재 코드 그대로 M01/M11을 200까지 돌려 winner로 채택하면 안 된다.** 우선 현재 코드로 seed 42의 `0/25/50`까지만 4-cell 진단을 돌려 M10과 M01의 분리를 확인할 수는 있지만, M01/M11 결과는 위 denominator bug 재현 자료일 뿐 과학적인 augmentation 판정이 아니다. denominator와 weighting/diagnostic을 고친 뒤 세 seed 본 실험을 시작한다.

평가는 on-policy 분포만 쓰면 cell마다 방문상태가 달라 mirror error를 유리하게 만들 수 있으므로 두 층으로 한다.

- **고정 bank primary:** model 0/H0 rollout에서 미리 동결한 10만 obs에 대해 기존 정의의 mirror mean error median/p90/p99와 `|logσ_L-logσ_R|`의 6 joint-pair median/max를 계산한다.
- **on-policy primary:** 각 cell의 clean 20 s와 combined(force+jitter) 20 s, 256 env에서 같은 mirror 지표를 다시 계산한다.
- **task 보존:** waypoint position median/p90, strict success, never-arrived, path speed, 1 m/s 도달률·p90 time, 전체/path falls per 1000.
- **stability/robustness:** touchdown L/R median bias, combined falls/env·min, body angular-velocity p90, force 5 s survival.
- **학습 진단:** ordinary actor/mirror actor/symmetry/value loss; 각 항 단독 gradient norm과 pairwise cosine; total pre-clip norm·clip share; original/mirror KL와 LR; mirror valid share와 z p50/p90/max; `Δlogp=log π_old(Ma|Ms)-log π_old(a|s)` p50/p90/max; mirrored critic loss/explained variance; L/R logstd pair 값. `Δlogp`가 0이 아닌 정도가 바로 현재 denominator bias의 크기다.

조기중단은 다음처럼 사전 고정한다.

1. nonfinite, mirror valid share `<0.90`, checkpoint 누락/seed 불일치/model-0 hash 불일치는 즉시 중단한다.
2. iteration 25/50에서 paired M00 및 model 0보다 fixed-bank mirror p90이 `20% 이상` 악화하면서 동시에 path falls `>1.5배`, path speed `<95%`, strict success `<70%`, never-arrived `+10%p` 중 하나라도 발생하면 해당 cell을 STOP한다.
3. iteration 100에서 세 seed 집계 mirror p90이 M00보다 `≥10%` 개선되지 않으면 200으로 연장하지 않는다. 개선하더라도 position p90 `≤105%`, path speed `≥95%`, falls `≤105%`, touchdown bias 비열세를 모두 만족해야 한다.
4. winner는 3 seed 중 최소 2개에서 같은 방향이고 paired bootstrap 95% CI가 mirror 개선 0을 넘지 않으며, 위 task/safety 비열세를 모두 만족한 cell이다. 이 survivor만 iteration 200에서 H 공통 full suite를 실행한다.

**최종 판단:** fine-tuning 자체를 버릴 이유는 없다. 현재 결과는 12k가 필요하다는 뜻이 아니라 iteration 100 전 이미 잘못된 방향을 충분히 식별할 수 있었다는 뜻이다. 먼저 mirror augmentation estimator를 바로잡고, H0 조건의 200-iteration paired factorial로 학습 신호를 확인한 뒤 통과한 cell만 길게 학습하는 것이 시간과 인과성 모두에서 가장 안전하다.

## 2026-08-01 하단 코멘트 실행안 — M-cell causal screen

앞 절의 `M00/M10/M01/M11` mirror-only 4-cell 안은 **실행 전에 폐기·대체했다.** 이유는 두 가지다.

1. 실제 서버의 G1 run config를 다시 대조하니 G1은 이미 `symmetry_coef=0.5`였다. 따라서 mirror 0→0.5를 “추가 효과”로 시험하면 실제 warm-start history와 맞지 않는다.
2. H0 YAML은 G1과 asset뿐 아니라 `stand_posture -1`, stop angular threshold 0.3, path controller/goal noise/외란/optimizer가 달라 pure control이 아니다. H0를 기반으로 한 실험은 공통 붕괴를 다시 묶는다.

새 screen은 recorder가 저장한 실제 G1 `config.yaml`의 SHA-256 `5eb9aa12a46759624babe1b9d7a3c1c52028b2c3c5f243e6512cc7fa47e3910c`를 입력에서 강제한다. hash가 다르면 GPU를 띄우지 않는다. G1의 command/path/reward/noise 의미를 보존하고, 필수 팔 asset·HBatch task class·fresh optimizer만 공통으로 바꾼다.

| cell | M0 대비 유일한 처치 | 직접 묻는 질문 |
|---|---|---|
| `M0_control-codex` | 없음 | minimum-allowed G1 continuation 자체가 200 iteration 안에 무너지나 |
| `M1_force-codex` | scenario-aware disturbance | 현실적 다방향 외란 노출이 clean 성능 비용 없이 5 s 생존/회복을 높이나 |
| `M2_jointdr-codex` | encoder ±0.015, target ±0.010 rad | persistent joint calibration DR가 held-out offset에 실제 이득을 주나 |
| `M3_mirror_off-codex` | G1 mirror loss 0.5→0 | 기존 mirror loss를 유지할 근거가 있나 |

공통 optimizer는 saved G1 Adam을 복원하지 않는 fresh Adam이며 LR `2e-6`, adaptive bounds `5e-7–2e-6`, desired KL `0.003`이다. `runner.save_interval=25`, `runner.load_optimizer_state=false`를 올바른 schema 위치에 강제한다. model 0은 각 run에 같은 warm-start bytes를 복사하고 네 SHA가 다르면 eval을 거부한다.

1차 diagnostic에서는 causal isolation을 위해 legacy global kick/push, external scenario, persistent joint offsets를 M0에서 모두 0으로 둔다. 이는 “이후 모든 **생산 후보**에는 외란과 jitter가 있어야 한다”는 원칙의 예외가 아니라, 레버 효과를 식별하기 위한 짧은 control이다. survivor를 합친 production candidate에는 low-dose scenario disturbance와 goal jitter/flicker를 반드시 다시 넣고 interaction을 확인한다.

### 외란 구현·검증 변경

- `scenario_aware.enabled`가 설정만 있고 scheduler fire path에서 호출되지 않던 연결 오류를 수정했다. event 생성 시 `_sample_scenario_events`가 실제 `pushing_forces/torques`를 채우고, Isaac Gym `ENV_SPACE` tensor API까지 이어진다.
- event onset에서 robot-local 방향을 world로 한 번 회전한 뒤 contact 동안 고정한다. 로봇이 회전할 때 외력 방향이 같이 도는 비물리 현상을 막는다.
- `omni_shove 0.50`, `rear_push 0.30`, `arm_entanglement 0.20`이며 정면 전속력 collision class는 없다. rear push는 robot `+x` 주변 ±22.5°, 나머지 두 class는 수평 전방향이다.
- 높이 tier는 shank 0.05, hip 0.05, chest 0.30, arm proxy 0.60이다. scenario 조건까지 합치면 상체/팔 event의 기대비중은 95%다. fixed arm link가 Trunk에 collapse되므로 chest/arm 접촉은 Trunk COM force와 `r×F + twist`의 등가 wrench로 표현한다. 실제 두 로봇 contact dynamics라고 과장하지 않는다.
- physics 500 Hz substep마다 실제 제출된 force/torque tensor를 적분하고 event analytic impulse와 비교한다. smoke는 max relative error `≤5e-4`를 요구한다.
- eval report에는 scenario/tier/body count, robot-local 8 octant count, z contact offset, expected/submitted impulse 오차와 scenario별 5 s survival/회복을 남긴다. 이 값으로 “외란을 랜덤화했다”는 설정문이 실제 표본분포와 전달량으로 확인된다.

### mirror와 joint DR 변경

transition mirror PPO의 잘못된 denominator `log π_old(Ma|Ms)`를 source behavior density `log π_old(a|s)`로 수정했다. signed permutation의 Jacobian 절댓값이 1이기 때문이다. old mirrored policy는 tail-support 검사와 KL 진단에만 쓴다. 하지만 1차 M-cell은 이 수정과 별개로 augmentation coefficient를 전부 0으로 유지해 새 코드와 loss ablation을 섞지 않는다.

joint DR는 세 층을 분리한다.

- `init_dof_pos`: G1에도 있던 episode reset pose noise이며 전 cell 공통이다.
- `joint_encoder_bias`: policy observation의 q에만 들어가는 episode-constant calibration error다.
- `joint_target_offset`: PD motor target에만 들어가는 episode-constant actuator zero error다.

M2는 뒤의 두 항만 추가한다. eval에서는 clean을 모두 0으로 만들고, 별도 joint mode에서 정확히 encoder ±0.015/target ±0.010 rad를 M0와 M2에 같은 seed로 준다. 따라서 “DR를 넣은 policy가 자기 training randomization에서만 좋아 보이는가”가 아니라 같은 held-out 오차에서 p90/never-arrived/path speed/fall이 실제 개선되는지를 본다.

### 두 A6000 실행·평가 계획

과거 동일 서버에서 카드당 2 process가 GPU 96–98%, VRAM 약 8 GB/49 GB였으므로 네 training cell을 한 wave로 둔다.

- GPU 0: M0 + M2
- GPU 1: M1 + M3

각 cell은 4,096 env, horizon 24, 5 mini-epochs, 4 minibatches, 200 iteration이다. inference/mechanics smoke를 먼저 cell별로 통과시키고, 학습 시작 뒤 2 iteration의 post-update finite/gradient/LR health marker를 240초 안에 요구한다. 실패 cell만 별도 failure log로 보내고 통과 cell은 계속한다.

평가는 “라운드로빈이라고 적고 실제로 순차 실행”하던 오류를 고쳐 GPU당 persistent worker queue 하나를 병렬 실행한다.

- clean: 네 cell × iteration 0/50/100/200, observation/joint/external perturbation 모두 0.
- force: M0/M1만 같은 held-out scenario force로 평가.
- joint: M0/M2만 같은 held-out encoder/target offset으로 평가.
- mirror: clean rollout의 `RMS(π(Ms)-Mπ(s))` p90을 M0/M3에서 비교.

총 eval은 48회가 아니라 27회이고 model 0은 mode당 한 번만 평가한다. 1차 결과는 screening으로만 사용한다. clean task non-inferiority와 직접 목표(force survival, joint probe p90, mirror p90)를 동시에 만족한 레버만 train seeds `31415, 27182`를 추가한다. 그 뒤 survivor interaction과 mandatory low-dose 외란+jitter를 넣은 production candidate를 학습한다.
