# H-batch 설계·구현·검증 계획 — Codex

이 문서는 E0/E1/E2/V7과 G1–G4의 모든 결론, 사용자의 하위 질문, H0–H3 정의, 학습/평가 하네스, sim-to-real 조사 결과를 한 곳에 집대성한다.

## Codex 서버 작업 경계

- 서버에서는 사용자 소유 경로 `/mnt/DATA/workspace/ws_eungkyu/k1-goalpose` 안만 조회한다. 다른 사용자의 workspace는 열람하지 않는다.
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
cd /mnt/DATA/workspace/ws_eungkyu/k1-goalpose
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

## humanoid locomotion sim-to-real에서 가장 자주 부딪히는 문제

우선순위는 다음과 같다.

1. **actuator dynamics와 control interface**: 실제 torque-speed/current limit, PD gain, motor lag, deadzone/backlash, battery voltage sag가 simulator의 이상적인 position target과 다르다. actuator model과 latency가 빠지면 real transfer가 실패할 수 있다. [Tan et al., RSS 2018](https://www.roboticsproceedings.org/rss14/p10.html).
2. **지연과 clock jitter**: 50 Hz policy, sensor sampling, 20 Hz goal pipeline, inference/transport 지연이 고정값이 아니다. timing variation은 단순 DR만으로 충분하지 않을 수 있다. [Sandha et al., CoRL/PMLR 2021](https://proceedings.mlr.press/v155/sandha21a.html).
3. **contact reality gap**: sole geometry, friction, compliance, floor unevenness, contact solver가 humanoid의 좁은 support polygon에서 큰 차이를 만든다. torso push만으로 link-level collision/contact를 대체할 수 없다.
4. **mass/CoM/inertia와 URDF mismatch**: 팔 자세, 케이블, 카메라, 배터리, fastener가 CoM과 yaw inertia를 바꾼다. 이번 arm overlap은 단순 visual 문제가 아니라 dynamics 문제다.
5. **encoder/IMU calibration**: joint zero, IMU mounting quaternion, bias/filter, observation order/normalization/default-q/action-unit 불일치가 policy 입력 전체를 어긋나게 한다.
6. **robustness–optimality trade-off**: 넓은 randomization은 정책을 보수적으로 만들 수 있다. 프로젝트 내부의 E2/G2가 같은 경고이고, 고전 실험도 randomization의 robustness와 peak performance trade-off를 보고한다. [Tan et al., RSS 2018 PDF](https://roboticsproceedings.org/rss14/p10.pdf).
7. **외력 randomization의 한계와 효용**: random force와 episodic actuation offset은 transfer에 도움이 될 수 있지만 event 크기/위치/nominal 비중과 실제 평가가 필요하다. [Campanaro et al., L4DC/PMLR 2024](https://proceedings.mlr.press/v242/campanaro24a.html).
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

- 고속 lean은 나쁜가? **가속 중 lean은 허용, steady cruise lean/ωxy만 줄인다.**
- 기존 collision force는 충분히 컸나? **명목 설정은 컸지만 substep 적용 오류로 실제 impulse는 약 1/10이어서 충분했다고 볼 수 없다. H부터 전 substep 적용으로 수정했다.**
- 그 force에도 괜찮았나? **외력 ON 평가가 없어 모른다.**
- 외력은 한 군데였나? **기존은 Trunk 한 곳. H는 5개 body 중 하나로 분산한다.**
- heel-ahead reward로 8번이 해결되나? **보장 못 하며 overstride 위험이 있어 H3로만 격리한다.**
- joint position DR은 충분했나? **아니다. persistent encoder/motor offset이 빠져 있어 추가했다.**
- 옆/뒤 goal에서 왜 느리고 같은 방향인데도 가끔 빠른가? **pose goal의 이동방향과 final heading이 분리되고 constellation이 둘을 묶어, 빠른 몸회전보다 느린 side/back-step이 보상상 합리적이기 때문이다. 반대로 path heading/combined `dtheta`가 진행방향과 맞거나 현재 yaw·momentum·gait phase가 이미 유리하면 빠르다. viewer의 world 방향과 robot-local goal 방향이 다른 것도 육안상 “같은 방향” 편차를 만든다.**
- G1 압승인가? **G군 내부에서 정확도와 속도를 함께 유지한 usable arm으로는 yes. raw speed만 보면 G3/G4 일부 수치가 더 높지만 overspeed·정확도 붕괴·낙상 때문에 winner가 아니다. E0 대비 accuracy/falls는 열세이고 기존 path_lag/grid로 숙련까지 주장할 수 없다.**
- E1/E0가 좋은가? **E0는 종합 accuracy 1위, E1은 구형 path 속도 1위지만 현재 deploy winner 근거는 없다.**
- E2 robust는 왜 느려졌고 H에서 어떻게 막나? **요구속도는 같았는데 body p90이 44.8% 줄고 53.34%가 never-arrived한 저이득 collapse다. bundled no-ramp robustness와 action-rate/progress 구조가 가장 유력하며, H는 low-dose/ramp, early-checkpoint selection, 1.5% never-arrived·speed/acceleration gate와 공통 force-ON eval로 재발을 막는다.**
