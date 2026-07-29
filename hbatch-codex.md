# H-batch 설계·구현·검증 계획 — Codex

이 문서는 E0/E1/E2/V7과 G1–G4의 모든 결론, 사용자의 하위 질문, H0–H3 정의, 학습/평가 하네스, sim-to-real 조사 결과를 한 곳에 집대성한다.

## 한 줄 결정

**G1@10700의 speed/path 계보를 warm-start로 쓰되, E2/G2처럼 robust 레버를 한꺼번에 세게 넣지 않는다.** 모든 H 버전에 낮은 dose의 외란과 goal jitter를 의무화하고, H1/H2에는 y-axis mirror augmentation+loss, H2에는 가속 lean을 허용하는 순항 전용 안정화, H3에는 gait touchdown 하나만 격리한다.

## 데이터가 말하는 현재 best

- 정확도 best: E0@6200, `2.72/5.01 cm`, heading `2.52°`, strict `89.29%`, falls 2.
- speed best: G1@10700, path mean-speed median `1.038 m/s`, body p90 `1.50 m/s`.
- G1의 비용: waypoint `5.52/7.42 cm`, strict `34.22%`, falls 38 중 path 34.
- robust 실패 경고: E2 body p90 `0.32 m/s`, never-arrived `53.34%`; G2 body p90 `0.19 m/s`, never-arrived `63.27%`.

따라서 H의 출발점은 G1이고, E0 수치는 clean accuracy reject gate로 사용한다. E0와 G1을 weight-level에서 동시에 “합치는” 것은 불가능하므로, G1이 잃은 waypoint 정확도를 dwell/selection gate로 되찾는 방식이다.

## H0–H3 frozen 정의

공통:

- task `K1/Goal_Pose_HBatch`, observation/action `54/12` 유지.
- warm start G1@10700.
- waypoint/path `0.65/0.35`, G1 speed×curvature grid+dwell 유지.
- legacy synchronized velocity kick는 0으로 끈다.
- 새 팔 asset `K1_locomotion_hbatch-codex.urdf` 사용, arm script와 16-DOF armswing은 사용하지 않는다.
- 모든 버전에 goal observation jitter, segment bias, 2–3 step hold, rare flicker와 multi-body force를 nonzero로 넣는다.
- 모든 버전에 reset pose DR + episode-constant encoder bias + motor-target offset을 넣는다.

| 버전 | modification | 가설 | 다른 버전과의 차이 |
|---|---|---|---|
| **H0** | G1 + 정확한 팔 + low-dose 외란/jitter + mild joint offsets | 현재 best speed를 최소 필수 robustness와 함께 보존 가능한가 | 통합 기준선 |
| **H1** | H0 + y mirror transition augmentation + mirror loss + stronger joint offset DR | 좌우 편향과 calibration gap을 nominal speed 손실 없이 줄이는가 | stability/gait reward 없음 |
| **H2** | H1 + steady-high-speed stability reward + 더 긴 disturbance ramp | 가속 lean/가속시간은 유지하면서 순항 pitch/roll/ωxy를 낮추는가 | H1 대비 stability 하나가 핵심 |
| **H3** | H0 + forward-path touchdown placement reward 하나 | heel/capture-point형 foot placement가 고속 낙상을 줄이는가 | H1/H2를 상속하지 않는 gait-only ablation |

정확한 config:

- H0 force: interval 8–14 s, event probability 0.25, collision share 0.25, collision `40–100 N`, `3–12 N·m`, `0.05–0.10 s`, support `3–8 N`, `0.2–1 N·m`, `0.5–1.5 s`.
- H1: H0 force 그대로, encoder bias ±0.025 rad, target offset ±0.020 rad, init q σ 0.075 rad, mirror augmentation 0.5, mirror loss 0.5.
- H2: interval 6–12 s, base probability 0.35, collision share 0.35, 72,000 control-step ramp; path `v≥0.8 m/s`에서는 event probability를 2배(상한 1.0)로 높이고 high-speed stability scale은 −0.5.
- 학습 ramp는 학습에만 적용한다. `--keep_perturbations` 평가는 새 프로세스의 step 0에서 시작하므로, 평가 시 `ramp_steps=1`로 덮어써 설정된 최종 외란 분포를 실제로 검사한다.
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

기존 V7의 collision은 `40–150 N × 0.05–0.15 s`, 즉 충격량 `2–22.5 N·s`로 **크기 자체는 robot collision급**이다. 그러나 다음 이유로 실제 robot–robot collision이라 부르면 안 된다.

- 기존 구현은 `Trunk/base_indice` 한 rigid-body COM에만 force와 독립 random torque를 `LOCAL_SPACE`로 가했다.
- 두 번째 로봇 actor, 형상 접촉, 팔/다리 타격, 접촉점 moment arm, 상대 로봇 dynamics가 없다.
- 긴 support force도 로봇이 돌면 방향이 같이 돌았다.
- 외란 중 fall/reset 뒤 active force가 새 episode에 남을 수 있었다.

그리고 clean 및 jitter stress에서 모두 외력이 OFF였으므로 **괜찮았는지는 미검증**이다. G2/E2의 0-fall clean은 force recovery 증거가 아니다.

HBatch는 다음을 수정했다.

- Trunk, 양 hip-roll, 양 knee body 중 하나를 event마다 선택한다. fixed 팔 link는 `collapse_fixed_joints=true`에서 Trunk로 합쳐지므로 upper-arm을 별도 body로 거짓 계수하지 않았다. 로드 후 5개 이름 중 하나라도 없으면 env 생성을 즉시 실패시킨다.
- ENV_SPACE force로 world 방향을 유지.
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
- asymmetric critic의 14 privileged channels도 mirror한다: COM-y, linear-vy, force-y sign flip; torque는 axial vector라 Tx/Tz sign flip, Ty 유지.
- H0/H3는 G1 계보의 기존 mirror loss는 유지하지만 transition augmentation는 0. H1/H2에서 augmentation 순효과를 본다.

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
- path-only commanded/achieved speed, lag, keepup, falls per attempt.
- reset 뒤 0.25 s를 제외한 pose-difference speed. path category가 아닌 sample은 speed-tracking에서 강제 제외.

고속 phase:

- acceleration: `v>0.3` 및 forward acceleration `>0.3`; pitch median/p90, time-to-0.8/1.0.
- cruise: `v≥0.8` 및 `|axy|≤0.3`; pitch/roll abs p90, `|ωxy|` p90, `|vz|` p90, cruise exposure.
- H2 채택 시 acceleration pitch 자체는 reject 지표가 아니고 가속시간 퇴화 여부를 본다.

외란:

- event count/active duty, force body/direction/magnitude, force-active falls.
- impulse/torque-impulse, max tilt, speed loss, 90% speed recovery time·≤5 s recovery share, 2/5 s survival을 collision/support별로 분리해 report에 저장한다. event와 5 s outcome record 수가 다르면 censored/overlap로 명시한다.

방향전환:

- `--goal_pattern lateral`, `--goal_pattern reverse`를 별도 실행.
- lateral은 근접 0 m 목표가 섞이지 않게 좌/우 부호를 무작위로 고른 `|dy|=1–2 m`, reverse는 `dx=−1–−2 m`로 고정한다. 둘 다 `dtheta=0`이다.
- switch 후 0–2 s min speed, time-to-0.5/0.8/1.0, initial bearing, gait-phase quarter별 응답을 `segments.csv`/JSON에 저장한다. heading/travel alignment time-series는 추가 후속 지표다.

symmetry/DR:

- mirror involution `M(Mx)=x`, action/obs permutation bijection, policy equivariance error.
- fixed encoder bias grid와 joint-group별 성능 저하.
- feet fore-aft asymmetry, lateral offset, hip-yaw bias는 마지막 snapshot이 아니라 rollout time-series로 승격해야 한다.

## 영상: top view 대신 simulator 시점에 표시

새 RGB logger나 debug actor를 추가하지 않았다. 기존 Isaac Gym `IMAGE_COLOR` RGBA sensor와 frame 수집을 그대로 쓰고, 매 frame follow-camera pose/FOV 숫자만 저장한다. 후처리에서 world 3D를 perspective image로 투영한다.

- 최근 moving-carrot trace: 녹색 path.
- 현재 path lookahead/carrot: amber `PATH CARROT`.
- waypoint: 녹색 `WAYPOINT GOAL`과 heading arrow.
- 외력: 선택된 body COM에서 시작하는 빨간 화살표. 길이는 force 크기의 제곱근으로 scale.
- path mode에는 별도 “final goal”이 없으므로 존재하지 않는 goal을 거짓으로 그리지 않는다.
- H config에서 top-down constellation inset은 끄고 실제 simulator perspective overlay를 사용한다.

2초 RGBA video smoke는 학습 전 실행되며 mp4 존재와 force event>0을 검사한다. 이는 과거 RGB/RGBA와 “record_video를 env build 뒤 켜서 graphics device가 −1이 된” 실패를 조기에 막는다.

## train + eval 단일 하네스

`tools/run_hbatch_suite.sh`:

1. H0–H3 config 생성.
2. arm별 static → 300-step dynamic → 2 s perspective video smoke를 독립 실행.
3. 통과한 arm만 GPU에 올린다.
4. 성공 smoke log는 삭제하고, 실패한 arm만 `logs/hbatch/smoke_failures/Hx-codex.log`에 남긴다.

`tools/train_and_eval_hbatch.sh`:

1. train.
2. warm-start를 `model_0.pth` 후보로 넣고 100/200/… 초기 checkpoint까지 selection에 강제 포함한다. 기존 tail 60% 편향을 제거한다.
3. clean, force ON, goal-jitter, jitter+force, lateral, reverse, perspective force-video를 같은 run config로 평가한다.
4. selection/report/segments/video를 arm별 공유 결과 폴더로 묶는다.
5. 완료된 최신 H0–H3를 lock 하에 다시 비교해 `shared_eval_videos/hbatch/hbatch-comparison-codex.md`/`.json`을 갱신한다. H1/H2의 95% speed 비열세·10% 가속시간 회귀·순항 안정성, H3의 H0 대비 path fall 감소를 여기서 cross-arm gate로 판정한다.

서버 실행:

```bash
cd htwk-gym
conda activate k1goalpose
bash tools/run_hbatch_suite.sh
```

현재 로컬에는 Isaac Gym/CUDA/PyYAML runtime이 없어 GPU dynamic/video smoke와 학습은 실행하지 않았다. Python syntax, shell syntax, config/URDF static invariant는 로컬에서 검사한다. 실제 GPU launch는 위 하네스가 smoke 통과 arm에만 수행한다.

## H 버전별 채택 기준

| 판정 축 | H0 | H1 | H2 | H3 |
|---|---|---|---|---|
| waypoint | G1 5.52/7.42 cm보다 악화 금지, 목표는 median≤5 cm | H0 비열세 | H1 비열세 | H0 비열세 |
| speed | path mean median ≥0.95 m/s | H0의 95% 이상 | H1의 95% 이상; time-to-1.0 퇴화≤10% | H0의 95% 이상 |
| falls | G1 path fall 2.06%보다 감소 | H0 비열세 | H1보다 감소 | H0보다 감소해야 gait 가설 지지 |
| force | event>0, 5 s survival 목표≥98% | 동일 | speed-conditioned recovery 개선 | H0와 동일 |
| stability | baseline 확보 | symmetry만 개선 | cruise pitch/roll/ωxy 개선, accel pitch 허용 | touchdown/impact만 개선 |
| symmetry | 기록 | mirror error p90≤0.10 및 gait 좌우 bias 감소 | H1 유지 | 가설 아님 |

H0가 G1 speed/accuracy를 보존하지 못하면 H1–H3의 해석 전에 dose를 더 낮춘다. H1이 speed를 해치면 mirror coefficient를 0.5→0.25로 낮춘다. H2가 가속을 10% 이상 늦추면 global pitch를 만지지 않고 steady gate/scale만 낮춘다. H3는 H0 대비 fall/impact 개선이 없으면 폐기한다.

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
- 기존 collision force는 충분히 컸나? **크기는 충돌급이나 trunk 한 점 wrench라 현실성은 부족하다.**
- 그 force에도 괜찮았나? **외력 ON 평가가 없어 모른다.**
- 외력은 한 군데였나? **기존은 Trunk 한 곳. H는 5개 body 중 하나로 분산한다.**
- heel-ahead reward로 8번이 해결되나? **보장 못 하며 overstride 위험이 있어 H3로만 격리한다.**
- joint position DR은 충분했나? **아니다. persistent encoder/motor offset이 빠져 있어 추가했다.**
- 옆/뒤 goal에서 왜 느린가? **pose goal의 이동방향과 final heading이 분리되고 constellation이 둘을 묶어, 빠른 몸회전보다 느린 side/back-step이 보상상 합리적이기 때문이다.**
- G1 압승인가? **G군 내부 yes, E0 대비 accuracy/falls는 열세.**
- E1/E0가 좋은가? **E0는 종합 accuracy 1위, E1은 구형 path 속도 1위지만 현재 deploy winner 근거는 없다.**
