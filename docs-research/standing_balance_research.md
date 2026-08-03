# K1 single-support instability on hardware — research report

**Date:** 2026-08-03 · **Subject:** E0 GoalPose policy stands/walks in Isaac Gym, is unstable standing on the real Booster K1 and falls when one foot lifts.

**Confidence key**
`[WS]` well-supported — published result, or directly verified in this repo's code/URDF/config.
`[PL]` plausible — consistent with multiple sources and with the measured numbers, but not directly demonstrated for K1.
`[SP]` speculative — my inference; test before acting.

---

## 0. Verdict on "add a reward on body height"

**Reject as the primary fix. `[WS]` for the facts, `[PL]` for the conclusion.**

Three independent reasons, all checkable in the repo:

1. **It is already there, at full strength.** `htwk-gym/envs/K1/Goal_Pose_V7.yaml` has `rewards.scales.base_height: -20.` and `rewards.base_height_target: 0.52`. That is the *same weight Booster ships in their own reference* ([`booster_gym/envs/T1.yaml`](https://github.com/BoosterRobotics/booster_gym/blob/main/envs/T1.yaml) also uses `base_height: -20.`) and the same weight in the Booster Gym paper's Table II (`(h_des − h)²`, weight −20.0, [arXiv:2506.15132](https://arxiv.org/html/2506.15132v1)). `orientation: -20.` is 4× Booster's `-5.`. The friend's fix is a no-op here.

2. **It is numerically inert in the failure regime.** I ran forward kinematics on `resources/K1/K1_locomotion_armsdown.urdf` at E0's `default_joint_angles` (Hip_Pitch −0.2, Knee_Pitch 0.4, Ankle_Pitch −0.25): **trunk sits at 0.5144 m**, versus the 0.52 m target. The nominal stance already satisfies the reward to within 6 mm. Per-step reward magnitudes:

   | base-height error | reward contribution | vs. `survival` = +0.25 | vs. `constellation` = up to +3.5 |
   |---|---|---|---|
   | 6 mm (nominal pose) | −0.0007 | 0.3 % | 0.02 % |
   | 20 mm | −0.008 | 3 % | 0.2 % |
   | 35 mm | −0.025 | 10 % | 0.7 % |
   | 112 mm | −0.25 | 100 % | 7 % |

   And `rewards.only_positive_rewards: true` (verified at `goal_pose.py:739-740`, clips the reward *sum* at 0) can mask small penalties entirely. `base_height` at −20 is a **crouch/over-extension guard**, not a balance term. It bites at ≥3.5 cm of sag. Falling sideways off one foot is not a sag failure.

3. **A base-height reward cannot install a height *regulator*, only a posture *prior*.** The reward reads privileged ground truth (`goal_pose.py:885-888`: `base_pos[:,2] − terrain_heights`). Base height is **not in the 54-d observation**. In double support the policy can infer height from `dof_pos` + `projected_gravity` by kinematics, so a posture prior transfers. **In single support the leg kinematics no longer determine base height** — the swing leg's angles say nothing — so the one signal the policy could have used is unobservable exactly in the regime that is failing. `[WS]` for the observability argument, `[PL]` for "therefore it cannot help here".

**Mild counter-argument, for fairness.** If a policy's failure mode is *progressive knee collapse* (sag → knee/ankle joint limits → loss of authority → fall), a height term genuinely fixes it, and that is probably what the friend saw on their robot. Symptom check: does the K1 visibly sink before it topples? If yes, the friend is right and this report is wrong. If it topples at roughly constant height, it is a lateral-CoP problem and height is irrelevant.

**And a reason it could make things slightly worse.** `[PL]` Pushing the base higher straightens the knee, raising CoM height `l`. The maximum lateral CoM velocity the ankle can arrest in single support is `v_max = w·√(g/l)` (capture-point bound, [Pratt et al. 2006](https://ieeexplore.ieee.org/document/4115602)), with `w` = foot half-width. For K1: `w` = 0.035 m, `l` ≈ 0.40 m → **v_max ≈ 0.173 m/s**. Raise the CoM to `l` = 0.45 m and `v_max` drops to 0.163 m/s. Straighter knees also push the leg Jacobian toward the extended singularity, reducing the policy's ability to modulate CoM height and lateral position through the stance leg.

**What I would do instead, in order:** (1) fix the deploy-side gait-clock gating bug (§5.1) — it is a one-line change and it is very likely the whole story; (2) run the ankle/torque diagnostics in §8 before touching any reward; (3) widen latency/actuator/foot-geometry domain randomization (§7); (4) only then consider new reward terms, and the useful ones are ankle-CoP / foot-force-distribution / posture-regularization, not height (§6).

---

## 1. Repo-derived baseline and physics (all numbers verified here)

### 1.1 What E0 actually is

| Item | Value | Source |
|---|---|---|
| Policy rate / physics | 50 Hz / 500 Hz (`dt 0.002`, `decimation 10`) | `Goal_Pose_V7.yaml` |
| Observation | 54 = grav(3) + ang_vel(3) + commands(10) + clock(2) + dof_pos(12) + dof_vel(12) + prev_action(12) | `goal_pose.py:793-808` |
| Privileged (critic only) | 14 = base_mass(1) + base_lin_vel(3) + base_height(1) + push F(3) + push τ(3) … | `goal_pose.py:809-818` |
| Sim PD | Hip/Knee **Kp 100, Kd 2**; Ankle **Kp 50, Kd 1** | `Goal_Pose_V7.yaml` `control` |
| Booster reference PD (T1) | Hip/Knee Kp 200 Kd 5; Ankle Kp 50 Kd 1 | [booster_gym T1.yaml](https://github.com/BoosterRobotics/booster_gym/blob/main/envs/T1.yaml) |
| Sim torque limits (from URDF `effort`) | HipP 30, HipR 20, HipY 20, Knee 40, **AnkP 20, AnkR 15** N·m | `K1_locomotion_armsdown.urdf` |
| **Deploy** torque limits | 60, 25, 30, 60, **24, 15** N·m | `deploy/configs/Goal_Pose_E0.yaml` |
| Foot collision box | **0.16 × 0.07 × 0.032 m**, origin (0.014, 0, −0.008) | URDF `left_foot_link` |
| `asset.feet_edge_pos` used by rewards | x ∈ [−0.1015, +0.1215], y = ±0.05, z = −0.03 | `Goal_Pose_V7.yaml` |
| Mass | 18.714 kg total | URDF |
| Ankle joint range | Pitch [−0.87, +0.345], Roll **[−0.345, +0.345]** rad | URDF |
| Both ankle DOFs | parallel mechanism (`parallel_mech_indexes: [15,16,21,22]`) | `Goal_Pose_E0.yaml` |

### 1.2 Two config inconsistencies inherited from the T1 fork `[WS]`

- **`feet_edge_pos` is T1's foot, not K1's.** The values `[±0.1215/−0.1015, ±0.05, −0.03]` are byte-identical to Booster's T1 yaml. K1's actual collision box gives corners at x ∈ [−0.066, +0.094], y = ±0.035, sole at z = −0.024 in the foot frame. So the reward/contact code assumes a foot **39 % longer, 43 % wider, and 6 mm deeper** than the one the physics engine collides. Consequence: `_refresh_feet_state` (`goal_pose.py:685-699`) sets `feet_contact` when *any* of those four phantom corners is within 1 cm of the ground, so **the sim declares foot contact earlier and holds it longer than physical contact**, and `feet_slip`, `feet_swing` gating and the swing/stance bookkeeping are all computed on an oversized foot. This is precisely the "sim foot is effectively larger / stickier" failure in §3.
- **Deploy ankle-pitch torque clamp (24 N·m) exceeds the URDF/sim clamp (20 N·m)**, and hip/knee clamps are 2× the URDF. These are T1's numbers. The clamp is probably inert (hardware clamps first) but it means nobody has checked K1's real motor limits against sim. Verify.

### 1.3 The single-support physics, computed for K1

FK at the nominal stance (script run against the URDF; head subtree excluded from the linked chain, so CoM height is a slight under-estimate):

| Quantity | Value |
|---|---|
| Weight `W = mg` | **183.6 N** |
| CoM height above ground | ≈ 0.423–0.435 m |
| CoM above ankle-roll axis, `l` | **≈ 0.40 m** |
| Ankle-roll axis height | 0.024 m |
| Foot lateral half-width `w` | **0.035 m** |
| Toe / heel margin from ankle axis | 0.094 m / 0.066 m |
| Nominal foot separation | 0.192 m |

Derived:

| Quantity | Formula | Value | Comment |
|---|---|---|---|
| Gravitational *destabilizing* stiffness | `m·g·l` | **73–76 N·m/rad** | vs. ankle `Kp` = **50 N·m/rad** |
| Divergence time constant | `√(l/g)` | **0.202 s** | one e-fold |
| Max lateral restoring moment | `W·w` | **6.43 N·m** | **foot-geometry limited, not motor limited** (roll limit 15 N·m) |
| Max lateral CoM acceleration | `g·w/l` | **0.86 m/s²** | 0.088 g |
| Max recoverable lateral CoM velocity | `w·√(g/l)` | **0.173 m/s** | capture-point bound |
| Ankle-roll PD error to command full 6.43 N·m at Kp 50 | `τ/Kp` | **0.129 rad = 7.4°** | = 37 % of the entire ±19.8° roll range |
| Max sagittal moment (toe) | `W·0.094` | 17.3 N·m | vs. 20 N·m limit → 87 % of limit |

**Three conclusions that matter:**

- **`m·g·l` (73–76) > `Kp_ankle` (50).** The ankle PD *alone is statically unstable* in single support: the passive joint stiffness does not even cancel gravity's toppling stiffness. All single-support balance therefore comes from the policy's active IMU feedback, at 50 Hz, against a 0.202 s divergence. `[WS]`
- **The binding constraint is the foot, not the motor.** Both ankle motors can saturate the CoP at the foot edge with margin (roll: 15 N·m available vs. 6.43 needed; pitch: 20 vs. 17.3). So "buy more ankle torque" is not the fix — but "do not waste the little CoP authority that exists on a 2 Hz march" is. `[WS]`
- **At 2 Hz gait the single-support phase is ≈ 0.25 s = 1.24 divergence time constants.** An uncorrected lateral CoM offset grows **e^1.24 ≈ 3.5×** per step. A 1 cm lateral error at toe-off is 3.5 cm — outside the 3.5 cm foot half-width — at touchdown. `[WS]`

---

## 2. Q1 — Why a policy that balances in sim falls on hardware when a foot lifts

Ranked by how often each is reported as *the* cause in humanoid sim-to-real writeups. General ranking `[PL]`; individual entries as marked.

| # | Cause | How often reported | Evidence | K1 status |
|---|---|---|---|---|
| 1 | **Actuator/transmission gap** — torque-speed envelope, gear friction/stiction, backlash, and especially *parallel/linkage ankles* | Most-reported for humanoids | ASAP measured per-joint sim-real gaps and found "the ankle and knee joints show the most pronounced discrepancies", and that G1's ankle linkage "introduces a significant sim-to-real gap difficult to bridge with conventional modeling techniques" ([arXiv:2502.01143](https://arxiv.org/html/2502.01143v2)) `[WS]` | K1 ankles are parallel; sim models them as ideal serial PD |
| 2 | **Latency / command filtering not modeled** | Very frequent | Booster Gym randomizes latency 0–20 ms ([arXiv:2506.15132](https://arxiv.org/html/2506.15132v1)); HuB uses **U(20, 60) ms** ([arXiv:2505.07294](https://arxiv.org/html/2505.07294v1)); ASAP U(20, 40) ms; NeRF2Real 10–50 ms + 5 ms jitter ([arXiv:2210.04932](https://arxiv.org/abs/2210.04932)) `[WS]` | Sim: 0–18 ms, constant per episode, **no jitter**. Deploy adds an unmodeled 0.8/0.2 EMA on targets (§2.1) |
| 3 | **Deploy-time command/regime mismatch** — the sim's stable behavior is never presented on hardware | Under-reported in papers, common in practice | Booster Gym: "for the standing gait, the gait cycle is set to zero"; legged_gym gates `feet_air_time` on `‖cmd‖ > 0.1` ([legged_robot.py](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py)) `[WS]` | **Broken here — see §5.1. My primary hypothesis.** |
| 4 | **Contact/foot model gap** — patch size, friction, sole compliance, restitution | Frequent | HuB: instability "arise[s] from modeling discrepancies … particularly in the simulation of ground contact and frictional interactions" `[WS]` | `feet_edge_pos` is T1's larger foot (§1.2) |
| 5 | **Sensor noise/bias/lag driving an action-jitter feedback loop** | Frequent, and specifically named for single support | HuB: "minor initial oscillations can progressively amplify due to unmodeled dynamics"; noisy IMU "causes jitter in the action outputs and can trigger a vicious feedback loop of instability" `[WS]` | Sim gravity noise σ = 0.01 (≈0.6°), white, no bias, no mounting rotation, no filter lag |
| 6 | **Mass/inertia/CoM error** (battery, cables, added sensors, head/arm payload) | Common | Standard DR item everywhere | Randomized: base_mass ×[0.8,1.2], base_com ±0.1 m — adequate |
| 7 | **Degenerate sim-only behavior** — chatter, micro-marching, edge-riding that is free in sim and destructive on hardware | Common | HuB's "close feet" penalty (weight −1000) exists because policies collapse the stance | Plausible: `feet_swing: +3` is a strong stepping incentive |

### 2.1 Deploy-side detail worth its own line `[WS]`

`deploy/deploy_goal_pose.py:441` applies `filtered_dof_target = 0.8·prev + 0.2·new` at the 500 Hz publish rate. Time constant **τ ≈ 9.0 ms**. This is Booster's own pattern (identical line in [booster_gym/deploy/deploy.py](https://github.com/BoosterRobotics/booster_gym/blob/main/deploy/deploy.py)) and **is not modeled in the sim at all**. For a corrective ankle action that alternates sign every policy tick (25 Hz fundamental), `ωτ = 1.41` → **gain 0.58, phase lag 54°**. `[PL]` The real ankle therefore receives roughly *half* the fast corrective authority the sim's ankle received, with extra phase lag, on the one joint that is already the sole authority in single support. This alone is a credible sim-real gap.

Also `deploy_goal_pose.py:450-457`: for the four parallel-mechanism indices, the deploy layer sets `kp = 0` and injects `tau = (filtered_target − measured) · stiffness`, clipped. `kd` retains `common.damping` = 1.0 and `dq` target is 0, so a D-term still exists — but it acts through the SDK's parallel↔serial Jacobian, not on the serial ankle velocity the sim damped. `[PL]` Verify with a measurement, not by reading.

---

## 3. Q2 — Ankle torque and actuator authority

### 3.1 What is needed

For a flat-footed single-support humanoid the ankle *is* the CoP actuator: `τ_ankle = W · d_CoP`, so the CoP range and the torque range are the same constraint expressed twice ([Stephens, "Integral Control of Humanoid Balance", IROS 2007](https://www.cs.cmu.edu/~bstephe1/papers/iros07.pdf); ankle strategy ≡ CoP balancing) `[WS]`.

For K1 (§1.3) the **foot geometry saturates before the motors do**, so raw torque is not the deficiency. The deficiencies are:

| Deficiency | Number | Why it matters |
|---|---|---|
| Ankle stiffness below gravity's toppling stiffness | Kp 50 < m·g·l ≈ 74 N·m/rad | Passive ankle cannot hold the inverted pendulum; only active feedback can `[WS]` |
| PD error needed for full CoP authority | 7.4° of ankle roll | The stance ankle physically rolls under load; on hardware, backlash + linkage compliance make the *actual* deflection larger than the encoder reports `[PL]` |
| Bandwidth budget | 0.202 s divergence vs 20 ms tick + ~9 ms EMA + transport | ~10 control steps per e-fold; each 20 ms of extra unmodeled lag costs ~10 % of the recovery window `[PL]` |
| Ankle Kd | 1 N·m·s/rad in both sim and K1 deploy | Booster's own T1 **deploy** config uses **ankle Kd = 3** while their sim uses 1 — i.e. the vendor triples ankle damping on hardware. K1's deploy kept 1. `[WS]` fact, `[SP]` significance |

### 3.2 How sim PD vs real torque-speed limits produce exactly this failure `[PL]`

In sim, `dof_torques = Kp·(target − q) − Kd·q̇`, clipped to the URDF `effort`, applied every 2 ms with a perfect, instantaneous, zero-friction actuator (`goal_pose.py:577-581`). The policy therefore learns to *command large instantaneous ankle target excursions and rely on getting the corresponding torque within one physics step*. On hardware the same command produces: (a) less torque at speed (torque-speed envelope; the URDF's flat `velocity=18 rad/s` and constant `effort` is a rectangle, the motor's is a trapezoid); (b) a dead zone from stiction/backlash in the parallel linkage; (c) the torque arriving ~10–30 ms late. Under a 0.202 s divergence that converts a stabilizing correction into a destabilizing one within two or three steps — the "vicious feedback loop" HuB describes.

### 3.3 Standard diagnostic `[WS]`

The field-standard test is the **commanded-vs-achieved torque/position sweep under load**, i.e. what ASAP does systematically (delta-action model fitted to real rollouts) and what actuator-network work formalized ([Hwangbo et al., *Learning agile and dynamic motor skills for legged robots*, Science Robotics 2019, arXiv:1901.08652](https://arxiv.org/abs/1901.08652)). Practical minimum version in §8, steps D3–D5.

---

## 4. Q3 — Foot / sole modeling

| Aspect | Isaac Gym as configured here | Real K1 | Effect on single support |
|---|---|---|---|
| Contact geometry | Rigid box 0.16 × 0.07 × 0.032, `contact_offset 0.02`, `rest_offset 0.0` | Rubber-ish sole, non-flat, edge chamfer | Sim contact begins 2 cm early and is perfectly rigid `[WS]` |
| Contact patch | Box faces → PhysX generates ≤ 4–8 contact points at the box corners | Distributed pressure over a compliant patch | Sim CoP can jump discontinuously between corners; real CoP moves continuously `[PL]` |
| **Effective foot size used by rewards** | **x ∈ [−0.1015, +0.1215], y = ±0.05 (T1's foot)** | x ∈ [−0.066, +0.094], y = ±0.035 | **Sim's contact flag uses a foot 39 % longer / 43 % wider** → contact declared earlier, held longer, `feet_slip`/`feet_swing` mis-gated `[WS]` |
| Friction | `static/dynamic 1.0`, randomized `+U(0.1, 2.0)` | Turf/carpet, ≈0.6–1.0, direction-dependent | Compare HuB, which trains at **U(2.5, 3.5)** deliberately so the policy never learns to rely on marginal friction `[WS]` |
| Restitution / compliance | randomized `+U(0.1, 0.9)` / `+U(0.5, 1.5)` | low restitution, real compliance | Reasonable coverage |
| Foot roll compliance about the sole | **none** — rigid box on rigid plane | ankle linkage + sole flex ≈ a real series spring | The real robot has an extra unmodeled compliant DOF between shank and ground exactly where the CoP authority lives `[PL]` |

**Why this specifically breaks single support:** in double support, two contact patches with a 0.192 m lever produce lateral moments through *vertical force redistribution*, so foot-patch modeling errors are second-order. The moment the second foot leaves, all lateral authority collapses onto the CoP inside a **7 cm-wide** patch, and every one of the errors above lands directly on the only remaining actuator. A 43 % over-estimate of foot width is a 43 % over-estimate of `v_max` in the capture-point bound: sim believes it can arrest 0.247 m/s of lateral CoM velocity; the robot can arrest 0.173 m/s. `[PL]`

---

## 5. Q4 — State estimation, and the 54-d observation

### 5.1 The finding: the standing regime is unreachable on hardware `[WS]`

**Training** (`goal_pose.py:433-491`): `commands.goal_categories.stand: 0.1`. For a stand goal the env sets `gait_frequency[env] = 0.0` (line 490), which (a) writes 0 into `commands[3]` → observation index 9, and (b) **freezes `gait_process`** (line 616: `gait_process += dt · gait_frequency`), so obs[16], obs[17] = `cos/sin` become constants, and (c) disables the `feet_swing` bonus (lines 1044-1045 gate on `gait_frequency > 1e-8`). This is the vendor-intended design: Booster Gym states "for the standing gait, the gait cycle is set to zero", and Booster's T1 yaml has `commands.still_proportion: 0.1`.

**Deployment** (`deploy/utils/policy_goal_pose.py`): the docstring says it outright —

> "E0 does NOT gate the gait clock / commands by gait_frequency the way `utils/policy.py` does for ParameterWalk."

`gait_process = fmod(time_now · 2.0, 1.0)` unconditionally, `commands[3] = 2.0` unconditionally. Compare the ParameterWalk deploy wrapper (`deploy/utils/policy.py:43-59`), which sets `gait_frequency = 0` on a near-zero command and multiplies both clock channels by `(gait_frequency > 1e-8)`.

**Consequence:** on the real robot E0 can never enter the stand regime it was trained for. With a zero goal it is running the *walk* behavior at 2 Hz — a march in place. That is 4 single-support entries per second, each 0.25 s = 1.24 divergence time constants (§1.3), on a foot 7 cm wide, with an ankle whose fast corrections are attenuated ~42 % by an unmodeled 9 ms filter. **This matches the reported symptom exactly: "standing posture is unstable, and when one foot lifts, the robot falls over."** The robot is not failing to stand; it was never asked to stand.

**Fix (minutes, no retraining):** in `policy_goal_pose.py`, gate on the goal magnitude the way `policy.py` gates on the velocity command —

```python
still = (abs(gx) < 0.05) and (abs(gy) < 0.05) and (abs(h) < 0.05)   # tune thresholds
f = 0.0 if still else self.gait_frequency
self.gait_process = 0.0 if still else np.fmod(time_now * f, 1.0)
self.commands[3] = f
self.obs[16] = np.cos(2*np.pi*self.gait_process) * (f > 1e-8)
self.obs[17] = np.sin(2*np.pi*self.gait_process) * (f > 1e-8)
```

Note the training/deploy contract detail: in training a *stand* goal has `commands[0:3] = 0` **and** `f = 0` simultaneously (lines 444-448, 490), so gate on the goal, and add hysteresis + a phase-continuous ramp so the transition is not a step discontinuity in the observation. `[PL]` for the exact thresholds — sweep them in sim first.

### 5.2 What the missing channels do

| Channel | In obs? | Real-robot source | Consequence |
|---|---|---|---|
| `projected_gravity` (3) | yes | `rotate_vector_inverse_rpy(imu.rpy, [0,0,−1])` (`deploy_goal_pose.py:270-275`) | Carries **all** the mounting error, bias and filter lag of the IMU. Sim adds only σ = 0.01 white noise (≈0.6°), zero-mean, no bias, no mounting rotation, no lag `[WS]` |
| `base_ang_vel` (3) | yes | raw `imu_state.gyro` | Sim σ = 0.1 rad/s white; real gyro has bias + the same mounting rotation `[WS]` |
| **base linear velocity** | **no** | not measured | Standard and correct — this is why the critic gets it as privileged obs. The policy must infer CoM velocity from the gravity/gyro/joint history, which is *exactly* the signal an IMU mounting error corrupts `[WS]` |
| **base height** | **no** | not measured | See §0.3: height is kinematically inferable in double support, **not** in single support |

**Why the missing height matters for the *reward* question specifically.** During training the height reward is computed from privileged truth, so it shapes the policy's *action prior*: "adopt joint angles that, in the sim's contact model, put the trunk at 0.52 m." On hardware, with a different sole thickness (sim sole 24 mm below the foot frame but the reward geometry assumes 30 mm), different foot compliance and a possibly non-zero joint zero-offset, that same joint prior produces a different height — and the policy has no way to notice or correct. So the base-height reward transfers as an **open-loop posture bias whose calibration is wrong by exactly the sim-real kinematic offset**. Turning the weight up sharpens a bias that is mis-calibrated. `[PL]`

**IMU mounting error is the single highest-leverage state-estimation item.** A constant mounting rotation of `ε` rad on the IMU appears to the policy as a permanent world tilt, i.e. a permanent bias in `projected_gravity` and therefore a permanent bias in the CoM-position estimate it implies. On a 0.40 m pendulum a **1° mounting error ≈ 7 mm of apparent lateral CoM offset — 20 % of the 35 mm foot half-width.** `[WS]` arithmetic. In double support, the wide support polygon absorbs it. In single support, it eats a fifth of the margin, permanently, always in the same direction — which is why such robots reliably fall to the *same side*. **Ask the team: does it always fall the same way?** If yes, this is near-diagnostic.

Recommended DR: NeRF2Real randomizes the IMU by "shifting it up to 0.5 cm and tilting it by up to 2 degrees" per episode ([arXiv:2210.04932](https://arxiv.org/abs/2210.04932)) `[WS]`. HuB adds *temporally correlated* Ornstein-Uhlenbeck noise to IMU Euler angles (θ = 25, σ = 250, in degrees) rather than white noise, explicitly because "real-world IMU noise exhibits significant temporal correlation" `[WS]`. E0 has neither.

---

## 6. Q5 — Standing and walking as distinct regimes

The claim in the prompt is correct and well documented. Standing is not the `v = 0` limit of walking; it is a different control problem (no swing-leg placement authority, CoP-only, and the "gait" reward machinery actively fights it).

| Fix | Who reports it | Reported to transfer? | K1 status |
|---|---|---|---|
| **Explicit zero-command / stand share in training** | Booster Gym: "with a certain probability, the command is set to 'stand still'", `still_proportion: 0.1`; Digit paper samples commands from 5 categories incl. standing, every 2–6 s ([arXiv:2404.19173](https://arxiv.org/abs/2404.19173)) | Yes `[WS]` | **Present** (`goal_categories.stand: 0.1`) but **unreachable at deploy** (§5.1) |
| **Zero the gait clock in the stand regime** | Booster Gym: "for the standing gait, the gait cycle is set to zero" | Yes `[WS]` | Present in sim, **missing in deploy** |
| **Gate the stepping bonus on non-zero command** | legged_gym gates `feet_air_time` on `‖cmd_xy‖ > 0.1`; universal in forks | Yes `[WS]` | Gated on `gait_frequency > 1e-8` — correct in sim, but deploy pins `gait_frequency` at 2.0 |
| **Explicit contact-pattern / single-contact reward** | Digit paper's "Feet contact" term: 1 if standing **or** single contact, 0.2 s grace, weight 0.1. Their best controller is literally named *Single Contact++* and achieved "perfect disturbance rejection" over 79–214 N, 200–500 ms impulses | Yes, on hardware `[WS]` | Absent |
| **Multi-phase curriculum: learn to stand, then to walk** | [arXiv:2505.20619](https://arxiv.org/abs/2505.20619) | Claimed; full text not verified here `[PL]` | Absent |
| **Posture regularization toward a nominal stance, gated near the goal** | Common | Yes `[PL]` | **Present**: `stand_posture: -1.0` inside `stand_posture_radius: 0.3` m |
| **CoM-over-support-foot reward** | HuB's dominant term (weight 160) — see §7 | Yes, on G1 single-leg tasks `[WS]` | Absent |
| **ZMP-in-support-polygon reward from privileged sim state** | Narrow-terrain H1-2 work ([arXiv:2502.17219](https://arxiv.org/abs/2502.17219)); asymmetric actor-critic so ZMP is critic-side only | Reported on H1-2; full text not verified here `[PL]` | Absent |
| **Foot-force distribution / symmetry reward** | Less common in RL; standard in model-based (force distribution QPs) | `[SP]` for RL transfer | Absent — and **K1 has no foot force/torque sensors**, so this would have to be a sim-privileged term only |

---

## 7. Q6 — Reward and regularizer catalogue

Weights are only comparable *within* a paper's own scale. Read the "trades off against" column, not the absolute number.

| Term | Published form | Typical weight | Trades off against | Verdict for K1 |
|---|---|---|---|---|
| **Base height** | `(h_des − h)²` (Booster Gym Tab. II); `e^(−20·|p_z − c_h|)` (Digit, weight 0.05) | −20 (Booster); +0.05 (Digit) | Crouch-based robustness; knee-singularity risk if pushed high | **Already at −20. Do not raise.** `[WS]` |
| **Orientation / gravity alignment** | `roll² + pitch²` (this repo) or `‖g_xy‖²` | −5 (Booster T1) … −20 (E0) | Hip strategy, which *requires* trunk lean. Over-weighting removes the robot's second balance strategy | E0 is at **4× Booster's**. Consider **lowering to −5 … −10** and letting the trunk counter-rotate `[PL]` |
| **CoM over support foot** | `exp(−‖p^com_xy − p^lower-foot_xy‖²/σ²)·1(‖ẑ_l − ẑ_r‖ > 0.05)`, σ = 0.1 | **160** (HuB's largest balance term) | Task tracking; deliberately relaxed to σ_pos = 0.6 m so balance wins | **Best single addition.** Direct, uses privileged sim state, gated to single support `[WS]` |
| **ZMP inside support polygon** | ZMP from privileged contact forces, penalize distance to polygon centre | n/a (H1-2 narrow-terrain work) | Agility; can over-conservatize gait | Good second choice. Critic-side/privileged only `[PL]` |
| **Foot contact-pattern mismatch** | XOR of actual vs reference contact state | **−250** (HuB); +0.1 as "single contact" bonus (Digit) | Free-form footfalls | Cheap and effective; directly punishes the unintended double-tap/scuff `[WS]` |
| **Close-feet penalty** | `max(0.16 − ‖p_l − p_r‖, 0)` | **−1000** (HuB) | Narrow stance | K1 has `feet_distance_ref: 0.18` but **no `feet_distance` scale** — Booster's T1 yaml *does* (`feet_distance: -1.`). **This term was dropped in the fork.** Re-add `[WS]` |
| **Feet orientation (sole flat)** | `‖g^feet_z‖·1(p^feet_z < 0.05)` | −62.5 (HuB) | Toe-off/heel-strike | Present as `feet_roll: -0.2`, `feet_pitch: -0.1` — very weak by comparison `[PL]` |
| **Slippage** | `‖v^feet‖²·1(F ≥ 1)` | −30 (HuB) | Friction reliance | Present: `feet_slip: -0.1` |
| **Feet air time** | `T_air − 0.25` | +250 (HuB), 1.0 (Digit, "the only sparse reward") | **Standing** — must be gated on non-zero command | Present as `feet_swing: +3`, gated on `gait_frequency`. **The gate is defeated at deploy** (§5.1) |
| **Ankle torque regularization** | `‖τ‖` or `‖τ‖²` | −2.5e−5 (HuB); `e^(−0.02·mean|τ|/τ_max)` weight 0.02 (Digit) | Ankle authority — over-penalizing directly removes CoP authority | E0 has `torques: -3e-4`, `torque_tiredness: -1e-2`, `torque_margin: -0.002`, `power: -3e-3`, `electrical_power: -2e-3`. **That is five overlapping effort penalties.** Check their summed share of the ankle gradient; this is a real suspect for a policy that under-uses its ankles `[PL]` |
| **Torque-limit violation** | `1(τ ∉ [τ_min, τ_max])` | −0.5 (HuB) | — | E0: `torque_limits: -0.` (**off**); margin-based `torque_margin: -0.002` at 85 % is on |
| **dof_pos → nominal stance** | `Σ(q − q_nom)²`, gated | −1.0 (E0's `stand_posture`, gated to 0.3 m) | Task freedom | Present. Reasonable |
| **Action rate / smoothness** | `‖a_t − a_{t−1}‖²` | −0.01 (Humanoid-Gym); −1.0 (Booster T1); −1.5 (E0) | Reaction speed — over-penalizing costs exactly the fast ankle corrections single support needs | E0 at **−1.5 is 1.5× Booster's**. Suspect, given §2.1's 9 ms output filter already smooths `[PL]` |
| **Contact force magnitude** | penalize `F > F_max` | −0.01 (Humanoid-Gym) | Impact softness | Absent |
| **Survival** | 1 per step | +0.25 (Booster, E0) | — | Present |

**Synthesis for K1 — what I would actually add, in order:**
1. HuB-style **CoM-over-stance-foot** term, active only when `‖z_l − z_r‖ > 0.05` (i.e. genuinely single support). This is the term that most directly encodes the §1.3 physics.
2. Re-add the **`feet_distance`** penalty that was dropped from the T1 fork.
3. **Reduce** the effort-penalty stack and `action_rate`, don't add anything — and measure the change in ankle torque RMS during single support.
4. Consider **lowering `orientation` from −20 to −5…−10** to permit a hip strategy.
5. Base height: leave at −20. Do not touch.

---

## 8. Q7 — Domain randomization for standing robustness

E0's `randomization` block is byte-for-byte Booster's T1 defaults (I diffed them) with only the disturbance model changed. The gaps are on exactly the axes §1–§5 identify.

| Parameter | E0 today | Reported successful ranges | Gap |
|---|---|---|---|
| Ground friction | `+U(0.1, 2.0)` on 1.0 | Humanoid-Gym `U(0.1, 2.0)`; HuB **`U(2.5, 3.5)`** | OK, but low end (0.1!) may be teaching the policy to expect ice. HuB's high-friction choice is deliberate `[WS]` |
| Restitution / compliance | `+U(0.1, 0.9)` / `+U(0.5, 1.5)` | Booster default | OK |
| **Control / comms latency** | `delay_steps = randint(0,10)` substeps = **0–18 ms, constant per episode, no jitter** | Booster Gym 0–20 ms; ASAP **U(20,40) ms**; HuB **U(20,60) ms**; NeRF2Real 10–50 ms **+ 5 ms jitter** | **Under-modeled.** Add a floor (real loop latency is never 0) and per-step jitter `[WS]` |
| **Deploy output EMA (0.8/0.2, τ ≈ 9 ms)** | **not modeled at all** | — | **Model it in sim.** Cheapest high-value change after §5.1 `[WS]` |
| PD gain scaling | `dof_stiffness/damping ×U(0.95, 1.05)` (**±5 %**) | HuB `U(0.75, 1.25)`; ASAP `U(0.925, 1.05)`; common `U(0.9, 1.1)` | **Far too narrow.** ±5 % is essentially no randomization. Widen to at least ±20 % `[WS]` |
| Motor strength / torque limit | **not randomized** | Humanoid-Gym motor strength `[95, 105] %`; HuB torque RFI `0.1 × τ_limit` | **Missing.** Add `τ_limit ×U(0.8, 1.0)` and RFI noise `[WS]` |
| Joint friction | `+U(0, 2.0)` N·m | Booster default | Present, and generous. Good |
| **Joint zero offset (encoder bias)** | **`joint_encoder_bias` supported in code (`goal_pose.py:353`) but absent from `Goal_Pose_V7.yaml` → zero** | Foot-IMU humanoid work uses joint encoder offset `[−0.01, 0.01]` rad; Codex's H-batch already uses **±0.025 rad** (`sweeps/hbatch/H1-codex.yaml:524`) | **Off in E0.** Turn it on — the code path exists `[WS]` |
| **Joint target offset** | same — supported, absent from E0's yaml | H-batch uses ±0.02 rad | **Off in E0.** Turn it on `[WS]` |
| **IMU mounting rotation** | **not randomized** (only white σ = 0.01 on gravity) | NeRF2Real: tilt up to **2°**, shift up to 5 mm, per episode | **Missing, and this is the one I would add first among DR items** (§5.2 arithmetic) `[WS]` |
| **IMU bias / correlated noise / filter lag** | white noise only | HuB: OU noise on Euler angles, θ = 25, σ = 250 (deg) | **Missing.** White noise is the wrong noise model `[WS]` |
| Link mass / CoM | base ×`U(0.8,1.2)`, CoM ±0.1 m | HuB link mass `U(0.7,1.3)`, torso CoM ±0.1 m | Adequate |
| Pushes | collision 40–150 N for 50–150 ms; support 3–15 N for 1.5–3 s | Digit: 200–800 N single 20 ms step, or 20–200 N for 200–500 ms; HuB pushes 0.5 m/s every 1 s | Reasonable; consider HuB's high-rate small pushes, which specifically target single-support jitter `[PL]` |
| **Foot geometry** | fixed (and wrong, §1.2) | rarely randomized, but a ±10 % sole-size randomization is cheap insurance | `[SP]` |

---

## 9. Q8 — Ordered diagnostic checklist (do all of this before touching a reward)

Each step is cheap and falsifies a specific hypothesis. Stop when one fires.

**D0 — Free, 5 minutes: characterize the fall.**
- Does it fall to the *same side* every time? → IMU mounting/bias (§5.2), or a left/right joint-offset asymmetry.
- Does it *sink* before toppling? → then the friend is right after all; height/knee-collapse.
- Does it *march in place* when the goal is (0,0,0)? → §5.1 confirmed, go straight to D1.
- Does it chatter/buzz at the ankles before going? → §2.1 lag/gain loop.

**D1 — Free, one line: confirm the gait clock is spinning during "standing".**
The deploy loop already publishes `/locomotion_test/policy_debug` at 10 Hz (`debug_period_s: 0.10`). Add `gait_process`, `commands[3]`, and per-leg `dof_pos` to that payload (`deploy_goal_pose.py:_publish_debug`) and watch it with a zero goal. If `gait_process` sweeps 0→1 at 2 Hz, **§5.1 is confirmed and is almost certainly the whole bug.** Fix the gating, retest before anything else.

**D2 — Sim replay of real observations (the decisive sim-vs-real test).**
Log the full 54-d `self.obs` vector every tick on hardware (it is already assembled in `policy_goal_pose.inference`). Then: (a) feed the logged obs sequence through the *same* TorchScript offline and diff the actions against what the robot actually commanded — this isolates any deploy-side numerical/ordering bug; (b) feed the *same* obs to the policy inside Isaac Gym and compare. If actions match but the robot falls, the gap is in the plant (D3–D6), not the policy or the observation.

**D3 — Commanded vs measured joint angles under load.**
On the robot in single support (hold it by the torso, lift one foot), log `motor_cmd[i].q` (or `filtered_dof_target`) against `motor_state_serial[i].q` for both ankles. Compute the tracking error and compare to the *sim's* error in the same pose. Expected sim value from §1.3: **≈7.4° of ankle-roll deflection at full CoP authority.** If the real deflection is materially larger, you have linkage compliance/backlash; if smaller, your effective real Kp is higher than 50 and the policy's learned gain is wrong.

**D4 — Ankle torque saturation during single support.**
Log the commanded `motor_cmd[i].tau` for indices 15, 16, 21, 22 (the deploy layer already computes it explicitly at `deploy_goal_pose.py:452`) plus any `tau_est` the SDK exposes. Compute the fraction of single-support samples at the ±clip. Predicted from §1.3: **roll should peak near 6.4 N·m (foot-limited), well under the 15 N·m clip.** If you see sustained clipping at 15 N·m, your real CoM is further outboard than the model says (mass/CoM error) or the foot is smaller/rolled. If you see roll torque *never* exceeding ~2 N·m, the policy is not using its ankle at all — look at the effort-penalty stack (§7).

**D5 — Ankle step-response / bandwidth check.**
With the robot hanging (feet off ground), command a small square wave on ankle roll at 1, 2, 5, 10 Hz and measure the achieved amplitude and phase. This directly measures the combined effect of the 0.8/0.2 EMA, SDK latency, and motor dynamics. Predicted from §2.1: **≈0.58 gain and ≈54° lag at 25 Hz.** Whatever you measure, put *that* filter in the sim.

**D6 — IMU frame and sign convention.**
- Static: place the robot on a level surface in the nominal stance; read `projected_gravity` from the deploy loop. It must be ≈`[0, 0, −1]`. **Any residual x/y component is your mounting error** — convert with `ε_rad ≈ atan2(‖g_xy‖, |g_z|)`, and multiply by 0.40 m to get the equivalent CoM offset (35 mm = the whole foot half-width).
- Sign: tilt the robot nose-down by hand; `projected_gravity[0]` must move in the same direction Isaac Gym produces for the same tilt. Do the same for roll and for all three gyro axes. Verify against `quat_rotate_inverse(base_quat, [0,0,−1])` in sim, not against intuition. `rotate_vector_inverse_rpy` (`deploy/utils/rotate.py`) uses `(R_z R_y R_x)ᵀ v` — confirm the SDK's `imu_state.rpy` is the matching intrinsic Z-Y-X convention and not Z-X-Y.
- Latency: compare the timestamp of a sharp gyro transient to the physical event.

**D7 — Foot geometry audit.** Measure the real sole: length, width, and height below the ankle axis. Fix `asset.feet_edge_pos` and the URDF collision box to match (§1.2). Then re-run the sim evaluation — if the 89 % success rate drops, you have quantified how much of it was the phantom foot.

**D8 — Only now, re-train.** Order: (1) deploy gating fix (no retrain); (2) DR widening per §7 (latency + EMA + PD ±20 % + torque limits + encoder/target offsets + IMU tilt); (3) foot geometry correction; (4) CoM-over-stance-foot reward + re-add `feet_distance`; (5) *reduce* `action_rate` and the effort stack; (6) leave `base_height` alone.

---

## 10. Sources

- HuB: Learning Extreme Humanoid Balance — [arXiv:2505.07294](https://arxiv.org/html/2505.07294v1)
- Booster Gym: End-to-End RL for Humanoid Locomotion (Booster T1) — [arXiv:2506.15132](https://arxiv.org/html/2506.15132v1) · [github.com/BoosterRobotics/booster_gym](https://github.com/BoosterRobotics/booster_gym) · [envs/T1.yaml](https://github.com/BoosterRobotics/booster_gym/blob/main/envs/T1.yaml) · [deploy/deploy.py](https://github.com/BoosterRobotics/booster_gym/blob/main/deploy/deploy.py)
- Humanoid-Gym: Zero-Shot Sim2Real Transfer — [arXiv:2404.05695](https://ar5iv.labs.arxiv.org/html/2404.05695) · [github.com/roboterax/humanoid-gym](https://github.com/roboterax/humanoid-gym)
- Revisiting Reward Design and Evaluation for Robust Humanoid Standing and Walking (Digit) — [arXiv:2404.19173](https://arxiv.org/abs/2404.19173)
- ASAP: Aligning Simulation and Real-World Physics (Unitree G1) — [arXiv:2502.01143](https://arxiv.org/html/2502.01143v2)
- Humanoid Whole-Body Locomotion on Narrow Terrain (ZMP reward, H1-2) — [arXiv:2502.17219](https://arxiv.org/abs/2502.17219)
- NeRF2Real: Sim2real Transfer of Vision-guided Bipedal Motion Skills — [arXiv:2210.04932](https://arxiv.org/abs/2210.04932)
- Hwangbo et al., Learning agile and dynamic motor skills for legged robots (actuator networks) — [arXiv:1901.08652](https://arxiv.org/abs/1901.08652)
- legged_gym reference implementation — [legged_robot.py](https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py)
- Stephens, Integral Control of Humanoid Balance (ankle strategy ≡ CoP control) — [IROS 2007 PDF](https://www.cs.cmu.edu/~bstephe1/papers/iros07.pdf)
- Pratt et al., Capture Point: A Step toward Humanoid Push Recovery — [Humanoids 2006](https://ieeexplore.ieee.org/document/4115602)
- Gait-Conditioned RL with Multi-Phase Curriculum — [arXiv:2505.20619](https://arxiv.org/abs/2505.20619) *(abstract-level only; full text not verified)*
- Sim-to-Real Transfer of Compliant Bipedal Locomotion on Torque Sensor-Less Gear-Driven Humanoid — [arXiv:2204.03897](https://arxiv.org/abs/2204.03897) *(not verified in full text)*
- Reactive Stepping for Humanoid Robots using RL (Atalante) — [arXiv:2203.01148](https://arxiv.org/abs/2203.01148)

**Repo files referenced:** `htwk-gym/envs/K1/Goal_Pose_V7.yaml`, `htwk-gym/envs/K1/goal_pose.py`, `htwk-gym/deploy/configs/Goal_Pose_E0.yaml`, `htwk-gym/deploy/deploy_goal_pose.py`, `htwk-gym/deploy/utils/policy_goal_pose.py`, `htwk-gym/deploy/utils/policy.py`, `htwk-gym/deploy/utils/command.py`, `htwk-gym/deploy/utils/rotate.py`, `htwk-gym/resources/K1/K1_locomotion_armsdown.urdf`, `htwk-gym/sweeps/hbatch/H1-codex.yaml`.
