# Deploy ↔ training contract audit (2026-08-04)

Written after the gait-clock bug: codex's wrapper asserted in a comment that E0
does not gate the gait clock, the training code says the opposite, and it was a
colleague who found it rather than any check of ours. This is the systematic
pass for anything else of that shape, done against the frozen config of the
checkpoint actually installed (I2a_dr `model_150`, task `K1/Goal_Pose_V7`).

## Verified correct

| Item | Training | Deploy | |
|---|---|---|---|
| gait phase | `fmod(gait_process + dt*freq, 1)`, an integrator, never reset per env (`goal_pose.py:621`) | was `fmod(time_now*freq, 1)` | **fixed** |
| obs layout | `3 grav + 3 angvel + 10 cmd + 2 clock + 12 dofpos + 12 dofvel + 12 act` | same | ok |
| V7 obs override | `goal_pose_v7.py:746` calls `super()` then only writes `extras["v7"]` | n/a | ok |
| leg slice | `_obs_dofs` is identity unless `arm_script_on`; this run has `arm_script.enabled: false` and the armsdown URDF, so 12 DOFs | `[10:22]` of 22 | ok |
| action clip | `clip(actions, ±clip_actions)` stored, then fed back as obs | same | ok |
| action → target | `default_dof_pos + action_scale * actions` | same | ok |
| policy rate | `sim.dt 0.002 × decimation 10` = 50 Hz | `common.dt × decimation` = 0.02 | ok |
| goal staleness | `goal_obs_hold_steps: [0,0]` — not modelled in this run | n/a | ok |

`joint_encoder_bias` and `joint_target_offset` appear in the training
observation and action paths but are `apply_randomization` around zero, i.e.
domain randomisation for quantities that are physically real on hardware.
Deploy correctly has no counterpart.

## Defect found: the gait phase was a closed form, not an integrator

Training integrates the phase and never resets it per env; a zero frequency
therefore *freezes* the phase where it stood. The wrapper computed
`fmod(time_now * gait_frequency, 1)` instead. The two agree exactly while the
frequency is constant, which is why this survived — but the arrival gate added
in the previous fix makes the frequency change, and there they diverge:

| | closed form | integrator |
|---|---|---|
| phase while stopped | snaps to 0 | frozen where it arrived |
| `(cos, sin)` step at arrival | mean 1.272, max 1.996 | 0 |
| `(cos, sin)` step at departure | mean 1.272, max 1.996 | one ordinary step |

For scale, one ordinary 50 Hz walking step moves that pair by 0.251 and the
largest possible move is 2.0. So the clock teleported by about five ordinary
steps on average, up to eight, at both arrival and departure — in the same frame
that `commands[3]` drops 2.0 → 0. Swept over 2250 arrival/stop timings.

The gating fix introduced this. It would have shown up on the next hardware run
as a disturbance exactly at the goal and again on departure, which is precisely
where we would have been looking for policy problems.

Fixed in `advance_gait_clock`, which also measures elapsed time rather than
assuming the nominal interval (so 2 Hz is 2 Hz in wall clock under jitter) and
clamps it (so an 8 s GetUp pause cannot wind the phase forward by 8 s).
Regression test: `htwk-gym/deploy/tests/test_gait_clock.py`, 6 cases, including
one that asserts the old closed form would fail the others.

## Confirms the arrival-gating fix

Two independent reasons a non-zero gait clock at the goal is wrong, beyond the
stand-goal distribution argument:

- `_reward_goal_reached` pays only while **stopped** inside the radius
  (`stop_speed_threshold: 0.1` m/s, `goal_reach_radius: 0.1` m). The policy is
  trained to arrive *and* stop.
- `feet_swing: 3.0` is gated on `gait_frequency > 1e-8`
  (`goal_pose.py:1048`). Holding 2 Hz at the goal sits directly on a stepping
  incentive.

## Notes on this checkpoint

Goal tracking is `constellation: 3.5`, the "No More Marching" kernel that
couples position and heading in one term. The zeroed `goal_position`,
`goal_heading` and `goal_progress` are its alternatives, not a gap — an initial
reading of the scales as "goal rewards are off" was wrong.

Two differences from the V7 baseline worth knowing when reading behaviour:

- `stand_posture: 0.0` (baseline `-1.0`). Nothing pulls the stance back to the
  default pose near the goal, so posture at arrival is whatever the other terms
  produce.
- `base_height_target: 0.55` (baseline `0.52`), still at `-20.0` scale.

## Method note

Both gait-clock defects were of one shape: a deploy-side reimplementation that
matched training in the common case and diverged in a corner, with a comment
asserting the divergence was intended. The remaining wrapper arithmetic was
re-derived from the training source rather than read for plausibility, which is
what turned up the second one.

## Not answerable without hardware

- Whether the crank↔ankle Jacobian matters. The E1 wrapper commands those joints
  by position and works, which is evidence enough to follow it, but it is not a
  measurement of the linkage.
- The gain step at `r` (legs 350/250 → 100/50 in one frame). E1 does the same
  and is verified, so it is presumed fine; `_startup_hold_s` /
  `_rl_stance_transition_s` exist there as knobs if it ever is not.
