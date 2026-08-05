"""Generate the v7 ablation ladder from Goal_Pose_V7.yaml.

Why a ladder and not just "run v7": v7 changes ~9 things at once relative to
armB (arms-down URDF, path mode, speed curriculum, BT flicker, perception noise,
10x disturbance, protection penalties, symmetry loss, minibatch PPO). That is
exactly the shape of armD, which changed 12 levers at once, collapsed to 24.3 cm
and told us nothing about which lever did it. We only recovered the cause
afterwards because the per-category breakdown happened to exonerate the noise.

So each arm below turns on ONE group on top of the previous one, and every arm
is evaluated with the same harness. If v7 underperforms armB we will know where.

  E0  armB reproduced on the v3 runner + arms-down URDF
      -> does the URDF dynamics change survive the warm start? does symmetry
         loss (never actually active in armA-D) help or hurt?
  E1  E0 + path mode + speed curriculum
      -> THE question: does a receding lookahead raise body speed at all?
  E2  E0 + disturbance + BT flicker + perception noise
      -> what does robustness cost on the gates?
  V7  everything (E1 + E2 + protection penalties + settled-stop)
      -> the integrated candidate

E1 and E2 are siblings on E0, not a chain, so their effects are separable and
they can run concurrently.

Usage (server):
    python tools/make_v7_arms.py --checkpoint <armB best>.pth
    python tools/make_v7_arms.py --only E1
"""

import argparse
import copy
import math
import os

# yaml is only needed to WRITE configs. Importing it at module scope made
# `--gpu-of` (and therefore the pre-launch arm check) fail on any machine
# without PyYAML, which is exactly where you want that check to still run.
try:
    import yaml
except ImportError:
    yaml = None

BASE = os.path.join("envs", "K1", "Goal_Pose_V7.yaml")
BASE_V8 = os.path.join("envs", "K1", "Goal_Pose_V8.yaml")
# SmoothTurn needs its own env class and task name, so it is generated from the
# V8 base rather than patched onto V7.
V8_ARMS = {
    "G4_smoothturn": {},
}

# armB @ 11500 iter -- the best goal-pose policy measured so far
# (3.9 cm median, 52.8% strict success). E0-E3/V7_full continue from it.
DEFAULT_CKPT = "logs/K1/K1/Goal_Pose/2026-07-24-17-22-03_armB_goal_reached/nn/model_11500.pth"

# E0's own final checkpoint: 12000 iters already adapted to the arms-down URDF.
# The F-batch (2026-07-27) warm-starts from THIS, not from armB directly, so it
# is not re-absorbing the T-pose-to-arms-down shock a second time. Note this is
# the 80-degree splay geometry E0 actually trained on; K1_locomotion_armsdown.urdf
# was rebuilt to 90 degrees + a rearward shoulder-pitch tuck AFTER E0 finished
# (commit f038e36), so the F-batch takes on one more small dynamics change
# (80->90 deg) on top -- watch early episode length the same way, just expect a
# smaller transient than the original armB->arms-down jump.
# E0's best, re-selected on its OWN config after the harness fix. The earlier
# pick (12000) came from the contaminated evaluation; the corrected re-eval puts
# the winner at 6200 with 2.7 cm / 89% strict / 2 falls -- the best goal-pose
# policy this project has produced, and the right base for everything below.
ARMSDOWN_CKPT = "logs/K1/K1/Goal_Pose_V7/2026-07-26-19-36-15_E0_armB_armsdown/nn/model_6200.pth"

_OFF_PATH = {
    "commands.goal_mode_mixture": {"waypoint": 1.0, "path": 0.0},
}
_OFF_ROBUST = {
    "randomization.disturbance.enabled": False,
    "noise.goal_bt_flicker.prob_per_step": 0.0,
    "noise.goal_pos.range": [0.0, 0.0],
    "noise.goal_heading.range": [0.0, 0.0],
    "noise.goal_pos_bias.range": [0.0, 0.0],
    "noise.goal_heading_bias.range": [0.0, 0.0],
    "noise.goal_obs_hold_steps": [0, 0],
}
_OFF_PROTECT = {
    "rewards.scales.dof_pos_margin": 0.0,
    "rewards.scales.dof_vel_margin": 0.0,
    "rewards.scales.torque_margin": 0.0,
    "rewards.scales.electrical_power": 0.0,
    "rewards.stop_ang_speed_threshold": 0.0,   # 0 = exact armB goal_reached
    "rewards.scales.stand_posture": 0.0,
}

ARMS = {
    # armB reproduced: no path, no extra robustness, no protection. The only
    # deltas vs armB itself are the arms-down URDF and the v3 runner.
    "E0_armB_armsdown": dict(**_OFF_PATH, **_OFF_ROBUST, **_OFF_PROTECT),
    # + the speed machinery, nothing else.
    "E1_path": dict(**_OFF_ROBUST, **_OFF_PROTECT),
    # + the robustness machinery, nothing else.
    "E2_robust": dict(**_OFF_PATH, **_OFF_PROTECT),
    # E1 with the SCHEDULER REMOVED: one wide fixed commanded-speed distribution
    # instead of a curriculum that creeps the ceiling upward.
    #
    # Why this might beat E1 rather than just differ from it:
    #   * The curriculum is a single global float shared by all 4096 envs, so it
    #     throws away every per-env signal. legged_gym's terrain curriculum is
    #     per-env; this one is cruder than that.
    #   * A curriculum stops raising the demand once the policy stops keeping up,
    #     so it CANNOT distinguish "cannot do it yet" from "cannot ever do it".
    #     Wherever it settles tells us nothing about K1's physical ceiling.
    #   * With a wide fixed range the achieved-vs-commanded curve saturates, and
    #     the knee IS the physical ceiling -- measured, for free, in the same run.
    #     That was masterplan2's open item #2 (a separate MaxSpeed probe).
    # The leash is what makes this safe: a robot handed an impossible speed just
    # parks the goal at lookahead_max, so those samples degrade into "run flat
    # out" episodes rather than into dead gradient.
    #
    # speed_init is pinned to 1.0 because path_speed is drawn from
    # [smin, smin + (smax-smin)*speed_level]; with the curriculum off, the level
    # never moves, so anything below 1.0 would silently clamp the range.
    "E3_wide_nosched": dict(
        **_OFF_ROBUST, **_OFF_PROTECT,
        **{
            "commands.path.speed_curriculum": False,
            "commands.path.speed_init": 1.0,
            "commands.path.speed_range_mps": [0.2, 2.0],
        }),
    # everything (base Goal_Pose_V7.yaml as written).
    "V7_full": {},
}

# ---- 2026-07-27 batch: masterplan2.md section 8 -----------------------------
# All four continue from ARMSDOWN_CKPT (E0's final model), not from armB, so
# they inherit an already-adapted-to-arms-down policy instead of re-absorbing
# that shock a second time. Base is E0 (path/robust/protect all off) unless
# noted, so each F-arm again isolates ONE new mechanism.
F_ARMS = {
    # + Rudin timed-window gate. Never once switched on in v3 either -- this is
    # its first real measurement anywhere in the project.
    "F1_timed": dict(**_OFF_PATH, **_OFF_ROBUST, **_OFF_PROTECT,
                     **{"rewards.final_window_s": 2.0}),
    # + path mode on the FIXED lookahead (pace/floor/leash), the grid-adaptive
    # (speed x curvature) curriculum, and carrot dwell.
    #
    # The v7 batch made this the load-bearing arm. It confirmed path mode raises
    # speed (segment peak median 0.35 -> 1.20 m/s) but also exposed that path
    # training wrecks waypoint accuracy (6.3 cm -> 37.9 cm) because the goal is
    # never reachable, and left the sustained ceiling unresolved because the
    # tracking ratio came out NON-MONOTONIC (0.92 -> 0.45 -> 0.77). That
    # non-monotonicity is the speed/curvature confound in raw form: speed and
    # curvature were sampled independently, so one commanded-speed bin mixes
    # easy wide curves with impossible tight ones. The grid separates them, and
    # dwell (in the base config) removes the waypoint conflict -- so F2 is now
    # testing three coupled repairs, not one lever. That is deliberate: they are
    # mutually dependent (path_lag needs the floor, the grid promotes on
    # path_lag, dwell keeps the waypoint gate meaningful), and E1 already
    # provides the without-any-of-them control.
    "F2_grid": dict(**_OFF_ROBUST, **_OFF_PROTECT,
                    **{"commands.path.speed_grid.enabled": True}),
    # + robustness, with BT flicker turned up (0.004 -> 0.01).
    #
    # The v7 batch left this question unresolved rather than answered: E2
    # (flicker on) did get the lowest stress |omega| p90 at 2.46 vs E0's 2.76,
    # but E2 was also by far the slowest arm (3.6% of time above 1.0 m/s vs
    # E0's 14.4%), so "filters the jitter" and "simply moves less" are not
    # separable in that comparison. F3 keeps the disturbance suite identical to
    # E2's and changes only the flicker rate, so if F3 lands at E2-like |omega|
    # WITHOUT E2's speed collapse, filtering is real.
    "F3_stress": dict(**_OFF_PATH, **_OFF_PROTECT,
                      **{"noise.goal_bt_flicker.prob_per_step": 0.01}),
    # ---- G batch (2026-07-28): everything rebased on E0, which measured
    # 2.7 cm / 89% strict / 2 falls once evaluated on its own task. E1's
    # path numbers cannot be reused -- the lookahead floor changed the task
    # underneath that checkpoint, so its re-eval scored an old policy against
    # new semantics (path 24.8 -> 165.9 cm, falls 5 -> 346). Path mode therefore
    # has to be re-asked from scratch, with the corrected code, on the E0 base.
    "G1_speed": dict(**_OFF_ROBUST, **_OFF_PROTECT,
                     **{"commands.path.speed_grid.enabled": True}),
    # E2's own-task number was never produced (its re-eval did not finish), so
    # the robustness cost is still unmeasured. Same disturbance suite as E2 plus
    # the harder flicker rate that F3 was going to ask.
    "G2_robust": dict(**_OFF_PATH, **_OFF_PROTECT,
                      **{"noise.goal_bt_flicker.prob_per_step": 0.01}),
    # Integration candidate: speed + robustness + protection + the scripted
    # elbows. The static arms-down pose already banked the inertia (I_zz -69%);
    # what it could not do is change with motion state, which is what the
    # scripted elbows add -- straightened into a repeatable stance when parked
    # for a clean RLKick handoff, tucked rearward once moving so the hands
    # cannot catch on another robot.
    "G3_full": {
        "asset.file": "resources/K1/K1_locomotion_armswing.urdf",
        "arm_script.enabled": True,
        # armsdown collapses the (fixed) elbows into Trunk's own rigid body, so
        # self-collision there is a no-op regardless of this flag -- there is
        # only one body, nothing to collide with itself. armswing makes the
        # elbows revolute, so left/right_hand_link become SEPARATE rigid bodies,
        # and the parked pose sits the hand 3.1 cm behind the hip -- inside the
        # torso collision mesh. With self-collision on (0 = enabled, matching
        # legged_gym convention) PhysX measured 444,180 N pushing the hands
        # apart from Trunk on literally the first control step: a scripted arm
        # is not something the reward or the policy is meant to keep clear of
        # its own body, so there is nothing to learn here, only an explosion to
        # suppress. 1 = disabled.
        "asset.self_collisions": 1,
    },
    # F4: reserved. Filled in after the v7 + F1-F3 results are in hand, as the
    # single arm that folds in whatever those runs recommend. Not generated by
    # this script -- there is nothing to generate yet.
}



# ---- I batch R0 (2026-08-03): reproduce E0, then correct the foot ------------
# Everything before this rolled forward from a worse starting point three
# generations running (E0 2.72 cm -> G1 5.52 -> H 6.90). R0 does not try to
# improve anything: it establishes that the current harness can still reproduce
# E0 from E0's own weights. If I0a cannot, no later result means anything.
FOOT_EDGE_REAL = [[0.094, 0.035, -0.024], [0.094, -0.035, -0.024],
                  [-0.066, 0.035, -0.024], [-0.066, -0.035, -0.024]]

I_ARMS = {
    # Byte-identical task to E0, continued from E0's own checkpoint. Zero levers.
    "I0a_repro": dict(**_OFF_PATH, **_OFF_ROBUST, **_OFF_PROTECT),
    # + the foot the robot actually has. asset.feet_edge_pos ships T1's foot:
    # the configured corners span 0.223 x 0.100 m while left_foot_link's
    # collision box is 0.16 x 0.07 at origin (0.014, 0, -0.008), i.e. 39% too
    # long and 43% too wide. Every balance policy this project has trained
    # learned on a support polygon it does not have, and the heel x = -0.1015
    # that any touchdown reward would key off is 3.6 cm behind the real box.
    # Expect I0b to score WORSE than I0a -- that is the point; sim becomes as
    # hard as the robot. It is the honest baseline every later round builds on.
    "I0b_foot": dict(**_OFF_PATH, **_OFF_ROBUST, **_OFF_PROTECT,
                     **{"asset.feet_edge_pos": FOOT_EDGE_REAL}),
    # base_height_target sweep. This is NOT the posture reward the user ruled
    # out as hardcoding -- the term already exists at scale -20, and the config
    # contradicts itself: init_state.pos z spawns the robot at 0.58 while the
    # reward drags it to 0.52. Six centimetres. On hardware the robot visibly
    # sinks before toppling and stands at 0.58, so the constant is simply wrong
    # and this measures which value is right. Correcting a wrong constant is the
    # same class of change as feet_edge_pos, not a new shaping term.
    "I0c_h055": dict(**_OFF_PATH, **_OFF_ROBUST, **_OFF_PROTECT,
                     **{"rewards.base_height_target": 0.55}),
    "I0d_h058": dict(**_OFF_PATH, **_OFF_ROBUST, **_OFF_PROTECT,
                     **{"rewards.base_height_target": 0.58}),
}

# ---- I batch R1 (2026-08-03) ------------------------------------------------
# R0 carried two winners that were never run TOGETHER: the corrected foot (falls
# 15 -> 4 per ~740 attempts) and base_height_target 0.55 (best p90 4.81, best
# strict 91.7%, falls 4). I1a is that combination and is R1's control.
#
# The two levers on top are a 2x2 factorial, not four unrelated arms, because
# the standing question and the speed question are suspected to interact: the
# cadence fix removes the forced torso lean, and disturbance is what should make
# an upright stance necessary rather than rewarded. I1d is the interaction cell
# and is labelled as such -- it is the one arm that is deliberately not a single
# lever.
# save_interval 100 gave R0 exactly two checkpoints (0 and 200), so watch_eval
# scored one of them and the stop rule could only fire at the very end -- the
# early-stop budget was zero. 25 gives eight scoring points across the round.
def merge(*layers):
    """Later layers override earlier ones.

    dict(**a, **b) raises on a shared key, and the robustness levers legitimately
    re-specify keys that _OFF_ROBUST already sets to their off value -- that is
    the whole point of a lever. Overriding must be allowed and must be explicit
    about who wins.
    """
    out = {}
    for d in layers:
        out.update(d)
    return out


_I1_BASE = merge(_OFF_PATH, _OFF_ROBUST, _OFF_PROTECT,
                 {"asset.feet_edge_pos": FOOT_EDGE_REAL,
                  "rewards.base_height_target": 0.55,
                  "runner.save_interval": 25})
# Low dose. E2 and G2 both collapsed to a near-stationary policy when the whole
# robustness bundle went on at once, so this turns on ONLY the two-class contact
# wrench -- no goal jitter, no bias, no flicker, no staleness.
_I1_FORCE = {
    "randomization.disturbance.enabled": True,
    "randomization.disturbance.interval_s": [8.0, 14.0],
    "randomization.disturbance.collision.force_n": [40.0, 100.0],
    "randomization.disturbance.collision.duration_s": [0.06, 0.10],
    "randomization.disturbance.support.force_n": [3.0, 8.0],
    "randomization.disturbance.support.duration_s": [0.5, 1.5],
}
_I1_CADENCE = {"commands.cadence_coupling.enabled": True}

I1_ARMS = {
    "I1a_base": merge(_I1_BASE),
    "I1b_force": merge(_I1_BASE, _I1_FORCE),
    "I1c_cadence": merge(_I1_BASE, _I1_CADENCE),
    "I1d_both": merge(_I1_BASE, _I1_FORCE, _I1_CADENCE),
}

# ---- I batch R2 (2026-08-03) ------------------------------------------------
# R1 adopted disturbance (falls -77%) and held cadence (it cancels that gain).
# R2 is not a new hypothesis pair; both arms correct conditions the simulator
# was inventing, in the same class as the feet_edge_pos fix.
#
# base_com randomises the Trunk's centre of mass by +/-0.1 m on every axis. The
# Trunk's own collision box is 0.12 x 0.18, i.e. half-extents of 0.060 and
# 0.090, so the sampled CoM lands OUTSIDE the trunk -- 1.7x its half-length in
# x. The foot spans -0.066 to +0.094, so a 0.1 m shift in x puts the CoM past
# the edge of the support polygon: a robot that cannot balance standing still,
# demanded not to fall. Real CoM uncertainty from cables, battery and camera
# placement is 1-3 cm. This is not a robustness trade; it is a fiction being
# removed, which is why it belongs before the speed round rather than after.
_I2_BASE = merge(_I1_BASE, _I1_FORCE)          # R1 winner: foot + h055 + force
I2_ARMS = {
    "I2a_dr": merge(_I2_BASE, {
        "randomization.base_com.range": [-0.025, 0.025],
        "randomization.base_mass.range": [0.92, 1.08],
        "randomization.dof_stiffness.range": [0.85, 1.15],
        "randomization.dof_damping.range": [0.85, 1.15],
    }),
    # feet_swing pays for "swing foot not in contact", and contact is any corner
    # within 1 cm of the ground, so 1.1 cm of clearance already earns full marks
    # -- on a perfectly flat plane nothing ever asks for more. Uneven ground asks
    # for it without naming a number, which is the difference between creating a
    # condition and hardcoding a target.
    "I2b_terrain": merge(_I2_BASE, {
        "terrain.type": "trimesh",
    }),
}

# R2 accepted the calibration DR: it cost nothing measurable and buys sim2real
# margin, so everything after this carries it.
_I2_DR = {
    "randomization.base_com.range": [-0.025, 0.025],
    "randomization.base_mass.range": [0.92, 1.08],
    "randomization.dof_stiffness.range": [0.85, 1.15],
    "randomization.dof_damping.range": [0.85, 1.15],
}
_I3_BASE = merge(_I2_BASE, _I2_DR)

I3_ARMS = {
    # I2b_terrain switched terrain on and inherited the base config's numbers
    # without reading them: random_height 0.1 over a horizontal_scale of 0.1 is
    # +/-10 cm of relief on a 10 cm grid -- local slopes near 45 deg, for a robot
    # with a 0.52 m leg and a 2.4 cm thick foot. That is rubble, not a floor. A
    # RoboCup artificial-turf pitch is +/-1-2 cm, and the discrete/slope classes
    # are a different skill (climbing) that nothing in this project asks for.
    #
    # 2 cm is not a clearance target dressed up as terrain. Nothing rewards
    # lifting to any height; the ground simply stops paying for 1.1 cm, which is
    # all the flat plane ever required (feet_contact = any corner within 1 cm, so
    # 1.1 cm already scores full marks on feet_swing). The policy picks the height.
    #
    # REJECT IF strict drops more than 2 %p against I2a_dr ON THE SAME GROUND.
    # I2b lost 6.2 %p, but it was scored on its own rubble while I2a was scored on
    # a plane, so that number never meant what it was read as. Both arms must be
    # evaluated with --terrain plane.
    "I3_rough": merge(_I3_BASE, {
        "terrain.type": "trimesh",
        "terrain.terrain_proportions": [0.0, 0.0, 1.0, 0.0],   # random only
        "terrain.random_height": 0.02,                          # +/-2 cm, turf-like
        "terrain.discrete_height": 0.0,
        "terrain.slope": 0.0,
    }),
}

# R3a -- joint zero drift.  The user reports K1's joint zero going off often
# enough that posture visibly degrades and a re-calibration is needed, and asked
# whether randomisation can absorb it.  Partly: the bias is NOT in the
# observation, so the policy cannot learn to correct a direction it cannot see.
# What it can learn is a posture that survives any offset -- which is
# conservative, and 8-17 showed conservatism costs speed.  So clean
# non-inferiority is part of the verdict, not an afterthought.
#
# encoder_bias moves what the policy SEES; target_offset moves where the PD
# actually aims.  A real zero drift causes both, so they go on together at the
# same magnitude (ibatch 484).
#
# Sizing is a search, not a claim.  hbatch's M2_jointdr proposed +/-0.015 rad
# (0.86 deg) and was never run, and no measurement of real drift exists in this
# repo.  The user chose to bracket from above at 10 deg and walk it down.  Note
# what 10 deg means: every one of the 22 joints is drawn independently, so with
# six leg joints the foot lands about 22 cm from where the policy thinks -- more
# than the whole support polygon (foot half-length 9.4 cm).  Expect collapse
# rather than a slowdown; that is still a useful upper bracket.  3 deg is run
# alongside so one round brackets from both sides instead of two.
def _jointcal(deg):
    r = round(math.radians(deg), 5)
    return {
        "randomization.joint_encoder_bias": {
            "range": [-r, r], "operation": "additive", "distribution": "uniform"},
        "randomization.joint_target_offset": {
            "range": [-r, r], "operation": "additive", "distribution": "uniform"},
    }


I3A_ARMS = {
    "I3a_jointcal10": merge(_I3_BASE, _jointcal(10.0)),
    "I3a_jointcal3": merge(_I3_BASE, _jointcal(3.0)),
}

# R3b -- stance width.  The deployed policy walks with its feet 7.0 cm apart
# centre to centre (8-31), and the foot is 7.0 cm wide, so the commanded gait has
# the feet touching.  On the real robot they collide and it falls after three
# steps.  Nothing asks for anything else: feet_offset_x and feet_offset_y are
# both at scale 0, so a narrow stance is free -- it cuts lateral CoM travel and
# saves feet_slip and energy, and on a flat floor with no latency it costs
# nothing at all.
#
# Only the scale needs changing.  _reward_feet_offset_y already exists, the
# target channel commands[:,9] is live at 0, feet_distance_ref is 0.18 so a
# target of 0 means exactly an 18 cm stance, and the lateral-speed rolloff
# (feet_offset_vel_scale_y 0.3) already relaxes the term for sidesteps.
#
# The scale must be NEGATIVE: the function returns clip(|error|, 0, 0.1), an
# unsigned magnitude, so a positive scale would pay the policy to stand wrong.
#
# Sizing is measured, not guessed.  _prepare_reward_function multiplies by dt
# (0.02), and the term saturates at 0.1, so the per-step contribution is
# scale * 0.002.  Against the live terms at their operating point --
# constellation +0.070, feet_swing +0.060, goal_reached +0.020 -- a scale of
# -0.5 would be -0.001, which is nothing.  -10 gives -0.020, the size of
# goal_reached; -30 gives -0.060, the size of feet_swing.  That brackets
# "noticeable" against "as strong as the gait term itself".
#
# Note the saturation: at the current 7 cm stance the error is 11 cm, past the
# 0.1 clip, so there is no local gradient there -- only a constant penalty that
# makes the sub-10 cm region preferable. The p90 tail at 11.5 cm is inside the
# clip and does have one.
_I3_STANCE_BASE = merge(_I2_BASE, _I2_DR)

I3B_ARMS = {
    "I3b_stance10": merge(_I3_STANCE_BASE, {"rewards.scales.feet_offset_y": -10.0}),
    "I3b_stance30": merge(_I3_STANCE_BASE, {"rewards.scales.feet_offset_y": -30.0}),
}


# ---- L batch: the first run longer than a screening -------------------------
# Every round so far has been 200 iterations from E0's warm start, which is a
# screening budget: it answers "is this lever going the right way", not "where
# does it converge". Two things now make a long run worth its cost.
#
# First, the H batch's 12000-iteration collapse has identified causes that are
# fixed -- 3-1's asset contamination and 8-10's base_com +/-0.1 fiction -- so
# "long runs degrade here" is a claim that deserves retesting rather than
# inheriting.
#
# Second, cadence coupling cannot be judged at 200. It changes the gait's
# defining parameter, and I1c was warm-started from a fixed-2 Hz policy and
# asked to relearn stride timing in 200 iterations. Its falls regression is
# exactly the re-adaptation transient watch_eval's own docstring warns about:
# I1b was over the line at 125 and 150 and clean at 175. 12000 iterations is the
# first budget where "cadence hurts" and "cadence had not finished landing" are
# distinguishable.
#
# base_height_target goes back to 0.52. The leg is 0.4607 m from hip to sole and
# the trunk sits 0.062 above the hip, so 0.5227 m is the tallest this robot can
# stand -- 0.55 is unreachable and its -20 penalty can never be satisfied, which
# leaves a permanent "stand taller" pull with no saturation point. This is a
# defect correction, not a lever, so it is in BOTH arms. It also restores the
# assumption cadence_coupling was sized under: max_stride_m 0.28 is documented as
# "~0.55 x leg length (base_height_target 0.52)".
#
# save_interval 250 instead of 25: 12000 iterations would otherwise write 480
# checkpoints, and the selector samples a tail of 12 regardless.
_L_BASE = merge(_I3_BASE, {
    "rewards.scales.feet_offset_y": -10.0,     # R3b adopted (8-32)
    "rewards.base_height_target": 0.52,        # 도달 가능한 최대. 8-33
    "runner.save_interval": 250,
    # 발 관성 교정. 기존 asset의 발은 주모멘트 삼각부등식을 2.6배 위반한다 --
    # 어떤 강체도 가질 수 없는 값이고, foot_yaw_L/R이 -1.0으로 그 위에서 학습돼 왔다.
    # 벤더 파일(k1/K1_locomotion.urdf)의 발 inertial만 이식했다. 그 파일은 엉덩이가
    # 15 mm 낮고 링크마다 4-5% 무거운 다른 리비전이라 통째로는 쓰지 않는다.
    "asset.file": "resources/K1/K1_locomotion_armsdown_footfix.urdf",
})

L_ARMS = {
    "L1_long": merge(_L_BASE),
    "L2_cadence": merge(_L_BASE, {"commands.cadence_coupling.enabled": True}),
    # 위상 없는 보행. feet_swing은 gait_process 위에 정의돼 있어서 클럭을 지우면
    # 발을 들 이유가 사라진다 -- 그래서 삭제가 아니라 교체다. 관측 채널(9, 16, 17)은
    # 그대로 두므로 obs 54가 유지되고 warm start가 산다: 정책은 클럭을 계속 보지만
    # 그걸 따를 이유가 없어진다. 스스로 타이밍을 만들면 위상 자유가 증명되고, 그때
    # 채널을 지우는 것은 값싼 후속 작업이다.
    #
    # 스케일은 추정이 아니라 실측 비율로 맞췄다. 절대값은 에피소드 길이에 따라
    # 움직이므로 같은 run 안에서 constellation 대비 비율을 본다:
    #   stance10  feet_swing/constellation = 32.04/102.26 = 0.313
    #   L3 smoke(375) feet_air_time/constellation = 3.96/5.58 = 0.709  -> 2.3배 과함
    # 175면 0.331로 거의 같아진다. 희소 보상이라 스케일 숫자가 커 보이는 것이지
    # 세다는 뜻이 아니다 -- 착지에서만 값이 나온다.
    # L1_long과 config가 같다. 바뀐 것은 config가 아니라 판정이다.
    # L1은 --ref 0.0253 으로 죽었는데 그 값은 옛 발 관성에서 나온 것이고, 발이 29%
    # 무거워진 로봇에게 옛 기준을 들이댄 것이었다. 낙상이 iteration 500에 이미 18로
    # 시작부터 높았던 것도 점진적 악화가 아니라 재적응의 시작점이다. 이번에는 새
    # 동역학에서 뽑은 기준(L1 @500 = 0.0284)으로, 훨씬 관대하게 보면서 끝까지 간다.
    "L4_settle": merge(_L_BASE),
    "L3_phasefree": merge(_L_BASE, {
        "rewards.scales.feet_swing": 0.0,
        "rewards.scales.feet_air_time": 175.0,
    }),
}


# ---- M 배치: L에서 교란된 것을 풀고, 처음부터 학습을 처음으로 시도한다 ---------
#
# M1 — L 배치의 교란 해소. _L_BASE는 stance10 대비 **두 가지**를 동시에 바꿨다:
#   (a) asset.file -> footfix urdf,  (b) base_height_target 0.55 -> 0.52.
# 그래서 §8-35의 "발 관성 교정이 대가를 치른다"는 (a) 때문인지 (b) 때문인지 갈리지
# 않는다. M1은 (a)만 바꾼다. stance10과의 차이가 발 관성 하나가 된다.
#
# M2 — 외력이 부족했는가. 지금 학습 외란은 collision 40-100 N / interval 8-14 s이고,
# eval의 held-out은 50-120 N / 4-8 s다. 12배 노출에서도 판별력이 없었다(§8-37,
# p=0.15). 두 해석이 남는다: "정책이 정말 강인하다" 또는 "이 크기가 애초에 이 로봇을
# 흔들지 못한다". 후자면 더 큰 외력에서 갈려야 한다. 18.71 kg에 200 N을 0.15 s 주면
# 충격량 30 N*s -> Δv 1.6 m/s로, 순항속도(1.48)를 넘겨 세우는 크기다.
# 방향도 같이 넓힌다: 지금은 크기만 랜덤이고 방향은 코드가 정한다(사용자 지적).
#
# M3/M4 — 처음부터 학습. 지금까지 모든 arm이 warm start였고, 그래서 봉우리가 1250
# 부근에서 서고 그 뒤로 흐른다. 이건 "학습"이 아니라 "미세조정"의 모양이다.
# Booster 공식 T1.yaml은 num_envs 4096 / max_iterations 10000으로 **처음부터** 돈다.
# L3(phase-free)가 1000에서 무너진 것도, 클럭을 따르도록 학습된 정책에게 클럭을 무시
# 하라고 1000 iteration 준 것이라 재적응이 끝날 리가 없었다. 위상 자유가 가능한지는
# 처음부터 돌려야 답이 나온다. M3는 같은 예산의 위상 기반 대조군이다 -- 둘 다 나쁘면
# 원인은 위상이 아니라 from-scratch 예산이다.
_M_BASE = merge(_I3_STANCE_BASE, {"rewards.scales.feet_offset_y": -10.0})

_PHASEFREE = {
    "rewards.scales.feet_swing": 0.0,
    "rewards.scales.feet_air_time": 175.0,
}

# ⛔ M1_footfix가 쓰는 footfix urdf는 **틀린 수정**이었다. 벤더 MJCF와 대조해서
# 알아낸 것:
#
#   벤더 MJCF(K1_serial.xml)의 발을 링크 프레임으로 환산  ixx 0.000202  iyy 0.000741
#   학습 URDF(K1_locomotion_armsdown.urdf)의 발            ixx 0.002020  iyy 0.007410
#
# **ixx와 iyy만 정확히 10배이고 izz(0.000785)와 ixz(-0.000053)는 소수점까지 같다.**
# 모델링 차이가 아니라 소수점 오타 두 개다. 그래서 텐서가 삼각부등식을 2.64배 위반하고,
# MuJoCo는 이 로봇을 아예 거부한다("inertia must satisfy A + B >= C").
#
# 내가 만든 footfix.urdf는 질량 0.4940 kg짜리 **다른 리비전**의 발을 이식한 것이라
# 질량부터 틀렸다(정답은 0.38305 그대로). footfix2는 두 항을 10으로 나누기만 한다 --
# 결과가 벤더 MJCF 값과 전 자릿수 일치하고 총 질량이 18.7142로 보존된다.
#
# 발목에서의 의미: J가 10배 작아지면 wn = sqrt(kp/J)가 sqrt(10) = 3.16배 커진다.
# 학습 13.1 Hz -> 실제 41.3 Hz. 정책은 3.16배 느린 발목에 맞춰 착지 타이밍을 배웠다.
# 사용자가 실기에서 본 "발이 중심을 못 잡고 왔다갔다"가 그 모양이다.
_FOOTFIX2 = {"asset.file": "resources/K1/K1_locomotion_armsdown_footfix2.urdf"}

# ⛔⛔ footfix2도 틀렸다. 로봇에 직접 붙어 실물 URDF를 읽고 확정했다
# (~/Workspace/test_deploy/tasks/scratch_curriculum/assets/K1_locomotion.urdf):
#
#   | | 실물 | 우리 학습 | footfix(원본) | footfix2 |
#   | 총 질량 | 19.6660 | 18.7142 | 18.9328 | 18.7142 |
#   | 발 질량 | 0.49400 | 0.38305 | 0.49400 ✅ | 0.38305 |
#   | 발 ixx/iyy/izz | 0.000261/0.001170/0.001213 | 0.002020/0.007410/0.000785 | 동일 ✅ | 0.000202/0.000741/0.000785 |
#
# 즉 실물은 **더 무거운 다른 리비전**이고, 내 "10배 소수점 오타" 추론은 낡은 MJCF를
# 기준으로 삼아 틀렸다. footfix(원본)의 발이 실물이었다 -- 내가 그걸 "다른 리비전"이라며
# 폐기한 것이 오판이었다.
#
# 그리고 차이가 발뿐이 아니다:
#   * 링크 22개 전부 실물이 무겁다 (+0.952 kg, +5.1 %)
#   * Hip_Pitch 원점 z가 15 mm 낮다
#   * 팔 자세가 다르다 -- 실물 어깨 ±1.35 rad(77°), 우리 armsdown ±1.5708(90°).
#     배포가 실제로 잡는 값은 ±1.3이라 실물 쪽이 맞다. 3-1이 다시 나온 것이다.
#
# 그래서 발만 이식하지 않고 **자산을 통째로 바꾼다.** K1_robot_boxfoot.urdf는
# 실물 URDF 그대로에 발 충돌만 우리 상자(0.16 x 0.07 x 0.032 @ 0.014,0,-0.008)로
# 되돌린 것이다 -- 실물 URDF는 발 충돌이 메시인데, 그 메시는 발목 하우징까지 포함해
# 두께가 5.2 cm다. 실제 접촉면은 3.2 cm 상자이고 벤더 MJCF도 같은 상자를 쓴다.
# 검증: 총 19.6660 kg, 관성 위반 0건, 충돌 기하 구성은 기존 학습 자산과 동일.
_ROBOT_ASSET = {"asset.file": "resources/K1/K1_robot_boxfoot.urdf"}

M_ARMS = {
    "M1_footfix": merge(_M_BASE, {
        "asset.file": "resources/K1/K1_locomotion_armsdown_footfix.urdf",
    }),
    # 오타만 고친 발. stance10과의 차이가 이것 하나다.
    "M5_footfix2": merge(_M_BASE, _FOOTFIX2),
    "M2_forcewide": merge(_M_BASE, {
        "randomization.disturbance.interval_s": [3.0, 8.0],
        "randomization.disturbance.collision.force_n": [40.0, 200.0],
        "randomization.disturbance.collision.torque_nm": [3.0, 35.0],
        "randomization.disturbance.collision.duration_s": [0.05, 0.15],
        "randomization.disturbance.support.force_n": [3.0, 20.0],
        "randomization.push_force.range": [0.0, 30.0],
        "randomization.push_torque.range": [0.0, 4.0],
    }),
    # from-scratch 두 arm은 footfix2 위에서 돈다. 발 오타는 레버가 아니라 결함이고,
    # 물리적으로 불가능한 로봇으로 5.7시간을 처음부터 학습시킬 이유가 없다.
    # 둘 다 같은 발을 신으므로 M3 대 M4(위상 대 위상없음) 비교는 그대로 성립한다.
    # 실물 자산 위에서 처음부터. 이전 M3/M4는 footfix2(틀린 리비전)로 돌다가
    # 575 / 2175 iteration에서 중단했다.
    "M3_scratch_phase": merge(_M_BASE, _ROBOT_ASSET),
    "M4_scratch_phasefree": merge(_M_BASE, _ROBOT_ASSET, _PHASEFREE),
    # 실물 자산 + warm start. stance10과의 차이가 "자산" 하나다.
    "M7_robotasset": merge(_M_BASE, _ROBOT_ASSET),
    # 관절 영점 오차 ±2도. 폭이 추측이 아니라 MuJoCo 실측이다 --
    # 2026-08-05 스윕: ±1도 낙상 0, ±2도 낙상 0(다만 오른 Hip_Roll이 이미 20 %
    # 토크 포화), **±3도에서 낙상 12로 절벽**. 즉 ±2가 이 정책이 버티는 상한이다.
    # 기존 I3a_jointcal3(±3도)은 절벽 바로 위 값이라 그 결과를 다시 봐야 한다.
    #
    # 왜 이 채널인가: 실기 2.4초에서 Hip_Roll이 13.5도 벌어지고 안 돌아오며
    # 토크 p99가 sim의 1.6배인데 **부호전환 주파수는 sim과 같았다**(6.3 vs 6.5-8.1 Hz).
    # 진동이 아니라 "계속 세게 미는데 안 돌아온다"의 모양이고, MuJoCo에서 Hip_Roll에
    # -5도를 넣었을 때 그 모양이 재현됐다(roll median -20.2도, 토크 포화, 낙상 10).
    # 부호도 맞는다 -- +5도는 낙상 0인데 -5도(다리가 모이는 방향)만 무너진다.
    #
    # ⚠️ 이것은 "영점 오차가 이 증상을 만들 수 있다"이지 "실기 영점이 틀어져 있다"가
    # 아니다. 후자는 R4(실기 영점 실측)로만 닫힌다. 그래도 학습이 이 채널을
    # [0,0]으로 꺼둔 채 돌아온 것은 그 자체로 결함이다 -- 정책은 1도도 본 적이 없다.
    "M8_jointcal2": merge(_M_BASE, _ROBOT_ASSET, _jointcal(2.0)),
    # 발목 이득 오차. 실기 대조에서 발목 pitch 토크가 sim의 0.6-0.7배인데 궤적은
    # 1.3배였다 -- 각도는 크고 토크는 작다, 즉 발목이 지면을 못 민다. 원인은
    # 미확정이므로(평행 링크 가설은 교차결합 지표가 지지하지 않았다) 메커니즘을
    # 특정하지 않고 결과만 흔든다. 0.6이 실측 하단, 1.3은 반대쪽 여유다.
    "M9_anklegain": merge(_M_BASE, _ROBOT_ASSET, {
        "randomization.ankle_gain": {
            "range": [0.6, 1.3], "operation": "scaling", "distribution": "uniform"},
    }),
}

# M3/M4는 checkpoint 없이 돈다. utils/runner.py:167 `if not checkpoint: return`이
# 그 경로이므로 train 명령에서 --checkpoint를 빼면 된다. SCRATCH_ARMS가 그 표식이다.
SCRATCH_ARMS = {"M3_scratch_phase", "M4_scratch_phasefree"}


ARMS_ON_E0 = {"I0a_repro", "I0b_foot", "I0c_h055", "I0d_h058",
              "I1a_base", "I1b_force", "I1c_cadence", "I1d_both",
              "I2a_dr", "I2b_terrain", "G1_speed", "G2_robust", "G3_full"}


def set_dotted(cfg, dotted, value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node[k]
    if keys[-1] not in node:
        raise KeyError("override path not in base config: {}".format(dotted))
    node[keys[-1]] = value


# V8_ARMS (G4_smoothturn) was missing from this merge, so --only G4_smoothturn
# raised KeyError before it ever reached the per-arm is_v8 branch below --
# G4 has never successfully generated a config, let alone run its smoke test.
ALL_ARMS = dict(**ARMS, **F_ARMS, **I_ARMS, **I1_ARMS, **I2_ARMS, **I3_ARMS,
                **I3A_ARMS, **I3B_ARMS, **L_ARMS, **M_ARMS, **V8_ARMS)

# GPU 0 / GPU 1 split. F-batch: F1+F2 share GPU 0 (lighter, no disturbance),
# F3 gets GPU 1 to itself (disturbance + higher flicker rate is the heavier one).
GPU_OF = {
    "E0_armB_armsdown": "cuda:0", "E1_path": "cuda:0", "E3_wide_nosched": "cuda:0",
    "E2_robust": "cuda:1", "V7_full": "cuda:1",
    "F1_timed": "cuda:0", "F2_grid": "cuda:0", "F3_stress": "cuda:1",
    # R0 runs ONE process per card: the H data showed damage is decided by
    # iteration 100, so round latency matters more than aggregate throughput.
    # 2 cards, 4 arms, 200 iterations: ~13 min each alone, ~26 min two-up. The
    # "one process per card" rule was about round LATENCY, and at this length
    # doubling up costs nothing that matters. Each pair shares a card so the two
    # comparisons that matter most sit under identical thermal/clock conditions:
    # control vs foot on one, and the two height values on the other.
    "I0a_repro": "cuda:0", "I0b_foot": "cuda:0",
    "I0c_h055": "cuda:1", "I0d_h058": "cuda:1",
    # Factorial pairs share a card so control-vs-lever runs under one thermal
    # and clock condition: (control, force) on 0, (cadence, both) on 1.
    "I1a_base": "cuda:0", "I1b_force": "cuda:0",
    "I1c_cadence": "cuda:1", "I1d_both": "cuda:1",
    "I2a_dr": "cuda:0", "I2b_terrain": "cuda:1",
    "I3_rough": "cuda:0",
    "I3a_jointcal10": "cuda:0",
    "I3b_stance10": "cuda:0",
    "L1_long": "cuda:0",
    "L2_cadence": "cuda:0",
    "L3_phasefree": "cuda:1",
    "L4_settle": "cuda:1",
    "I3b_stance30": "cuda:1",
    "I3a_jointcal3": "cuda:1",
    "M1_footfix": "cuda:0",
    "M2_forcewide": "cuda:1",
    "M3_scratch_phase": "cuda:0",
    "M4_scratch_phasefree": "cuda:1",
    "M5_footfix2": "cuda:0",
    "M7_robotasset": "cuda:0",
    "M8_jointcal2": "cuda:1",
    "M9_anklegain": "cuda:0",
}


def default_checkpoint(arm):
    return ARMSDOWN_CKPT if (arm in F_ARMS or arm in I_ARMS or arm in I1_ARMS
                             or arm in I2_ARMS or arm in ARMS_ON_E0
                             or arm in V8_ARMS) else DEFAULT_CKPT


def _need_yaml():
    if yaml is None:
        raise SystemExit('PyYAML이 필요하다: pip install pyyaml')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="override for every generated arm; if omitted each arm "
                         "picks E0's final checkpoint (F-batch) or armB (E-batch)")
    ap.add_argument("--gpu-of", dest="gpu_of", default=None,
                    help="print the GPU index assigned to this arm, then exit")
    ap.add_argument("--num_envs", type=int, default=4096)
    ap.add_argument("--max_iterations", type=int, default=12000)
    ap.add_argument("--out_dir", default="sweeps")
    ap.add_argument("--only", help="generate just this arm")
    args = ap.parse_args()
    if args.gpu_of:
        # Reaching here proves every arm dict above was built without a
        # duplicate-key TypeError -- the failure that killed the I1 launch.
        print(GPU_OF.get(args.gpu_of, 'cuda:0').split(':')[-1])
        return 0


    _need_yaml()
    arms = ALL_ARMS if not args.only else {args.only: ALL_ARMS[args.only]}
    with open(BASE, "r", encoding="utf-8") as f:
        base_v7 = yaml.load(f.read(), Loader=yaml.FullLoader)
    base_v8 = None
    if os.path.exists(BASE_V8):
        with open(BASE_V8, "r", encoding="utf-8") as f:
            base_v8 = yaml.load(f.read(), Loader=yaml.FullLoader)
    os.makedirs(args.out_dir, exist_ok=True)

    for arm, patch in arms.items():
        is_v8 = arm in V8_ARMS
        if is_v8 and base_v8 is None:
            raise SystemExit("{} needs {} which is missing".format(arm, BASE_V8))
        cfg = copy.deepcopy(base_v8 if is_v8 else base_v7)
        cfg["basic"]["description"] = arm
        for dotted, value in patch.items():
            set_dotted(cfg, dotted, value)
        path = os.path.join(args.out_dir, "{}.yaml".format(arm))
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)
        dev = GPU_OF.get(arm, "cuda:0")
        ckpt = args.checkpoint or default_checkpoint(arm)
        print("# --- {} ({} overrides) -> {} ---".format(arm, len(patch) or "no", dev))
        task = "K1/Goal_Pose_V8" if is_v8 else "K1/Goal_Pose_V7"
        # SCRATCH_ARMS는 warm start를 하지 않는다. utils/runner.py:167이
        # `if not cfg["basic"]["checkpoint"]: return` 이므로 인자를 아예 빼면
        # 처음부터 학습한다. --checkpoint를 빈 문자열로 주는 것이 아니다 --
        # argparse가 그걸 문자열로 받아 config를 덮어쓰기 때문에 경로가 ""가 되고
        # 그 뒤 torch.load가 아니라 falsy 검사에 걸려 조용히 통과할 뿐, 의도가
        # 코드에 드러나지 않는다. 인자를 생략하는 쪽이 읽는 사람에게 정직하다.
        ckpt_arg = "" if arm in SCRATCH_ARMS else "--checkpoint {} ".format(ckpt)
        print("python train_v7.py --task={task} --config {cfg} --headless True "
              "{ckpt}--num_envs {ne} --max_iterations {mi} "
              "--sim_device {dev} --rl_device {dev}\n".format(
                  task=task, cfg=path, ckpt=ckpt_arg, ne=args.num_envs,
                  mi=args.max_iterations, dev=dev))


if __name__ == "__main__":
    main()
