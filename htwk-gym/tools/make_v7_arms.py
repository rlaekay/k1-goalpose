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
import os

import yaml

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

ARMS_ON_E0 = {"I0a_repro", "I0b_foot", "I0c_h055", "I0d_h058", "G1_speed", "G2_robust", "G3_full"}


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
ALL_ARMS = dict(**ARMS, **F_ARMS, **I_ARMS, **V8_ARMS)

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
}


def default_checkpoint(arm):
    return ARMSDOWN_CKPT if (arm in F_ARMS or arm in I_ARMS or arm in ARMS_ON_E0
                             or arm in V8_ARMS) else DEFAULT_CKPT


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
        print(GPU_OF.get(args.gpu_of, 'cuda:0').split(':')[-1])
        return 0


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
        print("python train_v7.py --task={task} --config {cfg} --headless True "
              "--checkpoint {ckpt} --num_envs {ne} --max_iterations {mi} "
              "--sim_device {dev} --rl_device {dev}\n".format(
                  task=task, cfg=path, ckpt=ckpt, ne=args.num_envs,
                  mi=args.max_iterations, dev=dev))


if __name__ == "__main__":
    main()
