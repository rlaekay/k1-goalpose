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

# armB @ 11500 iter -- the best goal-pose policy measured so far
# (3.9 cm median, 52.8% strict success). Every arm continues from it.
DEFAULT_CKPT = "logs/K1/K1/Goal_Pose/2026-07-24-17-22-03_armB_goal_reached/nn/model_11500.pth"

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


def set_dotted(cfg, dotted, value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node[k]
    if keys[-1] not in node:
        raise KeyError("override path not in base config: {}".format(dotted))
    node[keys[-1]] = value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--num_envs", type=int, default=4096)
    ap.add_argument("--max_iterations", type=int, default=12000)
    ap.add_argument("--out_dir", default="sweeps")
    ap.add_argument("--only", help="generate just this arm")
    args = ap.parse_args()

    arms = ARMS if not args.only else {args.only: ARMS[args.only]}
    with open(BASE, "r", encoding="utf-8") as f:
        base = yaml.load(f.read(), Loader=yaml.FullLoader)
    os.makedirs(args.out_dir, exist_ok=True)

    # GPU 0 gets the two arms that answer the primary questions.
    gpu_of = {"E0_armB_armsdown": "cuda:0", "E1_path": "cuda:0",
              "E2_robust": "cuda:1", "V7_full": "cuda:1",
              "E3_wide_nosched": "cuda:0"}

    for arm, patch in arms.items():
        cfg = copy.deepcopy(base)
        cfg["basic"]["description"] = arm
        for dotted, value in patch.items():
            set_dotted(cfg, dotted, value)
        path = os.path.join(args.out_dir, "{}.yaml".format(arm))
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)
        dev = gpu_of.get(arm, "cuda:0")
        print("# --- {} ({} overrides) -> {} ---".format(arm, len(patch) or "no", dev))
        print("python train_v7.py --task=K1/Goal_Pose_V7 --config {cfg} --headless True "
              "--checkpoint {ckpt} --num_envs {ne} --max_iterations {mi} "
              "--sim_device {dev} --rl_device {dev}\n".format(
                  cfg=path, ckpt=args.checkpoint, ne=args.num_envs,
                  mi=args.max_iterations, dev=dev))


if __name__ == "__main__":
    main()
