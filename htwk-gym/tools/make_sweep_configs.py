"""Generate parallel-sweep config arms from the base Goal_Pose.yaml.

Why: the GPU is CPU-single-core bound (one train process tops out ~67% util at
8192 envs, using only 6.4/49 GB VRAM). Running several independent processes on
the SAME GPU pushes aggregate throughput up (each gets its own CPU core / GIL and
their kernels time-slice), and turns "try the v2 tuning ideas one at a time over
days" into one parallel batch. See MASTERPLAN 2026-07-24 entries.

Each arm = base config + a small override patch, written to sweeps/<arm>.yaml with
basic.description=<arm> so its run dir self-labels. All arms keep task=K1/Goal_Pose
(the env class is resolved from --task; --config picks the yaml). Launch each in its
own tmux window on the SAME GPU so you can kill one individually if the shared server
gets busy (shared-server yield principle).

Usage (server):
    python tools/make_sweep_configs.py            # writes sweeps/*.yaml, prints commands
    python tools/make_sweep_configs.py --num_envs 4096 --max_iterations 20000 --device cuda:1
"""

import os
import copy
import argparse

import yaml

BASE = os.path.join("envs", "K1", "Goal_Pose.yaml")

# best goal-pose checkpoint so far (v1 @ 20000 iter). Each arm continues from it and
# changes exactly ONE lever, so the eval-harness deltas are attributable.
DEFAULT_CKPT = "logs/K1/K1/Goal_Pose/2026-07-23-21-54-01/nn/model_20000.pth"

# One-variable-at-a-time arms. Values are dotted paths into the config.
ARMS = {
    # A: pure continuation -- the "v1 just needed more iterations" bet (baseline).
    "armA_continue": {},
    # B: turn on the sparse goal_reached bonus (+1/step while stopped inside the goal
    #    radius) to attack the two stuck metrics: final position and not-quite-stopping.
    "armB_goal_reached": {
        "rewards.scales.goal_reached": 1.0,
    },
    # C: 200 Hz physics (dt 0.005, decimation 4) instead of 500 Hz (dt 0.002, dec 10).
    #    Control frequency stays 50 Hz (dt*decimation = 0.02 s) so the policy's action
    #    semantics are unchanged -- only sim substep cost drops 2.5x. Tests the speedup.
    "armC_200hz": {
        "sim.dt": 0.005,
        "control.decimation": 4,
    },
    # D: integrated "v2 ultimate" dress rehearsal (GPU 0, 2026-07-24 user request).
    #    Deliberately NOT one-variable: bundles every sim2real lever we believe in, to
    #    preview the final deployment candidate while arms A-C answer the controlled
    #    single-lever questions on GPU 1.
    "armD_v2_ultimate": {
        # BT/perception goal jitter: goal_rel_x/y std up to 10 cm, heading std ~6 deg.
        # (The many real jitter sources -- localization, odometry, ball detection --
        # all collapse into noise on the relative goal, so this one knob covers them.)
        "noise.goal_pos.range": [0.0, 0.10],
        "noise.goal_heading.range": [0.0, 0.10],
        # Arrive AND stop is the task: sparse success bonus + settle into stand pose
        # near the goal (clean RLKick handoff stance).
        "rewards.scales.goal_reached": 1.0,
        "rewards.scales.stand_posture": -2.0,
        # A* lookahead points can be up to ~3 m out; longer segments so far goals are
        # actually reachable before resample.
        "commands.goal_dx": [-3.0, 3.0],
        "commands.goal_dy": [-2.0, 2.0],
        "commands.resampling_time_s": [4.0, 10.0],
        # Sustained gentle wrench (someone holding/nudging the robot by the arms)
        # instead of only 1 s shoves.
        "randomization.push_duration_s": 3.0,
        # Official Booster USD sets joint armature 0.02 (we had 0) -- closer actuator
        # dynamics to the real robot.
        "asset.armature": 0.02,
    },
}


def set_dotted(cfg, dotted, value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node[k]
    if keys[-1] not in node:
        raise KeyError("override path not found in base config: {}".format(dotted))
    node[keys[-1]] = value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--num_envs", type=int, default=4096)
    ap.add_argument("--max_iterations", type=int, default=20000)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--out_dir", default="sweeps")
    ap.add_argument("--only", help="generate just this arm (e.g. armD_v2_ultimate); leaves the other arms' yaml files untouched")
    args = ap.parse_args()

    arms = ARMS
    if args.only:
        if args.only not in ARMS:
            raise SystemExit("unknown arm {!r}; choices: {}".format(args.only, ", ".join(ARMS)))
        arms = {args.only: ARMS[args.only]}

    with open(BASE, "r", encoding="utf-8") as f:
        base = yaml.load(f.read(), Loader=yaml.FullLoader)
    os.makedirs(args.out_dir, exist_ok=True)

    print("# paste each block into its own tmux window (same GPU {}):\n".format(args.device))
    for arm, patch in arms.items():
        cfg = copy.deepcopy(base)
        cfg["basic"]["description"] = arm
        for dotted, value in patch.items():
            set_dotted(cfg, dotted, value)
        path = os.path.join(args.out_dir, "{}.yaml".format(arm))
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)
        cmd = (
            "python train.py --task=K1/Goal_Pose --config {cfg} --headless True "
            "--checkpoint {ckpt} --num_envs {ne} --max_iterations {mi} "
            "--sim_device {dev} --rl_device {dev}"
        ).format(cfg=path, ckpt=args.checkpoint, ne=args.num_envs,
                 mi=args.max_iterations, dev=args.device)
        override_str = ", ".join("{}={}".format(k, v) for k, v in patch.items()) or "(base, no override)"
        print("# --- {}: {} ---".format(arm, override_str))
        print(cmd + "\n")

    print("# after they finish (or enough iters), eval each newest arm run:")
    print("#   python eval_goal_pose.py --task K1/Goal_Pose --checkpoint <run>/nn/<latest>.pth "
          "--sim_device {dev} --rl_device {dev}".format(dev=args.device))


if __name__ == "__main__":
    main()
