"""Generate the M-cell short factorial: disturbance / joint DR / mirror loss.

WHY THIS EXISTS
---------------
The 2026-07-30 H batch trained four arms to 12000 iterations and the checkpoint
selector picked `model_0.pth` -- the untouched G1 warm start -- for all four.
Reading the four selection.md candidate tables side by side shows why, and it is
the same curve every time:

    iter |  H0    H1    H2    H3    (waypoint position median, cm)
    -----+---------------------------------
       0 |  7.3   7.3   7.3   7.3
     100 | 10.7  12.3  13.2  10.5
     200 | 12.4  14.7  13.5  13.9
   2600+ | ~38   ~38   ~38   ~39

Position degrades monotonically from the first checkpoint while falls DROP
(29 -> 0..15). That is the E2/G2 signature: the policy buys safety by moving
less. It happened in H0 too, and H0 was supposed to be the control -- so the
cause lives in the layer all four share, not in any arm's own intervention.

Two consequences drive this file:

1. The signal is already unambiguous at iteration 100. Training to 12000 to
   learn it costs ~10 h/arm and taught us nothing extra. These cells stop at
   200 and checkpoint at 0/25/50/100/200.
2. Nobody ever ran "fine-tune G1 and change NOTHING". Without that cell we
   cannot tell an intervention's cost from the cost of fine-tuning itself.
   M0 is that cell, and every other cell is M0 plus exactly one lever.

THE FACTORIAL
-------------
    M0_control   G1 continued, every new lever off        <- the missing control
    M1_force     + scenario-aware disturbance
    M2_jointdr   + encoder bias and PD target offset
    M3_mirror    + symmetry (mean-equivariance) loss

Each cell differs from M0 by ONE lever, so each main effect is a paired
difference against a control that shares the seed, the warm start and the
protocol. This is deliberately NOT H1's design: H1 moved mirror loss, mirror
augmentation, encoder bias, target offset and init-q sigma together, so its
result cannot be attributed.

Mirror *augmentation* is excluded on purpose. The transition-augmentation path
computes its PPO denominator as log pi_old(Ma|Ms) while the samples are drawn
from pi_old(a|s) (runner_v3.py:183-195, 295-298); those agree only if the old
policy is already perfectly symmetric, so the ratio is biased by construction.
M3 uses symmetry_coef only, which does not touch that path. Fix the denominator
before asking whether augmentation helps.

RAMP
----
`disturbance.ramp_steps` is counted in CONTROL steps. A 200-iteration run is
200 * horizon(24) = 4800 control steps per env, so H0's inherited
`ramp_steps: 12000` would leave the disturbance at ~40% of nominal for the whole
experiment -- the treatment would never actually be applied. M1 sets it to 1.
Anything short must check this; it is silent otherwise.

Usage (server, where PyYAML exists):
    python tools/make_mcell_configs.py --checkpoint <G1 model_10700.pth>
    python tools/make_mcell_configs.py --check      # verify committed files
"""

import argparse
import copy
import os
import sys

import yaml


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
BASE = os.path.join(ROOT, "sweeps", "hbatch", "H0-codex.yaml")
OUTDIR = os.path.join(ROOT, "sweeps", "mcells")

ITERATIONS = 200
SAVE_INTERVAL = 25
# 0/25/50/100/200 are the comparison points. 25 exists because H0 had already
# lost 3.4 cm by iteration 100 and we need to see whether the loss starts
# immediately or builds.
CHECKPOINTS = (0, 25, 50, 100, 200)


def deep_set(cfg, dotted, value):
    node = cfg
    parts = dotted.split(".")
    for key in parts[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[parts[-1]] = value


def deep_get(cfg, dotted, default=None):
    node = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


# ---- the four cells -------------------------------------------------------
#
# COMMON is applied to every cell, including M0. It turns OFF everything the H
# batch turned on, so M0 really is "G1, continued". Each cell then re-enables
# exactly one thing.

COMMON = {
    "basic.max_iterations": ITERATIONS,
    "basic.save_interval": SAVE_INTERVAL,
    # H3's heel_strike_ahead reward is dropped for good: the user's call, and
    # the H3 selection table showed it neutral through iteration 100 (10.5 vs
    # H0's 10.7) while carrying an overstride risk nobody measured.
    "rewards.scales.heel_strike_ahead": 0.0,
    # The cruise-stability reward is gated on a speed x steady-acceleration
    # sigmoid, i.e. it is a schedule. Removed here so no cell contains an
    # implicit curriculum -- if high-speed stability needs a reward, that is a
    # separate question asked after the common layer is trusted again.
    "rewards.scales.high_speed_stability": 0.0,
    # mirror off by default; M3 re-enables the loss only
    "algorithm.symmetry_coef": 0.0,
    "algorithm.mirror_augmentation_coef": 0.0,
    # disturbance off by default; M1 re-enables it
    "randomization.disturbance.enabled": False,
    # joint DR off by default; M2 re-enables it
    "randomization.joint_encoder_bias.range": [0.0, 0.0],
    "randomization.joint_target_offset.range": [0.0, 0.0],
    # G1 never saw goal-observation noise. Leaving it on in "the control" would
    # reproduce E2's collapse and blame it on fine-tuning.
    "noise.goal_pos.range": [0.0, 0.0],
    "noise.goal_heading.range": [0.0, 0.0],
    "noise.goal_pos_bias.range": [0.0, 0.0],
    "noise.goal_heading_bias.range": [0.0, 0.0],
    "noise.goal_bt_flicker.prob_per_step": 0.0,
}

CELLS = {
    "M0_control": {},
    "M1_force": {
        "randomization.disturbance.enabled": True,
        # Scenario-aware sampling is already implemented in goal_pose_hbatch.py
        # but no released config ever switched it on. It replaces the 40-150 N
        # frontal-collision class -- which cannot happen to a robot that has a
        # camera and does not run head-first into another one -- with the three
        # contacts that actually occur: an omnidirectional shove, a push from
        # behind at walking speed, and an arm snagging on another robot or the
        # net. Height tiers put 90% of the wrench on chest/arm level.
        "randomization.disturbance.scenario_aware.enabled": True,
        # See module docstring: a 12000-step ramp inside a 4800-step run applies
        # ~40% of the treatment and calls it a result.
        "randomization.disturbance.ramp_steps": 1,
        "randomization.disturbance.event_probability": 0.35,
        "randomization.disturbance.interval_s": [6.0, 12.0],
    },
    "M2_jointdr": {
        # The mild H0 dose, not H1's stronger one: H1 confounded this with
        # mirror, so the mild level is the one with a clean prior.
        "randomization.joint_encoder_bias.range": [-0.015, 0.015],
        "randomization.joint_target_offset.range": [-0.010, 0.010],
    },
    "M3_mirror": {
        "algorithm.symmetry_coef": 0.5,
        "algorithm.mirror_augmentation_coef": 0.0,   # see module docstring
    },
}


def build(base, name, checkpoint, seed):
    cfg = copy.deepcopy(base)
    for dotted, value in COMMON.items():
        deep_set(cfg, dotted, value)
    for dotted, value in CELLS[name].items():
        deep_set(cfg, dotted, value)
    deep_set(cfg, "basic.description", name)
    deep_set(cfg, "basic.seed", seed)
    if checkpoint:
        deep_set(cfg, "basic.checkpoint", checkpoint)
    # Every cell must start from byte-identical weights or the paired
    # difference is not paired.
    deep_set(cfg, "algorithm.load_optimizer_state", False)
    return cfg


def verify(cfg, name):
    """Fail loudly on the things that would silently void the experiment."""
    problems = []
    if deep_get(cfg, "basic.max_iterations") != ITERATIONS:
        problems.append("max_iterations != {}".format(ITERATIONS))
    if deep_get(cfg, "algorithm.load_optimizer_state", False):
        problems.append("load_optimizer_state must be False (G1's saved Adam LR "
                        "is 1.71e-4, 34x the declared 5e-6)")
    if deep_get(cfg, "randomization.disturbance.enabled", False):
        horizon = int(deep_get(cfg, "algorithm.horizon_length", 24) or 24)
        ramp = int(deep_get(cfg, "randomization.disturbance.ramp_steps", 1) or 1)
        budget = ITERATIONS * horizon
        if ramp > budget // 4:
            problems.append(
                "ramp_steps {} vs {} control steps in the run: the disturbance "
                "never reaches nominal".format(ramp, budget))
    # exactly one lever off the control
    levers = sum([
        bool(deep_get(cfg, "randomization.disturbance.enabled", False)),
        float(deep_get(cfg, "randomization.joint_encoder_bias.range", [0, 0])[1] or 0) > 0,
        float(deep_get(cfg, "algorithm.symmetry_coef", 0.0) or 0.0) > 0,
        float(deep_get(cfg, "algorithm.mirror_augmentation_coef", 0.0) or 0.0) > 0,
    ])
    expected = 0 if name == "M0_control" else 1
    if levers != expected:
        problems.append("{} active levers, expected {}".format(levers, expected))
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None,
                    help="G1 warm start, e.g. logs/.../G1_speed/nn/model_10700.pth")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--check", action="store_true",
                    help="verify the committed files instead of writing them")
    args = ap.parse_args()

    with open(BASE) as fh:
        base = yaml.safe_load(fh)

    if not os.path.isdir(OUTDIR):
        os.makedirs(OUTDIR)

    bad = 0
    for name in CELLS:
        path = os.path.join(OUTDIR, name + ".yaml")
        cfg = build(base, name, args.checkpoint, args.seed)
        problems = verify(cfg, name)
        for p in problems:
            print("INVALID {}: {}".format(name, p))
            bad += 1
        if args.check:
            if not os.path.exists(path):
                print("MISSING {}".format(path))
                bad += 1
                continue
            with open(path) as fh:
                on_disk = yaml.safe_load(fh)
            if on_disk != cfg:
                print("DRIFT   {} differs from the generator".format(path))
                bad += 1
            else:
                print("ok      {}".format(os.path.relpath(path, ROOT)))
        else:
            with open(path, "w") as fh:
                yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
            print("wrote   {}".format(os.path.relpath(path, ROOT)))

    print("\ncheckpoints saved at iterations: {}".format(
        ", ".join(str(c) for c in CHECKPOINTS)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
