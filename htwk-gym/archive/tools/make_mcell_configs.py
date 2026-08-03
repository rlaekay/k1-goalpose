#!/usr/bin/env python3
"""Generate the short causal M-cell screen from the *actual* G1 run config.

The H batch cannot identify disturbance, joint DR, or mirror effects: every
arm changed a common bundle and every selector returned the untouched model 0.
This screen keeps the G1 objective/path distribution and changes one lever per
cell for only 200 PPO iterations:

  M0_control-codex    minimum-allowed G1 continuation
  M1_force-codex      M0 + scenario-aware contact wrenches
  M2_jointdr-codex    M0 + persistent encoder/target offsets
  M3_mirror_off-codex M0 - G1's existing symmetry loss

G1 already trained with symmetry_coef=0.5.  Therefore adding 0.5 is not a
mirror experiment; removing it is the causal ablation.  Transition mirror PPO
augmentation remains off in every cell.

The base is deliberately the recorder's immutable G1 ``config.yaml``, not a
current source YAML and not H0.  H0 changed path semantics, stop/stand rewards,
noise, disturbances and the optimizer, so it is not a valid no-treatment base.
"""

import argparse
import copy
import hashlib
import os
import sys

import yaml


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEFAULT_BASE = os.path.join(
    ROOT, "logs", "K1", "K1", "Goal_Pose_V7",
    "2026-07-28-17-02-35_G1_speed", "config.yaml")
TEMPLATE = os.path.join(ROOT, "sweeps", "hbatch", "H0-codex.yaml")
OUTDIR = os.path.join(ROOT, "sweeps", "mcells")

# Copied read-only from the completed server run.  A mismatch means the causal
# base changed and must be reviewed instead of silently becoming a new M0.
EXPECTED_G1_CONFIG_SHA256 = (
    "5eb9aa12a46759624babe1b9d7a3c1c52028b2c3c5f243e6512cc7fa47e3910c")
ITERATIONS = 200
SAVE_INTERVAL = 25
CHECKPOINTS = (0, 25, 50, 100, 200)

CELLS = (
    "M0_control-codex",
    "M1_force-codex",
    "M2_jointdr-codex",
    "M3_mirror_off-codex",
)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get(cfg, dotted, default=None):
    node = cfg
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node


def put(cfg, dotted, value):
    node = cfg
    keys = dotted.split(".")
    for key in keys[:-1]:
        node = node.setdefault(key, {})
    node[keys[-1]] = copy.deepcopy(value)


def zero_range(cfg, name):
    node = cfg.setdefault("randomization", {}).setdefault(name, {})
    node.update({
        "range": [0.0, 0.0],
        "operation": "additive",
        "distribution": "uniform",
    })


def assert_g1_base(base, path, expected_sha):
    problems = []
    actual_sha = file_sha256(path)
    if expected_sha and actual_sha != expected_sha:
        problems.append("G1 config sha256 {} != frozen {}".format(
            actual_sha, expected_sha))
    expected = {
        "basic.description": "G1_speed",
        "asset.file": "resources/K1/K1_locomotion_armsdown.urdf",
        "commands.path.speed_grid.enabled": True,
        "randomization.disturbance.enabled": False,
        "rewards.scales.stand_posture": 0.0,
        "rewards.stop_ang_speed_threshold": 0.0,
        "algorithm.symmetry_coef": 0.5,
    }
    for key, value in expected.items():
        if get(base, key) != value:
            problems.append("{}={!r}, expected {!r}".format(
                key, get(base, key), value))
    for key in (
            "noise.goal_pos.range", "noise.goal_heading.range",
            "noise.goal_pos_bias.range", "noise.goal_heading_bias.range"):
        if get(base, key) != [0.0, 0.0]:
            problems.append("{} must be G1-clean [0,0]".format(key))
    if float(get(base, "noise.goal_bt_flicker.prob_per_step", -1.0)) != 0.0:
        problems.append("G1 goal flicker must be off")
    if problems:
        raise ValueError("invalid G1 causal base:\n  - " + "\n  - ".join(problems))
    return actual_sha


def scenario_disturbance(template, event_probability):
    d = copy.deepcopy(template["randomization"]["disturbance"])
    d.update({
        "enabled": True,
        "interval_s": [6.0, 12.0],
        "event_probability": float(event_probability),
        "ramp_steps": 1,
        "high_speed_probability_boost": 1.0,
        "body_names": [
            "Trunk", "Left_Hip_Roll", "Right_Hip_Roll",
            "Left_Shank", "Right_Shank",
        ],
        "scenario_aware": {"enabled": True},
    })
    return d


def common(base, template, checkpoint, seed):
    cfg = copy.deepcopy(base)
    put(cfg, "basic.task", "K1/Goal_Pose_HBatch")
    put(cfg, "basic.max_iterations", ITERATIONS)
    put(cfg, "basic.seed", int(seed))
    if checkpoint:
        put(cfg, "basic.checkpoint", checkpoint)
    # The reference arm angles are mandatory, so this is explicitly a
    # minimum-allowed G1 continuation rather than a byte-identical dynamics run.
    put(cfg, "asset.file", "resources/K1/K1_locomotion_hbatch-codex.urdf")
    put(cfg, "runner.save_interval", SAVE_INTERVAL)
    put(cfg, "runner.load_optimizer_state", False)
    put(cfg, "runner.use_wandb", False)

    # Fresh, conservative Adam.  All cells share this; the saved G1 Adam had an
    # adaptive LR far above its declared value and is unsafe to restore.
    put(cfg, "algorithm.learning_rate", 2.0e-6)
    put(cfg, "algorithm.min_learning_rate", 5.0e-7)
    put(cfg, "algorithm.max_learning_rate", 2.0e-6)
    put(cfg, "algorithm.desired_kl", 0.003)
    put(cfg, "algorithm.finite_checks", True)
    put(cfg, "algorithm.max_abs_log_ratio", 10.0)
    put(cfg, "algorithm.mirror_augmentation_coef", 0.0)
    put(cfg, "algorithm.mirror_augmentation_max_std", 5.0)
    put(cfg, "algorithm.mirror_augmentation_min_valid_share", 0.90)

    # Diagnostic cells omit all legacy/global wrench sources.  Production
    # candidates re-enable a low mandatory scenario dose only after this screen.
    for name in ("push_force", "push_torque", "kick_lin_vel", "kick_ang_vel"):
        zero_range(cfg, name)
    put(cfg, "randomization.disturbance",
        scenario_disturbance(template, event_probability=0.0))
    put(cfg, "randomization.disturbance.enabled", False)
    put(cfg, "randomization.joint_encoder_bias", {
        "range": [0.0, 0.0], "operation": "additive",
        "distribution": "uniform"})
    put(cfg, "randomization.joint_target_offset", {
        "range": [0.0, 0.0], "operation": "additive",
        "distribution": "uniform"})

    # HBatch-only reward methods must exist in the table but remain off.
    put(cfg, "rewards.scales.heel_strike_ahead", 0.0)
    put(cfg, "rewards.scales.high_speed_stability", 0.0)
    put(cfg, "rewards.heel_strike", copy.deepcopy(
        template["rewards"].get("heel_strike", {})))
    put(cfg, "rewards.high_speed_stability", copy.deepcopy(
        template["rewards"].get("high_speed_stability", {})))

    # One frozen exam for every policy.  Replace its old two-class force with
    # the same scenario model, at a held-out probability independent of M1's
    # training dose.  Clean/joint evals explicitly disable it.
    put(cfg, "evaluation", copy.deepcopy(template["evaluation"]))
    put(cfg, "evaluation.hbatch_common_eval.disturbance",
        scenario_disturbance(template, event_probability=0.50))
    put(cfg, "evaluation.mcell_protocol", {
        "version": "2026-08-01-codex-v1",
        "base_config_sha256": EXPECTED_G1_CONFIG_SHA256,
        "iterations": list(CHECKPOINTS),
        "diagnostic_only": True,
    })
    return cfg


def build(base, template, name, checkpoint, seed):
    cfg = common(base, template, checkpoint, seed)
    put(cfg, "basic.description", name.replace("-codex", "_codex"))
    if name == "M1_force-codex":
        put(cfg, "randomization.disturbance",
            scenario_disturbance(template, event_probability=0.35))
    elif name == "M2_jointdr-codex":
        put(cfg, "randomization.joint_encoder_bias.range", [-0.015, 0.015])
        put(cfg, "randomization.joint_target_offset.range", [-0.010, 0.010])
    elif name == "M3_mirror_off-codex":
        put(cfg, "algorithm.symmetry_coef", 0.0)
    return cfg


def verify(cfg, name, base):
    problems = []
    if get(cfg, "basic.max_iterations") != ITERATIONS:
        problems.append("max_iterations")
    if get(cfg, "runner.save_interval") != SAVE_INTERVAL:
        problems.append("runner.save_interval")
    if get(cfg, "runner.load_optimizer_state") is not False:
        problems.append("runner.load_optimizer_state")
    if get(cfg, "algorithm.mirror_augmentation_coef") != 0.0:
        problems.append("mirror augmentation must remain off")
    if get(cfg, "commands") != get(base, "commands"):
        problems.append("G1 command/path distribution drifted")
    if get(cfg, "noise") != get(base, "noise"):
        problems.append("G1 observation-noise distribution drifted")
    force = bool(get(cfg, "randomization.disturbance.enabled", False))
    joint = get(cfg, "randomization.joint_encoder_bias.range") != [0.0, 0.0]
    mirror_off = float(get(cfg, "algorithm.symmetry_coef", 0.0)) == 0.0
    expected = {
        "M0_control-codex": (False, False, False),
        "M1_force-codex": (True, False, False),
        "M2_jointdr-codex": (False, True, False),
        "M3_mirror_off-codex": (False, False, True),
    }[name]
    if (force, joint, mirror_off) != expected:
        problems.append("lever tuple {} != {}".format(
            (force, joint, mirror_off), expected))
    if force:
        if not get(cfg, "randomization.disturbance.scenario_aware.enabled", False):
            problems.append("scenario force not enabled")
        horizon = int(get(cfg, "runner.horizon_length", 24))
        if int(get(cfg, "randomization.disturbance.ramp_steps", 0)) > (
                ITERATIONS * horizon // 4):
            problems.append("force ramp exceeds short-run budget")
    for legacy in ("push_force", "push_torque", "kick_lin_vel", "kick_ang_vel"):
        if get(cfg, "randomization.{}.range".format(legacy)) != [0.0, 0.0]:
            problems.append("legacy source {} is nonzero".format(legacy))
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_config", default=DEFAULT_BASE)
    ap.add_argument("--template_config", default=TEMPLATE)
    ap.add_argument("--checkpoint")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--expected_base_sha", default=EXPECTED_G1_CONFIG_SHA256)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    with open(args.base_config, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    with open(args.template_config, encoding="utf-8") as f:
        template = yaml.safe_load(f)
    try:
        base_sha = assert_g1_base(
            base, args.base_config, args.expected_base_sha or None)
    except ValueError as exc:
        print("INVALID {}".format(exc), file=sys.stderr)
        return 1
    os.makedirs(OUTDIR, exist_ok=True)

    bad = 0
    for name in CELLS:
        cfg = build(base, template, name, args.checkpoint, args.seed)
        problems = verify(cfg, name, base)
        path = os.path.join(OUTDIR, name + ".yaml")
        if problems:
            bad += len(problems)
            for problem in problems:
                print("INVALID {}: {}".format(name, problem), file=sys.stderr)
            continue
        if args.check:
            if not os.path.isfile(path):
                print("MISSING {}".format(path), file=sys.stderr)
                bad += 1
            else:
                with open(path, encoding="utf-8") as f:
                    disk = yaml.safe_load(f)
                if disk != cfg:
                    print("DRIFT {}".format(path), file=sys.stderr)
                    bad += 1
                else:
                    print("ok {}".format(os.path.relpath(path, ROOT)))
        else:
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
            print("wrote {}".format(os.path.relpath(path, ROOT)))
    print("base sha256 {}".format(base_sha))
    print("checkpoints {}".format(",".join(map(str, CHECKPOINTS))))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
