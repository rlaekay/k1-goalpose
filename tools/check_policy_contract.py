#!/usr/bin/env python3
"""Cross-check a training run's frozen config against a deploy config.

A .pt file carries only weights and graph. Everything that turns those weights
into correct robot motion -- observation scaling, joint defaults, action scale,
control decimation, PD gains, gait frequency -- lives in the YAML next to it and
is mirrored by hand into the deploy config. When those two drift apart nothing
errors: the robot simply moves differently, and the cause is invisible at
runtime. (This has already bitten once: deploy shipped Hip/Knee PD 200/5 while
the frozen run trained at 100/2.)

So compare them explicitly before installing a policy.

Usage:
    check_policy_contract.py --frozen-config <run config.yaml> \\
                            --deploy-config <deploy/configs/X.yaml>

Pass "-" for --frozen-config to read it from stdin.
Exit status is 0 when everything matches, 1 on any mismatch.
"""

import argparse
import sys

try:
    import yaml
except ImportError:
    print("PyYAML is required (pip3 install pyyaml)", file=sys.stderr)
    sys.exit(2)


RED, YELLOW, GREEN, RESET = "\033[31m", "\033[33m", "\033[32m", "\033[0m"

# K1 leg joints in URDF order, i.e. deploy indices 11..22. Taken from
# resources/K1/K1_locomotion.urdf (movable joints after world_joint).
LEG_JOINTS = [
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]


def resolve_joint_map(mapping, joints, default_key=None):
    """Expand a training-style substring map into one value per joint.

    Training configs do not list 23 numbers; they write things like
    ``stiffness: {"Hip": 100., "Knee": 100., "Ankle": 50.}`` and the env matches
    each key as a *substring* of the real joint name. goal_pose.py iterates the
    keys without breaking, so when several match, the last one in insertion
    order wins -- replicated here so this check reflects what actually ran.

    Returns None if any joint is unmatched, which is exactly the case the env
    itself raises on.
    """
    if not isinstance(mapping, dict):
        return None
    out = []
    for joint in joints:
        value = None
        for key, val in mapping.items():
            if key == default_key:
                continue
            if key in joint:
                value = val
        if value is None and default_key is not None and default_key in mapping:
            value = mapping[default_key]
        if value is None:
            return None
        out.append(float(value))
    return out


def get(d, path, default=None):
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def approx(a, b, tol=1e-6):
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b


def compare_seq(a, b, tol=1e-6):
    if a is None or b is None:
        return False
    if len(a) != len(b):
        return False
    return all(approx(x, y, tol) for x, y in zip(a, b))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frozen-config", required=True,
                    help="training run config.yaml ('-' for stdin)")
    ap.add_argument("--deploy-config", required=True,
                    help="deploy/configs/<Policy>.yaml")
    args = ap.parse_args()

    if args.frozen_config == "-":
        frozen = yaml.safe_load(sys.stdin.read())
    else:
        with open(args.frozen_config) as fh:
            frozen = yaml.safe_load(fh)
    with open(args.deploy_config) as fh:
        deploy = yaml.safe_load(fh)

    if not isinstance(frozen, dict) or not isinstance(deploy, dict):
        print("could not parse one of the configs", file=sys.stderr)
        return 2

    leg_start = int(get(deploy, "policy.leg_dof_start", 11))
    dep_qpos = get(deploy, "common.default_qpos") or []
    dep_stiff = get(deploy, "common.stiffness") or []
    dep_damp = get(deploy, "common.damping") or []

    checks = []

    def check(label, frozen_val, deploy_val, ok):
        checks.append((label, frozen_val, deploy_val, ok))

    # --- shapes -------------------------------------------------------------
    check("num_observations",
          get(frozen, "env.num_observations"),
          get(deploy, "policy.num_observations"),
          approx(get(frozen, "env.num_observations"),
                 get(deploy, "policy.num_observations")))
    check("num_actions",
          get(frozen, "env.num_actions"),
          get(deploy, "policy.num_actions"),
          approx(get(frozen, "env.num_actions"),
                 get(deploy, "policy.num_actions")))

    # --- control ------------------------------------------------------------
    check("action_scale",
          get(frozen, "control.action_scale"),
          get(deploy, "policy.control.action_scale"),
          approx(get(frozen, "control.action_scale"),
                 get(deploy, "policy.control.action_scale")))
    check("decimation",
          get(frozen, "control.decimation"),
          get(deploy, "policy.control.decimation"),
          approx(get(frozen, "control.decimation"),
                 get(deploy, "policy.control.decimation")))

    # --- PD gains on the policy-controlled joints ---------------------------
    # Only the leg slice is compared: the frozen config describes the 12 DOFs the
    # policy owns, while deploy carries all 23 (arms/head are held, not learned).
    # This is the check that matters most -- a PD mismatch here (deploy once had
    # Hip/Knee 200/5 against a frozen 100/2) changes how the robot moves with no
    # runtime symptom.
    def joint_slice_check(label, frozen_node, deploy_list, default_key=None):
        if deploy_list is None or len(deploy_list) < leg_start + len(LEG_JOINTS):
            return
        dep_slice = [float(v) for v in deploy_list[leg_start:leg_start + len(LEG_JOINTS)]]
        if isinstance(frozen_node, list):
            fz_slice = [float(v) for v in frozen_node][-len(LEG_JOINTS):]
        else:
            fz_slice = resolve_joint_map(frozen_node, LEG_JOINTS, default_key)
        if fz_slice is None:
            check(label, None, None, False)
            return
        check(label, fz_slice, dep_slice, compare_seq(fz_slice, dep_slice))

    joint_slice_check("leg stiffness", get(frozen, "control.stiffness"), dep_stiff)
    joint_slice_check("leg damping", get(frozen, "control.damping"), dep_damp)
    joint_slice_check("leg default angles", get(frozen, "init_state.default_joint_angles"),
                      dep_qpos, default_key="default")

    # --- normalization ------------------------------------------------------
    for key, dep_path in (
        ("goal_pos", "policy.normalization.goal_pos"),
        ("goal_heading", "policy.normalization.goal_heading"),
        ("dof_pos", "policy.normalization.dof_pos"),
        ("dof_vel", "policy.normalization.dof_vel"),
        ("ang_vel", "policy.normalization.ang_vel"),
        ("gravity", "policy.normalization.gravity"),
        ("clip_actions", "policy.normalization.clip_actions"),
    ):
        fz = get(frozen, "normalization.obs_scales." + key)
        if fz is None:
            fz = get(frozen, "normalization." + key)
        dep = get(deploy, dep_path)
        if fz is None or dep is None:
            continue
        check("normalization." + key, fz, dep, approx(fz, dep))

    # --- goal envelope ------------------------------------------------------
    for label, fz_path, dep_path in (
        ("goal clamp x", "commands.goal_dx", "policy.goal_clamp.x_m"),
        ("goal clamp y", "commands.goal_dy", "policy.goal_clamp.y_m"),
    ):
        fz = get(frozen, fz_path)
        dep = get(deploy, dep_path)
        if fz is None or dep is None:
            continue
        # Frozen usually stores a [-a, a] range; deploy stores the magnitude.
        fz_mag = max(abs(float(v)) for v in fz) if isinstance(fz, list) else abs(float(fz))
        check(label, fz_mag, dep, approx(fz_mag, dep))

    # --- gait frequency (range vs the single deploy value) ------------------
    fz_gait = get(frozen, "commands.gait_frequency")
    dep_gait = get(deploy, "policy.gait_frequency")
    if isinstance(fz_gait, list) and len(fz_gait) == 2 and dep_gait is not None:
        lo, hi = float(fz_gait[0]), float(fz_gait[1])
        check("gait_frequency in trained range",
              "[%g, %g]" % (lo, hi), dep_gait, lo - 1e-9 <= float(dep_gait) <= hi + 1e-9)

    # --- report -------------------------------------------------------------
    compared = [c for c in checks if c[1] is not None and c[2] is not None]
    failed = [c for c in compared if not c[3]]
    skipped = [c for c in checks if c[1] is None or c[2] is None]

    for label, fz, dep, ok in compared:
        mark = GREEN + "ok  " + RESET if ok else RED + "FAIL" + RESET
        print("  [%s] %-32s frozen=%s deploy=%s" % (mark, label, fz, dep))
    for label, _, _, _ in skipped:
        print("  [%sskip%s] %-32s (not present in one of the configs)"
              % (YELLOW, RESET, label))

    if failed:
        print("\n%s%d field(s) disagree between the frozen run and the deploy "
              "config.%s" % (RED, len(failed), RESET), file=sys.stderr)
        print("The robot would move differently than the policy was trained to, "
              "with no runtime error.", file=sys.stderr)
        return 1

    if not compared:
        print("\n%snothing could be compared -- check the config layout%s"
              % (YELLOW, RESET), file=sys.stderr)
        return 1

    print("\n%sall %d comparable fields match%s" % (GREEN, len(compared), RESET))
    return 0


if __name__ == "__main__":
    sys.exit(main())
