#!/usr/bin/env python3
"""Generate the four frozen HBatch arms from the audited V7/G1 backbone."""

import argparse
import copy
import hashlib
import os
import xml.etree.ElementTree as ET

import yaml


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "envs", "K1", "Goal_Pose_V7.yaml")
ASSET = "resources/K1/K1_locomotion_hbatch-codex.urdf"
WARM_START = (
    "logs/K1/K1/Goal_Pose_V7/2026-07-28-17-02-35_G1_speed/"
    "nn/model_10700.pth"
)


def uniform(lo, hi):
    return {"range": [float(lo), float(hi)], "operation": "additive",
            "distribution": "uniform"}


def gaussian(std):
    return {"range": [0.0, float(std)], "operation": "additive",
            "distribution": "gaussian"}


def common(base, name):
    c = copy.deepcopy(base)
    c["basic"].update({
        "task": "K1/Goal_Pose_HBatch",
        "checkpoint": WARM_START,
        "description": name,
        "max_iterations": 12000,
    })
    c["asset"]["file"] = ASSET
    c["arm_script"]["enabled"] = False
    # G1 -> H changes the objective/distribution.  Reusing G1's Adam moments
    # and checkpoint LR (observed 1.7086e-4) would make the first H update 34x
    # larger than the declared 5e-6 and confound all four arm comparisons.
    c["runner"]["load_optimizer_state"] = False
    c["commands"]["goal_mode_mixture"] = {"waypoint": 0.65, "path": 0.35}
    path = c["commands"]["path"]
    path["speed_grid"]["enabled"] = True
    # H resumes from a G1 policy whose long-running environment had already
    # exposed all 30 speed/curvature cells.  Old checkpoints did not serialize
    # that task state, so explicitly restore the current-best distribution
    # instead of silently restarting from only the 0.3 m/s seed cell.
    path["speed_grid"]["initial_active"] = "all"
    path.update({
        "constraint_mode": "radial_rate_limited",
        "drag_speed_cap_mps": 2.1,
        "floor_recovery_rate_mps": 2.1,
        "floor_recovery_grace_s": 2.0,
        "goal_rate_max_mps": 3.2,
        "floor_tolerance_m": 0.02,
        "arrival_reward_mode": "dwell_only",
        "pause_gait_during_dwell": True,
    })
    # G1's protect bundle was off.  Do not silently reintroduce G3's failed
    # multi-lever integration into the H baseline.
    for key in ("dof_pos_margin", "dof_vel_margin", "torque_margin", "electrical_power"):
        c["rewards"]["scales"][key] = 0.0
    c["rewards"]["scales"].update({
        "high_speed_stability": 0.0,
        "heel_strike_ahead": 0.0,
    })
    c["rewards"]["high_speed_stability"] = {
        "min_speed_mps": 0.8, "speed_width_mps": 0.10,
        "max_accel_mps2": 0.3, "accel_width_mps2": 0.08,
        "accel_filter_alpha": 0.10,
        "angular_rate_weight": 0.10, "vertical_velocity_weight": 0.02,
    }
    c["rewards"]["heel_strike"] = {
        "min_forward_speed_mps": 0.6, "velocity_gain_s": 0.08,
        "target_min_m": 0.02, "target_max_m": 0.12, "sigma_m": 0.04,
    }
    # H0/H3 are the non-mirror controls.  Goal_Pose_V7.yaml already carries a
    # nonzero mirror-loss coefficient, so both levers must be reset here before
    # H1/H2 opt into them; otherwise the supposed H0 baseline silently trains
    # with half of H1's intervention and H3 is no longer gait-only.
    c["algorithm"]["symmetry_coef"] = 0.0
    c["algorithm"]["mirror_augmentation_coef"] = 0.0
    c["algorithm"].update({
        "finite_checks": True,
        "min_learning_rate": 1.0e-6,
        "max_learning_rate": 1.0e-5,
        "max_abs_log_ratio": 10.0,
        "mirror_augmentation_max_std": 5.0,
        "mirror_augmentation_min_valid_share": 0.10,
    })
    # Mandatory low-dose observation jitter in every H arm.
    c["noise"]["goal_pos"] = gaussian(0.015)
    c["noise"]["goal_heading"] = gaussian(0.020)
    c["noise"]["goal_pos_bias"] = gaussian(0.020)
    c["noise"]["goal_heading_bias"] = gaussian(0.020)
    c["noise"]["goal_obs_hold_steps"] = [2, 3]
    c["noise"]["goal_bt_flicker"] = {
        "prob_per_step": 0.001, "radius_m": 0.30, "heading_rad": 0.20}
    # Disable the synchronized legacy velocity kick; H's event model is the
    # sole external-force lever and is evaluated explicitly with forces ON.
    c["randomization"]["kick_lin_vel"] = gaussian(0.0)
    c["randomization"]["kick_ang_vel"] = gaussian(0.0)
    c["randomization"]["push_force"] = gaussian(0.0)
    c["randomization"]["push_torque"] = gaussian(0.0)
    c["randomization"]["disturbance"] = {
        "enabled": True,
        "interval_s": [8.0, 14.0],
        "event_probability": 0.25,
        "ramp_steps": 12000,
        "collision_share": 0.25,
        # Fixed arm links are collapsed into Trunk by this asset.  Use five
        # bodies that remain independently addressable after asset loading.
        "body_names": ["Trunk", "Left_Hip_Roll", "Right_Hip_Roll",
                       "Left_Shank", "Right_Shank"],
        "collision": {"force_n": [40.0, 100.0], "torque_nm": [3.0, 12.0],
                      "duration_s": [0.05, 0.10]},
        "support": {"force_n": [3.0, 8.0], "torque_nm": [0.2, 1.0],
                    "duration_s": [0.5, 1.5]},
    }
    c["randomization"]["joint_encoder_bias"] = uniform(-0.015, 0.015)
    c["randomization"]["joint_target_offset"] = uniform(-0.010, 0.010)
    c["randomization"]["init_dof_pos"] = gaussian(0.050)
    # Final cross-arm evaluation must not inherit the arm-specific training
    # distribution.  In particular H2 deliberately trains with a denser force
    # schedule, and H1/H2 train with wider joint offsets; scoring each policy
    # on its own distribution would confound the intervention with test
    # difficulty.  The evaluator applies this shared held-out profile to every
    # H arm and fingerprints the effective config in each report.
    eval_disturbance = copy.deepcopy(c["randomization"]["disturbance"])
    eval_disturbance.update({
        "interval_s": [6.0, 12.0],
        "event_probability": 0.5,
        "ramp_steps": 1,
        "collision_share": 0.35,
        "high_speed_probability_boost": 2.0,
        "high_speed_threshold_mps": 0.8,
    })
    c["evaluation"].update({
        "hbatch_protocol_version": "2026-07-30-codex-v3",
        "hbatch_common_eval": {
            "noise_overrides": {
                "goal_bt_flicker": {
                    "prob_per_step": 0.001,
                    "radius_m": 0.30,
                    "heading_rad": 0.20,
                },
            },
            "randomization_overrides": {
                "joint_encoder_bias": uniform(-0.025, 0.025),
                "joint_target_offset": uniform(-0.020, 0.020),
                "init_dof_pos": gaussian(0.075),
            },
            "disturbance": eval_disturbance,
        },
        "perspective_overlays": True,
        "high_speed_threshold_mps": 0.8,
        "steady_accel_threshold_mps2": 0.3,
        "reset_guard_s": 0.25,
        "hbatch_gates": {
            "waypoint_pos_median_max_m": 0.0552,
            "waypoint_pos_p90_max_m": 0.0742,
            "waypoint_heading_median_max_deg": 2.54,
            "waypoint_never_arrived_share_max": 0.015,
            "overall_falls_per_1000_max": 5.0,
            "waypoint_falls_per_1000_max": 2.0,
            "time_to_1mps_regression_max": 0.10,
            "path_falls_per_1000_max": 5.0,
            "path_speed_median_min": 0.95,
            "path_floor_below_0p75_max": 0.10,
            "path_dwell_resume_recovery_share_max": 0.15,
            "path_outside_leash_max": 0.01,
            "time_to_1mps_reached_share_min": 0.80,
            "time_to_1mps_p90_s_max": 3.0,
            "cruise_share_of_valid_min": 0.05,
            "cruise_pitch_p90_max_deg": 20.0,
            "cruise_roll_p90_max_deg": 15.0,
            "cruise_ang_xy_p90_max_radps": 3.0,
            "directional_t0p5_reached_share_min": 0.80,
            "directional_t0p5_p90_s_max": 2.0,
            "force_survival_5s_min": 0.98,
            "speed_recovery_within_5s_min": 0.90,
            "speed_recovery_p90_s_max": 2.0,
            "mirror_error_p90_max": 0.10,
            "touchdown_samples_min": 100,
            "touchdown_lr_bias_max_m": 0.02,
            "touchdown_target_share_min": 0.40,
            "touchdown_overstride_share_max": 0.10,
            "jitter_falls_per_env_min_max": 0.50,
            "combined_falls_per_env_min_max": 0.50,
            "jitter_body_angvel_p90_max": 3.0,
            "combined_body_angvel_p90_max": 3.0,
        },
    })
    return c


def build(base):
    h0 = common(base, "H0_current_best_low_dose_robust")

    h1 = copy.deepcopy(h0)
    h1["basic"]["description"] = "H1_mirror_and_joint_calibration"
    h1["algorithm"]["symmetry_coef"] = 0.5
    h1["algorithm"]["mirror_augmentation_coef"] = 0.5
    h1["randomization"]["joint_encoder_bias"] = uniform(-0.025, 0.025)
    h1["randomization"]["joint_target_offset"] = uniform(-0.020, 0.020)
    h1["randomization"]["init_dof_pos"] = gaussian(0.075)

    h2 = copy.deepcopy(h1)
    h2["basic"]["description"] = "H2_accel_preserving_high_speed_stability"
    h2["rewards"]["scales"]["high_speed_stability"] = -0.5
    h2["randomization"]["disturbance"].update({
        "interval_s": [6.0, 12.0], "event_probability": 0.35,
        "ramp_steps": 72000, "collision_share": 0.35,
        "high_speed_probability_boost": 2.0, "high_speed_threshold_mps": 0.8})
    h2["noise"]["goal_bt_flicker"]["prob_per_step"] = 0.002

    # H3 is deliberately H0 + one gait lever.  It must not inherit H1/H2, or a
    # result could not be attributed to touchdown placement.
    h3 = copy.deepcopy(h0)
    h3["basic"]["description"] = "H3_gait_only_touchdown_ablation"
    h3["rewards"]["scales"]["heel_strike_ahead"] = 0.10
    return {"H0": h0, "H1": h1, "H2": h2, "H3": h3}


def verify_asset():
    generated = ET.parse(os.path.join(ROOT, ASSET)).getroot()
    reference = ET.parse(os.path.join(ROOT, "..", "k1", "K1_locomotion.urdf")).getroot()
    names = ["ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch",
             "Left_Elbow_Yaw", "ARight_Shoulder_Pitch", "Right_Shoulder_Roll",
             "Right_Elbow_Pitch", "Right_Elbow_Yaw"]
    for name in names:
        got = generated.find("joint[@name='{}']".format(name))
        ref = reference.find("joint[@name='{}']".format(name))
        if got is None or ref is None:
            raise RuntimeError("missing arm joint {}".format(name))
        got_rpy = [float(x) for x in got.find("origin").attrib["rpy"].split()]
        ref_rpy = [float(x) for x in ref.find("origin").attrib["rpy"].split()]
        if (got.attrib["type"] != "fixed"
                or max(abs(a - b) for a, b in zip(got_rpy, ref_rpy)) > 1.0e-6):
            raise RuntimeError("arm mismatch {}: generated={} reference={}".format(
                name, got_rpy, ref_rpy))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "sweeps", "hbatch"))
    ap.add_argument(
        "--check", action="store_true",
        help="verify committed H*-codex.yaml files match the generator without rewriting them")
    args = ap.parse_args()
    verify_asset()
    with open(BASE, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    os.makedirs(args.out, exist_ok=True)
    for name, cfg in build(base).items():
        path = os.path.join(args.out, "{}-codex.yaml".format(name))
        if args.check:
            if not os.path.isfile(path):
                raise FileNotFoundError("missing frozen HBatch config: {}".format(path))
            with open(path, encoding="utf-8") as f:
                committed = yaml.safe_load(f)
            if committed != cfg:
                raise RuntimeError(
                    "{} drifted from its generator; review and regenerate explicitly "
                    "before launch".format(path))
        else:
            with open(path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]
        print("{}  {}  {}{}".format(
            name, digest, path, "  OK" if args.check else ""))


if __name__ == "__main__":
    main()
