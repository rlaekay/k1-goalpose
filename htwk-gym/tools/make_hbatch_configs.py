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
    c["commands"]["goal_mode_mixture"] = {"waypoint": 0.65, "path": 0.35}
    c["commands"]["path"]["speed_grid"]["enabled"] = True
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
    c["algorithm"]["mirror_augmentation_coef"] = 0.0
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
    c["randomization"]["disturbance"] = {
        "enabled": True,
        "interval_s": [8.0, 14.0],
        "event_probability": 0.25,
        "ramp_steps": 12000,
        "collision_share": 0.25,
        # Fixed arm links are collapsed into Trunk by this asset.  Use five
        # bodies that remain independently addressable after asset loading.
        "body_names": ["Trunk", "Left_Hip_Roll", "Right_Hip_Roll",
                       "Left_Knee_Pitch", "Right_Knee_Pitch"],
        "collision": {"force_n": [40.0, 100.0], "torque_nm": [3.0, 12.0],
                      "duration_s": [0.05, 0.10]},
        "support": {"force_n": [3.0, 8.0], "torque_nm": [0.2, 1.0],
                    "duration_s": [0.5, 1.5]},
    }
    c["randomization"]["joint_encoder_bias"] = uniform(-0.015, 0.015)
    c["randomization"]["joint_target_offset"] = uniform(-0.010, 0.010)
    c["randomization"]["init_dof_pos"] = gaussian(0.050)
    c["evaluation"].update({
        "perspective_overlays": True,
        "high_speed_threshold_mps": 0.8,
        "steady_accel_threshold_mps2": 0.3,
        "reset_guard_s": 0.25,
        "hbatch_gates": {
            "time_to_1mps_regression_max": 0.10,
            "path_falls_per_1000_max": 5.0,
            "force_survival_5s_min": 0.98,
            "speed_recovery_p90_s_max": 2.0,
            "mirror_error_p90_max": 0.10,
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
    args = ap.parse_args()
    verify_asset()
    with open(BASE, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    os.makedirs(args.out, exist_ok=True)
    for name, cfg in build(base).items():
        path = os.path.join(args.out, "{}-codex.yaml".format(name))
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        digest = hashlib.sha256(open(path, "rb").read()).hexdigest()[:12]
        print("{}  {}  {}".format(name, digest, path))


if __name__ == "__main__":
    main()
