#!/usr/bin/env python3
"""Static HBatch assertions, followed by the existing 300-step dynamic smoke."""

import argparse
import copy
import os
import json
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import yaml


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def require(ok, message):
    print("{}  {}".format("PASS" if ok else "FAIL", message))
    if not ok:
        raise AssertionError(message)


def arm_rpy(path, name):
    joint = ET.parse(path).getroot().find("joint[@name='{}']".format(name))
    return joint.attrib["type"], [float(x) for x in joint.find("origin").attrib["rpy"].split()]


def static_checks(path):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    desc = cfg["basic"]["description"]
    version = os.path.basename(path).split("-")[0]
    require(cfg["basic"]["task"] == "K1/Goal_Pose_HBatch", "HBatch task class selected")
    require(cfg["env"]["num_observations"] == 54 and cfg["env"]["num_actions"] == 12,
            "warm-start interface stays 54 observations / 12 actions")
    require(cfg["commands"]["path"]["speed_grid"]["enabled"], "G1 speed-curvature grid retained")
    require(cfg["randomization"]["disturbance"]["enabled"] and
            cfg["randomization"]["disturbance"]["event_probability"] > 0,
            "mandatory external disturbance is nonzero")
    require(cfg["randomization"]["disturbance"]["body_names"] == [
                "Trunk", "Left_Hip_Roll", "Right_Hip_Roll",
                "Left_Shank", "Right_Shank"],
            "five independently loadable disturbance bodies are frozen")
    require(cfg["noise"]["goal_pos"]["range"][1] > 0 and
            cfg["noise"]["goal_bt_flicker"]["prob_per_step"] > 0,
            "mandatory goal jitter/flicker is nonzero")
    require(cfg["randomization"]["joint_encoder_bias"]["range"] != [0, 0] and
            cfg["randomization"]["joint_target_offset"]["range"] != [0, 0],
            "persistent encoder and motor-target offsets are enabled")
    if version in ("H1", "H2"):
        require(cfg["algorithm"]["symmetry_coef"] > 0 and
                cfg["algorithm"]["mirror_augmentation_coef"] > 0,
                "H1/H2 mirror loss and transition augmentation both enabled")
    require((cfg["rewards"]["scales"]["heel_strike_ahead"] > 0) == (version == "H3"),
            "heel touchdown lever exists only in H3")
    require((cfg["rewards"]["scales"]["high_speed_stability"] < 0) == (version == "H2"),
            "steady high-speed stability lever exists only in H2")

    asset = os.path.join(ROOT, cfg["asset"]["file"])
    reference = os.path.join(ROOT, "..", "k1", "K1_locomotion.urdf")
    for name in ("ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch",
                 "Left_Elbow_Yaw", "ARight_Shoulder_Pitch", "Right_Shoulder_Roll",
                 "Right_Elbow_Pitch", "Right_Elbow_Yaw"):
        got, ref = arm_rpy(asset, name), arm_rpy(reference, name)
        require(got[0] == "fixed" and max(abs(a - b) for a, b in zip(got[1], ref[1])) <= 1e-6,
                "URDF arm matches reference: {}".format(name))
    print("PASS  static HBatch checks: {} ({})".format(version, desc))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sim_device", default="cuda:0")
    ap.add_argument("--rl_device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--static_only", action="store_true")
    args = ap.parse_args()
    cfg = static_checks(args.config)
    if args.static_only:
        return 0

    # Force both disturbance classes to appear inside a short smoke.  This
    # changes only the temporary smoke config, never the training arm.
    smoke_cfg = copy.deepcopy(cfg)
    smoke_cfg["randomization"]["disturbance"].update({
        "interval_s": [0.2, 0.6], "event_probability": 1.0, "ramp_steps": 1})
    with tempfile.NamedTemporaryFile(mode="w", suffix="-codex.yaml", delete=False) as f:
        yaml.safe_dump(smoke_cfg, f, sort_keys=False)
        temp_path = f.name
    try:
        cmd = [sys.executable, os.path.join(ROOT, "tools", "smoke_v7.py"),
               "--config", temp_path, "--task", "K1/Goal_Pose_HBatch",
               "--checkpoint", args.checkpoint, "--sim_device", args.sim_device,
               "--rl_device", args.rl_device, "--steps", str(args.steps)]
        rc = subprocess.call(cmd, cwd=ROOT)
        if rc:
            return rc
        video_dir = tempfile.mkdtemp(prefix="hbatch-video-smoke-")
        video_cmd = [sys.executable, os.path.join(ROOT, "eval_goal_pose.py"),
                     "--task", "K1/Goal_Pose_HBatch", "--config", temp_path,
                     "--checkpoint", args.checkpoint, "--sim_device", args.sim_device,
                     "--rl_device", args.rl_device, "--num_envs", "16",
                     "--duration_s", "10", "--keep_perturbations",
                     "--record_video", "--record_video_s", "2", "--out", video_dir]
        rc = subprocess.call(video_cmd, cwd=ROOT)
        mp4 = os.path.join(video_dir, "rollout_env0.mp4")
        report = os.path.join(video_dir, "report.json")
        if rc == 0 and os.path.exists(mp4) and os.path.getsize(mp4) > 0 and os.path.exists(report):
            data = json.load(open(report, encoding="utf-8"))
            rc = 0 if data.get("disturbance_eval", {}).get("events", 0) > 0 else 1
        else:
            rc = 1
        print("{}  perspective RGBA video + force-event smoke".format("PASS" if rc == 0 else "FAIL"))
        shutil.rmtree(video_dir, ignore_errors=True)
        return rc
    finally:
        os.unlink(temp_path)


if __name__ == "__main__":
    raise SystemExit(main())
