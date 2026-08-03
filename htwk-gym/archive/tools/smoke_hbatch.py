#!/usr/bin/env python3
"""Run isolated HBatch smoke stages before spending GPU-days on training.

The path controller, frozen warm-start policy, artificial disturbances and
video logger answer different questions.  Keep them in separate temporary
configs so a weak warm start cannot be misreported as a mechanics failure (or
vice versa), and run every viable stage before returning a combined verdict.
"""

import argparse
import copy
import glob
import os
import secrets
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

import yaml


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def require(ok, message):
    print("{}  {}".format("PASS" if ok else "FAIL", message), flush=True)
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
    require(cfg["runner"].get("load_optimizer_state") is False and
            cfg["algorithm"].get("finite_checks") is True and
            float(cfg["algorithm"].get("min_learning_rate", 0.0)) == 1.0e-6 and
            float(cfg["algorithm"].get("max_learning_rate", 0.0)) == 1.0e-5 and
            float(cfg["algorithm"].get("max_abs_log_ratio", 0.0)) == 10.0,
            "fresh H optimizer, bounded adaptive LR and finite PPO are mandatory")
    require(cfg["randomization"].get("base_com", {}).get("distribution") == "uniform" and
            cfg["randomization"].get("base_com", {}).get("range") == [-0.1, 0.1],
            "COM latent mirror assumes the frozen symmetric uniform range")
    path_cfg = cfg["commands"]["path"]
    require(path_cfg["speed_grid"]["enabled"], "G1 speed-curvature grid retained")
    require(path_cfg["speed_grid"].get("initial_active") == "all",
            "full G1 warm-start grid is restored explicitly")
    require(path_cfg.get("constraint_mode") == "radial_rate_limited",
            "2-D rate-limited floor/leash controller is mandatory")
    require(abs(float(path_cfg.get("drag_speed_cap_mps", 0.0)) - 2.1) < 1.0e-9 and
            abs(float(path_cfg.get("floor_recovery_rate_mps", 0.0)) - 2.1) < 1.0e-9 and
            abs(float(path_cfg.get("floor_recovery_grace_s", 0.0)) - 2.0) < 1.0e-9 and
            abs(float(path_cfg.get("goal_rate_max_mps", 0.0)) - 3.2) < 1.0e-9,
            "robot-drag, bounded dwell recovery and absolute goal-rate caps are frozen")
    max_floor = min(
        float(path_cfg["lookahead_m"][1]),
        float(path_cfg.get("lookahead_scale_frac", 0.6))
        / min(float(x) for x in path_cfg["speed_grid"]["curvatures"]),
    )
    require(max_floor / float(path_cfg["floor_recovery_rate_mps"]) <= 1.25,
            "stationary-robot nominal floor recovery is at most 1.25 s")
    require(float(path_cfg.get("floor_tolerance_m", 0.0)) > 0.0,
            "signed floor tolerance is nonzero")
    require(path_cfg.get("arrival_reward_mode") == "dwell_only" and
            path_cfg.get("pause_gait_during_dwell") is True,
            "arrival rewards and gait pause are restricted to dwell")
    gates = cfg["evaluation"]["hbatch_gates"]
    require(cfg["evaluation"].get("hbatch_protocol_version") ==
            "2026-07-30-codex-v3",
            "frozen HBatch campaign protocol version is explicit")
    common_eval = cfg["evaluation"].get("hbatch_common_eval") or {}
    eval_disturbance = common_eval.get("disturbance") or {}
    eval_joint = common_eval.get("randomization_overrides") or {}
    eval_noise = common_eval.get("noise_overrides") or {}
    require(eval_disturbance.get("interval_s") == [6.0, 12.0] and
            float(eval_disturbance.get("event_probability", 0.0)) == 0.5 and
            int(eval_disturbance.get("ramp_steps", 0)) == 1 and
            float(eval_disturbance.get("collision_share", -1.0)) == 0.35 and
            float(eval_disturbance.get(
                "high_speed_probability_boost", 0.0)) == 2.0,
            "shared held-out force evaluation distribution is frozen")
    require(eval_joint.get("joint_encoder_bias", {}).get("range") ==
            [-0.025, 0.025] and
            eval_joint.get("joint_target_offset", {}).get("range") ==
            [-0.02, 0.02] and
            eval_joint.get("init_dof_pos", {}).get("range") == [0.0, 0.075],
            "shared held-out joint calibration evaluation is frozen")
    require(float(eval_noise.get("goal_bt_flicker", {}).get(
                "prob_per_step", 0.0)) == 0.001,
            "shared held-out goal flicker evaluation is frozen")
    require(0.0 < float(gates.get("path_floor_below_0p75_max", 0.0)) <= 0.10 and
            0.0 < float(gates.get("path_dwell_resume_recovery_share_max", 0.0)) <= 0.15 and
            0.0 < float(gates.get("path_outside_leash_max", 0.0)) <= 0.01,
            "post-train floor, bounded-recovery and leash reject gates are frozen")
    require(abs(float(gates.get("waypoint_pos_median_max_m", 0.0)) - 0.0552) < 1e-9 and
            abs(float(gates.get("waypoint_pos_p90_max_m", 0.0)) - 0.0742) < 1e-9 and
            abs(float(gates.get(
                "waypoint_heading_median_max_deg", 0.0)) - 2.54) < 1e-9,
            "G1 waypoint median/p90/heading preservation gates are frozen")
    require(abs(float(gates.get(
                "waypoint_never_arrived_share_max", 0.0)) - 0.015) < 1e-9,
            "G1 never-arrived anti-collapse gate is frozen")
    require(float(gates.get("overall_falls_per_1000_max", 99.0)) <= 5.0 and
            float(gates.get("waypoint_falls_per_1000_max", 99.0)) <= 2.0,
            "overall and waypoint survivor-bias fall gates are frozen")
    require(float(gates.get("time_to_1mps_reached_share_min", 0.0)) >= 0.80 and
            float(gates.get("speed_recovery_within_5s_min", 0.0)) >= 0.90 and
            float(gates.get("directional_t0p5_reached_share_min", 0.0)) >= 0.80,
            "acceleration, force recovery and goal-direction response coverage gates are frozen")
    require(float(gates.get("jitter_falls_per_env_min_max", 1.0)) <= 0.50 and
            float(gates.get("combined_falls_per_env_min_max", 1.0)) <= 0.50 and
            float(gates.get("jitter_body_angvel_p90_max", 99.0)) <= 3.0 and
            float(gates.get("combined_body_angvel_p90_max", 99.0)) <= 3.0,
            "jitter and combined survival/non-divergence gates are frozen")
    require(int(gates.get("touchdown_samples_min", 0)) >= 100 and
            0.0 < float(gates.get("touchdown_lr_bias_max_m", 0.0)) <= 0.02 and
            float(gates.get("touchdown_target_share_min", 0.0)) >= 0.40 and
            float(gates.get("touchdown_overstride_share_max", 1.0)) <= 0.10,
            "H1/H3 first-contact coverage, symmetry and gait gates are frozen")
    for name in ("push_force", "push_torque", "kick_lin_vel", "kick_ang_vel"):
        require(cfg["randomization"][name]["range"] == [0.0, 0.0],
                "legacy global wrench source disabled: {}".format(name))
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
                cfg["algorithm"]["mirror_augmentation_coef"] > 0 and
                float(cfg["algorithm"].get(
                    "mirror_augmentation_max_std", 0.0)) == 5.0 and
                float(cfg["algorithm"].get(
                    "mirror_augmentation_min_valid_share", 0.0)) == 0.10,
                "H1/H2 mirror loss and support-bounded transition augmentation enabled")
    else:
        require(cfg["algorithm"].get("symmetry_coef", 0.0) == 0.0 and
                cfg["algorithm"].get("mirror_augmentation_coef", 0.0) == 0.0,
                "H0/H3 remain non-mirror controls")
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
    print("PASS  static HBatch checks: {} ({})".format(version, desc), flush=True)
    return cfg


def write_temp_config(cfg, stage):
    """Write a disposable config whose name still follows the -codex rule."""
    with tempfile.NamedTemporaryFile(
            mode="w", prefix="hbatch-{}-".format(stage),
            suffix="-codex.yaml", delete=False, encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        return f.name


def disable_observation_noise(cfg):
    """Remove observation/perceived-goal noise from an isolated mechanism run."""
    noise = cfg.setdefault("noise", {})
    for value in noise.values():
        if isinstance(value, dict) and "range" in value:
            value["range"] = [0.0, 0.0]
    noise["goal_obs_hold_steps"] = [0, 0]
    flicker = noise.get("goal_bt_flicker")
    if isinstance(flicker, dict):
        flicker.update({"prob_per_step": 0.0, "radius_m": 0.0,
                        "heading_rad": 0.0})


def zero_randomization_range(cfg, name):
    value = (cfg.get("randomization") or {}).get(name)
    if isinstance(value, dict) and "range" in value:
        value["range"] = [0.0, 0.0]


def codex_video_artifacts(directory):
    """Suffix eval's fixed temporary output names before inspecting them."""
    renamed = {}
    for name in ("rollout_env0.mp4", "report.json", "report.md", "segments.csv"):
        source = os.path.join(directory, name)
        stem, ext = os.path.splitext(name)
        target = os.path.join(directory, stem + "-codex" + ext)
        if os.path.exists(source):
            os.replace(source, target)
        renamed[name] = target
    return renamed


def dynamic_command(config, checkpoint, args):
    return [
        sys.executable, os.path.join(ROOT, "tools", "smoke_v7.py"),
        "--config", config, "--task", "K1/Goal_Pose_HBatch",
        "--checkpoint", checkpoint, "--sim_device", args.sim_device,
        "--rl_device", args.rl_device, "--steps", str(args.steps),
    ]


def call_stage(name, cmd, env=None):
    print("\n[{}]".format(name), flush=True)
    try:
        rc = subprocess.call(cmd, cwd=ROOT, env=env)
    except Exception as exc:
        print("FAIL  {} could not run: {}".format(name, exc), flush=True)
        return 1
    print("{}  {} stage".format("PASS" if rc == 0 else "FAIL", name), flush=True)
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--sim_device", default="cuda:0")
    ap.add_argument("--rl_device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--train_smoke_envs", type=int, default=4096)
    ap.add_argument("--train_smoke_iterations", type=int, default=2)
    ap.add_argument("--static_only", action="store_true")
    args = ap.parse_args()

    failures = []
    cfg = None
    print("[STATIC]", flush=True)
    try:
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as exc:
        failures.append("STATIC")
        print("FAIL  STATIC config load: {}".format(exc), flush=True)
    if cfg is not None:
        try:
            static_checks(args.config)
        except Exception as exc:
            failures.append("STATIC")
            print("FAIL  STATIC: {}".format(exc), flush=True)
    if args.static_only:
        return 1 if failures else 0
    if cfg is None:
        print("\n[PATH_MECHANICS]\nSKIP  no valid config after STATIC", flush=True)
        print("\n[TRAIN_UPDATE]\nSKIP  no valid config after STATIC", flush=True)
        print("\n[DISTURBANCE]\nSKIP  no valid config after STATIC", flush=True)
        print("\n[VIDEO]\nSKIP  no valid config after STATIC", flush=True)
        return 1

    temp_paths = []
    video_dir = None
    train_health_dir = None
    disturbance_path = None
    video_path = None
    try:
        # PATH_MECHANICS: no artificial wrench.  This stage validates path
        # update invariants and warm-start compatibility without collisions
        # changing the robot trajectory underneath the path checks.
        try:
            path_cfg = copy.deepcopy(cfg)
            path_cfg["randomization"]["disturbance"]["enabled"] = False
            # HBatch falls back to GoalPose's legacy global push when its event
            # model is disabled, so zero that fallback explicitly as well.
            for name in ("push_force", "push_torque", "kick_lin_vel", "kick_ang_vel"):
                zero_randomization_range(path_cfg, name)
            path_path = write_temp_config(path_cfg, "path-mechanics")
            temp_paths.append(path_path)
            path_rc = call_stage(
                "PATH_MECHANICS", dynamic_command(path_path, args.checkpoint, args))
        except Exception as exc:
            print("\n[PATH_MECHANICS]", flush=True)
            print("FAIL  PATH_MECHANICS setup: {}".format(exc), flush=True)
            path_rc = 1
        if path_rc:
            failures.append("PATH_MECHANICS")

        # TRAIN_UPDATE: inference smoke cannot exercise H1/H2's mirrored PPO
        # transition augmentation, critic reflection or backward pass.  Match
        # the production 4096-env PPO shape for two complete iterations: a
        # one-iteration smoke never forwards the policy after its final update
        # and therefore missed the H1/H2 NaN seen on the first server launch.
        train_tag = "smoke_train_{}_{}".format(
            os.path.basename(args.config).split("-")[0], os.getpid())
        train_root = os.path.join(
            ROOT, "logs", "K1", "K1", "Goal_Pose_HBatch")
        before_train_dirs = set(glob.glob(
            os.path.join(train_root, "*_{}".format(train_tag))))
        try:
            train_cfg = copy.deepcopy(cfg)
            train_cfg["basic"]["description"] = train_tag
            train_cfg["basic"]["max_iterations"] = args.train_smoke_iterations
            train_cfg["runner"]["use_wandb"] = False
            train_cfg["runner"]["save_interval"] = 1000000
            train_path = write_temp_config(train_cfg, "train-update")
            temp_paths.append(train_path)
            train_cmd = [
                sys.executable, os.path.join(ROOT, "train_hbatch.py"),
                "--task", "K1/Goal_Pose_HBatch", "--config", train_path,
                "--headless", "True", "--checkpoint", args.checkpoint,
                "--num_envs", str(args.train_smoke_envs),
                "--max_iterations", str(args.train_smoke_iterations),
                "--sim_device", args.sim_device,
                "--rl_device", args.rl_device,
            ]
            train_health_dir = tempfile.mkdtemp(
                prefix="hbatch-train-health-", suffix="-codex")
            train_health_path = os.path.join(
                train_health_dir, "health-codex.json")
            train_health_token = secrets.token_hex(16)
            train_env = os.environ.copy()
            train_env.update({
                "HBATCH_HEALTH_MARKER": train_health_path,
                "HBATCH_HEALTH_TOKEN": train_health_token,
                "HBATCH_HEALTH_ITERATIONS": str(args.train_smoke_iterations),
            })
            train_rc = call_stage("TRAIN_UPDATE", train_cmd, env=train_env)
            if train_rc == 0:
                verify_health_cmd = [
                    sys.executable,
                    os.path.join(ROOT, "tools", "verify_hbatch_health.py"),
                    "--marker", train_health_path,
                    "--health_token", train_health_token,
                    "--num_envs", str(args.train_smoke_envs),
                    "--min_iterations", str(args.train_smoke_iterations),
                ]
                train_rc = subprocess.call(verify_health_cmd, cwd=ROOT)
        except Exception as exc:
            print("\n[TRAIN_UPDATE]", flush=True)
            print("FAIL  TRAIN_UPDATE setup: {}".format(exc), flush=True)
            train_rc = 1
        finally:
            after_train_dirs = set(glob.glob(
                os.path.join(train_root, "*_{}".format(train_tag))))
            for directory in sorted(after_train_dirs - before_train_dirs):
                shutil.rmtree(directory, ignore_errors=True)
            if train_health_dir is not None:
                shutil.rmtree(train_health_dir, ignore_errors=True)
        if train_rc:
            failures.append("TRAIN_UPDATE")

        # DISTURBANCE: remove path mode, observation noise and joint offsets so
        # this run asks only whether both wrench classes reach exactly the
        # configured rigid bodies with the configured magnitudes. One event per env in the
        # six-second smoke gives ample deterministic-seed coverage, while the
        # 3-4 s interval remains longer than the maximum 1.5 s support event.
        try:
            disturbance_cfg = copy.deepcopy(cfg)
            disturbance_cfg["commands"]["goal_mode_mixture"] = {
                "waypoint": 1.0, "path": 0.0}
            disturbance_cfg["commands"]["path"]["speed_grid"]["enabled"] = False
            disable_observation_noise(disturbance_cfg)
            for name in (
                    "push_force", "push_torque", "kick_lin_vel", "kick_ang_vel",
                    "joint_encoder_bias", "joint_target_offset", "init_dof_pos"):
                zero_randomization_range(disturbance_cfg, name)
            disturbance_cfg["randomization"]["disturbance"].update({
                "enabled": True, "interval_s": [3.0, 4.0],
                "event_probability": 1.0, "ramp_steps": 1,
            })
            disturbance_path = write_temp_config(disturbance_cfg, "disturbance")
            temp_paths.append(disturbance_path)
            # Use the longer, low-force support class for the visual check so
            # env0 cannot terminate on a collision before a red-arrow frame is
            # captured.  Force every env into path mode so this same artifact
            # also proves the simulator-view path/carrot overlay is live. Both
            # wrench classes remain mandatory in DISTURBANCE above.
            video_cfg = copy.deepcopy(disturbance_cfg)
            video_cfg["commands"]["goal_mode_mixture"] = {
                "waypoint": 0.0, "path": 1.0}
            video_cfg["randomization"]["disturbance"]["collision_share"] = 0.0
            video_path = write_temp_config(video_cfg, "video")
            temp_paths.append(video_path)
            disturbance_rc = call_stage(
                "DISTURBANCE",
                dynamic_command(disturbance_path, args.checkpoint, args))
        except Exception as exc:
            print("\n[DISTURBANCE]", flush=True)
            print("FAIL  DISTURBANCE setup: {}".format(exc), flush=True)
            disturbance_rc = 1
        if disturbance_rc:
            failures.append("DISTURBANCE")

        # VIDEO: six recorded seconds include env0's guaranteed first event at
        # 3-4 s.  A report-wide event from another env is insufficient: require
        # at least one recorded env0 frame whose force-arrow input was active.
        print("\n[VIDEO]", flush=True)
        if video_path is None:
            print("SKIP  VIDEO has no valid disturbance config", flush=True)
            failures.append("VIDEO")
            video_rc = 1
        else:
            video_rc = 0
        if video_rc == 0:
            video_dir = tempfile.mkdtemp(prefix="hbatch-video-smoke-", suffix="-codex")
            video_token = secrets.token_hex(16)
            video_cmd = [sys.executable, os.path.join(ROOT, "eval_goal_pose.py"),
                         "--task", "K1/Goal_Pose_HBatch", "--config", video_path,
                         "--checkpoint", args.checkpoint, "--sim_device", args.sim_device,
                         "--rl_device", args.rl_device, "--num_envs", "16",
                         "--duration_s", "10", "--keep_perturbations",
                         "--record_video", "--record_video_s", "6",
                         "--force_visualization_probe", "--out", video_dir,
                         "--completion_token", video_token]
            try:
                eval_rc = subprocess.call(video_cmd, cwd=ROOT)
            except Exception as exc:
                print("FAIL  VIDEO could not run: {}".format(exc), flush=True)
                eval_rc = 1
            verify_cmd = [
                sys.executable,
                os.path.join(ROOT, "tools", "verify_hbatch_video.py"),
                "--directory", video_dir,
                "--completion_token", video_token,
            ]
            try:
                video_rc = subprocess.call(verify_cmd, cwd=ROOT)
                artifacts = codex_video_artifacts(video_dir)
                renamed_ok = all(os.path.isfile(path) and os.path.getsize(path) > 0
                                 for path in artifacts.values())
                if not renamed_ok:
                    video_rc = 1
                detail = ("eval rc {}; completion marker, hashes, full MP4 decode, "
                          "renderer counters and -codex artifact rename {}"
                          .format(eval_rc, "verified" if video_rc == 0 else "failed"))
            except Exception as exc:
                video_rc = 1
                detail = "artifact/report inspection failed: {}".format(exc)
            print("{}  perspective RGBA video + visible force event ({})".format(
                "PASS" if video_rc == 0 else "FAIL", detail), flush=True)
            if video_rc:
                failures.append("VIDEO")
    finally:
        for path in temp_paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        if video_dir is not None:
            shutil.rmtree(video_dir, ignore_errors=True)

    print("\n[SUMMARY]", flush=True)
    if failures:
        print("FAIL  {}".format(", ".join(failures)), flush=True)
        return 1
    print("PASS  STATIC, PATH_MECHANICS, TRAIN_UPDATE, DISTURBANCE, VIDEO", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
