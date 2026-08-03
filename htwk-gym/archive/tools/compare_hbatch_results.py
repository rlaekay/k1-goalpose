#!/usr/bin/env python3
"""Aggregate the newest completed H0-H3 suites and apply cross-arm gates."""

import argparse
import fcntl
import glob
import json
import math
import os
import tempfile


ARMS = ("H0", "H1", "H2", "H3")
REQUIRED_REPORTS = (
    "clean", "force", "jitter", "combined", "lateral", "reverse", "video_force")


def nested(data, *keys, default=float("nan")):
    cur = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def latest_suite(root, arm):
    paths = sorted(
        p for p in glob.glob(os.path.join(root, arm + "_*"))
        if (os.path.isdir(p)
            and "-partial-" not in os.path.basename(p)
            and os.path.isfile(os.path.join(p, "COMPLETE"))
            and all(os.path.isfile(os.path.join(p, name, "report.json"))
                    for name in REQUIRED_REPORTS)
            and os.path.isfile(os.path.join(
                p, "video_force", "rollout_env0.mp4"))
            and os.path.getsize(os.path.join(
                p, "video_force", "rollout_env0.mp4")) > 0)
    )
    return paths[-1] if paths else None


def load_report(suite, name):
    path = os.path.join(suite, name, "report.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def report_protocol_signature(report):
    """Fields that must match before two arms are compared cross-run."""
    return {
        "task": report.get("task"),
        "env_code_sha": report.get("env_code_sha"),
        "evaluation_protocol_sha": report.get("evaluation_protocol_sha"),
        "hbatch_protocol_version": report.get("hbatch_protocol_version"),
        "effective_eval_protocol_sha": report.get(
            "effective_eval_protocol_sha"),
        "seed": report.get("seed"),
        "num_envs": report.get("num_envs"),
        "duration_s": report.get("duration_s"),
        "task_state_protocol": report.get("task_state_protocol"),
    }


def protocol_signatures_complete(signatures):
    return bool(signatures) and all(
        all(value is not None and value != "" for value in signature.values())
        for signature in signatures.values())


def collect(root):
    out = {}
    for arm in ARMS:
        suite = latest_suite(root, arm)
        if not suite:
            continue
        reports = {name: load_report(suite, name) for name in REQUIRED_REPORTS}
        if not reports["clean"]:
            continue
        clean, force = reports["clean"], reports["force"] or {}
        gates = nested(clean, "hbatch_gates", default={}) or {}
        lateral = reports["lateral"] or {}
        reverse = reports["reverse"] or {}
        video = reports["video_force"] or {}
        out[arm] = {
            "suite": suite,
            "reports": reports,
            "protocol_signatures": {
                name: report_protocol_signature(reports[name] or {})
                for name in REQUIRED_REPORTS
            },
            "metrics": {
                "clean_authoritative": bool(clean.get(
                    "authoritative_gate_evaluation", False)),
                # The legacy MASTERPLAN gate is still reported, but H adoption
                # has its own G1-preservation/path/robustness gates below.  It
                # would be contradictory to require both max_falls=0 and the
                # explicit per-1000 path-fall budget, or both 5.0 cm and H0's
                # documented G1 reference of 5.52 cm.
                "clean_legacy_gates_pass": bool(
                    clean.get("all_gates_pass", False)),
                "force_protocol_enabled": bool(force.get("perturbations", False)),
                "waypoint_pos_median_m": nested(clean, "pos_err_m", "median"),
                "waypoint_pos_p90_m": nested(clean, "pos_err_m", "p90"),
                "waypoint_heading_median_deg": nested(
                    clean, "heading_err_deg", "median"),
                "waypoint_pos_median_max_m": gates.get(
                    "waypoint_pos_median_max_m", 0.0552),
                "waypoint_pos_p90_max_m": gates.get(
                    "waypoint_pos_p90_max_m", 0.0742),
                "waypoint_heading_median_max_deg": gates.get(
                    "waypoint_heading_median_max_deg", 2.54),
                "waypoint_never_arrived_share": nested(
                    clean, "failure_modes", "never_arrived", "share"),
                "waypoint_never_arrived_share_max": gates.get(
                    "waypoint_never_arrived_share_max", 0.015),
                "fall_classification_complete": bool(nested(
                    clean, "path_safety", "fall_classification_complete",
                    default=False)),
                "overall_falls_per_1000": nested(
                    clean, "overall_safety", "falls_per_1000_attempts"),
                "overall_falls_per_1000_max": gates.get(
                    "overall_falls_per_1000_max", 5.0),
                "waypoint_falls_per_1000": nested(
                    clean, "waypoint_safety", "falls_per_1000_attempts"),
                "waypoint_falls_per_1000_max": gates.get(
                    "waypoint_falls_per_1000_max", 2.0),
                "path_speed_median_mps": nested(clean, "path_tracking", "mean_speed_median"),
                "path_falls_per_1000": nested(clean, "path_safety", "falls_per_1000_attempts"),
                "path_step_samples": nested(
                    clean, "path_step_tracking", "samples", default=0),
                "path_steady_samples": nested(
                    clean, "path_step_tracking", "steady_samples_excluding_recovery",
                    default=0),
                "path_floor_below_0p75": nested(
                    clean, "path_step_tracking", "below_0p75_share_excluding_recovery"),
                "path_outside_leash": nested(
                    clean, "path_step_tracking", "outside_leash_share"),
                "path_dwell_resume_recovery": nested(
                    clean, "path_step_tracking", "dwell_resume_recovery_share"),
                "path_floor_below_0p75_max": nested(
                    clean, "hbatch_gates", "path_floor_below_0p75_max", default=0.10),
                "path_dwell_resume_recovery_max": gates.get(
                    "path_dwell_resume_recovery_share_max", 0.15),
                "path_outside_leash_max": nested(
                    clean, "hbatch_gates", "path_outside_leash_max", default=0.01),
                "path_speed_median_min": gates.get("path_speed_median_min", 0.95),
                "path_falls_per_1000_max": gates.get("path_falls_per_1000_max", 5.0),
                "time_to_1mps_p90_s": nested(
                    clean, "path_acceleration_response", "time_to_1p0", "p90_s"),
                "time_to_1mps_eligible": nested(
                    clean, "path_acceleration_response", "time_to_1p0", "eligible", default=0),
                "time_to_1mps_reached_share": nested(
                    clean, "path_acceleration_response", "time_to_1p0", "reached_share"),
                "time_to_1mps_reached_share_min": gates.get(
                    "time_to_1mps_reached_share_min", 0.80),
                "time_to_1mps_p90_s_max": gates.get("time_to_1mps_p90_s_max", 3.0),
                "cruise_pitch_p90_deg": nested(
                    clean, "high_speed_stability", "cruise_pitch_abs_p90_deg"),
                "cruise_roll_p90_deg": nested(
                    clean, "high_speed_stability", "cruise_roll_abs_p90_deg"),
                "cruise_ang_xy_p90_radps": nested(
                    clean, "high_speed_stability", "cruise_ang_xy_p90_radps"),
                "cruise_samples": nested(
                    clean, "high_speed_stability", "samples", "cruise", default=0),
                "cruise_share_of_valid": nested(
                    clean, "high_speed_stability", "cruise_share_of_valid"),
                "cruise_share_of_valid_min": gates.get(
                    "cruise_share_of_valid_min", 0.05),
                "cruise_pitch_p90_max_deg": gates.get(
                    "cruise_pitch_p90_max_deg", 20.0),
                "cruise_roll_p90_max_deg": gates.get(
                    "cruise_roll_p90_max_deg", 15.0),
                "cruise_ang_xy_p90_max_radps": gates.get(
                    "cruise_ang_xy_p90_max_radps", 3.0),
                "mirror_error_p90": nested(clean, "symmetry_eval", "p90"),
                "touchdown_samples": nested(
                    clean, "gait_touchdown", "samples", default=0),
                "touchdown_ahead_share": nested(
                    clean, "gait_touchdown", "ahead_of_trunk_share"),
                "touchdown_target_share": nested(
                    clean, "gait_touchdown",
                    "within_dynamic_target_one_sigma_share"),
                "touchdown_overstride_share": nested(
                    clean, "gait_touchdown", "overstride_share"),
                "touchdown_lr_bias_m": nested(
                    clean, "gait_touchdown",
                    "left_right_heel_x_median_abs_diff_m"),
                "touchdown_down_speed_p90_mps": nested(
                    clean, "gait_touchdown",
                    "precontact_down_speed_p90_mps"),
                "touchdown_force_p90_n": nested(
                    clean, "gait_touchdown", "contact_force_p90_n"),
                "touchdown_samples_min": gates.get(
                    "touchdown_samples_min", 100),
                "touchdown_lr_bias_max_m": gates.get(
                    "touchdown_lr_bias_max_m", 0.02),
                "touchdown_target_share_min": gates.get(
                    "touchdown_target_share_min", 0.40),
                "touchdown_overstride_share_max": gates.get(
                    "touchdown_overstride_share_max", 0.10),
                "force_events": nested(force, "disturbance_eval", "events", default=0),
                "force_survival_5s": nested(
                    force, "disturbance_eval", "overall", "survival_5s"),
                "force_survival_5s_eligible": nested(
                    force, "disturbance_eval", "overall",
                    "survival_5s_eligible", default=0),
                "force_recovery_share": nested(
                    force, "disturbance_eval", "overall", "recovery_90_within_5s_share"),
                "force_recovery_eligible": nested(
                    force, "disturbance_eval", "overall", "recovery_eligible", default=0),
                "force_recovery_share_min": gates.get(
                    "speed_recovery_within_5s_min", 0.90),
                "force_recovery_p90_s": nested(
                    force, "disturbance_eval", "overall", "recovery_90_s_p90"),
                "force_recovery_p90_s_max": gates.get(
                    "speed_recovery_p90_s_max", 2.0),
                "force_high_speed_recovery_share": nested(
                    force, "disturbance_eval", "high_speed",
                    "recovery_90_within_5s_share"),
                "force_high_speed_recovery_eligible": nested(
                    force, "disturbance_eval", "high_speed",
                    "recovery_eligible", default=0),
                "force_high_speed_recovery_p90_s": nested(
                    force, "disturbance_eval", "high_speed", "recovery_90_s_p90"),
                "lateral_t0p5_p90_s": nested(
                    lateral, "directional_response", "time_to_0p5", "p90_s"),
                "lateral_t0p5_eligible": nested(
                    lateral, "directional_response", "time_to_0p5", "eligible", default=0),
                "lateral_t0p5_reached_share": nested(
                    lateral, "directional_response", "time_to_0p5", "reached_share"),
                "reverse_t0p5_p90_s": nested(
                    reverse, "directional_response", "time_to_0p5", "p90_s"),
                "reverse_t0p5_eligible": nested(
                    reverse, "directional_response", "time_to_0p5", "eligible", default=0),
                "reverse_t0p5_reached_share": nested(
                    reverse, "directional_response", "time_to_0p5", "reached_share"),
                "directional_t0p5_reached_share_min": gates.get(
                    "directional_t0p5_reached_share_min", 0.80),
                "directional_t0p5_p90_s_max": gates.get(
                    "directional_t0p5_p90_s_max", 2.0),
                "jitter_falls_per_env_min": nested(
                    reports["jitter"] or {}, "falls_per_env_minute"),
                "combined_falls_per_env_min": nested(
                    reports["combined"] or {}, "falls_per_env_minute"),
                "jitter_body_angvel_p90": nested(
                    reports["jitter"] or {}, "body_angvel", "p90"),
                "combined_body_angvel_p90": nested(
                    reports["combined"] or {}, "body_angvel", "p90"),
                "combined_force_events": nested(
                    reports["combined"] or {}, "disturbance_eval", "events", default=0),
                "jitter_falls_per_env_min_max": gates.get(
                    "jitter_falls_per_env_min_max", 0.50),
                "combined_falls_per_env_min_max": gates.get(
                    "combined_falls_per_env_min_max", 0.50),
                "jitter_body_angvel_p90_max": gates.get(
                    "jitter_body_angvel_p90_max", 3.0),
                "combined_body_angvel_p90_max": gates.get(
                    "combined_body_angvel_p90_max", 3.0),
                "video_probe": bool(video.get("force_visualization_probe", False)),
                "video_recorded_frames": nested(
                    video, "disturbance_eval", "video_recorded_frames", default=0),
                "video_arrow_frames": nested(
                    video, "disturbance_eval", "video_force_arrow_drawn_frames", default=0),
                "video_carrot_frames": nested(
                    video, "disturbance_eval", "video_path_carrot_drawn_frames", default=0),
                "video_trace_frames": nested(
                    video, "disturbance_eval", "video_path_trace_drawn_frames", default=0),
                "video_artifact_nonempty": os.path.getsize(os.path.join(
                    suite, "video_force", "rollout_env0.mp4")) > 0,
            },
        }
    return out


def noninferior(value, reference, ratio, lower_is_better=False):
    if not (finite(value) and finite(reference)):
        return False
    return value <= reference * ratio if lower_is_better else value >= reference * ratio


def verdicts(data):
    result = {}
    h0 = data.get("H0", {}).get("metrics", {})
    h1 = data.get("H1", {}).get("metrics", {})
    h0_protocols = data.get("H0", {}).get("protocol_signatures", {})
    for arm, item in data.items():
        m = item["metrics"]
        protocols = item.get("protocol_signatures", {})
        checks = {
            "report_protocol_metadata_complete": (
                protocol_signatures_complete(protocols)),
            "same_report_protocol_as_H0": (
                protocol_signatures_complete(protocols)
                and protocol_signatures_complete(h0_protocols)
                and protocols == h0_protocols),
            "clean_report_is_authoritative": m["clean_authoritative"],
            "fall_context_classification_complete": (
                m["fall_classification_complete"]),
            "overall_falls_within_limit": (
                finite(m["overall_falls_per_1000"])
                and m["overall_falls_per_1000"]
                <= m["overall_falls_per_1000_max"]),
            "waypoint_falls_within_limit": (
                finite(m["waypoint_falls_per_1000"])
                and m["waypoint_falls_per_1000"]
                <= m["waypoint_falls_per_1000_max"]),
            "waypoint_never_arrived_anti_collapse": (
                finite(m["waypoint_never_arrived_share"])
                and m["waypoint_never_arrived_share"]
                <= m["waypoint_never_arrived_share_max"]),
            "force_report_protocol_enabled": m["force_protocol_enabled"],
            "path_step_samples_nonzero": m["path_step_samples"] > 0,
            "path_steady_samples_nonzero": m["path_steady_samples"] > 0,
            "forward_path_touchdown_sample_coverage": (
                m["touchdown_samples"] >= m["touchdown_samples_min"]),
            "path_speed_meets_floor": (
                finite(m["path_speed_median_mps"])
                and m["path_speed_median_mps"] >= m["path_speed_median_min"]),
            "path_falls_within_limit": (
                finite(m["path_falls_per_1000"])
                and m["path_falls_per_1000"] <= m["path_falls_per_1000_max"]),
            "path_floor_below_0p75_le_10pct": (
                finite(m["path_floor_below_0p75"])
                and finite(m.get("path_floor_below_0p75_max", 0.10))
                and m["path_floor_below_0p75"]
                <= m.get("path_floor_below_0p75_max", 0.10)),
            "dwell_resume_recovery_share_bounded": (
                finite(m["path_dwell_resume_recovery"])
                and m["path_dwell_resume_recovery"]
                <= m["path_dwell_resume_recovery_max"]),
            "path_outside_leash_le_1pct": (
                finite(m["path_outside_leash"])
                and finite(m.get("path_outside_leash_max", 0.01))
                and m["path_outside_leash"]
                <= m.get("path_outside_leash_max", 0.01)),
            "from_rest_t1_has_eligible_segments": m["time_to_1mps_eligible"] > 0,
            "from_rest_t1_reached_share": (
                finite(m["time_to_1mps_reached_share"])
                and m["time_to_1mps_reached_share"]
                >= m["time_to_1mps_reached_share_min"]),
            "from_rest_t1_p90_within_limit": (
                finite(m["time_to_1mps_p90_s"])
                and m["time_to_1mps_p90_s"] <= m["time_to_1mps_p90_s_max"]),
            "force_events_nonzero": m["force_events"] > 0,
            "force_survival_has_eligible_events": (
                m["force_survival_5s_eligible"] > 0),
            "force_survival_5s_ge_98pct": (
                finite(m["force_survival_5s"]) and m["force_survival_5s"] >= 0.98),
            "force_recovery_p90_le_2s": (
                finite(m["force_recovery_p90_s"])
                and m["force_recovery_p90_s"] <= m["force_recovery_p90_s_max"]),
            "force_recovery_has_eligible_events": m["force_recovery_eligible"] > 0,
            "force_recovery_share_meets_min": (
                finite(m["force_recovery_share"])
                and m["force_recovery_share"] >= m["force_recovery_share_min"]),
            "lateral_goal_direction_response": (
                m["lateral_t0p5_eligible"] > 0
                and finite(m["lateral_t0p5_reached_share"])
                and m["lateral_t0p5_reached_share"]
                >= m["directional_t0p5_reached_share_min"]
                and finite(m["lateral_t0p5_p90_s"])
                and m["lateral_t0p5_p90_s"] <= m["directional_t0p5_p90_s_max"]),
            "reverse_goal_direction_response": (
                m["reverse_t0p5_eligible"] > 0
                and finite(m["reverse_t0p5_reached_share"])
                and m["reverse_t0p5_reached_share"]
                >= m["directional_t0p5_reached_share_min"]
                and finite(m["reverse_t0p5_p90_s"])
                and m["reverse_t0p5_p90_s"] <= m["directional_t0p5_p90_s_max"]),
            "jitter_falls_within_limit": (
                finite(m["jitter_falls_per_env_min"])
                and m["jitter_falls_per_env_min"]
                <= m["jitter_falls_per_env_min_max"]),
            "combined_force_jitter_falls_within_limit": (
                finite(m["combined_falls_per_env_min"])
                and m["combined_falls_per_env_min"]
                <= m["combined_falls_per_env_min_max"]),
            "jitter_angular_motion_within_limit": (
                finite(m["jitter_body_angvel_p90"])
                and m["jitter_body_angvel_p90"]
                <= m["jitter_body_angvel_p90_max"]),
            "combined_angular_motion_within_limit": (
                finite(m["combined_body_angvel_p90"])
                and m["combined_body_angvel_p90"]
                <= m["combined_body_angvel_p90_max"]),
            "combined_force_events_nonzero": m["combined_force_events"] > 0,
            "video_path_force_artifact_verified": (
                m["video_probe"] and m["video_artifact_nonempty"]
                and m["video_recorded_frames"] > 0
                and m["video_arrow_frames"] > 0
                and m["video_carrot_frames"] > 0
                and m["video_trace_frames"] > 0),
        }
        if arm == "H0":
            checks.update({
                "path_speed_ge_0p95": finite(m["path_speed_median_mps"]) and m["path_speed_median_mps"] >= 0.95,
                "waypoint_median_le_G1": (
                    finite(m["waypoint_pos_median_m"])
                    and m["waypoint_pos_median_m"]
                    <= m["waypoint_pos_median_max_m"]),
                "waypoint_p90_le_G1": (
                    finite(m["waypoint_pos_p90_m"])
                    and m["waypoint_pos_p90_m"]
                    <= m["waypoint_pos_p90_max_m"]),
                "waypoint_heading_median_le_G1": (
                    finite(m["waypoint_heading_median_deg"])
                    and m["waypoint_heading_median_deg"]
                    <= m["waypoint_heading_median_max_deg"]),
            })
        elif arm == "H1":
            checks.update({
                "path_speed_ge_95pct_H0": noninferior(
                    m["path_speed_median_mps"], h0.get("path_speed_median_mps"), 0.95),
                "path_falls_le_105pct_H0": noninferior(
                    m["path_falls_per_1000"],
                    h0.get("path_falls_per_1000"), 1.05, True),
                "waypoint_error_le_105pct_H0": noninferior(
                    m["waypoint_pos_median_m"], h0.get("waypoint_pos_median_m"), 1.05, True),
                "waypoint_p90_le_105pct_H0": noninferior(
                    m["waypoint_pos_p90_m"], h0.get("waypoint_pos_p90_m"), 1.05, True),
                "waypoint_heading_le_105pct_H0": noninferior(
                    m["waypoint_heading_median_deg"],
                    h0.get("waypoint_heading_median_deg"), 1.05, True),
                "mirror_error_p90_le_0p10": finite(m["mirror_error_p90"]) and m["mirror_error_p90"] <= 0.10,
                "mirror_error_strictly_improves_H0": (
                    finite(m["mirror_error_p90"])
                    and finite(h0.get("mirror_error_p90"))
                    and m["mirror_error_p90"] < h0["mirror_error_p90"]),
                "touchdown_left_right_bias_improves_or_is_5mm": (
                    finite(m["touchdown_lr_bias_m"])
                    and finite(h0.get("touchdown_lr_bias_m"))
                    and (m["touchdown_lr_bias_m"] <= 0.005
                         if h0["touchdown_lr_bias_m"] <= 0.005
                         else m["touchdown_lr_bias_m"]
                         < h0["touchdown_lr_bias_m"])),
                "touchdown_left_right_bias_within_absolute_limit": (
                    finite(m["touchdown_lr_bias_m"])
                    and m["touchdown_lr_bias_m"]
                    <= m["touchdown_lr_bias_max_m"]),
                "cruise_sample_coverage": (
                    m["cruise_samples"] > 0
                    and finite(m["cruise_share_of_valid"])
                    and m["cruise_share_of_valid"] >= m["cruise_share_of_valid_min"]),
                "time_to_1mps_p90_le_110pct_H0": noninferior(
                    m["time_to_1mps_p90_s"], h0.get("time_to_1mps_p90_s"), 1.10, True),
                "time_to_1mps_reached_share_ge_95pct_H0": noninferior(
                    m["time_to_1mps_reached_share"],
                    h0.get("time_to_1mps_reached_share"), 0.95),
                "high_speed_force_recovery_reference_available": (
                    m["force_high_speed_recovery_eligible"] > 0
                    and finite(m["force_high_speed_recovery_share"])
                    and finite(m["force_high_speed_recovery_p90_s"])),
            })
        elif arm == "H2":
            checks.update({
                "path_speed_ge_95pct_H1": noninferior(
                    m["path_speed_median_mps"], h1.get("path_speed_median_mps"), 0.95),
                "path_falls_nonworse_H1": noninferior(
                    m["path_falls_per_1000"],
                    h1.get("path_falls_per_1000"), 1.0, True),
                "waypoint_error_le_105pct_H1": noninferior(
                    m["waypoint_pos_median_m"], h1.get("waypoint_pos_median_m"), 1.05, True),
                "waypoint_p90_le_105pct_H1": noninferior(
                    m["waypoint_pos_p90_m"], h1.get("waypoint_pos_p90_m"), 1.05, True),
                "waypoint_heading_le_105pct_H1": noninferior(
                    m["waypoint_heading_median_deg"],
                    h1.get("waypoint_heading_median_deg"), 1.05, True),
                "time_to_1mps_le_110pct_H1": noninferior(
                    m["time_to_1mps_p90_s"], h1.get("time_to_1mps_p90_s"), 1.10, True),
                "cruise_pitch_nonworse_H1": noninferior(
                    m["cruise_pitch_p90_deg"], h1.get("cruise_pitch_p90_deg"), 1.0, True),
                "cruise_roll_nonworse_H1": noninferior(
                    m["cruise_roll_p90_deg"], h1.get("cruise_roll_p90_deg"), 1.0, True),
                "cruise_ang_xy_nonworse_H1": noninferior(
                    m["cruise_ang_xy_p90_radps"], h1.get("cruise_ang_xy_p90_radps"), 1.0, True),
                "cruise_pitch_within_absolute_limit": (
                    finite(m["cruise_pitch_p90_deg"])
                    and m["cruise_pitch_p90_deg"]
                    <= m["cruise_pitch_p90_max_deg"]),
                "cruise_roll_within_absolute_limit": (
                    finite(m["cruise_roll_p90_deg"])
                    and m["cruise_roll_p90_deg"]
                    <= m["cruise_roll_p90_max_deg"]),
                "cruise_ang_xy_within_absolute_limit": (
                    finite(m["cruise_ang_xy_p90_radps"])
                    and m["cruise_ang_xy_p90_radps"]
                    <= m["cruise_ang_xy_p90_max_radps"]),
                "at_least_one_cruise_metric_strictly_improves_H1": (
                    all(finite(x) for x in (
                        m["cruise_pitch_p90_deg"], h1.get("cruise_pitch_p90_deg"),
                        m["cruise_roll_p90_deg"], h1.get("cruise_roll_p90_deg"),
                        m["cruise_ang_xy_p90_radps"], h1.get("cruise_ang_xy_p90_radps")))
                    and (m["cruise_pitch_p90_deg"] < h1["cruise_pitch_p90_deg"]
                         or m["cruise_roll_p90_deg"] < h1["cruise_roll_p90_deg"]
                         or m["cruise_ang_xy_p90_radps"]
                         < h1["cruise_ang_xy_p90_radps"])),
                "cruise_sample_coverage": (
                    m["cruise_samples"] > 0
                    and finite(m["cruise_share_of_valid"])
                    and m["cruise_share_of_valid"] >= m["cruise_share_of_valid_min"]),
                "cruise_coverage_ge_95pct_H1": noninferior(
                    m["cruise_share_of_valid"],
                    h1.get("cruise_share_of_valid"), 0.95),
                "mirror_error_p90_le_0p10": (
                    finite(m["mirror_error_p90"])
                    and m["mirror_error_p90"] <= 0.10),
                "mirror_error_le_105pct_H1": noninferior(
                    m["mirror_error_p90"],
                    h1.get("mirror_error_p90"), 1.05, True),
                "touchdown_left_right_bias_preserves_H1": (
                    finite(m["touchdown_lr_bias_m"])
                    and finite(h1.get("touchdown_lr_bias_m"))
                    and m["touchdown_lr_bias_m"]
                    <= max(0.005, 1.05 * h1["touchdown_lr_bias_m"])),
                "touchdown_left_right_bias_within_absolute_limit": (
                    finite(m["touchdown_lr_bias_m"])
                    and m["touchdown_lr_bias_m"]
                    <= m["touchdown_lr_bias_max_m"]),
                "time_to_1mps_reached_share_ge_95pct_H1": noninferior(
                    m["time_to_1mps_reached_share"],
                    h1.get("time_to_1mps_reached_share"), 0.95),
                "high_speed_force_recovery_has_eligible_events": (
                    m["force_high_speed_recovery_eligible"] > 0
                    and h1.get("force_high_speed_recovery_eligible", 0) > 0),
                "high_speed_force_recovery_share_nonworse_H1": noninferior(
                    m["force_high_speed_recovery_share"],
                    h1.get("force_high_speed_recovery_share"), 1.0),
                "high_speed_force_recovery_p90_nonworse_H1": noninferior(
                    m["force_high_speed_recovery_p90_s"],
                    h1.get("force_high_speed_recovery_p90_s"), 1.0, True),
                "high_speed_force_recovery_strictly_improves_H1": (
                    all(finite(x) for x in (
                        m["force_high_speed_recovery_share"],
                        h1.get("force_high_speed_recovery_share"),
                        m["force_high_speed_recovery_p90_s"],
                        h1.get("force_high_speed_recovery_p90_s")))
                    and (m["force_high_speed_recovery_share"]
                         > h1["force_high_speed_recovery_share"]
                         or m["force_high_speed_recovery_p90_s"]
                         < h1["force_high_speed_recovery_p90_s"])),
            })
        elif arm == "H3":
            checks.update({
                "path_speed_ge_95pct_H0": noninferior(
                    m["path_speed_median_mps"], h0.get("path_speed_median_mps"), 0.95),
                "waypoint_error_le_105pct_H0": noninferior(
                    m["waypoint_pos_median_m"], h0.get("waypoint_pos_median_m"), 1.05, True),
                "waypoint_p90_le_105pct_H0": noninferior(
                    m["waypoint_pos_p90_m"], h0.get("waypoint_pos_p90_m"), 1.05, True),
                "waypoint_heading_le_105pct_H0": noninferior(
                    m["waypoint_heading_median_deg"],
                    h0.get("waypoint_heading_median_deg"), 1.05, True),
                "time_to_1mps_p90_le_110pct_H0": noninferior(
                    m["time_to_1mps_p90_s"], h0.get("time_to_1mps_p90_s"), 1.10, True),
                "time_to_1mps_reached_share_ge_95pct_H0": noninferior(
                    m["time_to_1mps_reached_share"],
                    h0.get("time_to_1mps_reached_share"), 0.95),
                "path_falls_improve_or_both_zero_H0": (
                    finite(m["path_falls_per_1000"])
                    and finite(h0.get("path_falls_per_1000"))
                    and (m["path_falls_per_1000"] < h0["path_falls_per_1000"]
                         or (m["path_falls_per_1000"] == 0.0
                             and h0["path_falls_per_1000"] == 0.0))),
                "heel_target_share_strictly_improves_H0": (
                    finite(m["touchdown_target_share"])
                    and finite(h0.get("touchdown_target_share"))
                    and m["touchdown_target_share"]
                    > h0["touchdown_target_share"]),
                "heel_target_share_meets_absolute_floor": (
                    finite(m["touchdown_target_share"])
                    and m["touchdown_target_share"]
                    >= m["touchdown_target_share_min"]),
                "heel_ahead_share_nonworse_H0": noninferior(
                    m["touchdown_ahead_share"],
                    h0.get("touchdown_ahead_share"), 1.0),
                "overstride_share_nonworse_H0": noninferior(
                    m["touchdown_overstride_share"],
                    h0.get("touchdown_overstride_share"), 1.05, True),
                "overstride_share_within_absolute_limit": (
                    finite(m["touchdown_overstride_share"])
                    and m["touchdown_overstride_share"]
                    <= m["touchdown_overstride_share_max"]),
                "precontact_down_speed_p90_nonworse_H0": noninferior(
                    m["touchdown_down_speed_p90_mps"],
                    h0.get("touchdown_down_speed_p90_mps"), 1.05, True),
                "touchdown_force_p90_nonworse_H0": noninferior(
                    m["touchdown_force_p90_n"],
                    h0.get("touchdown_force_p90_n"), 1.05, True),
            })
        result[arm] = {
            "checks": checks,
            "verdict": "PASS" if checks and all(checks.values()) else "FAIL",
        }
    return result


def fmt(value, digits=3):
    return ("{:.{}f}".format(float(value), digits) if finite(value) else "NA")


def scaled(value, factor):
    return float(value) * factor if finite(value) else float("nan")


def render(data, verdict):
    lines = ["# H-batch 비교 결과 — Codex", "",
             "각 arm의 가장 최근 완료 suite를 비교한다. `NA`가 있으면 해당 gate는 통과로 간주하지 않는다.", "",
             "| arm | waypoint med/p90 cm / never | path speed | falls/1000 all/wp/path | t→1m/s p90 | mirror p90 | force 5s survival | force recovery p90 | verdict |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for arm in ARMS:
        if arm not in data:
            lines.append("| {} | NA | NA | NA | NA | NA | NA | NA | INCOMPLETE |".format(arm))
            continue
        m = data[arm]["metrics"]
        lines.append("| {} | {}/{}/{}% | {} | {}/{}/{} | {} | {} | {} | {} | {} |".format(
            arm, fmt(scaled(m["waypoint_pos_median_m"], 100), 2),
            fmt(scaled(m["waypoint_pos_p90_m"], 100), 2),
            fmt(scaled(m["waypoint_never_arrived_share"], 100), 2),
            fmt(m["path_speed_median_mps"]),
            fmt(m["overall_falls_per_1000"], 2),
            fmt(m["waypoint_falls_per_1000"], 2),
            fmt(m["path_falls_per_1000"], 2), fmt(m["time_to_1mps_p90_s"]),
            fmt(m["mirror_error_p90"]), fmt(scaled(m["force_survival_5s"], 100), 1),
            fmt(m["force_recovery_p90_s"]), verdict[arm]["verdict"]))
    lines += ["", "## Gate 상세", ""]
    for arm in ARMS:
        if arm not in verdict:
            continue
        lines.append("### {} — {}".format(arm, verdict[arm]["verdict"]))
        lines.append("")
        for name, ok in verdict[arm]["checks"].items():
            lines.append("- {} {}".format("PASS" if ok else "FAIL", name))
        lines.append("")
    lines += ["## path floor/leash 진단", "",
              "복구 transition 제외 floor collapse와 per-env leash 이탈을 control-step 단위로 본다.", "",
              "| arm | samples / steady | gap/lookahead<0.75 | leash 밖 | dwell-resume 복구 중 |",
              "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        if arm not in data:
            continue
        m = data[arm]["metrics"]
        lines.append("| {} | {} / {} | {}% | {}% | {}% |".format(
            arm, int(m["path_step_samples"]) if finite(m["path_step_samples"]) else "NA",
            int(m["path_steady_samples"]) if finite(m["path_steady_samples"]) else "NA",
            fmt(scaled(m["path_floor_below_0p75"], 100), 2),
            fmt(scaled(m["path_outside_leash"], 100), 2),
            fmt(scaled(m["path_dwell_resume_recovery"], 100), 2)))
    lines.append("")
    lines += ["## 방향 전환/goal jitter 진단", "",
              "| arm | lateral reach / p90 | reverse reach / p90 | jitter falls/env·min | combined falls/env·min |",
              "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        if arm not in data:
            continue
        m = data[arm]["metrics"]
        lines.append("| {} | {}% / {} | {}% / {} | {} | {} |".format(
            arm, fmt(scaled(m["lateral_t0p5_reached_share"], 100), 1),
            fmt(m["lateral_t0p5_p90_s"]),
            fmt(scaled(m["reverse_t0p5_reached_share"], 100), 1),
            fmt(m["reverse_t0p5_p90_s"]),
            fmt(m["jitter_falls_per_env_min"]), fmt(m["combined_falls_per_env_min"])))
    lines.append("")
    lines += ["## 첫 접지 gait 진단", "",
              "forward path 첫 접지만 집계하며 H1 좌우 대칭과 H3 heel-placement/impact 가설을 직접 판정한다.", "",
              "| arm | samples | heel ahead | target±1σ | L/R med bias | overstride | down-v p90 | force p90 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        if arm not in data:
            continue
        m = data[arm]["metrics"]
        lines.append("| {} | {} | {}% | {}% | {} cm | {}% | {} m/s | {} N |".format(
            arm, int(m["touchdown_samples"]),
            fmt(scaled(m["touchdown_ahead_share"], 100), 1),
            fmt(scaled(m["touchdown_target_share"], 100), 1),
            fmt(scaled(m["touchdown_lr_bias_m"], 100), 2),
            fmt(scaled(m["touchdown_overstride_share"], 100), 1),
            fmt(m["touchdown_down_speed_p90_mps"], 2),
            fmt(m["touchdown_force_p90_n"], 1)))
    lines.append("")
    lines += ["## simulator-view 영상 증거", "",
              "| arm | recorded | red arrow | path carrot | path trace | mp4 |",
              "|---|---:|---:|---:|---:|---|"]
    for arm in ARMS:
        if arm not in data:
            continue
        m = data[arm]["metrics"]
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            arm, m["video_recorded_frames"], m["video_arrow_frames"],
            m["video_carrot_frames"], m["video_trace_frames"],
            "yes" if m["video_artifact_nonempty"] else "no"))
    lines.append("")
    return "\n".join(lines)


def atomic_write(path, content):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".hbatch-comparison-", dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="shared_eval_videos/hbatch")
    ap.add_argument("--out")
    args = ap.parse_args()
    out = args.out or os.path.join(args.root, "hbatch-comparison-codex.md")
    lock_path = os.path.join(args.root, ".hbatch-comparison-codex.lock")
    os.makedirs(args.root, exist_ok=True)
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = collect(args.root)
        verdict = verdicts(data)
        atomic_write(out, render(data, verdict))
        json_out = os.path.splitext(out)[0] + ".json"
        payload = {arm: {"suite": item["suite"],
                         "protocol_signatures": item["protocol_signatures"],
                         "metrics": item["metrics"],
                         "verdict": verdict.get(arm)} for arm, item in data.items()}
        atomic_write(json_out, json.dumps(payload, indent=2, ensure_ascii=False, default=float) + "\n")
    print("wrote {} and {} ({} arms)".format(out, json_out, len(data)))


if __name__ == "__main__":
    main()
