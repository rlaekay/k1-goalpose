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
    paths = sorted(p for p in glob.glob(os.path.join(root, arm + "_*")) if os.path.isdir(p))
    return paths[-1] if paths else None


def load_report(suite, name):
    path = os.path.join(suite, name, "report.json")
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def collect(root):
    out = {}
    for arm in ARMS:
        suite = latest_suite(root, arm)
        if not suite:
            continue
        reports = {name: load_report(suite, name) for name in (
            "clean", "force", "jitter", "combined", "lateral", "reverse")}
        if not reports["clean"]:
            continue
        clean, force = reports["clean"], reports["force"] or {}
        out[arm] = {
            "suite": suite,
            "reports": reports,
            "metrics": {
                "waypoint_pos_median_m": nested(clean, "pos_err_m", "median"),
                "path_speed_median_mps": nested(clean, "path_tracking", "mean_speed_median"),
                "path_falls_per_1000": nested(clean, "path_safety", "falls_per_1000_attempts"),
                "time_to_1mps_p90_s": nested(
                    clean, "path_acceleration_response", "time_to_1p0", "p90_s"),
                "cruise_pitch_p90_deg": nested(
                    clean, "high_speed_stability", "cruise_pitch_abs_p90_deg"),
                "cruise_roll_p90_deg": nested(
                    clean, "high_speed_stability", "cruise_roll_abs_p90_deg"),
                "cruise_ang_xy_p90_radps": nested(
                    clean, "high_speed_stability", "cruise_ang_xy_p90_radps"),
                "mirror_error_p90": nested(clean, "symmetry_eval", "p90"),
                "force_events": nested(force, "disturbance_eval", "events", default=0),
                "force_survival_5s": nested(
                    force, "disturbance_eval", "overall", "survival_5s"),
                "force_recovery_share": nested(
                    force, "disturbance_eval", "overall", "recovery_90_within_5s_share"),
                "force_recovery_p90_s": nested(
                    force, "disturbance_eval", "overall", "recovery_90_s_p90"),
                "lateral_t0p5_p90_s": nested(
                    reports["lateral"] or {}, "directional_response", "time_to_0p5", "p90_s"),
                "reverse_t0p5_p90_s": nested(
                    reports["reverse"] or {}, "directional_response", "time_to_0p5", "p90_s"),
                "jitter_falls_per_env_min": nested(
                    reports["jitter"] or {}, "falls_per_env_minute"),
                "combined_falls_per_env_min": nested(
                    reports["combined"] or {}, "falls_per_env_minute"),
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
    for arm, item in data.items():
        m = item["metrics"]
        checks = {
            "force_events_nonzero": m["force_events"] > 0,
            "force_survival_5s_ge_98pct": (
                finite(m["force_survival_5s"]) and m["force_survival_5s"] >= 0.98),
            "force_recovery_p90_le_2s": (
                finite(m["force_recovery_p90_s"]) and m["force_recovery_p90_s"] <= 2.0),
        }
        if arm == "H0":
            checks.update({
                "path_speed_ge_0p95": finite(m["path_speed_median_mps"]) and m["path_speed_median_mps"] >= 0.95,
                "waypoint_median_le_G1": finite(m["waypoint_pos_median_m"]) and m["waypoint_pos_median_m"] <= 0.0552,
            })
        elif arm == "H1":
            checks.update({
                "path_speed_ge_95pct_H0": noninferior(
                    m["path_speed_median_mps"], h0.get("path_speed_median_mps"), 0.95),
                "waypoint_error_le_105pct_H0": noninferior(
                    m["waypoint_pos_median_m"], h0.get("waypoint_pos_median_m"), 1.05, True),
                "mirror_error_p90_le_0p10": finite(m["mirror_error_p90"]) and m["mirror_error_p90"] <= 0.10,
            })
        elif arm == "H2":
            checks.update({
                "path_speed_ge_95pct_H1": noninferior(
                    m["path_speed_median_mps"], h1.get("path_speed_median_mps"), 0.95),
                "time_to_1mps_le_110pct_H1": noninferior(
                    m["time_to_1mps_p90_s"], h1.get("time_to_1mps_p90_s"), 1.10, True),
                "cruise_pitch_improves_H1": noninferior(
                    m["cruise_pitch_p90_deg"], h1.get("cruise_pitch_p90_deg"), 1.0, True),
                "cruise_roll_improves_H1": noninferior(
                    m["cruise_roll_p90_deg"], h1.get("cruise_roll_p90_deg"), 1.0, True),
                "cruise_ang_xy_improves_H1": noninferior(
                    m["cruise_ang_xy_p90_radps"], h1.get("cruise_ang_xy_p90_radps"), 1.0, True),
            })
        elif arm == "H3":
            checks.update({
                "path_speed_ge_95pct_H0": noninferior(
                    m["path_speed_median_mps"], h0.get("path_speed_median_mps"), 0.95),
                "path_falls_improve_H0": (
                    finite(m["path_falls_per_1000"])
                    and finite(h0.get("path_falls_per_1000"))
                    and m["path_falls_per_1000"] < h0["path_falls_per_1000"]),
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
             "| arm | waypoint med cm | path speed | path falls/1000 | t→1m/s p90 | mirror p90 | force 5s survival | force recovery p90 | verdict |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for arm in ARMS:
        if arm not in data:
            lines.append("| {} | NA | NA | NA | NA | NA | NA | NA | INCOMPLETE |".format(arm))
            continue
        m = data[arm]["metrics"]
        lines.append("| {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            arm, fmt(scaled(m["waypoint_pos_median_m"], 100), 2), fmt(m["path_speed_median_mps"]),
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
    lines += ["## 방향 전환/goal jitter 진단", "",
              "| arm | lateral t→0.5 p90 | reverse t→0.5 p90 | jitter falls/env·min | combined falls/env·min |",
              "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        if arm not in data:
            continue
        m = data[arm]["metrics"]
        lines.append("| {} | {} | {} | {} | {} |".format(
            arm, fmt(m["lateral_t0p5_p90_s"]), fmt(m["reverse_t0p5_p90_s"]),
            fmt(m["jitter_falls_per_env_min"]), fmt(m["combined_falls_per_env_min"])))
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
        payload = {arm: {"suite": item["suite"], "metrics": item["metrics"],
                         "verdict": verdict.get(arm)} for arm, item in data.items()}
        atomic_write(json_out, json.dumps(payload, indent=2, ensure_ascii=False, default=float) + "\n")
    print("wrote {} and {} ({} arms)".format(out, json_out, len(data)))


if __name__ == "__main__":
    main()
