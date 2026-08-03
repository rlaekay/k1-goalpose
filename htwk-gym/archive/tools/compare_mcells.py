#!/usr/bin/env python3
"""Run and report the targeted two-GPU M-cell evaluation.

The two GPU workers are persistent queues, not a sequential loop with rotating
device labels.  Clean evaluates every cell.  Force evaluates only M0/M1 and
joint-offset evaluates only M0/M2; mirror error is already part of clean, so
this answers the three causal questions with 27 rather than 48 evaluations.
"""

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import subprocess
import sys
import time


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_BASE = os.path.join(ROOT, "logs", "mcells", "compare")
OUT = OUT_BASE
EVAL_CONFIG = os.path.join("sweeps", "mcells", "M0_control-codex.yaml")
ALL_CELLS = (
    "M0_control-codex", "M1_force-codex", "M2_jointdr-codex",
    "M3_mirror_off-codex")
ITERS = (0, 50, 100, 200)
DURATION_S = 40
NUM_ENVS = 256

MODE = {
    "clean": {
        "cells": ALL_CELLS,
        "args": ["--no_noise", "--joint_encoder_bias_rad", "0",
                 "--joint_target_offset_rad", "0", "--init_dof_std_rad", "0"],
    },
    "force": {
        "cells": ("M0_control-codex", "M1_force-codex"),
        "args": ["--keep_perturbations", "--no_noise",
                 "--joint_encoder_bias_rad", "0",
                 "--joint_target_offset_rad", "0", "--init_dof_std_rad", "0"],
    },
    "joint": {
        "cells": ("M0_control-codex", "M2_jointdr-codex"),
        "args": ["--no_noise", "--joint_encoder_bias_rad", "0.015",
                 "--joint_target_offset_rad", "0.010",
                 "--init_dof_std_rad", "0"],
    },
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_dir(state_dir, cell):
    path = os.path.join(state_dir, "run-{}.txt".format(cell))
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        value = f.readline().strip()
    return value if os.path.isdir(value) else None


def checkpoint(state_dir, cell, iteration):
    directory = run_dir(state_dir, cell)
    if not directory:
        return None
    return os.path.join(directory, "nn", "model_{}.pth".format(iteration))


def tag(cell, iteration, mode):
    return "{}_{}_{}-codex".format(cell, iteration, mode)


def report_path(cell, iteration, mode):
    return os.path.join(OUT, tag(cell, iteration, mode), "report.json")


def run_eval(state_dir, cell, iteration, mode, gpu):
    path = report_path(cell, iteration, mode)
    if os.path.isfile(path):
        return True, "cached {}".format(tag(cell, iteration, mode))
    ckpt = checkpoint(state_dir, cell, iteration)
    if not ckpt or not os.path.isfile(ckpt):
        return False, "missing checkpoint {}".format(ckpt)
    outdir = os.path.dirname(path)
    os.makedirs(outdir, exist_ok=True)
    cmd = [
        sys.executable, "-u", "eval_goal_pose.py",
        "--task", "K1/Goal_Pose_HBatch", "--config", EVAL_CONFIG,
        "--checkpoint", ckpt, "--num_envs", str(NUM_ENVS),
        "--duration_s", str(DURATION_S), "--seed", "0",
        "--sim_device", "cuda:{}".format(gpu),
        "--rl_device", "cuda:{}".format(gpu),
        "--exploratory", "--out", outdir,
    ] + MODE[mode]["args"]
    log = os.path.join(outdir, "eval-codex.log")
    with open(log, "w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    ok = proc.returncode == 0 and os.path.isfile(path)
    return ok, "{} {} on cuda:{} -> {}".format(
        "PASS" if ok else "FAIL", tag(cell, iteration, mode), gpu, log)


def verify_model_zero(state_dir, cells):
    values = {}
    for cell in cells:
        path = checkpoint(state_dir, cell, 0)
        if not path or not os.path.isfile(path):
            raise ValueError("{} has no copied model_0".format(cell))
        values[cell] = sha256(path)
    if len(set(values.values())) != 1:
        raise ValueError("model_0 hashes differ: {}".format(values))
    return next(iter(values.values()))


def jobs(cells):
    work = []
    for mode, spec in MODE.items():
        eligible = [cell for cell in cells if cell in spec["cells"]]
        for iteration in ITERS:
            for cell in eligible:
                if iteration == 0 and cell != eligible[0]:
                    continue
                work.append((cell, iteration, mode))
    return work


def worker(state_dir, gpu, queue):
    out = []
    for cell, iteration, mode in queue:
        result = run_eval(state_dir, cell, iteration, mode, gpu)
        print(result[1], flush=True)
        out.append(result)
    return out


def load(cell, iteration, mode):
    # All copied model_0 files are byte-identical, so each mode evaluates it
    # once from M0 and reuses that observation bank for the other cells.
    source = "M0_control-codex" if iteration == 0 else cell
    path = report_path(source, iteration, mode)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def nested(value, *keys):
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def fmt(value, digits=2, scale=1.0):
    return "-" if not finite(value) else ("{:.%df}" % digits).format(
        float(value) * scale)


def task_metrics(data):
    return {
        "pos_med_cm": (nested(data, "pos_err_m", "median") or 0.0) * 100.0,
        "pos_p90_cm": (nested(data, "pos_err_m", "p90") or 0.0) * 100.0,
        "strict_pct": (nested(data, "success_rate_strict") or 0.0) * 100.0,
        "never_pct": (nested(data, "failure_modes", "never_arrived", "share") or 0.0) * 100.0,
        "path_speed": nested(data, "path_tracking", "mean_speed_median"),
        "falls_1000": nested(data, "overall_safety", "falls_per_1000_attempts"),
        "path_falls_1000": nested(data, "path_safety", "falls_per_1000_attempts"),
        "mirror_p90": nested(data, "symmetry_eval", "p90"),
    } if data else {}


def noninferior(candidate, control):
    c, b = task_metrics(candidate), task_metrics(control)
    if not c or not b:
        return False
    checks = (
        c["pos_p90_cm"] <= 1.05 * b["pos_p90_cm"] + 0.25,
        c["never_pct"] <= b["never_pct"] + 2.0,
        c["strict_pct"] + 2.0 >= 0.95 * b["strict_pct"],
        finite(c["path_speed"]) and finite(b["path_speed"])
        and c["path_speed"] >= 0.95 * b["path_speed"],
        finite(c["falls_1000"]) and finite(b["falls_1000"])
        and c["falls_1000"] <= 1.10 * b["falls_1000"] + 1.0,
    )
    return all(checks)


def render(cells, model_zero_sha):
    lines = []
    add = lines.append
    add("# M-cell 빠른 결과 — 외란·joint DR·mirror loss")
    add("")
    add("- protocol: `2026-08-01-codex-v1`")
    add("- 공통 model 0 SHA-256: `{}`".format(model_zero_sha))
    add("- 1 seed·200 iteration의 **screening**이며, survivor만 paired seed 재확인한다.")
    add("- M0/M1/M2는 G1의 기존 mirror loss 0.5를 유지하고, M3만 이를 끈다. mirror PPO augmentation은 전 셀 0이다.")

    add("\n## Clean: fine-tune 비용과 task 보존")
    add("")
    add("| cell | iter | pos med/p90 cm | strict / never % | path speed m/s | falls all/path per1000 | mirror p90 |")
    add("|---|---:|---:|---:|---:|---:|---:|")
    for cell in cells:
        for iteration in ITERS:
            m = task_metrics(load(cell, iteration, "clean"))
            if not m:
                continue
            add("| {} | {} | {:.2f}/{:.2f} | {:.1f}/{:.1f} | {} | {}/{} | {} |".format(
                cell, iteration, m["pos_med_cm"], m["pos_p90_cm"],
                m["strict_pct"], m["never_pct"], fmt(m["path_speed"], 3),
                fmt(m["falls_1000"], 1), fmt(m["path_falls_1000"], 1),
                fmt(m["mirror_p90"], 3)))

    if all(x in cells for x in ("M0_control-codex", "M1_force-codex")):
        add("\n## 외란: 공통 다방향 scenario 시험")
        add("")
        add("| cell | iter | records | survival 5s / high-speed % | recovery≤5s % | force falls | delivery p90 err % |")
        add("|---|---:|---:|---:|---:|---:|---:|")
        for cell in ("M0_control-codex", "M1_force-codex"):
            for iteration in ITERS:
                d = load(cell, iteration, "force")
                overall = nested(d, "disturbance_eval", "overall") or {}
                high = nested(d, "disturbance_eval", "high_speed") or {}
                audit = nested(d, "disturbance_eval", "delivery_audit", "force") or {}
                add("| {} | {} | {} | {}/{} | {} | {} | {} |".format(
                    cell, iteration, overall.get("records", "-"),
                    fmt(overall.get("survival_5s"), 1, 100),
                    fmt(high.get("survival_5s"), 1, 100),
                    fmt(overall.get("recovery_90_within_5s_share"), 1, 100),
                    nested(d, "disturbance_eval", "falls_during_force") or 0,
                    fmt(audit.get("relative_error_p90"), 3, 100)))
        probe = load("M0_control-codex", 0, "force") or {}
        de = probe.get("disturbance_eval") or {}
        add("")
        add("- scenario counts: `{}`".format({
            k: v.get("records", 0) for k, v in
            (de.get("scenario_breakdown") or {}).items()}))
        add("- height-tier counts: `{}`".format({
            k: v.get("records", 0) for k, v in
            (de.get("height_tier_breakdown") or {}).items()}))
        add("- robot-local direction octants: `{}`".format(
            de.get("direction_octants_robot_local") or {}))

    if all(x in cells for x in ("M0_control-codex", "M2_jointdr-codex")):
        add("\n## Joint-offset probe: encoder ±0.015 / target ±0.010 rad")
        add("")
        add("| cell | iter | pos med/p90 cm | strict / never % | path speed | falls/1000 |")
        add("|---|---:|---:|---:|---:|---:|")
        for cell in ("M0_control-codex", "M2_jointdr-codex"):
            for iteration in ITERS:
                m = task_metrics(load(cell, iteration, "joint"))
                if m:
                    add("| {} | {} | {:.2f}/{:.2f} | {:.1f}/{:.1f} | {} | {} |".format(
                        cell, iteration, m["pos_med_cm"], m["pos_p90_cm"],
                        m["strict_pct"], m["never_pct"],
                        fmt(m["path_speed"], 3), fmt(m["falls_1000"], 1)))

    add("\n## 사전 고정 판정")
    add("")
    control0 = load("M0_control-codex", 0, "clean")
    control200 = load("M0_control-codex", 200, "clean")
    if control0 and control200:
        stable = noninferior(control200, control0)
        m0, m2 = task_metrics(control0), task_metrics(control200)
        add("- M0 fine-tune: **{}** — pos p90 {:.2f}→{:.2f} cm, never {:.1f}→{:.1f}%, path speed {}→{} m/s.".format(
            "PASS" if stable else "FAIL/STOP", m0["pos_p90_cm"],
            m2["pos_p90_cm"], m0["never_pct"], m2["never_pct"],
            fmt(m0["path_speed"], 3), fmt(m2["path_speed"], 3)))
    else:
        stable = False
        add("- M0 fine-tune: 데이터 부족")

    if "M1_force-codex" in cells:
        clean_ok = noninferior(load("M1_force-codex", 200, "clean"), control200)
        b = nested(load("M0_control-codex", 200, "force"),
                   "disturbance_eval", "overall", "survival_5s")
        c = nested(load("M1_force-codex", 200, "force"),
                   "disturbance_eval", "overall", "survival_5s")
        benefit = finite(b) and finite(c) and c >= b
        add("- scenario 외란: **{}** — clean 비열세 {}, 5s survival {}→{}%.".format(
            "SURVIVE" if clean_ok and benefit else "DROP/REDOSE",
            clean_ok, fmt(b, 2, 100), fmt(c, 2, 100)))
    if "M2_jointdr-codex" in cells:
        clean_ok = noninferior(load("M2_jointdr-codex", 200, "clean"), control200)
        b = task_metrics(load("M0_control-codex", 200, "joint"))
        c = task_metrics(load("M2_jointdr-codex", 200, "joint"))
        benefit = bool(b and c and c["pos_p90_cm"] <= b["pos_p90_cm"])
        add("- joint DR: **{}** — clean 비열세 {}, probe p90 {}→{} cm.".format(
            "SURVIVE" if clean_ok and benefit else "DROP/REDOSE", clean_ok,
            fmt(b.get("pos_p90_cm") if b else None),
            fmt(c.get("pos_p90_cm") if c else None)))
    if "M3_mirror_off-codex" in cells:
        on = task_metrics(control200)
        off_data = load("M3_mirror_off-codex", 200, "clean")
        off = task_metrics(off_data)
        keep = (on and off and finite(on["mirror_p90"])
                and finite(off["mirror_p90"])
                and on["mirror_p90"] <= off["mirror_p90"]
                and noninferior(control200, off_data))
        add("- G1 mirror loss 0.5: **{}** — loss ON/OFF mirror p90 {}/{}; task ON이 OFF 대비 비열세 {}.".format(
            "KEEP" if keep else "UNRESOLVED/REMOVE", fmt(on.get("mirror_p90") if on else None, 3),
            fmt(off.get("mirror_p90") if off else None, 3),
            noninferior(control200, off_data) if off_data else False))

    add("")
    add("> 이 표의 SURVIVE는 본학습 채택이 아니다. 같은 방향을 보인 레버만 M0과 paired seeds `31415, 27182`로 재확인하고, 생산 후보에는 low-dose 외란과 goal jitter를 의무적으로 다시 넣는다.")
    return "\n".join(lines) + "\n"


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--state_dir", required=True)
    ap.add_argument("--cells", nargs="+", default=list(ALL_CELLS))
    ap.add_argument("--report_only", action="store_true")
    args = ap.parse_args()
    OUT = os.path.join(
        OUT_BASE, os.path.basename(os.path.abspath(args.state_dir)))
    cells = [cell for cell in args.cells if cell in ALL_CELLS]
    if "M0_control-codex" not in cells:
        raise SystemExit("M0_control-codex is required for causal comparisons")
    model_zero_sha = verify_model_zero(args.state_dir, cells)
    failures = []
    if not args.report_only:
        os.makedirs(OUT, exist_ok=True)
        work = jobs(cells)
        queues = [work[0::2], work[1::2]]
        print("{} evaluations, two persistent GPU queues: {} / {}".format(
            len(work), len(queues[0]), len(queues[1])), flush=True)
        started = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, args.state_dir, gpu, queues[gpu])
                       for gpu in (0, 1)]
            for future in futures:
                failures.extend(message for ok, message in future.result() if not ok)
        print("evaluation wall {:.1f} min".format(
            (time.time() - started) / 60.0), flush=True)
    report = render(cells, model_zero_sha)
    path = os.path.join(OUT, "mcell-report-codex.md")
    os.makedirs(OUT, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print("written {}".format(path))
    if failures:
        print("FAILED evaluations:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
