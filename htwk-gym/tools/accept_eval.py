"""Acceptance evaluation over several seeds, because falls do not reproduce.

Measured 2026-08-03: I1c_cadence/model_95 evaluated twice under an identical
protocol -- same checkpoint, seed 0, 256 envs, 120 s, same config, same env SHA,
the only difference being that one run also recorded video -- returned position
medians of 2.64 and 2.66 cm and fall counts of 3 and 18. Position is stable to
under a percent; falls moved 6x. A common Poisson rate cannot produce both.

So a single run cannot decide the one gate this project has never passed. Every
fall comparison made from single runs -- E0's 2 against I1b's 4, the whole falls
column of the I1 factorial -- is weaker than it was presented as.

This runs the same checkpoint over N seeds and reports each metric with its
spread, so "falls = 0" is a claim about a distribution rather than about one
lucky rollout. Position and heading are reported the same way; they will look
boring, and that contrast is the point.

Usage:
    python tools/accept_eval.py --config sweeps/I1b_force.yaml \\
        --checkpoint logs/.../nn/model_175.pth --seeds 0 1 2 3 4
"""

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_one(cfg, ckpt, task, seed, envs, secs, device, out_root):
    out = os.path.join(out_root, "seed_%d" % seed)
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "-u", "eval_goal_pose.py",
           "--task", task, "--config", cfg, "--checkpoint", ckpt,
           "--num_envs", str(envs), "--duration_s", str(secs),
           "--seed", str(seed),
           "--sim_device", device, "--rl_device", device, "--out", out]
    t0 = time.time()
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    reps = glob.glob(os.path.join(out, "**", "report.json"), recursive=True)
    if not reps:
        with open(os.path.join(out, "eval_failed.log"), "w") as f:
            f.write("rc=%s\n%s" % (r.returncode, (r.stderr or "")[-4000:]))
        print("  seed %-3d 실패 (%s/eval_failed.log)" % (seed, out), flush=True)
        return None
    d = json.load(open(reps[0]))
    print("  seed %-3d  pos %.4f  p90 %.4f  hd %.2f  falls %-3s  strict %.3f  (%.0fs)"
          % (seed, d["pos_err_m"]["median"], d["pos_err_m"]["p90"],
             d["heading_err_deg"]["median"], d.get("falls"),
             d.get("success_rate_strict", float("nan")), time.time() - t0), flush=True)
    return d


def spread(name, vals, fmt="%.4f", scale=1.0):
    if not vals:
        return
    v = [x * scale for x in vals if x is not None]
    if not v:
        return
    lo, hi = min(v), max(v)
    med = statistics.median(v)
    rel = (hi - lo) / med if med else float("inf")
    warn = "   <<< 편차 큼, 단일 run으로 판정 불가" if rel > 0.5 else ""
    print(("  %-22s median " + fmt + "   범위 " + fmt + " ~ " + fmt + "   상대폭 %.0f%%%s")
          % (name, med, lo, hi, 100 * rel, warn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task", default="K1/Goal_Pose_V7")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--num_envs", type=int, default=256)
    ap.add_argument("--duration_s", type=float, default=120.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out_root = a.out or os.path.join(os.path.dirname(os.path.dirname(a.checkpoint)),
                                     "accept", time.strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(out_root, exist_ok=True)
    print("checkpoint : %s" % a.checkpoint)
    print("config     : %s" % a.config)
    print("seeds      : %s   (%d envs x %.0f s each)\n" % (a.seeds, a.num_envs, a.duration_s))

    reps = [r for r in (run_one(a.config, a.checkpoint, a.task, s, a.num_envs,
                                a.duration_s, a.device, out_root) for s in a.seeds)
            if r is not None]
    if not reps:
        print("\n전부 실패했다.")
        return 1

    print("\n=== %d seed 종합 ===" % len(reps))
    spread("위치 median [cm]", [r["pos_err_m"]["median"] for r in reps], "%.2f", 100)
    spread("위치 p90 [cm]", [r["pos_err_m"]["p90"] for r in reps], "%.2f", 100)
    spread("heading median [°]", [r["heading_err_deg"]["median"] for r in reps], "%.2f")
    spread("낙상 [회]", [r.get("falls") for r in reps], "%.1f")
    spread("낙상률 [/1000]", [r.get("fall_rate_per_attempt") for r in reps], "%.3f", 1000)
    spread("strict 성공률 [%]", [r.get("success_rate_strict") for r in reps], "%.1f", 100)

    falls = [r.get("falls") for r in reps if r.get("falls") is not None]
    att = sum((r.get("segments_completed") or 0) + (r.get("falls") or 0) for r in reps)
    print("\n  누적: 낙상 %d / 시도 %d = %.3f /1000" % (sum(falls), att,
                                                      1000.0 * sum(falls) / max(att, 1)))
    print("  게이트 '낙상 0'은 %s"
          % ("통과 (전 seed 0회)" if max(falls) == 0 else
             "미통과 — 최악 seed에서 %d회" % max(falls)))
    print("\n  결과: %s" % out_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
