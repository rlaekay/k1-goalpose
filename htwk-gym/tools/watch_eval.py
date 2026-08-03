"""Evaluate every checkpoint as it lands, and stop a run that is going wrong.

H spent roughly four GPU-days to learn one thing: all four arms got worse, and
model_0 won selection in 4/4. That was visible at iteration 100 -- position
median had already gone 7.25 -> 10.4..13.2 cm -- but nothing was looking, because
evaluation only ran after training finished. This looks continuously instead.

Two jobs, in this order of importance:

  1. Kill a run that has clearly gone the wrong way, using a verdict fixed
     BEFORE training started. `runner_v3.py` already stops cleanly on a `STOP`
     file in the run dir (it saves a checkpoint first), so stopping is a
     one-line write and needs no signal handling or process ownership.
  2. Feed the monitor a per-checkpoint score history, so the dashboard shows
     accuracy over iterations rather than only reward.

Design constraints that come from earlier failures:
  * The verdict is data, not code: `--ref` and `--stop-ratio` are recorded into
    every result line, so a stopped run says exactly which rule stopped it.
  * Evaluation runs SHORT (20 s x 256 env). This is a screening signal for
    "is this going backwards", never an acceptance number -- acceptance still
    comes from the full protocol in eval_goal_pose.py.
  * A failed eval never stops a run. Only a verdict that actually computed can.

Usage:
    python tools/watch_eval.py --run logs/K1/K1/Goal_Pose_V7/2026-..._I0a_repro \\
        --ref 0.0272 --stop-ratio 1.3 --device cuda:0
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def iters(run):
    out = {}
    for p in glob.glob(os.path.join(run, "nn", "model_*.pth")):
        m = re.search(r"model_(\d+)\.pth$", p)
        if m:
            out[int(m.group(1))] = p
    return out


def evaluate(ckpt, cfg, task, device, seconds, envs):
    """One short eval. Returns the parsed report dict, or None if it failed."""
    out_dir = os.path.join(os.path.dirname(os.path.dirname(ckpt)), "watch_eval",
                           os.path.basename(ckpt).replace(".pth", ""))
    os.makedirs(out_dir, exist_ok=True)
    cmd = [sys.executable, "-u", "eval_goal_pose.py",
           "--task", task, "--config", cfg, "--checkpoint", ckpt,
           "--num_envs", str(envs), "--duration_s", str(seconds),
           "--sim_device", device, "--rl_device", device,
           "--out", out_dir]
    try:
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return None
    for p in glob.glob(os.path.join(out_dir, "**", "report.json"), recursive=True):
        try:
            return json.load(open(p))
        except (ValueError, OSError):
            pass
    # Keep the reason on disk; a silently-failing evaluator is exactly the class
    # of bug that cost this project the whole v7 video set.
    with open(os.path.join(out_dir, "eval_failed.log"), "w") as f:
        f.write("rc=%s\n--- stderr ---\n%s" % (r.returncode, (r.stderr or "")[-4000:]))
    return None


def verdict(rep, ref, stop_ratio, ref_falls, fall_ratio):
    """Is this checkpoint over the line? (over: bool, reason: str)

    Being over the line once is NOT a reason to stop -- see main(). A warm-start
    policy re-adapting to a changed condition gets worse before it gets better,
    and I1b proved it: 3 falls at iteration 125, 2 at 150, then ZERO at 175 with
    every gate passing. I1a, I1c and I1d were all killed at iteration 50 on a
    single reading of 8-10 falls, which is the same transient I1b walked out of.
    """
    pos = (rep.get("pos_err_m") or {}).get("median")
    if pos is None:
        return False, "no position median"
    if ref and pos > ref * stop_ratio:
        return True, ("pos median %.4f m > %.2fx reference %.4f"
                      % (pos, stop_ratio, ref))
    # Compare RATES, not counts. The screening eval completes a different number
    # of segments per arm (731-743 observed), so a raw count silently compares
    # different denominators.
    falls = rep.get("falls")
    rate = rep.get("fall_rate_per_attempt")
    if ref_falls is not None and rate is not None:
        segs = rep.get("segments_completed") or 0
        ref_rate = (float(ref_falls) / segs) if segs else None
        if ref_rate and rate > ref_rate * fall_ratio:
            return True, ("fall rate %.4f > %.1fx reference %.4f (falls %s/%s)"
                          % (rate, fall_ratio, ref_rate, falls, segs))
    return False, "pos %.4f m, falls %s" % (pos, falls)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="training run directory (contains nn/)")
    ap.add_argument("--run-glob", dest="run_glob",
                    help="glob to resolve instead of --run; waits until it matches")
    ap.add_argument("--newer-than", dest="newer_than", type=float, default=0.0,
                    help="only accept a run dir created at or after this epoch time")
    ap.add_argument("--wait-run-s", dest="wait_run_s", type=float, default=600.0)
    ap.add_argument("--task", default="K1/Goal_Pose_V7")
    ap.add_argument("--config", default=None, help="default: <run>/config.yaml")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--ref", type=float, default=None,
                    help="reference position median [m] the run must not blow past")
    ap.add_argument("--stop-ratio", type=float, default=1.3)
    ap.add_argument("--ref-falls", type=int, default=None)
    ap.add_argument("--fall-ratio", type=float, default=3.0)
    ap.add_argument("--skip-first", type=int, default=1,
                    help="ignore the first N checkpoints (warm-start copy etc.)")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--poll", type=float, default=60.0)
    ap.add_argument("--warmup-iters", dest="warmup_iters", type=int, default=100,
                    help="never stop before this iteration; warm-start re-adaptation "
                         "is transiently worse and killing it discards the recovery")
    ap.add_argument("--strikes", type=int, default=2,
                    help="consecutive over-the-line checkpoints required to stop")
    a = ap.parse_args()

    # The launcher cannot pass a run dir because the trainer has not created it
    # yet -- Isaac Gym takes ~40 s to come up. Resolving it here, and REQUIRING
    # it to be newer than the launch, is what stops this from latching onto a
    # previous run of the same arm: the first attempt attached watch_eval to
    # 18:03's finished run while 18:43's actual training went unwatched.
    if a.run:
        run = os.path.abspath(a.run)
    elif a.run_glob:
        run, waited = None, 0.0
        while waited < a.wait_run_s:
            cands = [d for d in glob.glob(a.run_glob)
                     if os.path.isdir(d) and os.path.getmtime(d) >= a.newer_than]
            if cands:
                run = os.path.abspath(max(cands, key=os.path.getmtime))
                break
            time.sleep(5.0)
            waited += 5.0
        if run is None:
            print("run dir matching %r newer than %.0f never appeared in %.0fs"
                  % (a.run_glob, a.newer_than, a.wait_run_s), flush=True)
            return 1
        print("resolved run: %s" % run, flush=True)
    else:
        print("need --run or --run-glob", flush=True)
        return 2
    cfg = a.config or os.path.join(run, "config.yaml")
    hist = os.path.join(run, "watch_eval.jsonl")
    done, seen_any, strikes = set(), 0, 0
    print("watch_eval: %s  (ref %s, stop at %.2fx)" % (run, a.ref, a.stop_ratio), flush=True)

    while True:
        if os.path.exists(os.path.join(run, "STOP")):
            print("STOP file present; exiting.", flush=True)
            return 0
        cks = iters(run)
        todo = sorted(k for k in cks if k not in done)
        for it in todo:
            done.add(it)
            seen_any += 1
            if seen_any <= a.skip_first:
                continue
            rep = evaluate(cks[it], cfg, a.task, a.device, a.seconds, a.envs)
            if rep is None:
                print("  iter %d: eval failed (see watch_eval/); NOT stopping" % it, flush=True)
                continue
            over, why = verdict(rep, a.ref, a.stop_ratio, a.ref_falls, a.fall_ratio)
            strikes = strikes + 1 if over else 0
            stop = over and it >= a.warmup_iters and strikes >= a.strikes
            if over and not stop:
                why += "  (over the line, strike %d/%d%s)" % (
                    strikes, a.strikes,
                    ", warmup until %d" % a.warmup_iters if it < a.warmup_iters else "")
            rec = {"it": it, "t": time.time(), "stop": stop, "over": over,
                   "strikes": strikes, "why": why,
                   "pos_median": (rep.get("pos_err_m") or {}).get("median"),
                   "pos_p90": (rep.get("pos_err_m") or {}).get("p90"),
                   "heading_median": (rep.get("heading_err_deg") or {}).get("median"),
                   "falls": rep.get("falls"),
                   "strict": rep.get("success_rate_strict"),
                   "ref": a.ref, "stop_ratio": a.stop_ratio}
            with open(hist, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print("  iter %d: %s%s" % (it, why, "  -> STOP" if stop else ""), flush=True)
            if stop:
                with open(os.path.join(run, "STOP"), "w") as f:
                    f.write("watch_eval iter %d: %s\n" % (it, why))
                print("wrote STOP; the trainer saves a checkpoint and exits.", flush=True)
                return 0
        time.sleep(a.poll)


if __name__ == "__main__":
    sys.exit(main())
