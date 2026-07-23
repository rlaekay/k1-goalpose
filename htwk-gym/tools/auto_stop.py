"""Convergence watchdog: stop training at plateau, then auto-run the eval harness.

Flow (MASTERPLAN milestone 3 support):
  1. poll the newest (or given) run dir's TensorBoard scalars every --poll_s
  2. declare a plateau when the mean of the last --window rewards improved less
     than --rel_eps (relative) over the mean of the window before it, --patience
     times in a row (and at least --min_iters iterations have passed)
  3. drop a STOP file into the run dir -> utils/runner.py saves a checkpoint and
     exits gracefully (no SIGKILL, nothing lost)
  4. wait until the run dir goes quiet (training process exited, GPU freed)
  5. run eval_goal_pose.py on the final checkpoint and leave report.md/json there

Run it in a second tmux window next to the training:
    python tools/auto_stop.py --sim_device cuda:1 --rl_device cuda:1
"""

import os
import sys
import glob
import time
import argparse
import subprocess

import yaml
from tensorboard.backend.event_processing import event_accumulator


def latest_run_dir():
    candidates = sorted(glob.glob(os.path.join("logs", "**", "summaries"), recursive=True), key=os.path.getmtime)
    if not candidates:
        raise FileNotFoundError("no logs/**/summaries found — is a training run started?")
    return os.path.dirname(candidates[-1])


def read_scalars(run_dir, tag):
    ea = event_accumulator.EventAccumulator(
        os.path.join(run_dir, "summaries"), size_guidance={"scalars": 0}
    )
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [(e.step, e.value) for e in ea.Scalars(tag)]


def newest_mtime(root):
    newest = 0.0
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            try:
                newest = max(newest, os.path.getmtime(os.path.join(dirpath, name)))
            except OSError:
                pass
    return newest


def main():
    parser = argparse.ArgumentParser(description="Stop training at reward plateau, then auto-evaluate.")
    parser.add_argument("--run_dir", help="training run dir (default: newest under logs/)")
    parser.add_argument("--metric", default="reward")
    parser.add_argument("--window", type=int, default=50, help="iterations per comparison window")
    parser.add_argument("--rel_eps", type=float, default=0.02, help="relative improvement below which it's a plateau")
    parser.add_argument("--min_iters", type=int, default=2000, help="never stop before this many iterations")
    parser.add_argument("--patience", type=int, default=3, help="consecutive plateau checks required")
    parser.add_argument("--poll_s", type=float, default=120.0)
    parser.add_argument("--stale_min", type=float, default=30.0, help="no new data for this many minutes -> assume training already ended, go straight to eval")
    parser.add_argument("--quiet_s", type=float, default=60.0, help="run dir must be untouched this long before eval starts")
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--sim_device", help="passed to eval (default: from run config.yaml)")
    parser.add_argument("--rl_device", help="passed to eval (default: from run config.yaml)")
    args = parser.parse_args()

    run_dir = args.run_dir or latest_run_dir()
    print("watching run: {}".format(run_dir))
    with open(os.path.join(run_dir, "config.yaml"), "r", encoding="utf-8") as f:
        run_cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    task = run_cfg["basic"]["task"]
    sim_device = args.sim_device or run_cfg["basic"]["sim_device"]
    rl_device = args.rl_device or run_cfg["basic"]["rl_device"]

    stop_file = os.path.join(run_dir, "STOP")
    plateau_streak = 0
    last_count = 0
    last_growth_time = time.time()
    stopped = False

    while True:
        scalars = read_scalars(run_dir, args.metric)
        count = len(scalars)
        if count > last_count:
            last_count = count
            last_growth_time = time.time()

        if count >= max(args.min_iters, 2 * args.window):
            values = [v for _, v in scalars]
            recent = sum(values[-args.window:]) / args.window
            previous = sum(values[-2 * args.window:-args.window]) / args.window
            improvement = recent - previous
            threshold = args.rel_eps * max(abs(recent), 1e-3)
            is_plateau = improvement < threshold
            plateau_streak = plateau_streak + 1 if is_plateau else 0
            print("iter {} | {} last{}={:.4f} prev{}={:.4f} improve={:+.4f} (thr {:.4f}) | plateau {}/{}".format(
                scalars[-1][0], args.metric, args.window, recent, args.window, previous,
                improvement, threshold, plateau_streak, args.patience))
            if plateau_streak >= args.patience:
                with open(stop_file, "w", encoding="utf-8") as f:
                    f.write("plateau: improve {:+.4f} < {:.4f} x{}\n".format(improvement, threshold, args.patience))
                print("plateau confirmed -> STOP file written: {}".format(stop_file))
                stopped = True
                break
        else:
            print("iter {} — warming up (min_iters {}, window {})".format(
                scalars[-1][0] if scalars else 0, args.min_iters, args.window))

        if (time.time() - last_growth_time) > args.stale_min * 60:
            print("no new {} data for {:.0f} min — assuming training already ended".format(args.metric, args.stale_min))
            break

        time.sleep(args.poll_s)

    # wait for the training process to save its final checkpoint and exit
    print("waiting for run dir to go quiet ({}s)...".format(args.quiet_s))
    deadline = time.time() + 1800
    while time.time() < deadline:
        if time.time() - newest_mtime(run_dir) >= args.quiet_s:
            break
        time.sleep(10)

    models = sorted(glob.glob(os.path.join(run_dir, "nn", "*.pth")), key=os.path.getmtime)
    if not models:
        print("ERROR: no checkpoint found in {}/nn — nothing to evaluate".format(run_dir))
        sys.exit(1)
    final_ckpt = models[-1]
    print("final checkpoint: {}".format(final_ckpt))

    if args.skip_eval:
        print("(--skip_eval) done.")
        return

    cmd = [
        sys.executable, "eval_goal_pose.py",
        "--task", task,
        "--checkpoint", final_ckpt,
        "--sim_device", sim_device,
        "--rl_device", rl_device,
    ]
    print("running eval: {}".format(" ".join(cmd)))
    subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
