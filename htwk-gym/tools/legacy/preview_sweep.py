"""One-command mid-training preview for the three GoalPose sweep arms.

The script discovers each arm's newest run and a checkpoint that is old enough
not to still be inside torch.save(), prints TensorBoard progress, then evaluates
the checkpoints sequentially on a spare GPU.  Sequential evaluation keeps the
three reports comparable and avoids competing with the training jobs on GPU 1.

Quick visual preview (default: 64 envs x 30 simulated seconds, 12 s video):
    python tools/preview_sweep.py --device cuda:0

Full MASTERPLAN protocol (256 envs x 120 simulated seconds):
    python tools/preview_sweep.py --device cuda:0 --full --no_video

The default uses the common Goal_Pose.yaml dynamics for an apples-to-apples
comparison.  Add --native to use each run's config.yaml (notably armC's 200 Hz
physics) when inspecting native behavior rather than comparing policies.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time


DEFAULT_TASK = "K1/Goal_Pose"
DEFAULT_ARMS = ("armA_continue", "armB_goal_reached", "armC_200hz")
CHECKPOINT_RE = re.compile(r"model_(\d+)\.pth$")


def task_log_root(task):
    robot = task.split("/")[0]
    return os.path.join("logs", robot, task)


def find_latest_run(root, arm):
    candidates = [p for p in glob.glob(os.path.join(root, "*_{}".format(arm))) if os.path.isdir(p)]
    if not candidates:
        return None
    # Recorder prefixes runs with an ISO-like YYYY-MM-DD-HH-MM-SS timestamp.
    # Directory mtime is unsafe here because later eval/STOP children can touch
    # an older run after a newer training run has already started.
    return max(candidates, key=lambda p: os.path.basename(p))


def checkpoint_iteration(path):
    match = CHECKPOINT_RE.search(os.path.basename(path))
    return int(match.group(1)) if match else -1


def find_stable_checkpoint(run_dir, min_age_s):
    candidates = glob.glob(os.path.join(run_dir, "nn", "model_*.pth"))
    candidates.sort(key=lambda p: (checkpoint_iteration(p), os.path.getmtime(p)), reverse=True)
    now = time.time()
    for path in candidates:
        try:
            if os.path.getsize(path) <= 0 or now - os.path.getmtime(path) < min_age_s:
                continue
            first = (os.path.getsize(path), os.stat(path).st_mtime_ns)
            time.sleep(0.25)
            second = (os.path.getsize(path), os.stat(path).st_mtime_ns)
            if first != second:
                continue
            # A crashed torch.save can leave an old, non-empty truncated file.
            # Validate on CPU and fall back to the next checkpoint on any error.
            import torch

            payload = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(payload, dict) and "model" in payload:
                return path
        except Exception:
            continue
    return None


def tensorboard_progress(run_dir):
    """Return (last_step, reward) without making TensorBoard a hard dependency."""
    try:
        from tensorboard.backend.event_processing import event_accumulator

        accumulator = event_accumulator.EventAccumulator(
            os.path.join(run_dir, "summaries"), size_guidance={"scalars": 0}
        )
        accumulator.Reload()
        rewards = accumulator.Scalars("reward")
        if rewards:
            return rewards[-1].step + 1, rewards[-1].value
    except Exception:
        pass
    return None, None


def fmt(value, scale=1.0, digits=1):
    if value is None:
        return "-"
    return ("{:.%df}" % digits).format(value * scale)


def make_summary_markdown(rows, authoritative, native, num_envs, duration_s):
    lines = ["# GoalPose sweep 중간 비교", ""]
    if not authoritative:
        lines.append(
            "> 이 실행은 탐색용 preview이며 MASTERPLAN의 공식 게이트 판정용이 아니다."
        )
        lines.append("")
    lines.append("- 조건: {} envs x {:.0f} simulated s".format(num_envs, duration_s))
    lines.append("- 평가 dynamics: {}".format("각 run native config" if native else "공통 Goal_Pose.yaml (표준 비교)"))
    lines.append("- TB reward는 arm별 학습 진척 확인용이며 armB의 reward 정의가 달라 arm 사이 비교 금지")
    lines.append("")
    lines.append("| arm | train iter | TB reward (arm-local) | checkpoint | pos med/p90 | heading med | speed med | falls | rollout wall |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        if row.get("error"):
            lines.append("| {} | {} | {} | - | ERROR: {} | - | - | - | - |".format(
                row["arm"], row.get("train_iter") or "-", fmt(row.get("tb_reward"), digits=3), row["error"]))
            continue
        report = row["report"]
        lines.append(
            "| {arm} | {train_iter} | {reward} | {ckpt_iter} | {pos_med:.1f}/{pos_p90:.1f} cm | "
            "{heading:.1f} deg | {speed:.3f} m/s | {falls} | {wall:.1f}s |".format(
                arm=row["arm"],
                train_iter=row.get("train_iter") or "-",
                reward=fmt(row.get("tb_reward"), digits=3),
                ckpt_iter=row["checkpoint_iteration"],
                pos_med=report["pos_err_m"]["median"] * 100.0,
                pos_p90=report["pos_err_m"]["p90"] * 100.0,
                heading=report["heading_err_deg"]["median"],
                speed=report["final_speed_mps"]["median"],
                falls=report["falls"],
                wall=report.get("timing", {}).get("rollout_wall_s", 0.0),
            )
        )
    lines.append("")
    for row in rows:
        if row.get("report_path"):
            lines.append("- {}: `{}`".format(row["arm"], row["report_path"]))
        if row.get("warning"):
            lines.append("  - warning: {}".format(row["warning"]))
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Preview the latest checkpoints from all GoalPose sweep arms.")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--arms", default=",".join(DEFAULT_ARMS), help="comma-separated run suffixes")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--root", help="task log root (default: logs/<robot>/<task>)")
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--duration_s", type=float)
    parser.add_argument("--record_video_s", type=float, default=12.0)
    parser.add_argument("--min_checkpoint_age_s", type=float, default=10.0)
    parser.add_argument("--full", action="store_true", help="use the 256 env x 120 s standard protocol")
    parser.add_argument("--native", action="store_true", help="evaluate each arm with its run config.yaml")
    parser.add_argument("--status_only", action="store_true", help="print run/checkpoint progress without starting Isaac Gym")
    parser.add_argument("--dry_run", action="store_true", help="print eval commands without running them")
    video = parser.add_mutually_exclusive_group()
    video.add_argument("--record_video", dest="record_video", action="store_true")
    video.add_argument("--no_video", dest="record_video", action="store_false")
    parser.set_defaults(record_video=True)
    args = parser.parse_args()

    root = args.root or task_log_root(args.task)
    arms = [arm.strip() for arm in args.arms.split(",") if arm.strip()]
    num_envs = args.num_envs or (256 if args.full else 64)
    duration_s = args.duration_s or (120.0 if args.full else 30.0)
    authoritative = bool(args.full and not args.native and num_envs >= 256 and duration_s >= 120.0)
    stamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    selections = []

    print("arm progress (checkpoint numbers restart at 0 after warm-start):")
    for arm in arms:
        run_dir = find_latest_run(root, arm)
        if run_dir is None:
            selections.append({"arm": arm, "error": "run not found under {}".format(root)})
            print("  {:24s} run not found".format(arm))
            continue
        checkpoint = find_stable_checkpoint(run_dir, args.min_checkpoint_age_s)
        train_iter, reward = tensorboard_progress(run_dir)
        if checkpoint is None:
            selections.append({
                "arm": arm, "run_dir": run_dir, "train_iter": train_iter,
                "tb_reward": reward, "error": "no stable checkpoint yet",
            })
            print("  {:24s} iter={} reward={} (no stable checkpoint yet)".format(
                arm, train_iter or "-", fmt(reward, digits=3)))
            continue
        selection = {
            "arm": arm,
            "run_dir": run_dir,
            "checkpoint": checkpoint,
            "checkpoint_iteration": checkpoint_iteration(checkpoint),
            "train_iter": train_iter,
            "tb_reward": reward,
        }
        selections.append(selection)
        print("  {:24s} iter={} reward={} checkpoint=model_{}".format(
            arm, train_iter or "-", fmt(reward, digits=3), selection["checkpoint_iteration"]))

    if args.status_only:
        return

    rows = []
    for selection in selections:
        if selection.get("error"):
            rows.append(selection)
            continue
        arm = selection["arm"]
        out_dir = os.path.join(selection["run_dir"], "eval_preview", stamp)
        cmd = [
            sys.executable,
            "eval_goal_pose.py",
            "--task", args.task,
            "--checkpoint", selection["checkpoint"],
            "--sim_device", args.device,
            "--rl_device", args.device,
            "--num_envs", str(num_envs),
            "--duration_s", str(duration_s),
            "--out", out_dir,
        ]
        if args.native:
            cmd.extend(["--config", os.path.join(selection["run_dir"], "config.yaml")])
        if args.record_video:
            cmd.extend(["--record_video", "--record_video_s", str(args.record_video_s)])
        if not authoritative:
            cmd.append("--exploratory")
        print("\n[{}] {}".format(arm, " ".join(cmd)))
        if args.dry_run:
            rows.append(selection)
            continue
        try:
            subprocess.run(cmd, check=True)
            report_path = os.path.join(out_dir, "report.json")
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            row = dict(selection)
            row["report"] = report
            row["report_path"] = report_path
            rows.append(row)
        except subprocess.CalledProcessError as exc:
            # eval writes report.json before optional video encoding. Preserve
            # valid metrics when only imageio/ffmpeg post-processing failed.
            report_path = os.path.join(out_dir, "report.json")
            if os.path.isfile(report_path):
                try:
                    with open(report_path, "r", encoding="utf-8") as f:
                        report = json.load(f)
                    row = dict(selection)
                    row["report"] = report
                    row["report_path"] = report_path
                    row["warning"] = "eval post-processing exited {} (metrics preserved)".format(exc.returncode)
                    rows.append(row)
                    continue
                except (OSError, ValueError):
                    pass
            row = dict(selection)
            row["error"] = str(exc)
            rows.append(row)
        except (OSError, ValueError, KeyError) as exc:
            row = dict(selection)
            row["error"] = str(exc)
            rows.append(row)

    if args.dry_run:
        return

    summary_dir = os.path.join(root, "_preview", stamp)
    os.makedirs(summary_dir, exist_ok=True)
    markdown = make_summary_markdown(
        rows, authoritative=authoritative, native=args.native,
        num_envs=num_envs, duration_s=duration_s,
    )
    summary_md = os.path.join(summary_dir, "comparison.md")
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write(markdown)
    with open(os.path.join(summary_dir, "comparison.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print("\n" + markdown)
    print("comparison saved to: {}".format(summary_md))


if __name__ == "__main__":
    main()
