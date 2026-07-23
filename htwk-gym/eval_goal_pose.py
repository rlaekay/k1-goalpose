"""Evaluation harness for the K1 GoalPose task (MASTERPLAN milestone 1).

Loads a trained checkpoint, rolls the policy in headless sim, and measures the
actual task metrics the reward curve cannot show:
  - final position error per goal segment (median / p90)
  - final heading error per goal segment
  - final base speed (did it stop?)
  - falls (non-timeout terminations)
then judges them against the MASTERPLAN gates and writes report.md / report.json
/ segments.csv into <run_dir>/eval/<timestamp>/.

A "goal segment" ends when the env's 4-8 s resample timer fires; the final
position is measured against the goal that was just replaced, so the numbers
are exact (not one control step stale).

Usage (server):
    python eval_goal_pose.py --task K1/Goal_Pose --checkpoint -1 \
        --sim_device cuda:1 --rl_device cuda:1
"""

import os
import csv
import json
import glob
import time
import random
import argparse

import isaacgym  # noqa: F401  (must be imported before torch)
from envs import *  # noqa: F401,F403  (registers task classes)

import torch
import numpy as np
import yaml

from isaacgym.torch_utils import get_euler_xyz
from utils.model import ActorCritic
from utils.runner import get_task_class


def wrap(x):
    return (x + torch.pi) % (2 * torch.pi) - torch.pi


def find_latest_checkpoint(task_name):
    robot = task_name.split("/")[0]
    for pattern in [
        os.path.join("logs", robot, task_name, "**", "*.pth"),
        os.path.join("logs", "**", "*.pth"),
    ]:
        models = sorted(glob.glob(pattern, recursive=True), key=os.path.getmtime)
        if models:
            return models[-1]
    return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate a GoalPose checkpoint against the MASTERPLAN gates.")
    parser.add_argument("--task", default="K1/Goal_Pose")
    parser.add_argument("--checkpoint", default="-1", help=".pth path, or -1 for the latest under logs/")
    parser.add_argument("--num_envs", type=int, help="override evaluation.num_envs from the yaml")
    parser.add_argument("--duration_s", type=float, help="override evaluation.duration_s from the yaml")
    parser.add_argument("--sim_device", help="override yaml sim_device")
    parser.add_argument("--rl_device", help="override yaml rl_device")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of the deterministic mean")
    parser.add_argument("--keep_perturbations", action="store_true", help="keep random kicks/pushes on during eval")
    parser.add_argument("--no_noise", action="store_true", help="disable observation noise")
    parser.add_argument("--record_video", action="store_true", help="also record an mp4 of env 0 (first --record_video_s seconds)")
    parser.add_argument("--record_video_s", type=float, default=8.0)
    parser.add_argument("--out", help="output dir (default: <run_dir>/eval/<timestamp>)")
    args = parser.parse_args()

    with open(os.path.join("envs", "{}.yaml".format(args.task)), "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    eval_cfg = cfg.get("evaluation", {})
    gates = eval_cfg.get("gates", {})
    gate_pos_median = gates.get("pos_median_m", 0.05)
    gate_pos_p90 = gates.get("pos_p90_m", 0.10)
    gate_heading_median = gates.get("heading_median_deg", 10.0)
    gate_max_falls = gates.get("max_falls", 0)

    num_envs = args.num_envs or eval_cfg.get("num_envs", 256)
    duration_s = args.duration_s or eval_cfg.get("duration_s", 120.0)

    cfg["basic"]["task"] = args.task
    cfg["basic"]["headless"] = True
    cfg["env"]["num_envs"] = num_envs
    if args.sim_device:
        cfg["basic"]["sim_device"] = args.sim_device
    if args.rl_device:
        cfg["basic"]["rl_device"] = args.rl_device
    cfg["viewer"]["record_video"] = bool(args.record_video)
    cfg["viewer"]["record_env_idx"] = 0
    if not args.keep_perturbations:
        cfg["randomization"]["kick_interval_s"] = 1.0e9
        cfg["randomization"]["push_interval_s"] = 1.0e9
    if args.no_noise:
        cfg["noise"] = {}

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint = args.checkpoint
    if checkpoint in ("-1", -1, None, ""):
        checkpoint = find_latest_checkpoint(args.task)
        if checkpoint is None:
            raise FileNotFoundError("no .pth checkpoint found under logs/")
    print("Evaluating checkpoint: {}".format(checkpoint))

    task_class = get_task_class(args.task.split("/")[-1])
    if task_class is None:
        raise ValueError("unknown task: {}".format(args.task))
    env = task_class(cfg)
    device = cfg["basic"]["rl_device"]

    model = ActorCritic(env.num_actions, env.num_obs, env.num_privileged_obs).to(device)
    model_dict = torch.load(checkpoint, map_location=device, weights_only=True)
    load_result = model.load_state_dict(model_dict["model"], strict=False)
    if load_result.missing_keys or load_result.unexpected_keys:
        print("WARNING: partial checkpoint load ({} missing, {} unexpected keys)".format(
            len(load_result.missing_keys), len(load_result.unexpected_keys)))
    model.eval()

    obs, infos = env.reset()
    obs = obs.to(device)

    total_steps = int(duration_s / env.dt)
    pos_errs, head_errs, stop_speeds = [], [], []
    falls = 0
    censored = 0
    video_written = False

    for step_i in range(total_steps):
        prev_goal_pos = env.goal_pos_world.clone()
        prev_goal_heading = env.goal_heading_world.clone()
        with torch.no_grad():
            dist = model.act(obs)
            act = dist.sample() if args.stochastic else dist.loc
        obs, rew, done, infos = env.step(act.to(env.device))
        obs = obs.to(device)

        timeouts = infos["time_outs"].to(done.device)
        falls += (done & ~timeouts).sum().item()
        censored += (done & timeouts).sum().item()

        changed = (env.goal_pos_world != prev_goal_pos).any(dim=1) | (env.goal_heading_world != prev_goal_heading)
        completed = changed & ~done
        if completed.any():
            ids = completed.nonzero(as_tuple=False).flatten()
            d = torch.norm(env.base_pos[ids, :2] - prev_goal_pos[ids], dim=-1)
            _, _, yaw = get_euler_xyz(env.base_quat[ids])
            h = wrap(prev_goal_heading[ids] - yaw).abs()
            v = torch.norm(env.root_states[ids, 7:9], dim=-1)
            pos_errs.extend(d.cpu().tolist())
            head_errs.extend(h.cpu().tolist())
            stop_speeds.extend(v.cpu().tolist())

        if args.record_video and not video_written and (step_i + 1) * env.dt >= args.record_video_s:
            env.cfg["viewer"]["record_video"] = False  # stop accumulating frames (memory)
            video_written = True

        if (step_i + 1) % 500 == 0:
            print("  step {}/{} — segments so far: {}, falls: {}".format(step_i + 1, total_steps, len(pos_errs), falls))

    pos = np.array(pos_errs)
    head_deg = np.degrees(np.array(head_errs))
    speed = np.array(stop_speeds)
    n = len(pos)
    attempts = n + falls
    if n == 0:
        raise RuntimeError("no completed goal segments — duration too short or every env fell")

    stop_thr = cfg["rewards"].get("stop_speed_threshold", 0.1)
    success_strict = float(np.mean((pos <= gate_pos_median) & (head_deg <= gate_heading_median) & (speed <= stop_thr)))
    success_loose = float(np.mean((pos <= gate_pos_p90) & (head_deg <= gate_heading_median)))

    results = {
        "checkpoint": checkpoint,
        "task": args.task,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_envs": num_envs,
        "duration_s": duration_s,
        "deterministic": not args.stochastic,
        "perturbations": bool(args.keep_perturbations),
        "obs_noise": not args.no_noise,
        "segments_completed": n,
        "segments_censored_by_episode_end": censored,
        "falls": falls,
        "fall_rate_per_attempt": falls / attempts if attempts else 0.0,
        "pos_err_m": {
            "median": float(np.median(pos)), "p90": float(np.percentile(pos, 90)),
            "mean": float(np.mean(pos)), "max": float(np.max(pos)),
        },
        "heading_err_deg": {
            "median": float(np.median(head_deg)), "p90": float(np.percentile(head_deg, 90)),
            "mean": float(np.mean(head_deg)), "max": float(np.max(head_deg)),
        },
        "final_speed_mps": {
            "median": float(np.median(speed)), "p90": float(np.percentile(speed, 90)), "mean": float(np.mean(speed)),
        },
        "success_rate_strict": success_strict,
        "success_rate_loose": success_loose,
        "gates": {
            "pos_median": {"limit": gate_pos_median, "value": float(np.median(pos)), "pass": bool(np.median(pos) <= gate_pos_median)},
            "pos_p90": {"limit": gate_pos_p90, "value": float(np.percentile(pos, 90)), "pass": bool(np.percentile(pos, 90) <= gate_pos_p90)},
            "heading_median": {"limit": gate_heading_median, "value": float(np.median(head_deg)), "pass": bool(np.median(head_deg) <= gate_heading_median)},
            "falls": {"limit": gate_max_falls, "value": falls, "pass": bool(falls <= gate_max_falls)},
        },
    }
    results["all_gates_pass"] = all(g["pass"] for g in results["gates"].values())

    out_dir = args.out
    if not out_dir:
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint)))
        out_dir = os.path.join(run_dir, "eval", time.strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, "segments.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pos_err_m", "heading_err_deg", "final_speed_mps"])
        for row in zip(pos.tolist(), head_deg.tolist(), speed.tolist()):
            w.writerow(["{:.4f}".format(x) for x in row])

    md = []
    md.append("# GoalPose 평가 리포트 — {}".format(results["date"]))
    md.append("")
    md.append("- checkpoint: `{}`".format(checkpoint))
    md.append("- 조건: {} envs × {:.0f}s, {} 정책, 외란 {}, 관측노이즈 {}".format(
        num_envs, duration_s,
        "결정론적" if results["deterministic"] else "확률적",
        "ON" if results["perturbations"] else "OFF",
        "ON" if results["obs_noise"] else "OFF"))
    md.append("- 완료 구간 {}개 / 낙상 {}회 / 에피소드경계 절단 {}개".format(n, falls, censored))
    md.append("")
    md.append("## 게이트 판정 (MASTERPLAN §성공 기준)")
    md.append("")
    md.append("| 게이트 | 기준 | 측정값 | 판정 |")
    md.append("|---|---|---|---|")
    md.append("| 위치 오차 median | ≤ {:.0f} cm | {:.1f} cm | {} |".format(
        gate_pos_median * 100, np.median(pos) * 100, "✅ PASS" if results["gates"]["pos_median"]["pass"] else "❌ FAIL"))
    md.append("| 위치 오차 p90 | ≤ {:.0f} cm | {:.1f} cm | {} |".format(
        gate_pos_p90 * 100, np.percentile(pos, 90) * 100, "✅ PASS" if results["gates"]["pos_p90"]["pass"] else "❌ FAIL"))
    md.append("| heading 오차 median | ≤ {:.0f}° | {:.1f}° | {} |".format(
        gate_heading_median, np.median(head_deg), "✅ PASS" if results["gates"]["heading_median"]["pass"] else "❌ FAIL"))
    md.append("| 낙상 | ≤ {} | {} | {} |".format(
        gate_max_falls, falls, "✅ PASS" if results["gates"]["falls"]["pass"] else "❌ FAIL"))
    md.append("")
    md.append("**종합: {}**".format("✅ 전체 게이트 통과" if results["all_gates_pass"] else "❌ 미통과 게이트 있음"))
    md.append("")
    md.append("## 부가 지표")
    md.append("")
    md.append("- 도착 시 속도 median {:.2f} m/s (정지 기준 {:.2f} m/s)".format(np.median(speed), stop_thr))
    md.append("- 성공률(엄격: {:.0f}cm+{:.0f}°+정지): {:.1f}%".format(gate_pos_median * 100, gate_heading_median, success_strict * 100))
    md.append("- 성공률(완화: {:.0f}cm+{:.0f}°): {:.1f}%".format(gate_pos_p90 * 100, gate_heading_median, success_loose * 100))
    md.append("")
    md.append("## 다음에 확인/시도할 것")
    md.append("")
    if not results["gates"]["pos_median"]["pass"] or not results["gates"]["pos_p90"]["pass"]:
        md.append("- 위치 오차 미달 → `goal_progress` 보상(현재 0)을 0.5~1.0으로 켜서 원거리 유인 강화, 또는 `goal_position_sigma` 축소(0.5)로 근거리 정밀도 강화")
    if not results["gates"]["heading_median"]["pass"]:
        md.append("- heading 미달 → `heading_near_goal` 보상(현재 0)으로 교체(`goal_heading`은 0으로), 안 되면 sin/cos 2채널 업그레이드(MASTERPLAN 참고)")
    if not results["gates"]["falls"]["pass"]:
        md.append("- 낙상 발생 → push/kick 강도·빈도 확인, `terminate_height`/보상 균형 점검; 낙상 시점 영상 확인(`--record_video`)")
    if np.median(speed) > stop_thr:
        md.append("- 도착 후 안 멈춤 → `goal_stop` 스케일 강화(-1→-3) 또는 `goal_reached`(+) 켜기")
    if results["all_gates_pass"]:
        md.append("- 게이트 통과 → 영상 확인(`--record_video`) 후 milestone 4(export + MuJoCo 검증)로 진행")
    md.append("- 눈으로 확인: `python eval_goal_pose.py ... --record_video` (env 0을 mp4로 저장) 또는 로컬에서 `play.py`")

    report_md = "\n".join(md)
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(report_md + "\n")

    if args.record_video and hasattr(env, "camera_frames") and len(env.camera_frames) > 0:
        import imageio
        video_path = os.path.join(out_dir, "rollout_env0.mp4")
        with imageio.get_writer(video_path, fps=int(1.0 / env.dt)) as writer:
            for frame in env.camera_frames:
                writer.append_data(frame)
        print("video written: {}".format(video_path))

    print("")
    print(report_md)
    print("")
    print("report saved to: {}".format(out_dir))


if __name__ == "__main__":
    main()
