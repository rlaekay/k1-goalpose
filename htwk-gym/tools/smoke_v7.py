"""Numerically verify the v7 machinery before committing GPU-days to it.

v7 was written without ever being executed (no GPU on the authoring machine), so
every new mechanism here is unproven. These checks are chosen so that each one
fails LOUDLY for a specific bug rather than producing plausible-looking garbage:

  * path mode advancing more than once per control step (the goal would move at
    a multiple of the commanded speed and the speed curriculum would calibrate
    against a number that was never real)
  * the leash not holding (goal runs away, gradient dies)
  * goal_segment_id ticking every step (eval would record one "segment" per
    control step and every aggregate would be meaningless)
  * the two disturbance classes not firing, or firing at the wrong magnitude
  * the arms-down URDF changing the DOF/observation layout (would break warm start)
  * any reward returning NaN/inf

Usage:
    python tools/smoke_v7.py --sim_device cuda:0 --rl_device cuda:0
    python tools/smoke_v7.py --config envs/K1/Goal_Pose_V7.yaml --steps 400
Exit code 0 = safe to launch training. Non-zero = do not launch.
"""

import argparse
import os
import sys

import isaacgym  # noqa: F401  (must precede torch)
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import *  # noqa: F401,F403  (registers task classes)
from utils.model import ActorCritic
from utils.runner import get_task_class


FAILURES = []
CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    print("{}  {:<46} {}".format("PASS" if ok else "FAIL", name, detail))
    if not ok:
        FAILURES.append(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join("envs", "K1", "Goal_Pose_V7.yaml"))
    ap.add_argument("--task", default="K1/Goal_Pose_V7")
    ap.add_argument("--num_envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--sim_device", default="cuda:0")
    ap.add_argument("--rl_device", default="cuda:0")
    ap.add_argument("--checkpoint",
                    default="logs/K1/K1/Goal_Pose/2026-07-24-17-22-03_armB_goal_reached/nn/model_11500.pth",
                    help="drive the env with this policy. With zero actions the robot "
                         "falls every second and the constant resets make the segment-rate "
                         "check meaningless. Pass '' to force zero actions.")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    cfg["basic"]["task"] = args.task
    cfg["basic"]["headless"] = True
    cfg["basic"]["sim_device"] = args.sim_device
    cfg["basic"]["rl_device"] = args.rl_device
    cfg["env"]["num_envs"] = args.num_envs
    cfg["viewer"]["record_video"] = False

    torch.manual_seed(0)
    np.random.seed(0)

    env = get_task_class(args.task.split("/")[-1])(cfg)
    obs, _ = env.reset()

    model = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        device = cfg["basic"]["rl_device"]
        model = ActorCritic(env.num_actions, env.num_obs, env.num_privileged_obs).to(device)
        sd = torch.load(args.checkpoint, map_location=device, weights_only=True)
        res = model.load_state_dict(sd["model"], strict=False)
        model.eval()
        # A shape mismatch here is the warm-start canary: it means v7 moved the
        # obs/action layout and every armB weight is being silently discarded.
        check("warm-start checkpoint loads cleanly",
              not res.missing_keys and not res.unexpected_keys,
              "{} missing / {} unexpected".format(len(res.missing_keys), len(res.unexpected_keys)))
        print("     driving with policy: {}".format(args.checkpoint))
    else:
        if args.checkpoint:
            print("NOTE  checkpoint not found, falling back to zero actions: {}".format(args.checkpoint))
        print("     driving with ZERO actions (robot will fall often; segment-rate check relaxed)")

    # ---- asset / layout: must not have moved, or warm start is dead ---------
    check("URDF loads, 12 actuated DOFs", env.num_dofs == 12, "num_dofs={}".format(env.num_dofs))
    check("observation width unchanged (54)", env.num_obs == 54, "num_obs={}".format(env.num_obs))
    check("obs finite after reset", bool(torch.isfinite(obs).all()))
    urdf = cfg["asset"]["file"]
    check("using the arms-down URDF", "armsdown" in urdf, urdf)

    pcfg = cfg["commands"].get("path", {})
    share = cfg["commands"].get("goal_mode_mixture", {}).get("path", 0.0)
    n_path = int(env.is_path_env.sum().item())
    frac = n_path / float(env.num_envs)
    check("path-mode share matches config",
          abs(frac - share) < 0.12 or share == 0.0,
          "configured {:.2f}, got {:.2f} ({} envs)".format(share, frac, n_path))

    # ---- run ---------------------------------------------------------------
    gaps, seg_ticks, rew_bad = [], 0, 0
    goal_step_move = []
    push_f_seen, push_t_seen, push_active_steps = [], [], 0
    prev_seg = env.goal_segment_id.clone() if hasattr(env, "goal_segment_id") else None
    prev_goal = env.goal_pos_world.clone()
    path_mask = env.is_path_env.clone()

    falls = 0
    for i in range(args.steps):
        if model is not None:
            with torch.no_grad():
                act = model.act(obs.to(cfg["basic"]["rl_device"])).loc.to(env.device)
        else:
            act = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        obs, rew, done, infos = env.step(act)
        falls += int((done & ~infos["time_outs"].to(done.device)).sum().item())

        if not torch.isfinite(rew).all() or not torch.isfinite(obs).all():
            rew_bad += 1

        if hasattr(env, "goal_segment_id"):
            seg_ticks += int((env.goal_segment_id != prev_seg).sum().item())
            prev_seg = env.goal_segment_id.clone()

        alive = ~done
        if path_mask.any():
            m = path_mask & alive
            if bool(m.any()):
                gaps.append(torch.norm(
                    env.goal_pos_world[m] - env.base_pos[m, :2], dim=-1).cpu().numpy())
                moved = torch.norm(env.goal_pos_world[m] - prev_goal[m], dim=-1)
                goal_step_move.append(moved.cpu().numpy())
        prev_goal = env.goal_pos_world.clone()

        f = torch.norm(env.pushing_forces[:, env.base_indice, :], dim=-1)
        t = torch.norm(env.pushing_torques[:, env.base_indice, :], dim=-1)
        act_mask = f > 1e-3
        if bool(act_mask.any()):
            push_active_steps += 1
            push_f_seen.append(f[act_mask].cpu().numpy())
            push_t_seen.append(t[act_mask].cpu().numpy())

    check("rewards and observations stay finite", rew_bad == 0,
          "{} bad steps".format(rew_bad))

    # ---- path mode ---------------------------------------------------------
    if n_path > 0 and goal_step_move and gaps:
        move = np.concatenate(goal_step_move)
        move = move[move < 1.0]                     # drop re-roll/reset teleports
        speed_lo, speed_hi = pcfg.get("speed_range_mps", [0.3, 1.6])
        lvl = getattr(env, "speed_level", 1.0)
        cmd_hi = speed_lo + (speed_hi - speed_lo) * lvl
        observed = np.percentile(move, 99) / env.dt if len(move) else float("nan")
        # THE key check: if _advance_paths runs N times per control step the goal
        # travels N*path_speed*dt per step and this lands at N x the ceiling.
        check("path goal speed <= commanded ceiling (no double-advance)",
              observed <= cmd_hi * 1.35,
              "observed p99 {:.2f} m/s vs ceiling {:.2f} m/s (level {:.2f})".format(
                  observed, cmd_hi, lvl))

        g = np.concatenate(gaps)
        leash = pcfg.get("lookahead_max_m", 3.5)
        check("leash holds the lookahead point",
              np.percentile(g, 99) <= leash * 1.25,
              "gap p50 {:.2f} p99 {:.2f} m, leash {:.2f}".format(
                  np.percentile(g, 50), np.percentile(g, 99), leash))
    elif n_path > 0:
        check("path-mode samples collected", False,
              "no surviving path envs in {} steps -- policy may be falling immediately".format(args.steps))

    # ---- segment accounting ------------------------------------------------
    if hasattr(env, "goal_segment_id"):
        per_env_per_step = seg_ticks / float(args.steps * env.num_envs)
        lo, hi = cfg["commands"]["resampling_time_s"]
        expected = env.dt / (0.5 * (lo + hi))
        # Every reset also starts a new segment, so a policy that falls a lot
        # legitimately raises this rate; only the flooding case (~1.0) is a bug.
        tol = 5.0 if model is not None else 50.0
        check("goal_segment_id ticks at the resample rate, not every step",
              per_env_per_step < min(0.5, tol * expected),
              "{:.4f}/env/step (resample ~{:.4f}, {} falls); 1.0 = eval floods".format(
                  per_env_per_step, expected, falls))

    # ---- disturbance -------------------------------------------------------
    d = cfg["randomization"].get("disturbance", {})
    if d.get("enabled"):
        check("disturbance fires", push_active_steps > 0,
              "active on {}/{} steps".format(push_active_steps, args.steps))
        if push_f_seen:
            fv = np.concatenate(push_f_seen)
            tv = np.concatenate(push_t_seen)
            cmax = d["collision"]["force_n"][1]
            smin = d["support"]["force_n"][0]
            check("collision-class magnitudes present",
                  fv.max() >= d["collision"]["force_n"][0] * 0.8,
                  "max {:.1f} N (collision band {}-{} N)".format(
                      fv.max(), *d["collision"]["force_n"]))
            check("support-class magnitudes present",
                  fv.min() <= d["support"]["force_n"][1] * 1.5,
                  "min {:.1f} N (support band {}-{} N)".format(
                      fv.min(), *d["support"]["force_n"]))
            check("no force beyond the configured ceiling",
                  fv.max() <= cmax * 1.5, "max {:.1f} N vs ceiling {:.1f}".format(fv.max(), cmax))
            check("torque applied with force", tv.max() > 0.0, "max {:.2f} N*m".format(tv.max()))
            print("     force  p50 {:.1f}  p90 {:.1f}  max {:.1f} N".format(
                np.percentile(fv, 50), np.percentile(fv, 90), fv.max()))
            _ = smin

    # ---- reward wiring -----------------------------------------------------
    missing = [n for n, s in cfg["rewards"]["scales"].items()
               if float(s) != 0.0 and not hasattr(env, "_reward_" + n)]
    check("every nonzero reward scale has an implementation", not missing, str(missing))

    print("\n{}/{} checks passed".format(len(CHECKS) - len(FAILURES), len(CHECKS)))
    if FAILURES:
        print("FAILED: " + ", ".join(FAILURES))
        print("\nDo NOT launch training until these are resolved.")
        return 1
    print("v7 machinery verified — safe to launch training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
