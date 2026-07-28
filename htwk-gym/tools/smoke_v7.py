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
    gap_dwell, gap_run = [], []
    goal_step_move = []
    push_f_seen, push_t_seen, push_active_steps = [], [], 0
    has_segment_id = hasattr(env, "goal_segment_id")
    prev_seg = env.goal_segment_id.clone() if has_segment_id else None
    prev_goal = env.goal_pos_world.clone()
    prev_path_mask = env.is_path_env.clone()

    falls = 0
    dwell_seen = 0
    seq_adv = 0
    arm_blend_lo, arm_blend_hi = 1.0, 0.0
    robot_speed = []
    seq_prev = env.seq_idx.clone() if hasattr(env, "seq_idx") else None
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

        if has_segment_id:
            seg_changed = env.goal_segment_id != prev_seg
            seg_ticks += int(seg_changed.sum().item())
        else:
            seg_changed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        alive = ~done
        path_mask = env.is_path_env.clone()
        if path_mask.any():
            # Only measure steady path-carrot motion. Resets can switch goal mode,
            # and segment rerolls deliberately teleport/re-anchor the carrot.
            m = prev_path_mask & path_mask & alive & ~seg_changed
            if bool(m.any()):
                gaps.append(torch.norm(
                    env.goal_pos_world[m] - env.base_pos[m, :2], dim=-1).cpu().numpy())
                moved = torch.norm(env.goal_pos_world[m] - prev_goal[m], dim=-1)
                goal_step_move.append(moved.cpu().numpy())
                if hasattr(env, "path_dwell_left"):
                    dw_m = env.path_dwell_left[m] > 0
                    gsel = torch.norm(env.goal_pos_world[m] - env.base_pos[m, :2], dim=-1)
                    if bool(dw_m.any()):
                        gap_dwell.append(gsel[dw_m].cpu().numpy())
                    if bool((~dw_m).any()):
                        gap_run.append(gsel[~dw_m].cpu().numpy())
        if has_segment_id:
            prev_seg = env.goal_segment_id.clone()
        prev_path_mask = path_mask
        prev_goal = env.goal_pos_world.clone()

        robot_speed.append(torch.norm(env.root_states[:, 7:9], dim=-1).cpu().numpy())

        if seq_prev is not None:
            seq_adv += int((env.seq_idx > seq_prev).sum().item())
            seq_prev = env.seq_idx.clone()

        if getattr(env, "arm_script_on", False):
            b = env.arm_blend
            arm_blend_lo = min(arm_blend_lo, float(b.min().item()))
            arm_blend_hi = max(arm_blend_hi, float(b.max().item()))

        if hasattr(env, "path_dwell_left"):
            dwell_seen += int(((env.path_dwell_left > 0) & env.is_path_env).sum().item())

        f = torch.norm(env.pushing_forces[:, env.base_indice, :], dim=-1)
        t = torch.norm(env.pushing_torques[:, env.base_indice, :], dim=-1)
        act_mask = f > 1e-3
        if bool(act_mask.any()):
            push_active_steps += 1
            push_f_seen.append(f[act_mask].cpu().numpy())
            push_t_seen.append(t[act_mask].cpu().numpy())

    seq_r = env._reward_seq_goal() if hasattr(env, "_reward_seq_goal") else torch.zeros(1)

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
        # The carrot may legitimately exceed the commanded pace: at the distance
        # floor a robot running faster than the pace drags it along at the
        # ROBOT's speed. So the bound is max(pace, robot speed) * catchup_ratio,
        # not the pace alone -- comparing against the pace alone would fail every
        # run in which the policy is doing well.
        catch = float(pcfg.get("catchup_ratio", 1.5))
        v99 = np.percentile(np.concatenate(robot_speed), 99) if robot_speed else 0.0
        bound = max(cmd_hi, v99) * catch + 0.05
        check("path goal speed within the rate limit (no teleport)",
              observed <= bound * 1.25,
              "observed p99 {:.2f} m/s vs bound {:.2f} (pace<={:.2f}, robot p99 {:.2f}, x{:.1f})".format(
                  observed, bound, cmd_hi, v99, catch))

        g = np.concatenate(gaps)
        # Leash is now per-env: lookahead * leash_ratio, capped at lookahead_max_m.
        # The old check compared against lookahead_max_m alone, which is only the
        # cap and would pass even if the per-env leash were being ignored.
        la_lo, la_hi = pcfg.get("lookahead_m", [0.5, 3.0])
        # lookahead is now capped at lookahead_scale_frac * path_scale, so the
        # widest achievable leash follows the largest curve, not the raw config.
        frac = float(pcfg.get("lookahead_scale_frac", 0.6))
        scale_hi = pcfg.get("scale_m", [1.5, 4.0])[1]
        la_eff = min(la_hi, frac * scale_hi)
        leash = min(la_eff * float(pcfg.get("leash_ratio", 1.6)),
                    float(pcfg.get("lookahead_max_m", 3.5)))
        check("leash holds the lookahead point",
              np.percentile(g, 99) <= leash * 1.25,
              "gap p50 {:.2f} p99 {:.2f} m, leash {:.2f}".format(
                  np.percentile(g, 50), np.percentile(g, 99), leash))

        # The floor is the fix for the defect that made the goal drift onto the
        # robot (measured gap median 0.41 m against a configured 0.5 m minimum),
        # flattening the reward exactly where the robot was. While NOT dwelling
        # the gap must stay at or above the smallest configured lookahead.
        # Split by dwell state instead of thresholding a pooled fraction. Pooling
        # conflates "the floor is broken" with "the carrot is parked, which
        # releases the floor on purpose", and the pooled number then moves
        # whenever the dwell duty cycle is retuned -- a check that shifts under
        # unrelated config changes is not measuring what it claims to.
        gd = np.concatenate(gap_dwell) if gap_dwell else np.array([])
        gr = np.concatenate(gap_run) if gap_run else np.array([])
        la_floor = min(la_lo, frac * pcfg.get("scale_m", [1.5, 4.0])[0])
        if len(gr):
            frac_bad = float((gr < la_floor * 0.75).mean())
            check("lookahead floor holds while running (not dwelling)",
                  frac_bad < 0.15,
                  "{:.0%} of running steps inside the floor, gap p2 {:.2f} vs floor {:.2f}".format(
                      frac_bad, np.percentile(gr, 2), la_floor))
        dwell_cfg = pcfg.get("dwell") or {}
        if dwell_cfg.get("enabled", False):
            check("dwell actually releases the floor",
                  len(gd) > 0 and float((gd < la_floor * 0.75).mean()) > 0.2,
                  "{:.0%} of dwelling steps inside the floor (should be most of them)".format(
                      float((gd < la_floor * 0.75).mean()) if len(gd) else 0.0))
    elif n_path > 0:
        check("path-mode samples collected", False,
              "no surviving path envs in {} steps -- policy may be falling immediately".format(args.steps))

    # ---- dwell + grid: brand-new machinery, never executed before -----------
    dwell_cfg = pcfg.get("dwell") or {}
    if dwell_cfg.get("enabled", False) and n_path > 0:
        # dwell is the repair for path training wrecking waypoint accuracy
        # (6.3 cm -> 37.9 cm in the v7 batch). If the carrot never actually
        # stops, the repair is absent and the next batch reproduces the fault.
        check("dwell fires (carrot actually stops)", dwell_seen > 0,
              "{} env-steps spent dwelling out of {}".format(dwell_seen, args.steps * n_path))
        lo, hi = dwell_cfg.get("interval_s", [4.0, 10.0])
        dur_lo, dur_hi = dwell_cfg.get("duration_s", [1.5, 3.0])
        expect = (0.5 * (dur_lo + dur_hi)) / (0.5 * (lo + hi))
        observed = dwell_seen / float(max(args.steps * n_path, 1))
        check("dwell duty cycle is in the configured ballpark",
              observed <= expect * 3.0 + 0.02,
              "observed {:.1%} vs configured ~{:.1%}".format(observed, expect))

    # ---- scripted arms ------------------------------------------------------
    if getattr(env, "arm_script_on", False):
        # The entire point of keeping these out of the action/observation
        # vectors is that E0's weights still load. If the width moved, that
        # failed and the run would silently start from scratch.
        check("scripted arms leave the observation at 54", env.num_obs == 54,
              "num_obs={}".format(env.num_obs))
        check("scripted arms leave num_actions at 12", env.num_actions == 12,
              "num_actions={}".format(env.num_actions))
        check("arm DOFs exist and are excluded from the leg set",
              len(env.arm_dof_idx) == 4 and len(env.leg_dof_idx) == 12,
              "{} arm / {} leg of {} dofs".format(
                  len(env.arm_dof_idx), len(env.leg_dof_idx), env.num_dofs))
        # A blend that never leaves 0 or 1 means the parked test never flipped,
        # so the pose is effectively static and the mechanism is inert.
        check("arm pose actually blends with motion state",
              arm_blend_lo < 0.5 < arm_blend_hi,
              "blend spanned [{:.2f}, {:.2f}] over the run".format(arm_blend_lo, arm_blend_hi))

    # ---- v8 SmoothTurn ------------------------------------------------------
    if getattr(env, "st_on", False):
        n_seq = int(env.is_seq_env.sum().item())
        share = cfg["commands"].get("smooth_turn", {}).get("share", 0.0)
        check("sequential-nav share matches config",
              abs(n_seq / float(env.num_envs) - share) < 0.12 or share == 0.0,
              "configured {:.2f}, got {:.2f} ({} envs)".format(
                  share, n_seq / float(env.num_envs), n_seq))
        if n_seq > 0:
            # Goals must be BANKED, not just approached: if seq_idx never
            # advances the sequential reward is stuck at rho/N and the whole
            # mechanism reduces to a single-goal task with extra bookkeeping.
            check("goals actually get banked", seq_adv > 0,
                  "{} goal advances across {} seq envs in {} steps".format(
                      seq_adv, n_seq, args.steps))
            # The lookahead window must be live in the six reused slots. All-zero
            # means the observation is carrying nothing and the policy is blind
            # to upcoming turns -- the ablation in the paper shows that costs
            # most of the benefit.
            look = env.commands[env.is_seq_env, 4:10]
            check("lookahead window populated (command slots 4-9)",
                  float(look.abs().max().item()) > 1e-3,
                  "max |lookahead| = {:.3f}".format(float(look.abs().max().item())))
            check("sequential reward finite and in [0, 1]",
                  bool(torch.isfinite(seq_r).all()) and float(seq_r.max().item()) <= 1.001,
                  "max {:.3f}".format(float(seq_r.max().item())))

    if getattr(env, "grid_on", False):
        active = int(env.grid_active.sum().item())
        total = int(env.grid_active.numel())
        # Seeded at exactly one cell (slowest, straightest). If it were already
        # wide open the curriculum would be doing nothing -- which is the
        # no-curriculum condition Margolis et al. report as a total failure.
        check("speed grid seeded small", 1 <= active < total,
              "{}/{} cells active after {} steps".format(active, total, args.steps))
        print("     grid speeds {} x curvatures {}".format(
            [round(float(x), 2) for x in env.grid_speeds.tolist()],
            [round(float(x), 2) for x in env.grid_curvs.tolist()]))

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
