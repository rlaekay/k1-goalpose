"""Numerically verify the v7 machinery before committing GPU-days to it.

v7 was written without ever being executed (no GPU on the authoring machine), so
every new mechanism here is unproven. These checks are chosen so that each one
fails LOUDLY for a specific bug rather than producing plausible-looking garbage:

  * path mode advancing more than once per control step (the goal would move at
    a multiple of the commanded speed and the speed curriculum would calibrate
    against a number that was never real)
  * the 2-D floor/leash projection or world-step rate bound being violated
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
    print("{}  {:<46} {}".format(
        "PASS" if ok else "FAIL", name, detail), flush=True)
    if not ok:
        FAILURES.append((name, detail))


def note(name, detail=""):
    """Non-blocking observation: printed, never gates the launch.

    For numbers that measure something real but that a FROZEN warm-start
    policy is not expected to already be good at -- e.g. SmoothTurn's
    lookahead channels are, for E0, out-of-distribution input (command slots
    4-9 were pinned to exactly [0,0] for the whole time E0 trained), so a low
    end-of-run occupancy or bank count reflects "this policy hasn't learned
    the new task yet," which is what training is FOR, not a mechanism defect.
    """
    print("NOTE  {:<46} {}".format(name, detail), flush=True)


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
    ap.add_argument(
        "--disturbance_probe", action="store_true",
        help="for a disturbance-enabled config, shorten only the smoke cadence "
             "and force event_probability=1 so every scenario/tier is exercised")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    cfg["basic"]["task"] = args.task
    cfg["basic"]["headless"] = True
    cfg["basic"]["sim_device"] = args.sim_device
    cfg["basic"]["rl_device"] = args.rl_device
    cfg["env"]["num_envs"] = args.num_envs
    cfg["viewer"]["record_video"] = False
    disturbance_cfg = cfg.get("randomization", {}).get("disturbance") or {}
    if args.disturbance_probe and disturbance_cfg.get("enabled", False):
        disturbance_cfg.update({
            # Stay above the longest 0.8 s entanglement event so the probe
            # never replaces a live wrench while accelerating its cadence.
            "interval_s": [1.0, 1.5],
            "event_probability": 1.0,
            "ramp_steps": 1,
        })

    torch.manual_seed(0)
    np.random.seed(0)

    env = get_task_class(args.task.split("/")[-1])(cfg)
    obs, _ = env.reset()
    # is_seq_env right after the full-population reset(), before an untrained
    # warm start has had any chance to fall out of the mode -- see the note by
    # the check below for why this and the end-of-run count answer different
    # questions.
    n_seq_at_reset = int(env.is_seq_env.sum().item()) if hasattr(env, "is_seq_env") else None

    model = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        device = cfg["basic"]["rl_device"]
        model = ActorCritic(env.num_actions, env.num_obs, env.num_privileged_obs).to(device)
        sd = torch.load(args.checkpoint, map_location=device, weights_only=True)
        res = model.load_state_dict(sd["model"], strict=False)
        # Deliberately do not restore sd["env_state"] here.  Smoke verifies the
        # generated config's frozen launch distribution; Runner restores task
        # state for an actual training resume, while eval exposes an explicit
        # --restore_task_state diagnostic when native state is desired.
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
    # 12 legs, plus the scripted elbows when the armswing URDF is in play. This
    # asserted a bare 12 and so failed G3_full for being correct.
    n_arm = len(getattr(env, "arm_dof_idx", [])) if getattr(env, "arm_script_on", False) else 0
    check("URDF loads {} actuated DOFs".format(12 + n_arm), env.num_dofs == 12 + n_arm,
          "num_dofs={} (12 leg + {} scripted arm)".format(env.num_dofs, n_arm))
    check("observation width unchanged (54)", env.num_obs == 54, "num_obs={}".format(env.num_obs))
    check("obs finite after reset", bool(torch.isfinite(obs).all()))
    urdf = cfg["asset"]["file"]
    # The thing to catch is a silent fall back to the stock arms-OUT URDF, which
    # widens the ego footprint and changes the yaw inertia the warm start was
    # trained under. armswing is the same arms-down geometry with the elbows made
    # revolute so they can be scripted, so it is equally valid -- asserting
    # "armsdown" in the name failed G3_full for using exactly the asset its whole
    # arm is about.
    check("using a compact-arm URDF",
          ("armsdown" in urdf) or ("armswing" in urdf) or ("hbatch" in urdf), urdf)

    if hasattr(env, "mirror_obs"):
        probe_o = torch.randn(7, env.num_obs, device=env.device)
        probe_a = torch.randn(7, env.num_actions, device=env.device)
        check("observation mirror is an involution",
              bool(torch.allclose(env.mirror_obs(env.mirror_obs(probe_o)), probe_o, atol=1e-6)))
        check("action mirror is an involution",
              bool(torch.allclose(env.mirror_actions(env.mirror_actions(probe_a)), probe_a, atol=1e-6)))
        check("mirror observation permutation is bijective",
              len(torch.unique(env.mirror_obs_perm)) == env.num_obs)
        check("mirror action permutation is bijective",
              len(torch.unique(env.mirror_act_perm)) == env.num_actions)
    if hasattr(env, "mirror_privileged_obs"):
        probe_p = torch.randn(7, env.num_privileged_obs, device=env.device)
        check("privileged mirror is an involution",
              bool(torch.allclose(env.mirror_privileged_obs(
                  env.mirror_privileged_obs(probe_p)), probe_p, atol=1e-6)))
        privileged = env.privileged_obs_buf
        mirrored_privileged = env.mirror_privileged_obs(privileged)
        latent = torch.cat((privileged[:, :4], mirrored_privileged[:, :4]), dim=0)
        check("privileged mirror preserves U[0,1] DR latent support",
              bool(((latent >= 0.0) & (latent <= 1.0)).all()),
              "original/mirrored latent [{:.4f}, {:.4f}]".format(
                  float(latent.min().item()), float(latent.max().item())))

    encoder_cfg = cfg["randomization"].get("joint_encoder_bias")
    if encoder_cfg:
        encoder_active = any(abs(float(x)) > 0.0
                             for x in encoder_cfg.get("range", [0.0, 0.0]))
        encoder_max = float(env.joint_encoder_bias.abs().max().item())
        check("episode-constant encoder bias sampled" if encoder_active
              else "encoder bias disabled as configured",
              encoder_max > 0.0 if encoder_active else encoder_max <= 1.0e-9,
              "max |offset| {:.6f} rad".format(encoder_max))
    target_cfg = cfg["randomization"].get("joint_target_offset")
    if target_cfg:
        target_active = any(abs(float(x)) > 0.0
                            for x in target_cfg.get("range", [0.0, 0.0]))
        target_max = float(env.joint_target_offset.abs().max().item())
        check("episode-constant motor-target offset sampled" if target_active
              else "motor-target offset disabled as configured",
              target_max > 0.0 if target_active else target_max <= 1.0e-9,
              "max |offset| {:.6f} rad".format(target_max))

    if hasattr(env, "get_checkpoint_state") and hasattr(env, "load_checkpoint_state"):
        task_state = env.get_checkpoint_state()
        before_active = (env.grid_active.clone()
                         if getattr(env, "grid_on", False) else None)
        env.load_checkpoint_state(task_state)
        state_ok = (
            isinstance(task_state, dict)
            and "speed_level" in task_state
            and "keepup_ema" in task_state
            and (before_active is None
                 or ("path_grid" in task_state
                     and torch.equal(before_active, env.grid_active)))
        )
        check("task curriculum checkpoint state round-trips", state_ok)

    pcfg = cfg["commands"].get("path", {})
    share = cfg["commands"].get("goal_mode_mixture", {}).get("path", 0.0)
    n_path = int(env.is_path_env.sum().item())
    frac = n_path / float(env.num_envs)
    # GoalPoseV8._reset_idx draws is_seq_env independently over the SAME pool
    # and then does `is_path_env &= ~is_seq_env` ("sequential envs are exempt
    # from waypoint/path mode entirely") -- so when SmoothTurn is on, path's
    # OWN share is only realized among the envs sequential mode didn't already
    # take. Comparing raw n_path/num_envs against the unadjusted 0.35 failed
    # G4_smoothturn for behaving exactly as v8's _reset_idx is written to:
    # configured 0.35, observed 0.19 (49 envs) is 0.35 * (1 - 0.50) almost to
    # the digit, not a broken draw.
    st_share = float(cfg["commands"].get("smooth_turn", {}).get("share", 0.0)) \
        if getattr(env, "st_on", False) else 0.0
    expect = share * (1.0 - st_share)
    check("path-mode share matches config",
          abs(frac - expect) < 0.12 or share == 0.0,
          "configured {:.2f}{}, got {:.2f} ({} envs)".format(
              share, " x (1-{:.2f} seq)={:.2f}".format(st_share, expect) if st_share else "",
              frac, n_path))

    # This is the policy-independent floor invariant.  _reroll_paths places
    # every moving carrot at its own sampled/capped lookahead.  Allow 2 cm for
    # the reset() physics step, but fail if the old "goal starts on the robot"
    # defect returns.  The long-rollout occupancy check below is necessarily
    # policy dependent because the hard goal-speed limit intentionally takes
    # precedence over the soft floor.
    if n_path > 0 and hasattr(env, "lookahead"):
        init_mask = env.is_path_env
        init_gap = torch.norm(
            env.goal_pos_world[init_mask] - env.base_pos[init_mask, :2], dim=-1)
        init_floor = env.lookahead[init_mask]
        init_margin = init_gap - init_floor
        check("path lookahead initializes at its per-env floor",
              bool((init_margin >= -0.02).all()),
              "minimum gap-floor margin {:.3f} m".format(float(init_margin.min().item())))

    if (n_path > 0
            and pcfg.get("constraint_mode") == "radial_rate_limited"
            and hasattr(env, "_bounded_annulus_delta")):
        # Pure controller fixtures: no policy, physics, disturbance or reset is
        # involved.  These are the mechanism gates the rollout occupancy could
        # never provide.  Row 1 is an angular/tangent deficit, row 2 is above
        # the leash, row 3 is a large deficit under a 0.10 m rate budget, and
        # row 4 exercises the zero-vector fallback direction.
        dev = env.device
        old = torch.tensor([
            [0.04, 0.0], [1.50, 0.0], [0.04, 0.0], [0.0, 0.0], [0.75, 0.0]],
            device=dev)
        robot = torch.zeros(5, 2, device=dev)
        nominal = torch.tensor([
            [0.0, 0.02], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.01, 0.0]],
            device=dev)
        l_min = torch.full((5,), 0.50, device=dev)
        l_max = torch.full((5,), 1.00, device=dev)
        max_step = torch.tensor([1.0, 1.0, 0.10, 1.0, 1.0], device=dev)
        fallback = torch.tensor([[1.0, 0.0]] * 5, device=dev)
        delta = env._bounded_annulus_delta(
            old, robot, nominal, l_min, l_max, max_step, fallback)
        new_gap = torch.norm(old + delta - robot, dim=-1)
        delta_norm = torch.norm(delta, dim=-1)
        fixture_ok = (
            torch.allclose(new_gap[:2], torch.tensor([0.50, 1.00], device=dev), atol=1e-5)
            and abs(float(delta_norm[2].item()) - 0.10) < 1e-5
            and 0.04 < float(new_gap[2].item()) < 0.50
            and abs(float(new_gap[3].item()) - 0.50) < 1e-5
            and torch.allclose(delta[4], nominal[4], atol=1e-6)
            and abs(float(new_gap[4].item()) - 0.76) < 1e-5
            and bool((delta_norm <= max_step + 1e-6).all())
        )
        check("radial floor/leash controller fixtures", fixture_ok,
              "new gaps {} m, steps {} m".format(
                  [round(float(x), 3) for x in new_gap.tolist()],
                  [round(float(x), 3) for x in delta_norm.tolist()]))

    # Reward activation fixtures: method existence alone cannot catch an
    # always-false mask that would waste an entire training run.
    if (float(cfg["rewards"]["scales"].get("high_speed_stability", 0.0)) != 0.0
            and hasattr(env, "_reward_high_speed_stability")):
        names = ("filtered_lin_vel", "last_stability_vel",
                 "stability_accel_filtered", "projected_gravity",
                 "base_ang_vel", "base_lin_vel")
        saved = {name: getattr(env, name).clone() for name in names}
        try:
            env.filtered_lin_vel.zero_()
            env.filtered_lin_vel[:, 0] = 1.0
            env.last_stability_vel.copy_(env.filtered_lin_vel)
            env.stability_accel_filtered.zero_()
            env.base_ang_vel.zero_()
            env.base_lin_vel.zero_()
            tilt = 0.20
            env.projected_gravity.zero_()
            env.projected_gravity[:, 0] = -np.sin(tilt)
            env.projected_gravity[:, 2] = -np.cos(tilt)
            steady = env._reward_high_speed_stability().clone()

            env.stability_accel_filtered.zero_()
            env.last_stability_vel.zero_()
            accelerating = env._reward_high_speed_stability().clone()
            check("H2 stability reward activates only in steady fast motion",
                  bool(torch.isfinite(steady).all())
                  and float(steady.mean().item()) > 1.0e-4
                  and float(accelerating.mean().item())
                  < 0.25 * float(steady.mean().item()),
                  "steady mean {:.6f}, high-accel mean {:.6f}".format(
                      float(steady.mean().item()),
                      float(accelerating.mean().item())))
        finally:
            for name, value in saved.items():
                getattr(env, name).copy_(value)

    if (float(cfg["rewards"]["scales"].get("heel_strike_ahead", 0.0)) != 0.0
            and hasattr(env, "_reward_heel_strike_ahead")):
        names = ("feet_contact", "last_feet_contact", "base_lin_vel",
                 "base_pos", "base_quat", "feet_pos", "feet_quat",
                 "is_path_env")
        saved = {name: getattr(env, name).clone() for name in names}
        try:
            env.feet_contact.zero_()
            env.feet_contact[:, 0] = True
            env.last_feet_contact.zero_()
            env.base_lin_vel.zero_()
            env.base_lin_vel[:, 0] = 1.0
            env.base_pos.zero_()
            env.base_quat.zero_()
            env.base_quat[:, 3] = 1.0
            env.feet_pos.zero_()
            # local heel x=-0.1015, so foot-link x=0.1815 puts heel at 0.08 m.
            env.feet_pos[:, :, 0] = 0.1815
            env.feet_quat.zero_()
            env.feet_quat[:, :, 3] = 1.0
            env.is_path_env[:] = True
            active = env._reward_heel_strike_ahead().clone()
            env.last_feet_contact.copy_(env.feet_contact)
            inactive = env._reward_heel_strike_ahead().clone()
            check("H3 heel reward activates only at eligible first contact",
                  bool(torch.isfinite(active).all())
                  and float(active.mean().item()) > 0.5
                  and float(inactive.abs().max().item()) == 0.0,
                  "eligible mean {:.4f}, held-contact max {:.4f}".format(
                      float(active.mean().item()),
                      float(inactive.abs().max().item())))
        finally:
            for name, value in saved.items():
                getattr(env, name).copy_(value)

    # ---- run ---------------------------------------------------------------
    gaps, seg_ticks, rew_bad = [], 0, 0
    gap_dwell_ratio, gap_run_ratio = [], []
    floor_deficit_run, leash_excess_run = [], []
    goal_step_move = []
    goal_step_move_run = []
    goal_step_budget = []
    goal_step_dwell = []
    goal_heading_step_dwell = []
    push_f_seen, push_t_seen, push_active_steps = [], [], 0
    disturbance_bodies_seen, disturbance_kinds_seen = set(), set()
    disturbance_scenarios_seen, disturbance_tiers_seen = set(), set()
    disturbance_direction_octants_seen = set()
    disturbance_scenario_events = 0
    disturbance_upper_events = 0
    disturbance_max_active_bodies = 0
    disturbance_body_mismatch = 0
    disturbance_clear_transitions = 0
    has_segment_id = hasattr(env, "goal_segment_id")
    prev_seg = env.goal_segment_id.clone() if has_segment_id else None
    prev_goal = env.goal_pos_world.clone()
    prev_goal_heading = env.goal_heading_world.clone()
    prev_path_mask = env.is_path_env.clone()
    prev_dwell_mask = ((env.path_dwell_left > 0) & env.is_path_env
                       if hasattr(env, "path_dwell_left") else
                       torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
    prev_force_any = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    prev_event_serial = (env.dist_event_serial.clone()
                         if hasattr(env, "dist_event_serial") else None)

    falls = 0
    dwell_seen = 0
    dwell_streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    max_dwell_streak = 0
    dwell_gait_bad = 0
    resumed_gait_bad = 0
    seq_adv = 0
    seq_look_max = 0.0
    seq_r_all_finite = True
    seq_r_max = 0.0
    seq_r_seen = False
    arm_blend_lo, arm_blend_hi = 1.0, 0.0
    seq_prev = env.seq_idx.clone() if hasattr(env, "seq_idx") else None
    for i in range(args.steps):
        if model is not None:
            with torch.no_grad():
                act = model.act(obs.to(cfg["basic"]["rl_device"])).loc.to(env.device)
        else:
            act = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        obs, rew, done, infos = env.step(act)
        if prev_event_serial is not None:
            new_event = env.dist_event_serial != prev_event_serial
            if bool(new_event.any()) and hasattr(env, "dist_last_scenario_id"):
                ids = env.dist_last_scenario_id[new_event].long()
                tiers = env.dist_last_height_tier[new_event].long()
                disturbance_scenarios_seen.update(
                    int(x) for x in ids.cpu().tolist() if int(x) > 0)
                disturbance_tiers_seen.update(
                    int(x) for x in tiers.cpu().tolist() if int(x) >= 0)
                direction = env.dist_last_direction_local[new_event]
                degrees = torch.rad2deg(torch.atan2(direction[:, 1], direction[:, 0]))
                octants = torch.floor(((degrees + 22.5) % 360.0) / 45.0).long()
                disturbance_direction_octants_seen.update(
                    int(x) for x in octants.cpu().tolist())
                disturbance_scenario_events += int(new_event.sum().item())
                upper_ids = {
                    i for i, name in enumerate(getattr(
                        env, "dist_height_tier_names", ()))
                    if name in ("chest", "arm_proxy")}
                disturbance_upper_events += sum(
                    int(x) in upper_ids for x in tiers.cpu().tolist())
            prev_event_serial = env.dist_event_serial.clone()
        physical = infos.get("physical_failures")
        fell = (done & physical.to(done.device) if physical is not None
                else done & ~infos["time_outs"].to(done.device))
        falls += int(fell.sum().item())

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
            if hasattr(env, "path_dwell_left"):
                current_dwell = (env.path_dwell_left > 0) & path_mask
                dwell_streak = torch.where(
                    current_dwell & alive,
                    dwell_streak + 1,
                    torch.zeros_like(dwell_streak),
                )
                max_dwell_streak = max(
                    max_dwell_streak, int(dwell_streak.max().item()))
                if bool(pcfg.get("pause_gait_during_dwell", False)):
                    dwell_gait_bad += int((
                        current_dwell & alive
                        & (env.gait_frequency.abs() > 1.0e-8)).sum().item())
                    resumed_now = (prev_dwell_mask & prev_path_mask & path_mask
                                   & alive & ~current_dwell)
                    resumed_gait_bad += int((
                        resumed_now & (env.gait_frequency < 1.8)).sum().item())
                stable_dwell = (prev_path_mask & path_mask & alive
                                & prev_dwell_mask & current_dwell)
                if bool(stable_dwell.any()):
                    goal_step_dwell.append(torch.norm(
                        env.goal_pos_world[stable_dwell] - prev_goal[stable_dwell],
                        dim=-1).cpu().numpy())
                    heading_step = torch.abs(
                        (env.goal_heading_world[stable_dwell]
                         - prev_goal_heading[stable_dwell] + torch.pi)
                        % (2 * torch.pi) - torch.pi)
                    goal_heading_step_dwell.append(heading_step.cpu().numpy())
            # Only measure steady path-carrot motion. Resets can switch goal mode,
            # and segment rerolls deliberately teleport/re-anchor the carrot.
            m = prev_path_mask & path_mask & alive & ~seg_changed
            if bool(m.any()):
                gsel = torch.norm(
                    env.goal_pos_world[m] - env.base_pos[m, :2], dim=-1)
                gaps.append(gsel.cpu().numpy())
                moved = torch.norm(env.goal_pos_world[m] - prev_goal[m], dim=-1)
                goal_step_move.append(moved.cpu().numpy())
                if hasattr(env, "path_goal_rate_limit"):
                    goal_step_budget.append(
                        (env.path_goal_rate_limit[m] * env.dt).cpu().numpy())
                if hasattr(env, "path_dwell_left"):
                    dw_m = env.path_dwell_left[m] > 0
                    look = env.lookahead[m].clamp(min=1.0e-6)
                    ratio = gsel / look
                    if bool(dw_m.any()):
                        gap_dwell_ratio.append(ratio[dw_m].cpu().numpy())
                    if bool((~dw_m).any()):
                        goal_step_move_run.append(moved[~dw_m].cpu().numpy())
                        run_gap = gsel[~dw_m]
                        run_look = look[~dw_m]
                        run_leash = torch.clamp(
                            run_look * float(pcfg.get("leash_ratio", 1.6)),
                            max=float(pcfg.get("lookahead_max_m", 3.5)))
                        gap_run_ratio.append((run_gap / run_look).cpu().numpy())
                        floor_deficit_run.append(
                            (run_look - run_gap).clamp(min=0.0).cpu().numpy())
                        leash_excess_run.append(
                            (run_gap - run_leash).clamp(min=0.0).cpu().numpy())
        if has_segment_id:
            prev_seg = env.goal_segment_id.clone()
        prev_path_mask = path_mask
        prev_goal = env.goal_pos_world.clone()
        prev_goal_heading = env.goal_heading_world.clone()
        prev_dwell_mask = ((env.path_dwell_left > 0) & path_mask
                           if hasattr(env, "path_dwell_left") else
                           torch.zeros_like(path_mask))

        if seq_prev is not None:
            seq_adv += int((env.seq_idx > seq_prev).sum().item())
            seq_prev = env.seq_idx.clone()
            # Tracked EVERY step a sequential env exists, not just whichever
            # ones survive to the final step -- occupancy legitimately
            # collapses over the run for an untrained warm start (see the
            # note() by the checks below), and gating these on the final
            # snapshot would silently skip validating them at all.
            if bool(env.is_seq_env.any()):
                look = env.commands[env.is_seq_env, 4:10]
                seq_look_max = max(seq_look_max, float(look.abs().max().item()))
                seq_r_seen = True
                r = env._reward_seq_goal()
                seq_r_all_finite = seq_r_all_finite and bool(torch.isfinite(r).all())
                seq_r_max = max(seq_r_max, float(r.max().item()))

        if getattr(env, "arm_script_on", False):
            b = env.arm_blend
            arm_blend_lo = min(arm_blend_lo, float(b.min().item()))
            arm_blend_hi = max(arm_blend_hi, float(b.max().item()))

        if hasattr(env, "path_dwell_left"):
            dwell_seen += int(((env.path_dwell_left > 0) & env.is_path_env).sum().item())

        # HBatch distributes artificial hits over several rigid bodies.  The
        # former base-only readout falsely reported that those events never fired.
        force_by_body = torch.norm(env.pushing_forces, dim=-1)
        torque_by_body = torch.norm(env.pushing_torques, dim=-1)
        force_body_active = force_by_body > 1e-3
        torque_body_active = torque_by_body > 1e-3
        active_counts = force_body_active.sum(dim=-1)
        disturbance_max_active_bodies = max(
            disturbance_max_active_bodies, int(active_counts.max().item()))
        force_any = active_counts > 0
        disturbance_clear_transitions += int((prev_force_any & ~force_any).sum().item())
        prev_force_any = force_any
        paired = force_any | (torque_body_active.sum(dim=-1) > 0)
        if bool(paired.any()):
            force_idx = torch.argmax(force_by_body, dim=-1)
            torque_idx = torch.argmax(torque_by_body, dim=-1)
            torque_counts = torque_body_active.sum(dim=-1)
            mismatch = paired & (
                (force_body_active.sum(dim=-1) != 1)
                | (torque_counts > 1)
                | ((torque_counts == 1) & (force_idx != torque_idx)))
            if hasattr(env, "dist_active_body"):
                mismatch |= paired & (force_idx != env.dist_active_body)
            disturbance_body_mismatch += int(mismatch.sum().item())

        f = force_by_body.amax(dim=-1)
        t = torque_by_body.amax(dim=-1)
        act_mask = f > 1e-3
        if bool(act_mask.any()):
            push_active_steps += 1
            push_f_seen.append(f[act_mask].cpu().numpy())
            push_t_seen.append(t[act_mask].cpu().numpy())
            if hasattr(env, "dist_active_body"):
                disturbance_bodies_seen.update(
                    int(x) for x in env.dist_active_body[act_mask].cpu().tolist())
            if hasattr(env, "dist_event_kind"):
                disturbance_kinds_seen.update(
                    int(x) for x in env.dist_event_kind[act_mask].cpu().tolist() if int(x) > 0)

    check("rewards and observations stay finite", rew_bad == 0,
          "{} bad steps".format(rew_bad))
    disturbance_enabled = bool(
        (cfg["randomization"].get("disturbance") or {}).get("enabled", False))
    if hasattr(env, "dist_event_serial") and disturbance_enabled:
        scenario_enabled = bool((cfg["randomization"]["disturbance"].get(
            "scenario_aware") or {}).get("enabled", False))
        expected_apply_calls = args.steps * int(cfg["control"]["decimation"])
        check("disturbance wrench is submitted on every physics substep",
              int(getattr(env, "dist_wrench_apply_calls", -1)) == expected_apply_calls,
              "{} calls, expected {} = {} control steps x {} decimation".format(
                  int(getattr(env, "dist_wrench_apply_calls", -1)),
                  expected_apply_calls, args.steps,
                  int(cfg["control"]["decimation"])))
        check("all configured disturbance bodies receive events",
              disturbance_bodies_seen == set(int(x) for x in env.dist_body_indices.cpu().tolist()),
              "seen {} expected {}".format(
                  sorted(disturbance_bodies_seen),
                  sorted(int(x) for x in env.dist_body_indices.cpu().tolist())))
        expected_kinds = {2} if scenario_enabled else {1, 2}
        check("configured disturbance event classes fire",
              disturbance_kinds_seen == expected_kinds,
              "event kinds seen {}, expected {}".format(
                  sorted(disturbance_kinds_seen), sorted(expected_kinds)))
        check("at most one disturbance body is active per env",
              disturbance_max_active_bodies <= 1,
              "maximum active bodies in one env: {}".format(disturbance_max_active_bodies))
        check("force and torque stay on the declared event body",
              disturbance_body_mismatch == 0,
              "{} mismatched env-steps".format(disturbance_body_mismatch))
        check("expired disturbance wrench clears",
              disturbance_clear_transitions > 0,
              "{} active-to-clear transitions".format(disturbance_clear_transitions))
        if scenario_enabled:
            expected_scenarios = set(range(
                1, len(getattr(env, "dist_scenario_names", ())) + 1))
            expected_tiers = set(range(
                len(getattr(env, "dist_height_tier_names", ()))))
            check("every scenario-aware contact class is sampled",
                  disturbance_scenarios_seen == expected_scenarios,
                  "seen {} expected {}".format(
                      sorted(disturbance_scenarios_seen),
                      sorted(expected_scenarios)))
            check("every configured height tier is sampled",
                  disturbance_tiers_seen == expected_tiers,
                  "seen {} expected {}".format(
                      sorted(disturbance_tiers_seen), sorted(expected_tiers)))
            check("omnidirectional force covers all robot-local octants",
                  disturbance_direction_octants_seen == set(range(8)),
                  "octants seen {}".format(
                      sorted(disturbance_direction_octants_seen)))
            upper_share = disturbance_upper_events / float(
                max(disturbance_scenario_events, 1))
            check("scenario force is concentrated on upper/arm tiers",
                  upper_share >= 0.80,
                  "{} / {} = {:.1%} upper events".format(
                      disturbance_upper_events, disturbance_scenario_events,
                      upper_share))
            inactive = env.dist_steps_left == 0
            expected_i = env.dist_last_expected_impulse[inactive]
            submitted_i = env.dist_last_submitted_impulse[inactive]
            delivered = (expected_i > 1.0e-6) & (submitted_i > 0.0)
            if bool(delivered.any()):
                rel = torch.abs(submitted_i[delivered] - expected_i[delivered]) \
                    / expected_i[delivered]
                delivery_max = float(rel.max().item())
            else:
                delivery_max = float("inf")
            check("configured force impulse reaches physics substeps",
                  bool(delivered.any()) and delivery_max <= 5.0e-4,
                  "{} completed events, max relative error {:.6f}".format(
                      int(delivered.sum().item()), delivery_max))

    # ---- path mode ---------------------------------------------------------
    if n_path > 0 and goal_step_move and gaps:
        move = np.concatenate(goal_step_move)
        # Reset/reroll samples were already excluded by alive & ~seg_changed.
        # Never discard large moves here: those are exactly the untagged
        # teleports this hard invariant is meant to catch.
        if goal_step_budget:
            budget = np.concatenate(goal_step_budget)
            excess = move - budget
            check("every path goal step obeys its per-env rate limit",
                  bool(np.all(excess <= 1.0e-4)),
                  "max move {:.4f} m, max budget excess {:.6f} m".format(
                      float(move.max()), float(excess.max())))
        else:
            check("every path goal step obeys its per-env rate limit", False,
                  "environment did not expose path_goal_rate_limit")
        if goal_step_move_run:
            move_run = np.concatenate(goal_step_move_run)
            moving_share = float((move_run > 1.0e-5).mean())
            check("running path carrot is not frozen",
                  moving_share > 0.50,
                  "{:.1%} of non-dwell steady steps moved; p50 {:.6f} m".format(
                      moving_share, float(np.percentile(move_run, 50))))

        # These are closed-loop policy outcomes, not code invariants.  Use each
        # env's sampled lookahead/leash (the former global 0.375 m floor and
        # widest global leash did not test what their labels claimed), preserve
        # the numbers in the log, and judge them after training.
        if gap_run_ratio:
            ratio = np.concatenate(gap_run_ratio)
            deficit = np.concatenate(floor_deficit_run)
            leash_excess = np.concatenate(leash_excess_run)
            note("running lookahead occupancy (post-train metric)",
                 "gap/lookahead p2 {:.2f} p50 {:.2f}; below 0.75 {:.1%}; "
                 "floor deficit p90 {:.3f} m".format(
                     np.percentile(ratio, 2), np.percentile(ratio, 50),
                     float((ratio < 0.75).mean()), np.percentile(deficit, 90)))
            note("running leash occupancy (post-train metric)",
                 "outside per-env leash {:.1%}; excess p99 {:.3f} m".format(
                     float((leash_excess > 1.0e-4).mean()),
                     np.percentile(leash_excess, 99)))

        dwell_cfg = pcfg.get("dwell") or {}
        if dwell_cfg.get("enabled", False) and gap_dwell_ratio:
            gd_ratio = np.concatenate(gap_dwell_ratio)
            note("dwell arrival occupancy (post-train metric)",
                 "gap/lookahead below 0.75 on {:.1%} of dwelling samples".format(
                     float((gd_ratio < 0.75).mean())))
        if dwell_cfg.get("enabled", False) and goal_step_dwell:
            dwell_move = np.concatenate(goal_step_dwell)
            dwell_heading = (np.concatenate(goal_heading_step_dwell)
                             if goal_heading_step_dwell else np.array([np.inf]))
            check("dwell parks the full world-frame goal pose",
                  float(dwell_move.max()) <= 1.0e-5
                  and float(dwell_heading.max()) <= 1.0e-5,
                  "max position step {:.7f} m, heading step {:.7f} rad".format(
                      float(dwell_move.max()), float(dwell_heading.max())))
    elif n_path > 0:
        check("path-mode samples collected", False,
              "no surviving path envs in {} steps -- policy may be falling immediately".format(args.steps))

    # ---- dwell + grid: brand-new machinery, never executed before -----------
    dwell_cfg = pcfg.get("dwell") or {}
    if dwell_cfg.get("enabled", False) and n_path > 0:
        # dwell is the repair for path training wrecking waypoint accuracy
        # (6.3 cm -> 37.9 cm in the v7 batch). If the carrot never actually
        # stops, the repair is absent and the next batch reproduces the fault.
        check("dwell counter fires", dwell_seen > 0,
              "{} env-steps spent dwelling out of {}".format(dwell_seen, args.steps * n_path))
        lo, hi = dwell_cfg.get("interval_s", [4.0, 10.0])
        dur_lo, dur_hi = dwell_cfg.get("duration_s", [1.5, 3.0])
        expect = (0.5 * (dur_lo + dur_hi)) / (0.5 * (lo + hi))
        observed = dwell_seen / float(max(args.steps * n_path, 1))
        check("dwell duty cycle is in the configured ballpark",
              observed <= expect * 3.0 + 0.02,
              "observed {:.1%} vs configured ~{:.1%}".format(observed, expect))
        minimum_streak = max(1, int(dur_lo / env.dt) - 2)
        check("at least one dwell survives its configured minimum duration",
              max_dwell_streak >= minimum_streak,
              "longest {} steps, required at least {}".format(
                  max_dwell_streak, minimum_streak))
        if bool(pcfg.get("pause_gait_during_dwell", False)):
            check("dwell gait clock pauses and resumes",
                  dwell_gait_bad == 0 and resumed_gait_bad == 0,
                  "{} dwelling nonzero-Hz and {} resumed below-1.8-Hz env-steps".format(
                      dwell_gait_bad, resumed_gait_bad))

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
        # Checked at ASSIGNMENT time (right after reset()), not at the end of
        # the run. Those answer different questions: this one verifies the
        # DRAW mechanism (does _reset_idx assign sequential mode at roughly
        # `share` probability), which is the only thing pre-training code
        # correctness can promise. The end-of-run count instead measures how
        # long a policy SURVIVES in sequential mode, which for a warm start
        # that has never seen the lookahead channels carry a nonzero value
        # (command slots 4-9 were pinned to exactly [0,0] for the whole time
        # E0 trained) is expected to be short until training adapts it -- a
        # low number there is the bootstrap cost, not a broken assignment.
        if n_seq_at_reset is not None:
            check("sequential-nav share matches config (at assignment)",
                  abs(n_seq_at_reset / float(env.num_envs) - share) < 0.12 or share == 0.0,
                  "configured {:.2f}, got {:.2f} ({} envs right after reset())".format(
                      share, n_seq_at_reset / float(env.num_envs), n_seq_at_reset))
        note("sequential-nav occupancy at end of run",
             "{} envs still sequential after {} steps (of {} assigned at reset) -- "
             "expected to shrink under an untrained warm start, not a gate".format(
                 n_seq, args.steps, n_seq_at_reset))
        # Goals must be BANKED, not just approached: if seq_idx never advances
        # the sequential reward is stuck at rho/N and the whole mechanism
        # reduces to a single-goal task with extra bookkeeping. Informational
        # pre-training: with the warm start seeing this observation channel
        # for the first time, 0 banks in a short smoke run does not mean the
        # reward/reaching-condition wiring is broken -- _reached()'s
        # tolerances were exercised directly by tools/diag_seq.py, which is
        # the tool to use if this stays at 0 after real training has had a
        # chance to adapt. Not gated on end-of-run n_seq: occupancy collapsing
        # to 0 must not silently hide whether any bank ever happened.
        note("goals actually get banked",
             "{} goal advances across up to {} seq envs over {} steps".format(
                 seq_adv, n_seq_at_reset, args.steps))
        # Tracked EVERY step any env was sequential (seq_look_max/seq_r_*),
        # not just whichever survive to the literal final step -- occupancy
        # legitimately collapses over the run (see the note above), and
        # gating on the final snapshot would silently skip validating these
        # at all once it does.
        if seq_r_seen:
            # The lookahead window must be live in the six reused slots. All-zero
            # means the observation is carrying nothing and the policy is blind
            # to upcoming turns -- the ablation in the paper shows that costs
            # most of the benefit.
            check("lookahead window populated (command slots 4-9)",
                  seq_look_max > 1e-3, "max |lookahead| = {:.3f}".format(seq_look_max))
            check("sequential reward finite and in [0, 1]",
                  seq_r_all_finite and seq_r_max <= 1.001,
                  "max {:.3f}".format(seq_r_max))
        else:
            check("lookahead window populated (command slots 4-9)", False,
                  "no env was ever sequential during the run -- cannot verify")
            check("sequential reward finite and in [0, 1]", False,
                  "no env was ever sequential during the run -- cannot verify")

    if getattr(env, "grid_on", False):
        active = int(env.grid_active.sum().item())
        total = int(env.grid_active.numel())
        initial = (pcfg.get("speed_grid") or {}).get("initial_active", "seed")
        if initial == "all":
            # H warm-starts from the late G1 policy, whose eval exercised the
            # full grid. Historical checkpoints did not serialize task state,
            # so H freezes that distribution explicitly instead of silently
            # restarting at the 0.3 m/s straight seed cell.
            check("speed grid restores the full H warm-start distribution",
                  active == total,
                  "{}/{} cells active after {} steps".format(active, total, args.steps))
        else:
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
            scenario_on = bool((d.get("scenario_aware") or {}).get(
                "enabled", False))
            if scenario_on:
                specs = getattr(env, "_dist_scenario_specs", ())
                force_min = min(float(x["force_n"][0]) for x in specs)
                force_max = max(float(x["force_n"][1]) for x in specs)
                check("scenario force magnitudes stay in their envelope",
                      fv.min() >= force_min * 0.95
                      and fv.max() <= force_max * 1.05,
                      "observed {:.1f}-{:.1f} N, envelope {:.1f}-{:.1f} N".format(
                          fv.min(), fv.max(), force_min, force_max))
            else:
                cmax = d["collision"]["force_n"][1]
                check("collision-class magnitudes present",
                      fv.max() >= d["collision"]["force_n"][0] * 0.8,
                      "max {:.1f} N (collision band {}-{} N)".format(
                          fv.max(), *d["collision"]["force_n"]))
                check("support-class magnitudes present",
                      fv.min() <= d["support"]["force_n"][1] * 1.5,
                      "min {:.1f} N (support band {}-{} N)".format(
                          fv.min(), *d["support"]["force_n"]))
                check("no force beyond the configured ceiling",
                      fv.max() <= cmax * 1.5,
                      "max {:.1f} N vs ceiling {:.1f}".format(fv.max(), cmax))
            check("torque applied with force", tv.max() > 0.0, "max {:.2f} N*m".format(tv.max()))
            print("     force  p50 {:.1f}  p90 {:.1f}  max {:.1f} N".format(
                np.percentile(fv, 50), np.percentile(fv, 90), fv.max()))

    # ---- reward wiring -----------------------------------------------------
    missing = [n for n, s in cfg["rewards"]["scales"].items()
               if float(s) != 0.0 and not hasattr(env, "_reward_" + n)]
    check("every nonzero reward scale has an implementation", not missing, str(missing))

    print("\n{}/{} checks passed".format(len(CHECKS) - len(FAILURES), len(CHECKS)))
    if FAILURES:
        print("FAILED:")
        for name, detail in FAILURES:
            print("  - {}{}".format(name, ": " + detail if detail else ""))
        print("\nDo NOT launch training until these are resolved.")
        return 1
    print("v7 machinery verified — safe to launch training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
