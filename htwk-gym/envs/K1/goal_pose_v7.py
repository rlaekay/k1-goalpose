"""GoalPose v7 -- the consolidated "ultimate" task built ON TOP of goal_pose.py.

Rationale (2026-07-26 eval of armA/B/C/D + v3, see MASTERPLAN2.md):

  * armB won decisively on every category (3.9 cm vs armA 7.6 cm, strict success
    52.8% vs 13.9%) and A/B differ by exactly ONE lever: `goal_reached`. That
    lever is therefore kept, unchanged, as the backbone of v7.
  * The measured task NEVER demanded speed: over 4647 segments the maximum
    required speed was 0.574 m/s and the median 0.116 m/s, because 29.5% of
    segments (stand + turn) have a goal distance of exactly zero and the rest
    draw a <=2.5 m goal with a 4-8 s deadline. A 1.3-1.5 m/s walk cannot be
    learned or even measured under that distribution. -> `path` goal mode.
  * armD changed 12 levers at once and collapsed to 24.3 cm; its stand category
    was FINE (5.0 cm) while everything requiring locomotion failed, which points
    at the actuation change (2x PD stiffness on a warm start), not at the
    perception noise. -> v7 keeps armD's perception model, drops its PD change.

Everything here is a switch. With `commands.goal_mode_mixture.path: 0.0`,
`randomization.disturbance.enabled: false` and the v7 reward scales at zero, v7
reduces exactly to armB.

New mechanisms, each independently switchable:

  1. path goal mode -- a lookahead point that ADVANCES along a parametric path
     (figure-8 / circle / spiral / rose / random walk) at a commanded speed,
     leashed so it never runs more than `lookahead_max_m` ahead of the robot.
     A stationary goal can be reached by any speed > 0; a receding one can only
     be held by matching its speed. This is what creates the speed demand.
  2. speed curriculum -- raises the commanded path speed while the robot keeps
     the leash slack. Note this is the OPPOSITE of v3's curriculum, which shrank
     goal DISTANCE and so reduced the speed demand as the policy improved.
  3. BT-flicker goal observation -- occasional large discrete jumps of the
     PERCEIVED goal (behaviour-tree branch flip / ball re-detection), on top of
     armD's gaussian jitter + per-segment bias + staleness. Reward keeps reading
     ground truth, so the policy is forced to filter rather than to chase.
  4. two-class disturbance -- robot-robot COLLISION (short, strong, impulse
     derived from the URDF mass) and human SUPPORT (long, gentle), on
     independent per-env timers instead of one global clock that hit all envs on
     the same step.
  5. joint/actuator protection -- margin penalties that activate before the hard
     limit, plus an electrical-power proxy for the battery budget.
"""

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import torch_rand_float, get_euler_xyz

assert gymtorch

import numpy as np
import torch

from envs.K1.goal_pose_v3 import GoalPoseV3


# goal_category ids: 0-4 come from the base task's "No More Marching" mixture.
CATEGORY_PATH = 5


class GoalPoseV7(GoalPoseV3):
    """Inherits v3's L/R mirror maps (RunnerV3's symmetry loss -- the user asked
    for symmetric loss explicitly) and its timed-reward hook, but v3's
    `goal_curriculum` is switched OFF in Goal_Pose_V7.yaml: that curriculum
    shrinks goal DISTANCE as the policy improves, which lowers the speed demand.
    v7's curriculum raises path SPEED instead, which is the opposite direction.
    """


    def __init__(self, cfg):
        super().__init__(cfg)
        c = self.cfg["commands"]
        self.path_cfg = c.get("path", {}) or {}
        mix = c.get("goal_mode_mixture", {}) or {}
        self.path_share = float(mix.get("path", 0.0))

        n, dev = self.num_envs, self.device
        self.is_path_env = torch.zeros(n, dtype=torch.bool, device=dev)
        self.path_shape = torch.zeros(n, dtype=torch.long, device=dev)
        self.path_scale = torch.ones(n, dtype=torch.float, device=dev)
        self.path_origin = torch.zeros(n, 2, dtype=torch.float, device=dev)
        self.path_rot = torch.zeros(n, dtype=torch.float, device=dev)
        self.path_dir = torch.ones(n, dtype=torch.float, device=dev)
        self.path_u = torch.zeros(n, dtype=torch.float, device=dev)
        self.path_speed = torch.zeros(n, dtype=torch.float, device=dev)
        self.lookahead = torch.zeros(n, dtype=torch.float, device=dev)
        # tracking error against the carrot, and the per-segment counters the
        # grid curriculum promotes on. path_lag is exported for eval: goal
        # distance can never reach 0 in path mode (the goal is meant to sit
        # lookahead_min ahead), so raw distance is not the tracking error.
        self.path_lag = torch.zeros(n, dtype=torch.float, device=dev)
        self.path_dwell_left = torch.zeros(n, dtype=torch.long, device=dev)
        self.path_dwell_next = torch.zeros(n, dtype=torch.long, device=dev)
        self.track_ok_steps = torch.zeros(n, dtype=torch.float, device=dev)
        self.track_steps = torch.zeros(n, dtype=torch.float, device=dev)
        self.grid_on = False
        # curriculum state
        self.speed_level = float(self.path_cfg.get("speed_init", 0.4))
        self.keepup_ema = 0.0
        self._last_speed_adjust = 0
        # disturbance state (per-env timers)
        # In path mode the goal moves EVERY step, so eval_goal_pose.py's "did
        # goal_pos_world change?" test would close a segment on every control
        # step. Publish an explicit segment counter it can use instead.
        self.goal_segment_id = torch.zeros(n, dtype=torch.long, device=dev)
        self.dist_steps_left = torch.zeros(n, dtype=torch.long, device=dev)
        self.dist_next = torch.zeros(n, dtype=torch.long, device=dev)
        self._init_disturbance_schedule()
        self._init_speed_grid()
        self._init_arm_script()
        # step() calls _update_goal_state() twice and _resample_goals() once, so an
        # unguarded advance would run the path 3x per control step (i.e. 3x the
        # commanded speed). Advance at most once per common_step_counter tick.
        self._path_advanced_at = -1

    # ---- 1. path goal mode --------------------------------------------------

    def _path_point(self, u):
        """Closed-form curve families, evaluated per env at phase u.

        Shapes: 0 figure-8 (Gerono lemniscate), 1 circle, 2 spiral, 3 rose/star,
        4 smooth pseudo-random wander (sum of two incommensurate harmonics).
        Returned in the path's own frame; caller applies scale/rotation/origin.
        """
        s = self.path_shape
        x = torch.zeros_like(u)
        y = torch.zeros_like(u)

        m = s == 0                                        # figure-8
        x = torch.where(m, torch.sin(u), x)
        y = torch.where(m, torch.sin(u) * torch.cos(u), y)

        m = s == 1                                        # circle
        x = torch.where(m, torch.cos(u), x)
        y = torch.where(m, torch.sin(u), y)

        m = s == 2                                        # spiral (radius grows then unwinds)
        r = 0.35 + 0.65 * (0.5 - 0.5 * torch.cos(u * 0.25))
        x = torch.where(m, r * torch.cos(u), x)
        y = torch.where(m, r * torch.sin(u), y)

        m = s == 3                                        # 5-petal rose = star-like
        rr = torch.cos(2.5 * u)
        x = torch.where(m, rr * torch.cos(u), x)
        y = torch.where(m, rr * torch.sin(u), y)

        m = s == 4                                        # pseudo-random wander
        x = torch.where(m, torch.sin(u) + 0.5 * torch.sin(2.3 * u + 1.1), x)
        y = torch.where(m, torch.cos(0.7 * u) + 0.5 * torch.sin(1.7 * u), y)

        # 5 serpentine: translation advances monotonically while curvature flips
        # sign every half period. The canonical translation/rotation coupling
        # test -- the robot must keep a steady forward pace while yaw rate
        # reverses, which is precisely the regime where the two interfere.
        m = s == 5
        x = torch.where(m, 0.45 * u, x)
        y = torch.where(m, torch.sin(u), y)
        return x, y

    def _path_world(self, u):
        x, y = self._path_point(u)
        x = x * self.path_scale
        y = y * self.path_scale
        c, s = torch.cos(self.path_rot), torch.sin(self.path_rot)
        wx = self.path_origin[:, 0] + c * x - s * y
        wy = self.path_origin[:, 1] + s * x + c * y
        return wx, wy

    def _reroll_paths(self, env_ids):
        if len(env_ids) == 0:
            return
        p = self.path_cfg
        k = len(env_ids)
        shapes = p.get("shapes", [0, 1, 2, 3, 4])
        pick = torch.randint(0, len(shapes), (k,), device=self.device)
        self.path_shape[env_ids] = torch.tensor(shapes, device=self.device, dtype=torch.long)[pick]
        lo, hi = p.get("scale_m", [1.5, 4.0])
        self.path_scale[env_ids] = torch_rand_float(lo, hi, (k, 1), device=self.device).squeeze(1)
        self.path_rot[env_ids] = torch_rand_float(-np.pi, np.pi, (k, 1), device=self.device).squeeze(1)
        self.path_dir[env_ids] = torch.where(
            torch.rand(k, device=self.device) < 0.5,
            -torch.ones(k, device=self.device), torch.ones(k, device=self.device))
        self.path_u[env_ids] = torch_rand_float(-np.pi, np.pi, (k, 1), device=self.device).squeeze(1)
        lo, hi = p.get("lookahead_m", [0.5, 3.0])
        self.lookahead[env_ids] = torch_rand_float(lo, hi, (k, 1), device=self.device).squeeze(1)

        if self.grid_on:
            # Report on the segment that just ended BEFORE overwriting its cell,
            # then draw the next (speed, curvature) pair from the active set.
            self._grid_report(env_ids)
            flat, si, ci = self._sample_cells(k)
            self.env_cell[env_ids] = flat
            jitter = torch_rand_float(0.85, 1.15, (k, 1), device=self.device).squeeze(1)
            self.path_speed[env_ids] = self.grid_speeds[si] * jitter
            # kappa ~ 1/scale for these curve families, so curvature is commanded
            # through the path size. This is what couples yaw rate to speed:
            # omega = v * kappa, i.e. the grid axis pair IS (v_x, omega_z).
            self.path_scale[env_ids] = 1.0 / self.grid_curvs[ci].clamp(min=1e-3)
        else:
            # commanded path speed, scaled by the scalar curriculum level
            smin, smax = p.get("speed_range_mps", [0.3, 1.6])
            top = smin + (smax - smin) * self.speed_level
            self.path_speed[env_ids] = torch_rand_float(smin, max(smin + 1e-3, top), (k, 1), device=self.device).squeeze(1)
        self.track_ok_steps[env_ids] = 0.0
        self.track_steps[env_ids] = 0.0
        self.path_lag[env_ids] = 0.0
        # anchor the path so the lookahead point starts just ahead of the robot
        wx, wy = self._path_world(self.path_u)
        self.path_origin[env_ids, 0] += self.base_pos[env_ids, 0] - wx[env_ids]
        self.path_origin[env_ids, 1] += self.base_pos[env_ids, 1] - wy[env_ids]

    def _project_robot(self, u_goal, ds_du):
        """Arc-length phase of the point on the path nearest the robot.

        Needed for a real lookahead: "0.5-3.0 m ahead" is only meaningful
        relative to where the robot IS on the path, not relative to where the
        goal happened to drift to. Searched locally (a window behind the goal)
        rather than globally, because a global nearest-point search on a
        self-intersecting curve like a figure-8 can snap to the wrong lobe.
        """
        # Window must be genuinely LOCAL. It used to span +-1.2 * lookahead_max
        # (~4.9 m) while the curves themselves are only 1-4 m across, so on a
        # figure-8 or a 5-petal rose the nearest probe could snap to a different
        # lobe between steps -- the projection jumped, the floor clamp followed
        # it, and the carrot teleported (smoke measured a p99 goal speed of
        # 20.3 m/s against a 0.82 m/s ceiling). Keep it under half a leash.
        span = float(self.path_cfg.get("project_window_m", 0.8))
        probes = torch.linspace(-span, 0.25 * span, 9, device=self.device)
        best_u = u_goal.clone()
        best_d = torch.full_like(u_goal, float("inf"))
        for off in probes:
            uc = u_goal + self.path_dir * off / ds_du
            xc, yc = self._path_world(uc)
            d = torch.norm(torch.stack([xc, yc], dim=-1) - self.base_pos[:, :2], dim=-1)
            closer = d < best_d
            best_d = torch.where(closer, d, best_d)
            best_u = torch.where(closer, uc, best_u)
        return best_u

    def _advance_paths(self):
        """Advance the lookahead point.

        Three constraints act at once, and each fixes a different failure:

          pace   u advances at path_speed              -> creates the speed demand.
                 A goal that only sits `L` ahead of the robot moves at whatever
                 speed the robot chooses, so pure pursuit alone demands nothing.
          floor  goal stays >= lookahead_min ahead     -> keeps it a LOOKAHEAD.
                 Without this the goal drifts onto the robot (measured: gap
                 median 0.41 m, below the configured 0.5 m minimum), and there
                 constellation = exp(-w d^2) ~ 1 with gradient ~ 0 -- the reward
                 goes flat exactly where the robot is.
          leash  goal stays <= lookahead_max ahead     -> keeps the gradient alive
                 when the robot cannot keep up, instead of handing it a target
                 that recedes forever.

        The floor also means a robot faster than path_speed drags the carrot
        along faster than commanded, so path_speed is a floor on pace rather
        than a ceiling -- which is what lets the achieved-speed distribution
        reveal the physical limit instead of the commanded one.
        """
        ids = self.is_path_env
        if not bool(ids.any()) or self._path_advanced_at == self.common_step_counter:
            return
        self._path_advanced_at = self.common_step_counter
        p = self.path_cfg
        du = 1.0e-3
        u0 = self.path_u
        x0, y0 = self._path_world(u0)
        x1, y1 = self._path_world(u0 + du)
        ds_du = torch.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2).clamp(min=1.0e-4) / du

        # ---- dwell: the carrot periodically STOPS and must be parked on ----
        # The 2026-07-27 v7 eval showed path training wrecking waypoint accuracy:
        # waypoint-only position error was 6.3 cm for E0 (no path training) but
        # 37.9 cm for E1 and 38.7 cm for V7, both path-trained, while their stand
        # category stayed fine at 5.0 cm. A goal that is ALWAYS lookahead_min
        # ahead and never reachable teaches "hold station behind the target",
        # and that habit is exactly wrong for a waypoint, where the job is to
        # arrive and stop. Mixing the two 50/50 with one reward produced a policy
        # that did neither.
        # Dwell removes the conflict instead of trading it off: the carrot pauses
        # (and releases the minimum-distance floor, or arriving would still be
        # impossible), so path mode ends in the same "arrive and stop" state that
        # waypoint mode is entirely about. goal_reached then fires on both.
        dw = p.get("dwell") or {}
        if dw.get("enabled", False):
            self.path_dwell_left = (self.path_dwell_left - 1).clamp(min=0)
            self.path_dwell_next = self.path_dwell_next - 1
            fire = (self.path_dwell_next <= 0) & ids
            if bool(fire.any()):
                k = int(fire.sum().item())
                lo, hi = dw.get("duration_s", [1.5, 3.0])
                self.path_dwell_left[fire] = (torch_rand_float(
                    lo, hi, (k, 1), device=self.device).squeeze(1) / self.dt).long().clamp(min=1)
                lo, hi = dw.get("interval_s", [4.0, 10.0])
                self.path_dwell_next[fire] = torch.randint(
                    int(lo / self.dt), max(int(lo / self.dt) + 1, int(hi / self.dt)),
                    (k,), device=self.device)
        dwelling = self.path_dwell_left > 0

        l_min = self.lookahead                                   # per-env, [0.5, 3.0]
        l_min = torch.where(dwelling, torch.zeros_like(l_min), l_min)
        l_max = torch.clamp(self.lookahead * float(p.get("leash_ratio", 1.6)),
                            max=float(p.get("lookahead_max_m", 3.5)))
        l_max = torch.maximum(l_max, l_min + 0.1)

        u_proj = self._project_robot(u0, ds_du)
        pace = torch.where(dwelling, torch.zeros_like(self.path_speed), self.path_speed)
        u_paced = u0 + self.path_dir * pace * self.dt / ds_du
        # clamp in "progress" coordinates so the sign of path_dir drops out
        d = self.path_dir
        prog = torch.clamp(d * u_paced,
                           min=d * u_proj + l_min / ds_du,
                           max=d * u_proj + l_max / ds_du)

        # Rate limit, in metres of arc per control step. The clamp above is a
        # POSITION constraint, so any jump in u_proj passes straight through it
        # into the goal; this bounds how fast that correction is allowed to be
        # applied. The bound is max(pace, robot speed) rather than pace alone
        # because a robot moving faster than the commanded pace legitimately
        # drags the carrot along at its own speed to hold the floor -- capping
        # at pace would break the floor exactly when the robot is doing well.
        v_robot = torch.norm(self.root_states[:, 7:9], dim=-1)
        rate = torch.maximum(pace, v_robot) * float(self.path_cfg.get("catchup_ratio", 1.5))
        max_du = (rate + 0.05) * self.dt / ds_du
        prog0 = d * u0
        prog = prog0 + torch.clamp(prog - prog0, -max_du, max_du)
        self.path_u = torch.where(ids, d * prog, u0)

        gx, gy = self._path_world(self.path_u)
        gx1, gy1 = self._path_world(self.path_u + self.path_dir * du)
        tangent = torch.atan2(gy1 - gy, gx1 - gx)
        self.goal_pos_world[:, 0] = torch.where(ids, gx, self.goal_pos_world[:, 0])
        self.goal_pos_world[:, 1] = torch.where(ids, gy, self.goal_pos_world[:, 1])
        self.goal_heading_world = torch.where(ids, tangent, self.goal_heading_world)

        # How far the robot has fallen behind its own projection-based carrot.
        # This, not the raw goal distance, is the tracking error: the goal is
        # SUPPOSED to sit l_min ahead, so distance alone can never reach zero.
        gap_now = torch.norm(torch.stack([gx, gy], dim=-1) - self.base_pos[:, :2], dim=-1)
        self.path_lag = torch.where(ids, (gap_now - l_min).clamp(min=0.0), self.path_lag)
        keep = float(p.get("keepup_gap_m", 2.0))
        self.track_ok_steps = torch.where(ids & (self.path_lag < keep),
                                          self.track_ok_steps + 1, self.track_ok_steps)
        self.track_steps = torch.where(ids, self.track_steps + 1, self.track_steps)

        self._update_speed_curriculum(self.path_lag[ids])

    # ---- 2b. grid-adaptive curriculum over (speed x curvature) --------------
    #
    # Margolis et al., "Rapid Locomotion via Reinforcement Learning" (RSS 2022,
    # 3.9 m/s on Mini Cheetah) report two things that decide this design:
    #
    #   * "In the absence of any curriculum, no useful locomotion behavior is
    #     learned, even at low speeds" -- the robot jitters in place. Sampling
    #     uniformly from a wide command range fails because most commands are
    #     too hard to collect reward from. That is exactly what E3_wide_nosched
    #     does, so E3 is a genuine risk, not a safe simplification.
    #   * Their curriculum is a GRID over (v_x, omega_z), not a scalar, because
    #     "simultaneous high linear and angular velocities are more demanding
    #     due to centrifugal forces", giving roughly omega ~ 1/v at the
    #     feasibility boundary. A scalar level cannot represent that boundary.
    #
    # On a path, yaw rate is not free: omega = v * kappa. So (speed, curvature)
    # IS Margolis's (v, omega) grid, reparameterised -- and it is simultaneously
    # the translation/rotation coupling this project wanted trained explicitly.
    # Curvature is set through path_scale (kappa ~ 1/scale for these curves).

    def _init_speed_grid(self):
        g = self.path_cfg.get("speed_grid") or {}
        self.grid_on = bool(g.get("enabled", False))
        if not self.grid_on:
            return
        self.grid_speeds = torch.tensor(g.get("speeds", [0.3, 0.6, 0.9, 1.2, 1.5, 1.8]),
                                        dtype=torch.float, device=self.device)
        self.grid_curvs = torch.tensor(g.get("curvatures", [0.25, 0.4, 0.6, 0.8, 1.0]),
                                       dtype=torch.float, device=self.device)
        ns, nc = len(self.grid_speeds), len(self.grid_curvs)
        self.grid_active = torch.zeros(ns, nc, dtype=torch.bool, device=self.device)
        self.grid_active[0, 0] = True          # seed: slowest, straightest
        self.env_cell = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.grid_success = torch.zeros(ns, nc, dtype=torch.float, device=self.device)
        self.grid_trials = torch.zeros(ns, nc, dtype=torch.float, device=self.device)

    def _sample_cells(self, k):
        flat = self.grid_active.flatten().float()
        idx = torch.multinomial(flat / flat.sum(), k, replacement=True)
        nc = len(self.grid_curvs)
        return idx, idx // nc, idx % nc

    def _grid_report(self, env_ids):
        """Promote neighbours of any cell the policy is now tracking well."""
        if not self.grid_on or len(env_ids) == 0:
            return
        ok = (self.track_ok_steps[env_ids] /
              self.track_steps[env_ids].clamp(min=1.0)) > float(
                  self.path_cfg.get("speed_grid", {}).get("up_threshold", 0.8))
        nc = len(self.grid_curvs)
        cells = self.env_cell[env_ids]
        self.grid_trials.view(-1).index_add_(0, cells, torch.ones_like(ok, dtype=torch.float))
        self.grid_success.view(-1).index_add_(0, cells, ok.float())
        won = cells[ok]
        if won.numel() == 0:
            return
        si, ci = won // nc, won % nc
        ns = len(self.grid_speeds)
        # 4-neighbourhood, exactly as the paper expands into adjacent bins
        for dsi, dci in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            s2, c2 = (si + dsi).clamp(0, ns - 1), (ci + dci).clamp(0, nc - 1)
            self.grid_active[s2, c2] = True

    def _update_speed_curriculum(self, gap):
        p = self.path_cfg
        if self.grid_on:
            return   # the grid replaces the scalar level entirely
        if not p.get("speed_curriculum", False) or gap.numel() == 0:
            return
        target = float(p.get("keepup_gap_m", 2.0))
        e = float(p.get("ema", 0.995))
        self.keepup_ema = e * self.keepup_ema + (1.0 - e) * float((gap < target).float().mean().item())
        every = int(p.get("adjust_every_steps", 500))
        if self.common_step_counter - self._last_speed_adjust < every:
            return
        self._last_speed_adjust = self.common_step_counter
        step = float(p.get("speed_step", 0.05))
        if self.keepup_ema > float(p.get("up_threshold", 0.8)):
            self.speed_level = min(1.0, self.speed_level + step)
        elif self.keepup_ema < float(p.get("down_threshold", 0.5)):
            self.speed_level = max(float(p.get("speed_min_level", 0.1)), self.speed_level - step)

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        # goal mode is fixed for the whole episode so eval can attribute cleanly.
        # The path itself is NOT rolled here: base _reset_idx zeroes both
        # episode_length_buf and cmd_resample_time, so these envs are "due" in the
        # _resample_goals call that follows, and rolling there uses a base_pos that
        # has actually been refreshed from the new root state.
        self.is_path_env[env_ids] = torch.rand(len(env_ids), device=self.device) < self.path_share

    def _resample_goals(self):
        # base samples a waypoint for every due env; path envs then have their
        # goal overwritten by _advance_paths. The resample event is reused as the
        # natural moment to re-roll path shape/speed/lookahead.
        all_due = self.episode_length_buf == self.cmd_resample_time
        due = all_due & self.is_path_env
        super()._resample_goals()
        if not getattr(self, "manual_control", False):
            self.goal_segment_id[all_due] += 1
        ids = due.nonzero(as_tuple=False).flatten()
        if len(ids) > 0:
            self._reroll_paths(ids)
        self._advance_paths()
        if bool(self.is_path_env.any()):
            self.goal_category[self.is_path_env] = CATEGORY_PATH
            # a moving lookahead has no "stand still" case; keep the gait clock live
            self.gait_frequency[self.is_path_env] = self.gait_frequency[self.is_path_env].clamp(min=1.8)

    def _update_goal_state(self):
        self._advance_paths()
        super()._update_goal_state()

    # ---- instrumentation ----------------------------------------------------

    def _compute_observations(self):
        super()._compute_observations()
        # Coverage audit (2026-07-27): three v7 intents had no measurement at
        # all, so after a run there was no way to tell "the mechanism did
        # nothing" from "the mechanism was never exercised".
        e = self.extras.setdefault("v7", {})
        if self.grid_on:
            # Was the curriculum climbing, or stuck? A scalar level that never
            # got logged could not answer that after the fact.
            e["grid_cells_active"] = float(self.grid_active.sum().item())
            e["grid_cells_total"] = float(self.grid_active.numel())
            tried = self.grid_trials > 0
            e["grid_success_rate"] = float(
                (self.grid_success[tried] / self.grid_trials[tried]).mean().item()) if bool(tried.any()) else 0.0
            act = self.grid_active.nonzero()
            if act.numel():
                e["grid_max_speed"] = float(self.grid_speeds[act[:, 0]].max().item())
                e["grid_max_curv"] = float(self.grid_curvs[act[:, 1]].max().item())
        if bool(self.is_path_env.any()):
            e["path_lag_mean"] = float(self.path_lag[self.is_path_env].mean().item())

        # Joint protection: a zero penalty is ambiguous -- it means either
        # "well within limits" or "the term is switched off". Margin occupancy
        # disambiguates, and is the number that maps to the hardware spec.
        half = 0.5 * (self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0])
        mid = 0.5 * (self.dof_pos_limits[:, 1] + self.dof_pos_limits[:, 0])
        e["dof_pos_occupancy"] = float(((self.dof_pos - mid).abs() / half.clamp(min=1e-6)).max(dim=-1)[0].mean().item())
        e["torque_occupancy"] = float((self.torques.abs() / self.torque_limits.clamp(min=1e-6)).max(dim=-1)[0].mean().item())
        e["torque_saturated"] = float((self.torques.abs() > 0.95 * self.torque_limits).float().mean().item())

        # Symmetry: we finally switched the loss on, but had no way to check
        # whether left/right actually became symmetric. This is the direct
        # readout of armA's "left foot always slightly ahead" observation.
        if hasattr(self, "feet_pos") and self.feet_pos.shape[1] >= 2:
            # Measured in the BASE frame, not the world frame: in world frame a
            # turning robot makes every left/right difference oscillate with
            # heading and the mean washes out to zero regardless of the gait.
            rel = self.feet_pos[:, :2, :] - self.base_pos.unsqueeze(1)
            _, _, yaw = get_euler_xyz(self.base_quat)
            cy, sy = torch.cos(yaw), torch.sin(yaw)
            fx = cy.unsqueeze(1) * rel[:, :, 0] + sy.unsqueeze(1) * rel[:, :, 1]
            fy = -sy.unsqueeze(1) * rel[:, :, 0] + cy.unsqueeze(1) * rel[:, :, 1]
            # fore-aft: "left foot always slightly ahead" (armA)
            e["feet_fwd_asymmetry"] = float((fx[:, 0] - fx[:, 1]).mean().item())
            # LATERAL: how far the stance sits off the body centreline. The
            # reviewer's observation -- fore-aft looked fine but the right hip
            # yaw sat pulled toward the centre -- is invisible to the fore-aft
            # term above and only shows up here.
            e["feet_lat_offset"] = float((fy[:, 0] + fy[:, 1]).mean().item() / 2.0)
            e["feet_width"] = float((fy[:, 0] - fy[:, 1]).abs().mean().item())
            # per-joint left/right bias: the direct readout for hip yaw
            for i, nm in enumerate(self.dof_names):
                if not nm.startswith("Left_"):
                    continue
                j = self.dof_names.index("Right_" + nm[len("Left_"):])
                # Roll/Yaw mirror with a sign flip, Pitch/Knee sign-preserving
                sgn = -1.0 if ("Roll" in nm or "Yaw" in nm) else 1.0
                bias = (self.dof_pos[:, i] - sgn * self.dof_pos[:, j]).mean()
                e["bias_" + nm[len("Left_"):]] = float(bias.item())


    # ---- 7. scripted (non-learned) arm DOFs ---------------------------------
    #
    # The static arms-down pose already captures the inertia win (I_zz -69%).
    # What it cannot do is CHANGE with motion state, which is what was asked
    # for: elbows straightened into a clean, repeatable stance when parked (the
    # RLKick handoff pose), tucked rearward once moving so the hands cannot
    # catch on another robot.
    #
    # The elbows are revolute in K1_locomotion_armswing.urdf but stay OUT of the
    # action and observation vectors: they are driven from a closed form, not
    # learned. That keeps num_actions at 12 and the observation at 54, so E0's
    # weights still load -- which is the whole reason this is worth doing at all.
    #
    # The trigger is NOT a hand-picked speed cut-off. It reuses the exact
    # "parked" test the reward already uses, so there is one definition of
    # stopped in the system instead of two that can disagree:
    #     goal_dist < goal_reach_radius AND |v| < stop_speed AND |w| < stop_ang
    # On hardware the BT already knows goal_reached, and IMU gives the rest.

    def _init_arm_script(self):
        a = self.cfg.get("arm_script") or {}
        self.arm_script_on = bool(a.get("enabled", False))
        self.leg_dof_idx = None
        if not self.arm_script_on:
            return
        arm_names = a.get("joints", ["Left_Elbow_Pitch", "Right_Elbow_Pitch",
                                     "Left_Elbow_Yaw", "Right_Elbow_Yaw"])
        self.arm_dof_idx = torch.tensor(
            [self.dof_names.index(n) for n in arm_names if n in self.dof_names],
            dtype=torch.long, device=self.device)
        self.leg_dof_idx = torch.tensor(
            [i for i, n in enumerate(self.dof_names) if n not in arm_names],
            dtype=torch.long, device=self.device)
        if len(self.leg_dof_idx) != self.num_actions:
            raise ValueError(
                "arm_script expects {} leg DOFs to match num_actions={}, got {}. "
                "Check asset.file points at the armswing URDF.".format(
                    self.num_actions, self.num_actions, len(self.leg_dof_idx)))
        self.arm_parked = torch.tensor(
            [float(a.get("parked", {}).get(n, 0.0)) for n in arm_names],
            dtype=torch.float, device=self.device)
        self.arm_moving = torch.tensor(
            [float(a.get("moving", {}).get(n, 0.0)) for n in arm_names],
            dtype=torch.float, device=self.device)
        self.arm_blend = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def _arm_targets(self):
        """Blend between the parked and moving arm poses.

        Blended rather than switched: a step change in 2.9 kg of arm would kick
        the base every time the robot starts or stops, which is a disturbance we
        would then have to learn to reject for no reason.
        """
        a = self.cfg.get("arm_script") or {}
        r = self.cfg["rewards"]
        parked = ((self.goal_dist < r.get("goal_reach_radius", 0.1))
                  & (torch.norm(self.root_states[:, 7:9], dim=-1) < r.get("stop_speed_threshold", 0.1))
                  & (torch.norm(self.root_states[:, 10:13], dim=-1)
                     < max(r.get("stop_ang_speed_threshold", 0.3), 1e-6)))
        tau = float(a.get("blend_tau_s", 0.25))
        alpha = float(min(1.0, self.dt / max(tau, 1e-6)))
        self.arm_blend += alpha * (torch.where(parked, torch.zeros_like(self.arm_blend),
                                               torch.ones_like(self.arm_blend)) - self.arm_blend)
        b = self.arm_blend.unsqueeze(-1)
        return self.arm_parked.unsqueeze(0) * (1.0 - b) + self.arm_moving.unsqueeze(0) * b

    def _dof_targets_from_actions(self):
        if not self.arm_script_on:
            return super()._dof_targets_from_actions()
        full = self.default_dof_pos.repeat(self.num_envs, 1).clone()
        scale = self.cfg["control"]["action_scale"]
        full[:, self.leg_dof_idx] = (self.default_dof_pos[:, self.leg_dof_idx]
                                     + scale * self.actions)
        full[:, self.arm_dof_idx] = self._arm_targets()
        return full

    def _obs_dofs(self, x):
        return x if not self.arm_script_on else x[:, self.leg_dof_idx]

    def _leg_slice(self, x):
        """Keep the observation at 54 by hiding the scripted arm DOFs from it.

        Without this, adding four revolute elbows takes dof_pos and dof_vel from
        12 to 16 each and the observation from 54 to 62 -- which discards every
        warm start, the exact cost this design exists to avoid.
        """
        return x if self.leg_dof_idx is None else x[:, self.leg_dof_idx]

    def _expand_actions(self, dof_targets):
        """Scatter the 12 learned leg targets and the 4 scripted arm targets
        into the full num_dofs vector the PD loop expects."""
        if not self.arm_script_on or self.leg_dof_idx is None:
            return dof_targets
        full = self.default_dof_pos.repeat(self.num_envs, 1).clone()
        full[:, self.leg_dof_idx] = dof_targets[:, :len(self.leg_dof_idx)]
        full[:, self.arm_dof_idx] = self._arm_targets()
        return full

    # ---- 3. BT-flicker on the perceived goal --------------------------------

    def _update_perceived_goal(self):
        super()._update_perceived_goal()
        f = self.cfg["noise"].get("goal_bt_flicker") or {}
        p = float(f.get("prob_per_step", 0.0))
        if p <= 0.0:
            return
        # A behaviour-tree branch flip or a ball re-detection does not nudge the
        # goal by a few cm -- it MOVES it. Model that as a rare, large, discrete
        # jump of the PERCEIVED goal only; the reward still reads ground truth,
        # so the policy is rewarded for filtering rather than for chasing.
        hit = torch.rand(self.num_envs, device=self.device) < p
        if not bool(hit.any()):
            return
        radius = float(f.get("radius_m", 1.0))
        heading = float(f.get("heading_rad", 0.5))
        jump = torch_rand_float(-radius, radius, (self.num_envs, 2), device=self.device)
        jump_h = torch_rand_float(-heading, heading, (self.num_envs, 1), device=self.device).squeeze(1)
        self.goal_obs_cached[:, 0:2] = torch.where(
            hit.unsqueeze(-1), self.goal_obs_cached[:, 0:2] + jump, self.goal_obs_cached[:, 0:2])
        self.goal_obs_cached[:, 2] = torch.where(hit, self.goal_obs_cached[:, 2] + jump_h, self.goal_obs_cached[:, 2])

    # ---- 4. two-class disturbance model -------------------------------------

    def _init_disturbance_schedule(self):
        d = self.cfg["randomization"].get("disturbance") or {}
        if not d.get("enabled", False):
            return
        lo, hi = d.get("interval_s", [3.0, 8.0])
        self.dist_next = torch.randint(
            int(lo / self.dt), max(int(lo / self.dt) + 1, int(hi / self.dt)),
            (self.num_envs,), device=self.device)

    def _push_robots(self):
        d = self.cfg["randomization"].get("disturbance") or {}
        if not d.get("enabled", False):
            super()._push_robots()
            return

        # Per-env timers: the base task pushes EVERY env on the same global step,
        # which correlates the disturbance across the batch and wastes sample
        # diversity. Independent timers give ~num_envs uncorrelated events.
        self.dist_steps_left = (self.dist_steps_left - 1).clamp(min=0)
        expired = self.dist_steps_left == 0
        self.pushing_forces[expired, self.base_indice, :] = 0.0
        self.pushing_torques[expired, self.base_indice, :] = 0.0

        self.dist_next = self.dist_next - 1
        fire = (self.dist_next <= 0).nonzero(as_tuple=False).flatten()
        if len(fire) > 0:
            k = len(fire)
            collide_share = float(d.get("collision_share", 0.5))
            is_collision = torch.rand(k, device=self.device) < collide_share

            cf = d.get("collision", {})
            sf = d.get("support", {})
            fmag = torch.where(
                is_collision,
                torch_rand_float(cf.get("force_n", [40.0, 150.0])[0], cf.get("force_n", [40.0, 150.0])[1], (k, 1), device=self.device).squeeze(1),
                torch_rand_float(sf.get("force_n", [3.0, 15.0])[0], sf.get("force_n", [3.0, 15.0])[1], (k, 1), device=self.device).squeeze(1))
            tmag = torch.where(
                is_collision,
                torch_rand_float(cf.get("torque_nm", [3.0, 20.0])[0], cf.get("torque_nm", [3.0, 20.0])[1], (k, 1), device=self.device).squeeze(1),
                torch_rand_float(sf.get("torque_nm", [0.2, 2.0])[0], sf.get("torque_nm", [0.2, 2.0])[1], (k, 1), device=self.device).squeeze(1))
            dur_s = torch.where(
                is_collision,
                torch_rand_float(cf.get("duration_s", [0.05, 0.15])[0], cf.get("duration_s", [0.05, 0.15])[1], (k, 1), device=self.device).squeeze(1),
                torch_rand_float(sf.get("duration_s", [1.5, 3.0])[0], sf.get("duration_s", [1.5, 3.0])[1], (k, 1), device=self.device).squeeze(1))

            ang = torch_rand_float(-np.pi, np.pi, (k, 1), device=self.device).squeeze(1)
            self.pushing_forces[fire, self.base_indice, 0] = fmag * torch.cos(ang)
            self.pushing_forces[fire, self.base_indice, 1] = fmag * torch.sin(ang)
            tang = torch_rand_float(-1.0, 1.0, (k, 3), device=self.device)
            self.pushing_torques[fire, self.base_indice, :] = tang / tang.norm(dim=-1, keepdim=True).clamp(min=1e-6) * tmag.unsqueeze(-1)

            self.dist_steps_left[fire] = (dur_s / self.dt).long().clamp(min=1)
            lo, hi = d.get("interval_s", [3.0, 8.0])
            self.dist_next[fire] = torch.randint(
                int(lo / self.dt), max(int(lo / self.dt) + 1, int(hi / self.dt)),
                (k,), device=self.device)

        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.pushing_forces),
            gymtorch.unwrap_tensor(self.pushing_torques),
            gymapi.LOCAL_SPACE,
        )

    # ---- 6. goal_reached with an angular-rate term --------------------------

    def _reward_goal_reached(self):
        """armB's winning lever, plus the term it was missing.

        The base version tests root_states[:, 7:9] -- LINEAR velocity only. A
        policy can therefore rock, yaw and pitch in place and still collect the
        +1/step bonus. Two independent video observations converge on this: armB
        (the only arm with goal_reached on) wobbles in place more than armA, and
        armC (no goal_reached) wobbles less than armB. Gating on |omega| too
        closes the loophole without touching what makes the term work.

        stop_ang_speed_threshold <= 0 restores the exact armB behaviour.
        """
        base = super()._reward_goal_reached()
        thr = self.cfg["rewards"].get("stop_ang_speed_threshold", 0.0)
        if thr <= 0.0:
            return base
        settled = torch.norm(self.root_states[:, 10:13], dim=-1) < thr
        return base * settled.float()

    # ---- 5. joint / actuator protection -------------------------------------

    def _margin_penalty(self, value, limit, frac):
        """Quadratic cost on the part of |value| that exceeds frac*limit. Zero in
        the normal operating band, so it cannot distort a healthy gait, and it
        rises before the hard limit rather than at it (codex note, MASTERPLAN)."""
        soft = limit * frac
        return torch.square((value.abs() - soft).clamp(min=0.0)).sum(dim=-1)

    def _reward_dof_pos_margin(self):
        frac = self.cfg["rewards"].get("protect_pos_frac", 0.85)
        lo, hi = self.dof_pos_limits[:, 0], self.dof_pos_limits[:, 1]
        mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo)
        return self._margin_penalty(self.dof_pos - mid, half, frac)

    def _reward_dof_vel_margin(self):
        return self._margin_penalty(self.dof_vel, self.dof_vel_limits,
                                    self.cfg["rewards"].get("protect_vel_frac", 0.85))

    def _reward_torque_margin(self):
        return self._margin_penalty(self.torques, self.torque_limits,
                                    self.cfg["rewards"].get("protect_torque_frac", 0.85))

    def _reward_electrical_power(self):
        """|tau*dq| + copper loss proxy. The base task's `power` term counts only
        POSITIVE mechanical power, which is not what drains the battery: braking
        still burns current, and I^2 R dominates at high torque / low speed."""
        mech = torch.abs(self.torques * self.dof_vel).sum(dim=-1)
        kr = self.cfg["rewards"].get("copper_loss_coef", 0.0)
        return mech + kr * torch.square(self.torques).sum(dim=-1)
