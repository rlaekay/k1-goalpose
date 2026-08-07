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

  1. path goal mode -- the goal is a VIRTUAL LEADER: a pose (x, y, heading)
     carrying a twist (v, omega = v * kappa), integrated forward every control
     step, with kappa following a per-env curvature program (figure-8 / circle /
     spiral / rose / wander / serpentine). It is floored so it stays a genuine
     lookahead and leashed so it never runs more than `lookahead_max_m` ahead.
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
        self.path_scale = torch.ones(n, dtype=torch.float, device=dev)   # turn radius, 1/kappa
        self.path_dir = torch.ones(n, dtype=torch.float, device=dev)     # sign of kappa: L/R turn
        self.path_head = torch.zeros(n, dtype=torch.float, device=dev)   # leader heading, world
        self.path_phi = torch.zeros(n, dtype=torch.float, device=dev)    # curvature-program phase
        self.path_speed = torch.zeros(n, dtype=torch.float, device=dev)
        self.lookahead = torch.zeros(n, dtype=torch.float, device=dev)
        # tracking error against the carrot, and the per-segment counters the
        # grid curriculum promotes on. path_lag is exported for eval: goal
        # distance can never reach 0 in path mode (the goal is meant to sit
        # lookahead_min ahead), so raw distance is not the tracking error.
        self.path_lag = torch.zeros(n, dtype=torch.float, device=dev)
        # Keep the other side of the error too.  The former one-sided lag made
        # an overtaken/collapsed carrot (gap < lookahead) look perfect because
        # clamp(gap-lookahead, min=0) returned zero.  Curriculum and eval need
        # the signed margin to distinguish "behind" from "inside the floor".
        self.path_floor_margin = torch.zeros(n, dtype=torch.float, device=dev)
        self.path_floor_deficit = torch.zeros(n, dtype=torch.float, device=dev)
        self.path_goal_rate_limit = torch.zeros(n, dtype=torch.float, device=dev)
        self.path_floor_recovering = torch.zeros(n, dtype=torch.bool, device=dev)
        self.path_floor_recovery_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.path_dwell_left = torch.zeros(n, dtype=torch.long, device=dev)
        self.path_dwell_next = torch.zeros(n, dtype=torch.long, device=dev)
        self.path_was_dwelling = torch.zeros(n, dtype=torch.bool, device=dev)
        self.path_run_gait_frequency = torch.zeros(n, dtype=torch.float, device=dev)
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

    def _curvature(self):
        """kappa(phi) for each shape family, in units of 1/path_scale.

        The old design parametrised each curve in closed form and then tried to
        constrain the carrot's DISTANCE to the robot. Those live in two
        different spaces, and every teleport bug came from the conversion:
        the rate limit was applied to the phase step du, but phase -> world goes
        through ds_du, which varies along the curve, so bounding du did not
        bound the world-space step at all. That is the whole story behind the
        20.3 / 14.5 m/s carrots -- not the projection, not the bisection.

        So the curve is no longer parametrised. The goal is a virtual leader
        with a pose and a twist, integrated forward, and every constraint is a
        world-space distance. There is no second space left to disagree with.
        The shape families survive as curvature PROGRAMS, which is what they
        always were geometrically:

          0 figure-8    kappa flips smoothly, net turning zero over a period
          1 circle      kappa constant
          2 spiral      |kappa| grows then unwinds
          3 rose/star   kappa alternates fast -> petals
          4 wander      two incommensurate harmonics
          5 serpentine  near-square wave: steady pace, reversing yaw
        """
        s, phi = self.path_shape, self.path_phi
        f = torch.ones_like(phi)
        f = torch.where(s == 0, torch.cos(phi), f)
        # s == 1 keeps f = 1
        f = torch.where(s == 2, 0.35 + 0.65 * (0.5 - 0.5 * torch.cos(0.25 * phi)), f)
        f = torch.where(s == 3, torch.cos(2.5 * phi), f)
        f = torch.where(s == 4, 0.6 * torch.sin(phi) + 0.4 * torch.sin(1.7 * phi + 1.1), f)
        f = torch.where(s == 5, torch.tanh(3.0 * torch.sin(phi)), f)
        return self.path_dir * f / self.path_scale.clamp(min=0.2)

    @staticmethod
    def _bounded_annulus_delta(old_goal, robot_pos, nominal_delta,
                                l_min, l_max, max_step, fallback_dir):
        """Minimum 2-D correction toward a robot-centred distance annulus.

        Projecting the nominal candidate radially is the smallest Euclidean
        displacement that satisfies ``l_min <= gap <= l_max``.  The previous
        scalar ``step += l_min-gap`` moved only along leader heading and was
        exact only when that heading happened to be radial; at a tangent it
        under-corrected even before the hard rate limit was applied.

        The returned delta is vector-rate-limited, so a large deficit improves
        smoothly without reintroducing the historical goal teleports.
        """
        candidate = old_goal + nominal_delta
        rel = candidate - robot_pos
        dist = torch.norm(rel, dim=-1)
        fallback = fallback_dir / torch.norm(
            fallback_dir, dim=-1, keepdim=True).clamp(min=1.0e-6)
        direction = rel / dist.unsqueeze(-1).clamp(min=1.0e-6)
        direction = torch.where((dist > 1.0e-6).unsqueeze(-1), direction, fallback)
        target_dist = torch.minimum(torch.maximum(dist, l_min), l_max)
        projected = robot_pos + direction * target_dist.unsqueeze(-1)
        desired = projected - old_goal
        desired_norm = torch.norm(desired, dim=-1)
        scale = torch.minimum(
            torch.ones_like(desired_norm),
            max_step / desired_norm.clamp(min=1.0e-6),
        )
        return desired * scale.unsqueeze(-1)

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
        self.path_dir[env_ids] = torch.where(
            torch.rand(k, device=self.device) < 0.5,
            -torch.ones(k, device=self.device), torch.ones(k, device=self.device))
        self.path_phi[env_ids] = torch_rand_float(-np.pi, np.pi, (k, 1), device=self.device).squeeze(1)
        lo, hi = p.get("lookahead_m", [0.5, 3.0])
        la = torch_rand_float(lo, hi, (k, 1), device=self.device).squeeze(1)

        if self.grid_on:
            # Report on the segment that just ended BEFORE overwriting its cell,
            # then draw the next (speed, curvature) pair from the active set.
            self._grid_report(env_ids)
            flat, si, ci = self._sample_cells(k)
            self.env_cell[env_ids] = flat
            jitter = torch_rand_float(0.85, 1.15, (k, 1), device=self.device).squeeze(1)
            self.path_speed[env_ids] = self.grid_speeds[si] * jitter
            # path_scale IS the turn radius, so curvature is commanded directly.
            # This is what couples yaw rate to speed: the leader turns by
            # kappa * ds each step, i.e. omega = v * kappa, so the grid axis pair
            # IS (v_x, omega_z).
            self.path_scale[env_ids] = 1.0 / self.grid_curvs[ci].clamp(min=1e-3)
        else:
            # commanded path speed, scaled by the scalar curriculum level
            smin, smax = p.get("speed_range_mps", [0.3, 1.6])
            top = smin + (smax - smin) * self.speed_level
            self.path_speed[env_ids] = torch_rand_float(smin, max(smin + 1e-3, top), (k, 1), device=self.device).squeeze(1)

        # Cap the lookahead by the turn radius -- AFTER the grid has had its say.
        # This ran before the grid branch, so a grid segment at kappa = 1.0
        # (radius 1.0 m) kept a cap computed from the discarded random scale of
        # up to 4.0 m, i.e. a 2.4 m carrot around a 1 m turn. The floor is then
        # unsatisfiable, the carrot pins at full leash, and the speed demand the
        # whole mode exists to create quietly disappears. The physical statement
        # is simply: you cannot see 3 m ahead around a 1 m turn.
        frac = float(p.get("lookahead_scale_frac", 0.6))
        self.lookahead[env_ids] = torch.minimum(la, frac * self.path_scale[env_ids])
        self.track_ok_steps[env_ids] = 0.0
        self.track_steps[env_ids] = 0.0
        self.path_lag[env_ids] = 0.0
        self.path_floor_recovering[env_ids] = False
        self.path_floor_recovery_steps[env_ids] = 0
        # Place the leader exactly on the floor, l_min ahead of the robot, facing
        # away from it. It must START at the floor: climbing 0.5-3.0 m at the
        # rate limit takes 50-119 control steps, and for that whole stretch the
        # goal would sit inside the floor -- the flat-gradient state the floor
        # exists to prevent. With a leader this is one line instead of an
        # origin-shift plus an arc-length step.
        head = torch_rand_float(-np.pi, np.pi, (k, 1), device=self.device).squeeze(1)
        self.path_head[env_ids] = head
        la = self.lookahead[env_ids]
        self.goal_pos_world[env_ids, 0] = self.base_pos[env_ids, 0] + la * torch.cos(head)
        self.goal_pos_world[env_ids, 1] = self.base_pos[env_ids, 1] + la * torch.sin(head)
        self.goal_heading_world[env_ids] = head

        # A normal 4-8 s path segment boundary must not restart or terminate the
        # independent 12-24 s dwell schedule. Episode reset owns the first
        # stagger; a fired dwell owns every subsequent interval and duration.

    def _advance_paths(self):
        """Integrate the virtual leader one control step.

        The goal is a pose (x, y, heading) carrying a twist (v, omega=v*kappa).
        That twist pair IS the curriculum grid's two axes, so the coupling
        Margolis et al. (RSS 2022) curriculum over is expressed directly rather
        than encoded through a curve's size.

        Three constraints act, and each fixes a different failure:

          pace   the leader advances at path_speed  -> creates the SPEED DEMAND.
                 A goal that merely sits `L` ahead moves at whatever speed the
                 robot picks, so pure pursuit alone demands nothing. This is the
                 masterplan's core claim: a stationary goal is reachable at any
                 speed, a receding one only at ITS speed.
          floor  gap >= lookahead                   -> keeps it a LOOKAHEAD.
                 Without it the goal drifts onto the robot (measured gap median
                 0.41 m against a 0.5 m minimum) and constellation
                 exp(-w d_con) ~ 1 with gradient ~ 0 -- the reward goes flat
                 exactly where the robot is.
          leash  gap <= lookahead * leash_ratio     -> keeps the gradient alive
                 when the robot cannot keep up, instead of handing it a target
                 that recedes forever.

        All three are world-space distances now, measured the same way the
        reward measures them (goal_dist and constellation are both straight
        line). The old code enforced them on the curve phase u and converted
        through ds_du, which is exactly where the teleports came from.

        ONE hard invariant: the leader's world-space delta is clamped LAST, so
        goal speed can never exceed the rate limit. Floor and leash are soft --
        a brief floor miss only flattens the gradient while the carrot catches
        up and is self-correcting, whereas a teleporting goal is a
        discontinuity the policy cannot track and corrupts what the value
        function is fitted to.
        """
        ids = self.is_path_env
        if not bool(ids.any()) or self._path_advanced_at == self.common_step_counter:
            return
        self._path_advanced_at = self.common_step_counter
        p = self.path_cfg

        # ---- dwell: the carrot periodically STOPS and must be parked on ----
        # The 2026-07-27 v7 eval showed path training wrecking waypoint accuracy:
        # waypoint-only position error was 6.3 cm for E0 (no path training) but
        # 37.9 cm for E1 and 38.7 cm for V7, both path-trained, while their stand
        # category stayed fine at 5.0 cm. A goal that is ALWAYS lookahead ahead
        # and never reachable teaches "hold station behind the target", and that
        # habit is exactly wrong for a waypoint, where the job is to arrive and
        # stop. Mixing the two 50/50 under one reward produced a policy that did
        # neither. Dwell removes the conflict instead of trading it off: the
        # carrot pauses and releases the floor (or arriving would still be
        # impossible), so path mode ends in the same "arrive and stop" state
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
        was_dwelling = self.path_was_dwelling.clone()
        started = ids & dwelling & ~was_dwelling
        resumed = ids & ~dwelling & was_dwelling
        # A parked carrot can be on the robot when dwell ends. Recover the
        # lookahead smoothly under the same hard goal-rate limit, but do not
        # grade those transition steps as failed speed-grid trials.
        self.path_floor_recovering[resumed] = True
        self.path_floor_recovery_steps[resumed] = 0

        if bool(p.get("pause_gait_during_dwell", False)):
            if bool(started.any()):
                self.path_run_gait_frequency[started] = self.gait_frequency[started].clamp(min=1.8)
                self.gait_frequency[started] = 0.0
            if bool(resumed.any()):
                self.gait_frequency[resumed] = self.path_run_gait_frequency[resumed].clamp(min=1.8)
            changed = started | resumed
            if bool(changed.any()):
                self.commands[changed, 3] = self.gait_frequency[changed]
        self.path_was_dwelling = torch.where(ids, dwelling, self.path_was_dwelling)

        l_min = torch.where(dwelling, torch.zeros_like(self.lookahead), self.lookahead)
        l_max = torch.clamp(self.lookahead * float(p.get("leash_ratio", 1.6)),
                            max=float(p.get("lookahead_max_m", 3.5)))
        l_max = torch.maximum(l_max, l_min + 0.1)

        pace = torch.where(dwelling, torch.zeros_like(self.path_speed), self.path_speed)

        # Heading first: the leader turns, then advances along where it now
        # points. Curvature is per unit ARC, so the yaw increment is kappa * ds
        # -- omega = v * kappa falls out without being written separately.
        kappa = self._curvature()
        rx, ry = self.base_pos[:, 0], self.base_pos[:, 1]
        gx, gy = self.goal_pos_world[:, 0], self.goal_pos_world[:, 1]

        nominal_step = pace * self.dt
        head = self.path_head + kappa * nominal_step
        cx, cy = torch.cos(head), torch.sin(head)

        # A normal running robot may drag the floor; a collision spike may not.
        # The old min(v_robot, path_speed) made a 0.3 m/s seed cell cap the goal
        # at ~0.5 m/s even when the warm policy was already moving much faster,
        # so overtaking remained inside the floor for seconds.  H config caps
        # ordinary robot-follow speed at the largest jittered grid command
        # (2.1 m/s), while a separate absolute cap remains teleport-safe.
        v_robot = torch.norm(self.root_states[:, 7:9], dim=-1)
        drag_cap = p.get("drag_speed_cap_mps")
        if drag_cap is None:
            drag = torch.minimum(v_robot, self.path_speed)  # legacy V7 behaviour
        else:
            drag = v_robot.clamp(max=float(drag_cap))
        rate = torch.maximum(pace, drag) * float(p.get("catchup_ratio", 1.5)) + 0.05
        if p.get("floor_recovery_rate_mps") is not None:
            recovery_rate = torch.full_like(
                rate, float(p["floor_recovery_rate_mps"]))
            rate = torch.where(
                self.path_floor_recovering,
                torch.maximum(rate, recovery_rate),
                rate,
            )
        if p.get("goal_rate_max_mps") is not None:
            rate = rate.clamp(max=float(p["goal_rate_max_mps"]))
        self.path_goal_rate_limit = torch.where(ids, rate, self.path_goal_rate_limit)

        constraint_mode = p.get("constraint_mode", "heading_scalar_legacy")
        if constraint_mode == "radial_rate_limited":
            old_goal = torch.stack([gx, gy], dim=-1)
            heading_vec = torch.stack([cx, cy], dim=-1)
            nominal_delta = nominal_step.unsqueeze(-1) * heading_vec
            delta = self._bounded_annulus_delta(
                old_goal,
                self.base_pos[:, :2],
                nominal_delta,
                l_min,
                l_max,
                rate * self.dt,
                heading_vec,
            )
            # Dwell means the world-frame carrot is parked.  It releases both
            # distance constraints instead of chasing a robot that walks away.
            delta = torch.where(dwelling.unsqueeze(-1), torch.zeros_like(delta), delta)
            next_goal = old_goal + delta
            gx, gy = next_goal[:, 0], next_goal[:, 1]
            phase_step = nominal_step
        elif constraint_mode == "heading_scalar_legacy":
            def gap_after(s):
                return torch.sqrt((gx + s * cx - rx) ** 2 + (gy + s * cy - ry) ** 2)

            step = nominal_step + (l_min - gap_after(nominal_step)).clamp(min=0.0)
            step = step - (gap_after(step) - l_max).clamp(min=0.0)
            step = torch.clamp(step, -rate * self.dt, rate * self.dt)
            gx, gy = gx + step * cx, gy + step * cy
            phase_step = step
        else:
            raise ValueError("unknown commands.path.constraint_mode: {}".format(constraint_mode))

        self.path_head = torch.where(ids, head, self.path_head)
        self.path_phi = torch.where(ids, self.path_phi + phase_step / self.path_scale.clamp(min=0.2),
                                    self.path_phi)
        self.goal_pos_world[:, 0] = torch.where(ids, gx, self.goal_pos_world[:, 0])
        self.goal_pos_world[:, 1] = torch.where(ids, gy, self.goal_pos_world[:, 1])
        # The leader's own heading IS the goal heading -- no tangent to
        # differentiate, so no finite-difference noise when the step is tiny
        # (during dwell the old code divided a ~0 displacement to get a tangent).
        self.goal_heading_world = torch.where(ids, head, self.goal_heading_world)

        # How far the robot has fallen behind the carrot. This, not the raw goal
        # distance, is the tracking error: the goal is SUPPOSED to sit l_min
        # ahead, so distance alone can never reach zero.
        gap_now = torch.norm(torch.stack([gx, gy], dim=-1) - self.base_pos[:, :2], dim=-1)
        signed_margin = gap_now - self.lookahead
        self.path_floor_margin = torch.where(ids, signed_margin, self.path_floor_margin)
        self.path_floor_deficit = torch.where(
            ids, (-signed_margin).clamp(min=0.0), self.path_floor_deficit)
        self.path_lag = torch.where(ids, signed_margin.clamp(min=0.0), self.path_lag)
        keep = float(p.get("keepup_gap_m", 2.0))
        running = ids & ~dwelling
        floor_tol = float(p.get("floor_tolerance_m", 0.02))
        recovering = running & self.path_floor_recovering
        scored_running = running & ~recovering
        tracking_ok = (scored_running & (signed_margin >= -floor_tol)
                       & (self.path_lag < keep))
        self.track_ok_steps = torch.where(tracking_ok,
                                          self.track_ok_steps + 1, self.track_ok_steps)
        self.track_steps = torch.where(
            scored_running, self.track_steps + 1, self.track_steps)
        self.path_floor_recovery_steps = torch.where(
            recovering,
            self.path_floor_recovery_steps + 1,
            self.path_floor_recovery_steps,
        )
        recovered = recovering & (signed_margin >= -floor_tol)
        grace_s = float(p.get("floor_recovery_grace_s", 2.0))
        recovery_expired = recovering & (
            self.path_floor_recovery_steps >= max(1, int(np.ceil(grace_s / self.dt))))
        # Recovery is a bounded transition allowance, never a permanent
        # curriculum/eval exclusion.  After the grace period, an unresolved
        # floor deficit is graded as an ordinary tracking failure.
        recovery_done = recovered | recovery_expired
        self.path_floor_recovering[recovery_done] = False
        self.path_floor_recovery_steps[recovery_done] = 0

        self._update_speed_curriculum(signed_margin[scored_running])

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
        initial = g.get("initial_active", "seed")
        if initial == "all":
            self.grid_active[:] = True
        elif initial == "seed":
            self.grid_active[0, 0] = True      # slowest, straightest
        else:
            raise ValueError("speed_grid.initial_active must be 'seed' or 'all'")
        self.env_cell = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.grid_success = torch.zeros(ns, nc, dtype=torch.float, device=self.device)
        self.grid_trials = torch.zeros(ns, nc, dtype=torch.float, device=self.device)

    def get_checkpoint_state(self):
        """Task state that must travel with policy/optimizer checkpoints.

        Historical checkpoints saved only ``curriculum_prob`` from the base
        task, so the V7 speed/curvature grid silently restarted from one slow
        cell on every resume.  H starts from an explicit frozen initial grid;
        checkpoints produced from now on preserve subsequent task state too.
        """
        state = {
            "speed_level": float(self.speed_level),
            "keepup_ema": float(self.keepup_ema),
        }
        if self.grid_on:
            state["path_grid"] = {
                "active": self.grid_active.detach().clone(),
                "trials": self.grid_trials.detach().clone(),
                "success": self.grid_success.detach().clone(),
            }
        return state

    def load_checkpoint_state(self, state):
        if not isinstance(state, dict):
            return
        self.speed_level = float(state.get("speed_level", self.speed_level))
        self.keepup_ema = float(state.get("keepup_ema", self.keepup_ema))
        grid = state.get("path_grid")
        if not (self.grid_on and isinstance(grid, dict)):
            return
        for key, dst in (("active", self.grid_active),
                         ("trials", self.grid_trials),
                         ("success", self.grid_success)):
            src = grid.get(key)
            if src is None or tuple(src.shape) != tuple(dst.shape):
                raise ValueError("checkpoint path_grid.{} shape mismatch".format(key))
            dst.copy_(src.to(device=self.device, dtype=dst.dtype))

    def _sample_cells(self, k):
        flat = self.grid_active.flatten().float()
        idx = torch.multinomial(flat / flat.sum(), k, replacement=True)
        nc = len(self.grid_curvs)
        return idx, idx // nc, idx % nc

    def _grid_report(self, env_ids, terminal_failure=None):
        """Promote neighbours of any cell the policy is now tracking well."""
        if not self.grid_on or len(env_ids) == 0:
            return
        # The first _reroll_paths() after reset has no preceding segment.  It
        # used to record a zero-length failure for every env, biasing trials and
        # success before the policy had taken one action.  A fall after actual
        # rollout still has track_steps > 0 and is retained as a real failure.
        eligible = self.track_steps[env_ids] > 0
        if terminal_failure is not None:
            # A fall during dwell or dwell-resume recovery can occur before a
            # scored running step exists.  It is still a real attempted cell,
            # not the zero-length initialization case this filter removes.
            eligible |= terminal_failure
        env_ids = env_ids[eligible]
        if terminal_failure is not None:
            terminal_failure = terminal_failure[eligible]
        if len(env_ids) == 0:
            return
        ok = (self.track_ok_steps[env_ids] /
              self.track_steps[env_ids].clamp(min=1.0)) > float(
                  self.path_cfg.get("speed_grid", {}).get("up_threshold", 0.8))
        if terminal_failure is not None:
            ok &= ~terminal_failure
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

    def _update_speed_curriculum(self, signed_margin):
        p = self.path_cfg
        if self.grid_on:
            return   # the grid replaces the scalar level entirely
        if not p.get("speed_curriculum", False) or signed_margin.numel() == 0:
            return
        target = float(p.get("keepup_gap_m", 2.0))
        floor_tol = float(p.get("floor_tolerance_m", 0.02))
        tracked = (signed_margin >= -floor_tol) & (signed_margin < target)
        e = float(p.get("ema", 0.995))
        self.keepup_ema = e * self.keepup_ema + (1.0 - e) * float(
            tracked.float().mean().item())
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
        # Close the old path segment before base reset overwrites episode state.
        # A physical fall is never allowed to promote a speed/curvature cell,
        # even if its pre-fall gap ratio happened to look good.
        if (len(env_ids) > 0 and getattr(self, "grid_on", False)
                and hasattr(self, "is_path_env")):
            old_path = env_ids[self.is_path_env[env_ids]]
            if len(old_path) > 0:
                physical = self.extras.get("physical_failures")
                if isinstance(physical, torch.Tensor):
                    terminal_failure = physical[old_path]
                else:
                    episode_timeout = self.episode_length_buf[old_path] > np.ceil(
                        self.cfg["rewards"]["episode_length_s"] / self.dt)
                    terminal_failure = ~episode_timeout
                self._grid_report(old_path, terminal_failure=terminal_failure)
        if len(env_ids) > 0 and hasattr(self, "track_steps"):
            self.track_ok_steps[env_ids] = 0.0
            self.track_steps[env_ids] = 0.0
        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        # goal mode is fixed for the whole episode so eval can attribute cleanly.
        # The path itself is NOT rolled here: base _reset_idx zeroes both
        # episode_length_buf and cmd_resample_time, so these envs are "due" in the
        # _resample_goals call that follows, and rolling there uses a base_pos that
        # has actually been refreshed from the new root state.
        self.is_path_env[env_ids] = torch.rand(len(env_ids), device=self.device) < self.path_share
        dw = self.path_cfg.get("dwell") or {}
        if dw.get("enabled", False):
            hi = float(dw.get("interval_s", [4.0, 10.0])[1])
            self.path_dwell_next[env_ids] = torch.randint(
                1, max(2, int(hi / self.dt)), (len(env_ids),), device=self.device)
        else:
            self.path_dwell_next[env_ids] = 0
        self.path_dwell_left[env_ids] = 0
        self.path_was_dwelling[env_ids] = False
        self.path_floor_margin[env_ids] = 0.0
        self.path_floor_deficit[env_ids] = 0.0
        self.path_goal_rate_limit[env_ids] = 0.0
        self.path_floor_recovering[env_ids] = False
        self.path_floor_recovery_steps[env_ids] = 0

    def _resample_goals(self):
        # base samples a waypoint for every due env; path envs then have their
        # goal overwritten by _advance_paths. The resample event is reused as the
        # natural moment to re-roll path shape/speed/lookahead.
        all_due = self.episode_length_buf == self.cmd_resample_time
        due = all_due & self.is_path_env
        # Base resampling writes a temporary waypoint for every due env.  If a
        # path segment rolls while its dwell is active, preserve the parked
        # world position after drawing the next path parameters; otherwise the
        # supposedly stopped carrot teleports at an unrelated 4-8 s boundary.
        parked_due = due & (self.path_dwell_left > 0)
        parked_ids = parked_due.nonzero(as_tuple=False).flatten()
        parked_goal = self.goal_pos_world[parked_ids].clone()
        parked_heading = self.goal_heading_world[parked_ids].clone()
        parked_path_head = self.path_head[parked_ids].clone()
        super()._resample_goals()
        if not getattr(self, "manual_control", False):
            self.goal_segment_id[all_due] += 1
        ids = due.nonzero(as_tuple=False).flatten()
        if len(ids) > 0:
            self._reroll_paths(ids)
        if len(parked_ids) > 0:
            self.goal_pos_world[parked_ids] = parked_goal
            self.goal_heading_world[parked_ids] = parked_heading
            self.path_head[parked_ids] = parked_path_head
        self._advance_paths()
        if bool(self.is_path_env.any()):
            self.goal_category[self.is_path_env] = CATEGORY_PATH
            if bool(self.path_cfg.get("pause_gait_during_dwell", False)):
                parked = self.is_path_env & (self.path_dwell_left > 0)
                running = self.is_path_env & ~parked
                # Never clamp parked 0 Hz back to 1.8 and overwrite the saved
                # cadence: _resample_goals() is called every control step, not
                # only when a command is due.  Preserve the pre-dwell value
                # until _advance_paths() restores it on the transition out.
                self.gait_frequency[running] = self.gait_frequency[running].clamp(min=1.8)
                self.path_run_gait_frequency[running] = self.gait_frequency[running]
                self.gait_frequency[parked] = 0.0
            else:
                # A moving lookahead has no stand-still case in legacy mode.
                self.gait_frequency[self.is_path_env] = self.gait_frequency[
                    self.is_path_env].clamp(min=1.8)
                self.path_run_gait_frequency[self.is_path_env] = self.gait_frequency[
                    self.is_path_env]
            self.commands[self.is_path_env, 3] = self.gait_frequency[self.is_path_env]
        self._couple_cadence_to_speed()

    def _couple_cadence_to_speed(self):
        """Pick cadence from the speed the goal actually demands.

        gait_frequency is otherwise drawn uniformly from [1.8, 2.4] Hz with no
        reference to the goal, and one cycle is two steps, so stride length is
        forced to v / (2f). At 1.8 Hz a 1.5 m/s demand needs a 0.417 m stride
        against roughly 0.28 m of leg travel -- physically impossible, and the
        only compliant response left to the policy is to throw the torso forward
        and fall in the intended direction. That is the high-speed lean the user
        observed, and it is also why G1's achieved speed sat flat at 0.32-0.49
        m/s across every commanded bin: cadence, not effort, was the binding
        constraint.

        This is a COMMAND-DISTRIBUTION change, not a reward: nothing is being
        paid for holding a posture. The stride the task asks for simply becomes
        one the leg can take, and the lean stops being the optimal answer.
        Observation width is untouched -- gait_frequency already occupies
        command slot 3.
        """
        c = self.cfg["commands"].get("cadence_coupling") or {}
        if not c.get("enabled", False):
            return
        stride = float(c.get("max_stride_m", 0.28))
        lo, hi = c.get("hz_range", [1.6, 3.2])
        # Required speed for this segment: how far, over how long it has to last.
        secs = (self.cmd_resample_time - self.episode_length_buf).clamp(min=1).float() * self.dt
        need = (self.goal_dist / secs.clamp(min=1e-3))
        f = (need / (2.0 * stride)).clamp(min=float(lo), max=float(hi))
        # Standing goals keep a frozen clock; overriding it would re-enable the
        # stepping incentive that makes standing still non-optimal.
        live = self.gait_frequency > 1.0e-8
        self.gait_frequency = torch.where(live, f, self.gait_frequency)
        self.commands[:, 3] = self.gait_frequency

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
            running = self.is_path_env & (self.path_dwell_left == 0)
            if bool(running.any()):
                e["path_floor_deficit_mean"] = float(
                    self.path_floor_deficit[running].mean().item())
                e["path_inside_floor_share"] = float(
                    (self.path_floor_margin[running] < 0.0).float().mean().item())
                e["path_floor_recovering_share"] = float(
                    self.path_floor_recovering[running].float().mean().item())

        # Joint protection: a zero penalty is ambiguous -- it means either
        # "well within limits" or "the term is switched off". Margin occupancy
        # disambiguates, and is the number that maps to the hardware spec.
        half = 0.5 * (self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0])
        mid = 0.5 * (self.dof_pos_limits[:, 1] + self.dof_pos_limits[:, 0])
        e["dof_pos_occupancy"] = float(((self.dof_pos - mid).abs() / half.clamp(min=1e-6)).max(dim=-1)[0].mean().item())
        e["torque_occupancy"] = float((self.torques.abs() / self.torque_limits.clamp(min=1e-6)).max(dim=-1)[0].mean().item())
        e["torque_saturated"] = float((self.torques.abs() > 0.95 * self.torque_limits).float().mean().item())

        # ⛔ 위 두 줄은 **한 프레임**이다(eval 은 롤아웃 끝에 한 번 읽는다).
        # 아래가 롤아웃 전체 누산이고, 인용해도 되는 쪽이다 -- RETRACTIONS C11.
        n = max(1, int(getattr(self, "_tq_steps", 0)))
        if hasattr(self, "_tq_occ_sum"):
            names = [d.replace("_Pitch", "P").replace("_Roll", "R").replace("_Yaw", "Y")
                     for d in self.dof_names]
            e["torque_by_joint"] = {
                names[i]: {
                    "limit": round(float(self.torque_limits[i]), 1),
                    "mean_Nm": round(float(self._tq_abs_sum[i]) / n, 3),
                    "mean_occ": round(float(self._tq_occ_sum[i]) / n, 4),
                    "peak_occ": round(float(self._tq_occ_max[i]), 4),
                    "sat_share": round(float(self._tq_sat_sum[i]) / n, 5),
                }
                for i in range(self.num_dofs)
            }
            e["torque_steps_accumulated"] = n

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
                                     + scale * self.actions
                                     + self.joint_target_offset[:, self.leg_dof_idx])
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
        # Draw the FIRST event from a uniform PHASE within the interval, not from
        # the interval itself. Drawing randint(lo/dt, hi/dt) means no env can be
        # disturbed before `lo` seconds have passed, so with interval [8, 14] the
        # opening 8 s of training -- 27% of a 30 s episode -- was guaranteed
        # disturbance-free for every env at once. Smoke caught it as "active on
        # 0/300 steps"; the training-distribution defect is the bigger half.
        # Same fix, same reason, as the dwell stagger in _reroll_paths.
        self.dist_next = torch.randint(
            1, max(2, int(hi / self.dt)), (self.num_envs,), device=self.device)

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
        if self.path_cfg.get("arrival_reward_mode", "legacy") == "dwell_only":
            arrival_mode = (~self.is_path_env) | (self.path_dwell_left > 0)
            base = base * arrival_mode.float()
        thr = self.cfg["rewards"].get("stop_ang_speed_threshold", 0.0)
        if thr <= 0.0:
            return base
        settled = torch.norm(self.root_states[:, 10:13], dim=-1) < thr
        return base * settled.float()

    def _reward_stand_posture(self):
        reward = super()._reward_stand_posture()
        if self.path_cfg.get("arrival_reward_mode", "legacy") == "dwell_only":
            arrival_mode = (~self.is_path_env) | (self.path_dwell_left > 0)
            reward = reward * arrival_mode.float()
        return reward

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
