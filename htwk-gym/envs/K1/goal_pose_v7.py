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
from isaacgym.torch_utils import torch_rand_float

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
        # curriculum state
        self.speed_level = float(self.path_cfg.get("speed_init", 0.4))
        self.keepup_ema = 0.0
        self._last_speed_adjust = 0
        # disturbance state (per-env timers)
        self.dist_steps_left = torch.zeros(n, dtype=torch.long, device=dev)
        self.dist_next = torch.zeros(n, dtype=torch.long, device=dev)
        self._init_disturbance_schedule()

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
        # commanded path speed, scaled by the curriculum level
        smin, smax = p.get("speed_range_mps", [0.3, 1.6])
        top = smin + (smax - smin) * self.speed_level
        self.path_speed[env_ids] = torch_rand_float(smin, max(smin + 1e-3, top), (k, 1), device=self.device).squeeze(1)
        # anchor the path so the lookahead point starts just ahead of the robot
        wx, wy = self._path_world(self.path_u)
        self.path_origin[env_ids, 0] += self.base_pos[env_ids, 0] - wx[env_ids]
        self.path_origin[env_ids, 1] += self.base_pos[env_ids, 1] - wy[env_ids]

    def _advance_paths(self):
        """Move the lookahead point along the path at path_speed, but never let it
        get further than lookahead_max ahead of the robot (the "leash"). Without
        the leash a robot that cannot keep up is handed an ever-receding target
        and the gradient dies; with it, falling behind simply parks the goal."""
        ids = self.is_path_env
        if not bool(ids.any()):
            return
        p = self.path_cfg
        du = 1.0e-3
        u0 = self.path_u
        x0, y0 = self._path_world(u0)
        x1, y1 = self._path_world(u0 + du)
        ds_du = torch.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2).clamp(min=1.0e-4) / du

        step_u = self.path_dir * self.path_speed * self.dt / ds_du
        gap = torch.norm(torch.stack([x0, y0], dim=-1) - self.base_pos[:, :2], dim=-1)
        leash = float(p.get("lookahead_max_m", 3.5))
        # freeze the phase for envs whose lookahead point already ran too far
        step_u = torch.where(gap > leash, torch.zeros_like(step_u), step_u)
        self.path_u = torch.where(ids, u0 + step_u, u0)

        gx, gy = self._path_world(self.path_u)
        gx1, gy1 = self._path_world(self.path_u + self.path_dir * du)
        tangent = torch.atan2(gy1 - gy, gx1 - gx)
        self.goal_pos_world[:, 0] = torch.where(ids, gx, self.goal_pos_world[:, 0])
        self.goal_pos_world[:, 1] = torch.where(ids, gy, self.goal_pos_world[:, 1])
        self.goal_heading_world = torch.where(ids, tangent, self.goal_heading_world)

        self._update_speed_curriculum(gap[ids])

    def _update_speed_curriculum(self, gap):
        p = self.path_cfg
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
        # goal mode is fixed for the whole episode so eval can attribute cleanly
        self.is_path_env[env_ids] = torch.rand(len(env_ids), device=self.device) < self.path_share
        self._reroll_paths(env_ids[self.is_path_env[env_ids]])

    def _resample_goals(self):
        # base samples a waypoint for every due env; path envs then have their
        # goal overwritten by _advance_paths. The resample event is reused as the
        # natural moment to re-roll path shape/speed/lookahead.
        due = (self.episode_length_buf == self.cmd_resample_time) & self.is_path_env
        super()._resample_goals()
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
