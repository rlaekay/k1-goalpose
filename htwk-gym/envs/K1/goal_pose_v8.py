"""GoalPose v8 -- SmoothTurn: sequential goals with a lookahead window.

Ports "SmoothTurn: Learning to Turn Smoothly for Agile Navigation with
Quadrupedal Robots" (arXiv:2603.12842) onto the K1 goal-pose stack.

WHY THIS AND NOT MORE PATH MODE
-------------------------------
The v7 batch found that path mode raises speed (segment peak median 0.35 ->
1.20 m/s) but destroys waypoint accuracy (6.3 -> 37.9 cm), because a carrot that
is never reachable teaches the policy to hold station behind its target. v7's
`dwell` answers that by making the carrot stop. SmoothTurn answers the same
question from the opposite side, and the paper's framing is the sharper one:

    the failure is not "the goal moves", it is "the reward pays for LINGERING
    at a goal, so the policy decelerates into every one of them".

Their fix is a reward that pays for goals ALREADY BANKED plus progress toward
the current one, so there is nothing to gain by dawdling at any single goal:

    r_seq = (k_t + rho(e_t)) / N          k_t = goals reached so far

Since k_t only ever increases, arriving early is strictly better than arriving
late, and the discontinuity at a goal switch vanishes (rho -> 1 as the error
-> 0, and k_t increments by exactly the amount rho was contributing). That is
what removes the stop-and-go without needing the goal to stop at all.

THE OBSERVATION PROBLEM, AND HOW IT IS AVOIDED
----------------------------------------------
SmoothTurn conditions the policy on n FUTURE goals, taking their obs from
47 to 47+3n. Doing that here would take ours from 54 to 60 and destroy every
warm start -- the same trap flagged for heading sin/cos.

It is avoidable. `commands` has ten slots and only the first four carry
information: slots 4-9 (foot_yaw_L/R, body_pitch/roll_target,
feet_offset_x/y_target) are pinned to [0,0] in every config since milestone 2.
That is exactly 6 dead channels = 2 lookahead goals x (dx, dy, dtheta).

So v8 reuses them, precisely as the goal channels themselves were once carved
out of ParameterWalk's velocity-command slots. Observation width stays 54 and
E0's weights load unchanged.

GOAL ADVANCE: TWO CONDITIONS, NOT ONE
-------------------------------------
The paper's dual reaching condition is the other half of the mechanism:

  direct-switch  tight position AND heading, NO velocity requirement
                 -> lets the robot bank a goal at speed and keep its momentum
  stop-switch    loose position AND heading, but near-zero v AND omega
                 -> catches the case where it stopped close but not close enough

With only the tight condition the robot stalls forever next to a goal it cannot
quite satisfy; with only the loose one it must brake at every goal, which is the
behaviour we are trying to remove. Both together are what make the switch clean.
"""

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import torch_rand_float, get_euler_xyz

assert gymtorch and gymapi

import numpy as np
import torch

from envs.K1.goal_pose_v7 import GoalPoseV7


CATEGORY_SEQ = 6      # goal_category id for sequential-navigation segments


class GoalPoseV8(GoalPoseV7):

    def __init__(self, cfg):
        super().__init__(cfg)
        st = self.cfg["commands"].get("smooth_turn", {}) or {}
        self.st_cfg = st
        self.st_on = bool(st.get("enabled", False))
        self.st_share = float(st.get("share", 0.0))

        n, dev = self.num_envs, self.device
        self.seq_len = int(st.get("num_goals", 4))
        self.n_look = int(st.get("lookahead_goals", 2))
        # world-frame (x, y, theta) for every goal in the sequence
        self.seq_goals = torch.zeros(n, self.seq_len, 3, dtype=torch.float, device=dev)
        self.seq_idx = torch.zeros(n, dtype=torch.long, device=dev)
        self.is_seq_env = torch.zeros(n, dtype=torch.bool, device=dev)
        self.seq_banked = torch.zeros(n, dtype=torch.float, device=dev)
        # curriculum progress c in [0,1] (paper section IV-B)
        self.st_level = float(st.get("init_level", 0.0))
        self._st_reached = 0.0
        self._st_attempts = 0.0
        self._st_last_adjust = 0

        if self.n_look * 3 > 6:
            raise ValueError(
                "lookahead_goals={} needs {} command slots but only 6 are free "
                "(slots 4-9); raising this would change num_observations and "
                "break warm start.".format(self.n_look, self.n_look * 3))

    # ---- sequence sampling (paper eq. 7) ------------------------------------

    def _sample_sequence(self, env_ids):
        """Chain goals: each is placed relative to the previous one, after a
        0.5 m pre-step along the reference heading. The pre-step is what stops
        the sampler from ever producing a degenerate goal sitting on top of its
        predecessor, which would make the turn angle meaningless."""
        if len(env_ids) == 0:
            return
        k = len(env_ids)
        st = self.st_cfg
        c = self.st_level

        dth_max = float(st.get("dtheta_max", np.pi)) * c
        l_lo0, l_lo1 = st.get("len_lo", [1.5, 0.0])     # lo = l_lo0 - l_lo1*c
        l_hi0, l_hi1 = st.get("len_hi", [2.0, 2.5])     # hi = l_hi0 + l_hi1*c
        l_lo = max(0.3, l_lo0 - l_lo1 * c)
        l_hi = l_hi0 + l_hi1 * c

        _, _, yaw = get_euler_xyz(self.base_quat[env_ids])
        ref_th = (yaw + torch.pi) % (2 * torch.pi) - torch.pi
        ref_p = self.base_pos[env_ids, :2].clone()

        for g in range(self.seq_len):
            dth = torch_rand_float(-dth_max, dth_max, (k, 1), device=self.device).squeeze(1)
            ell = torch_rand_float(l_lo, l_hi, (k, 1), device=self.device).squeeze(1)
            th = (ref_th + dth + torch.pi) % (2 * torch.pi) - torch.pi
            # 0.5 m along the REFERENCE heading, then ell along the NEW heading
            px = ref_p[:, 0] + 0.5 * torch.cos(ref_th) + ell * torch.cos(th)
            py = ref_p[:, 1] + 0.5 * torch.sin(ref_th) + ell * torch.sin(th)
            self.seq_goals[env_ids, g, 0] = px
            self.seq_goals[env_ids, g, 1] = py
            self.seq_goals[env_ids, g, 2] = th
            ref_p = torch.stack([px, py], dim=-1)
            ref_th = th

        self.seq_idx[env_ids] = 0
        self.seq_banked[env_ids] = 0.0
        self.goal_category[env_ids] = CATEGORY_SEQ
        self.goal_start_pos[env_ids] = self.base_pos[env_ids, :2]
        self.goal_start_step[env_ids] = self.episode_length_buf[env_ids]

    # ---- reaching conditions (paper eq. 2 and 3) ----------------------------

    def _reached(self):
        st = self.st_cfg
        d = self.goal_dist
        h = self.heading_error.abs()
        v = torch.norm(self.root_states[:, 7:9], dim=-1)
        w = torch.norm(self.root_states[:, 10:13], dim=-1)

        direct = (d < float(st.get("eps_xy", 0.10))) & (h < float(st.get("eps_theta", np.pi / 36)))
        stop = ((d < float(st.get("eps_xy_stop", 0.5)))
                & (h < float(st.get("eps_theta_stop", np.pi / 3)))
                & (v < float(st.get("stop_v", 0.1)))
                & (w < float(st.get("stop_w", 0.1))))
        return direct | stop

    def _advance_sequence(self):
        if not bool(self.is_seq_env.any()):
            return
        hit = self._reached() & self.is_seq_env & (self.seq_idx < self.seq_len)
        if bool(hit.any()):
            self.seq_idx[hit] += 1
            self.seq_banked[hit] += 1.0
            self.goal_segment_id[hit] += 1
            self._st_reached += float(hit.sum().item())
        # a finished sequence immediately gets a new one, so the robot never
        # runs out of goals mid-episode and coasts
        done_seq = self.is_seq_env & (self.seq_idx >= self.seq_len)
        ids = done_seq.nonzero(as_tuple=False).flatten()
        if len(ids) > 0:
            self._st_attempts += float(len(ids)) * self.seq_len
            self._sample_sequence(ids)
        self._update_st_curriculum()

    def _update_st_curriculum(self):
        st = self.st_cfg
        if not st.get("curriculum", True):
            return
        every = int(st.get("adjust_every_steps", 500))
        if self.common_step_counter - self._st_last_adjust < every:
            return
        self._st_last_adjust = self.common_step_counter
        if self._st_attempts < 1.0:
            return
        rate = self._st_reached / max(self._st_attempts, 1.0)
        step = float(st.get("level_step", 0.05))
        if rate > float(st.get("up_threshold", 0.8)):
            self.st_level = min(1.0, self.st_level + step)
        elif rate < float(st.get("down_threshold", 0.2)):
            self.st_level = max(0.0, self.st_level - step)
        self._st_reached = 0.0
        self._st_attempts = 0.0

    # ---- goal + lookahead into the command vector ---------------------------

    def _update_goal_state(self):
        super()._update_goal_state()
        if not bool(self.is_seq_env.any()):
            return
        ids = self.is_seq_env
        idx = self.seq_idx.clamp(max=self.seq_len - 1)
        cur = self.seq_goals[torch.arange(self.num_envs, device=self.device), idx]

        self.goal_pos_world[:, 0] = torch.where(ids, cur[:, 0], self.goal_pos_world[:, 0])
        self.goal_pos_world[:, 1] = torch.where(ids, cur[:, 1], self.goal_pos_world[:, 1])
        self.goal_heading_world = torch.where(ids, cur[:, 2], self.goal_heading_world)

        # recompute the local-frame goal now that we replaced the world target
        to_goal = self.goal_pos_world - self.base_pos[:, :2]
        _, _, base_yaw = get_euler_xyz(self.base_quat)
        base_yaw = (base_yaw + torch.pi) % (2 * torch.pi) - torch.pi
        cy, sy = torch.cos(base_yaw), torch.sin(base_yaw)
        gx = cy * to_goal[:, 0] + sy * to_goal[:, 1]
        gy = -sy * to_goal[:, 0] + cy * to_goal[:, 1]
        self.goal_rel_pos[:, 0] = torch.where(ids, gx, self.goal_rel_pos[:, 0])
        self.goal_rel_pos[:, 1] = torch.where(ids, gy, self.goal_rel_pos[:, 1])
        self.goal_dist[:] = torch.where(ids, torch.norm(to_goal, dim=-1), self.goal_dist)
        he = (self.goal_heading_world - base_yaw + torch.pi) % (2 * torch.pi) - torch.pi
        self.heading_error[:] = torch.where(ids, he, self.heading_error)
        self.commands[:, 0] = self.goal_rel_pos[:, 0]
        self.commands[:, 1] = self.goal_rel_pos[:, 1]
        self.commands[:, 2] = self.heading_error

        # lookahead window -> the six dead style slots (paper eq. 8). Clamped at
        # the last goal, so the window repeats the final goal near the end of a
        # sequence rather than inventing one.
        ar = torch.arange(self.num_envs, device=self.device)
        for i in range(self.n_look):
            nxt = (self.seq_idx + i + 1).clamp(max=self.seq_len - 1)
            g = self.seq_goals[ar, nxt]
            dx = g[:, 0] - self.base_pos[:, 0]
            dy = g[:, 1] - self.base_pos[:, 1]
            lx = cy * dx + sy * dy
            ly = -sy * dx + cy * dy
            lh = (g[:, 2] - base_yaw + torch.pi) % (2 * torch.pi) - torch.pi
            s = 4 + 3 * i
            self.commands[:, s] = torch.where(ids, lx, self.commands[:, s])
            self.commands[:, s + 1] = torch.where(ids, ly, self.commands[:, s + 1])
            self.commands[:, s + 2] = torch.where(ids, lh, self.commands[:, s + 2])

    # ---- lifecycle ----------------------------------------------------------

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if len(env_ids) == 0 or not self.st_on:
            return
        pick = torch.rand(len(env_ids), device=self.device) < self.st_share
        self.is_seq_env[env_ids] = pick
        # sequential envs are exempt from waypoint/path mode entirely
        self.is_path_env[env_ids] = self.is_path_env[env_ids] & ~pick

    def _resample_goals(self):
        super()._resample_goals()
        if not self.st_on:
            return
        # envs that just became sequential, or whose sequence is unset, get one
        need = self.is_seq_env & (self.seq_goals.abs().sum(dim=(1, 2)) == 0)
        ids = need.nonzero(as_tuple=False).flatten()
        if len(ids) > 0:
            self._sample_sequence(ids)
        self._advance_sequence()
        if bool(self.is_seq_env.any()):
            self.goal_category[self.is_seq_env] = CATEGORY_SEQ
            self.gait_frequency[self.is_seq_env] = self.gait_frequency[self.is_seq_env].clamp(min=1.8)

    # ---- sequential goal-reaching reward (paper eq. 4-6) --------------------

    def _reward_seq_goal(self):
        """(k_t + rho(e_t)) / N.

        rho is the paper's smooth kernel 1/(1+(e/sigma)^2), and the error mixes
        heading into position with a distance gate:

            e = d_xy + lambda_theta * rho(d_xy) * d_theta

        The gate is what makes it work at range: far from the goal rho(d_xy) is
        near zero, so heading is almost free and the robot is told only to close
        the distance; heading tightens continuously as it arrives. Penalising
        heading at range would make it turn toward a goal it has not reached,
        which is armD's failure mode.
        """
        st = self.st_cfg
        sig_g = float(st.get("sigma_goal", 0.5))
        lam = float(st.get("lambda_theta", 0.5))
        d = self.goal_dist
        dth = self.heading_error.abs()
        rho_d = 1.0 / (1.0 + (d / sig_g) ** 2)
        e = d + lam * rho_d * dth
        sig_e = float(st.get("sigma_error", 0.5))
        rho_e = 1.0 / (1.0 + (e / sig_e) ** 2)
        r = (self.seq_banked + rho_e) / float(self.seq_len)
        return torch.where(self.is_seq_env, r, torch.zeros_like(r))
