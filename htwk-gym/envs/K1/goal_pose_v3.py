"""GoalPose v3 -- challenger task built ON TOP of goal_pose.py (which stays frozen).

Adds three research-backed mechanisms, each an independent switch in
Goal_Pose_V3.yaml so any of them can be turned off without touching code:

  1. success-adaptive goal curriculum: goal ranges scale with a level in
     [min_level, 1] driven by an EMA of segment success (arrived within
     success_pos_m AND success_heading_rad when the goal resamples). Standard
     automatic-curriculum practice for legged RL.
  2. Rudin-style timed task reward (arXiv:2209.12827 "Advanced Skills by
     Learning Locomotion and Local Navigation End-to-End"): optionally gate the
     task rewards (constellation / goal_reached) to the last final_window_s of
     each goal segment. The policy is then free to choose path/speed/gait and
     only the end pose matters -- the paper shows this time dependence is
     critical for goal-reaching (vs velocity-tracking) behavior, and it removes
     the "rush at max speed" overachievement incentive.
  3. left-right mirror maps (signed permutations of obs/action vectors),
     consumed by RunnerV3's symmetry loss (Abdolhosseini et al. 2019, "On
     Learning Symmetric Locomotion"). K1's URDF joint limits confirm the
     convention: mirrored Roll/Yaw joints flip sign, Pitch/Knee keep it.

Everything else (perceived-goal noise model, goal categories, constellation,
stand_posture, DR suite, ...) is inherited unchanged from GoalPose, so v3
automatically picks up fixes made to the base task.
"""

import torch

from envs.K1.goal_pose import GoalPose


class GoalPoseV3(GoalPose):

    def __init__(self, cfg):
        super().__init__(cfg)
        cc = self.cfg["commands"].get("goal_curriculum", {})
        self.base_goal_ranges = {k: tuple(self.cfg["commands"][k]) for k in ("goal_dx", "goal_dy", "goal_dtheta")}
        self.goal_level = float(cc.get("init_level", 1.0)) if cc.get("enabled", False) else 1.0
        self.goal_success_ema = 0.0
        self._last_level_adjust_step = 0
        self._build_mirror_maps()

    # ---- 1. success-adaptive goal curriculum --------------------------------
    def _resample_goals(self):
        cc = self.cfg["commands"].get("goal_curriculum", {})
        if cc.get("enabled", False) and not getattr(self, "manual_control", False):
            # measure success of the segments that are ending right now, BEFORE
            # the parent replaces their goals (episode_length>0 excludes resets)
            due = (self.episode_length_buf == self.cmd_resample_time) & (self.episode_length_buf > 0)
            ids = due.nonzero(as_tuple=False).flatten()
            if len(ids) > 0:
                ok = (self.goal_dist[ids] < cc["success_pos_m"]) & (self.heading_error[ids].abs() < cc["success_heading_rad"])
                e = cc["ema"]
                self.goal_success_ema = e * self.goal_success_ema + (1.0 - e) * ok.float().mean().item()
            # adjust the level on a slow clock so per-step resample events
            # (with 8192 envs, some env resamples nearly every step) can't
            # rail the level across its whole range in a few seconds
            if self.common_step_counter - self._last_level_adjust_step >= cc.get("adjust_every_steps", 500):
                self._last_level_adjust_step = self.common_step_counter
                if self.goal_success_ema > cc["up_threshold"]:
                    self.goal_level = min(1.0, self.goal_level + cc["step"])
                elif self.goal_success_ema < cc["down_threshold"]:
                    self.goal_level = max(cc["min_level"], self.goal_level - cc["step"])
            for key in ("goal_dx", "goal_dy", "goal_dtheta"):
                lo, hi = self.base_goal_ranges[key]
                self.cfg["commands"][key] = [lo * self.goal_level, hi * self.goal_level]
        super()._resample_goals()

    # ---- 2. Rudin-style timed task reward -----------------------------------
    def _timed_gate(self):
        window = self.cfg["rewards"].get("final_window_s", 0.0)
        if window <= 0.0:
            return 1.0
        remaining_s = (self.cmd_resample_time - self.episode_length_buf).clamp(min=0).float() * self.dt
        return (remaining_s <= window).float()

    def _reward_constellation(self):
        return super()._reward_constellation() * self._timed_gate()

    def _reward_goal_reached(self):
        return super()._reward_goal_reached() * self._timed_gate()

    # ---- 3. mirror maps for RunnerV3's symmetry loss ------------------------
    def _build_mirror_maps(self):
        # Build over the DOFs the POLICY sees, not every DOF in the URDF. With
        # the armswing asset the robot has 16 joints but the observation and the
        # action vector still carry 12: v7 scripts the four elbows and slices
        # them out to keep num_obs at 54 so warm starts survive.
        #
        # Indexing this by the full DOF count wrote 16 entries into 12-wide obs
        # blocks -- dof_pos spilled over dof_vel, and the actions block ran off
        # the end of a 54-long list for a bare IndexError. The permutation was
        # also expressed in full-URDF indices, which do not address a 12-wide
        # action tensor at all. Both were silent until an armswing arm ran.
        #
        # Read arm_script from the config rather than from self.leg_dof_idx:
        # _init_arm_script has not run yet at this point in __init__.
        arm_cfg = self.cfg.get("arm_script") or {}
        scripted = set(arm_cfg.get("joints", []) or []) if arm_cfg.get("enabled", False) else set()
        obs_dof_names = [n for n in self.dof_names if n not in scripted]
        if len(obs_dof_names) != self.num_actions:
            raise ValueError(
                "mirror maps expect {} observed DOFs to match num_actions={}, got {} "
                "(dof_names={}, scripted={})".format(
                    self.num_actions, self.num_actions, len(obs_dof_names),
                    self.dof_names, sorted(scripted)))
        num_dofs = len(obs_dof_names)
        dof_perm = [0] * num_dofs
        dof_sign = [1.0] * num_dofs
        for i, name in enumerate(obs_dof_names):
            if name.startswith("Left_"):
                partner = "Right_" + name[len("Left_"):]
            elif name.startswith("Right_"):
                partner = "Left_" + name[len("Right_"):]
            else:
                partner = name
            dof_perm[i] = obs_dof_names.index(partner)
            # URDF limit tables confirm: Roll/Yaw joints mirror with a sign flip
            # (e.g. Left_Hip_Roll [-0.4,1.57] vs Right_Hip_Roll [-1.57,0.4]),
            # Pitch/Knee joints mirror sign-preserving.
            dof_sign[i] = -1.0 if ("Roll" in name or "Yaw" in name) else 1.0

        # obs layout (54): gravity 0:3 | angvel 3:6 | commands 6:16 |
        #                  clock 16:18 | dof_pos 18:30 | dof_vel 30:42 | actions 42:54
        perm = list(range(self.num_obs))
        sign = [1.0] * self.num_obs
        sign[1] = -1.0                    # gravity_y
        sign[3], sign[5] = -1.0, -1.0     # angvel_x (roll rate), angvel_z (yaw rate)
        sign[7] = -1.0                    # goal_rel_y
        sign[8] = -1.0                    # heading_error
        perm[10], perm[11] = 11, 10       # foot_yaw_L <-> foot_yaw_R ...
        sign[10], sign[11] = -1.0, -1.0   # ... with sign flip (z-rotations)
        sign[13] = -1.0                   # body_roll_target
        sign[15] = -1.0                   # feet_offset_y_target
        # gait clock: swapping legs == half-period phase shift == negate cos & sin
        sign[16], sign[17] = -1.0, -1.0
        for block in (18, 30, 42):        # dof_pos, dof_vel, actions
            for i in range(num_dofs):
                perm[block + i] = block + dof_perm[i]
                sign[block + i] = dof_sign[i]

        self.mirror_obs_perm = torch.tensor(perm, dtype=torch.long, device=self.device)
        self.mirror_obs_sign = torch.tensor(sign, dtype=torch.float, device=self.device)
        self.mirror_act_perm = torch.tensor(dof_perm, dtype=torch.long, device=self.device)
        self.mirror_act_sign = torch.tensor(dof_sign, dtype=torch.float, device=self.device)

    def mirror_obs(self, obs):
        return obs[..., self.mirror_obs_perm] * self.mirror_obs_sign

    def mirror_actions(self, actions):
        return actions[..., self.mirror_act_perm] * self.mirror_act_sign
