"""HBatch: conservative G1-derived goal-pose experiments.

The class keeps GoalPoseV7's 54-observation/12-action interface and isolates the
new mechanisms requested after the E/G audit: episode-constant joint offsets,
multi-body disturbances, acceleration-preserving high-speed stabilization and
an H3-only touchdown placement hypothesis.  Every mechanism is config gated.
"""

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import (
    get_euler_xyz,
    quat_rotate,
    quat_rotate_inverse,
    torch_rand_float,
)

assert gymtorch

import numpy as np
import torch

from envs.K1.goal_pose_v7 import GoalPoseV7


class GoalPoseHBatch(GoalPoseV7):

    def _init_buffers(self):
        super()._init_buffers()
        self.last_feet_contact = torch.zeros_like(self.feet_contact)
        self.last_stability_vel = torch.zeros_like(self.filtered_lin_vel)
        self.stability_accel_filtered = torch.zeros_like(self.filtered_lin_vel)

    def __init__(self, cfg):
        super().__init__(cfg)
        names = (self.cfg["randomization"].get("disturbance") or {}).get(
            "body_names", [self.cfg["asset"]["base_name"]])
        missing = [name for name in names if name not in self.body_names]
        if missing:
            raise ValueError(
                "disturbance body names were removed or misspelled: {} (loaded: {})".format(
                    missing, self.body_names))
        indices = [self.body_names.index(name) for name in names]
        if not indices:
            indices = [int(self.base_indice)]
        self.dist_body_indices = torch.tensor(indices, dtype=torch.long, device=self.device)
        self.dist_active_body = torch.full(
            (self.num_envs,), int(self.base_indice), dtype=torch.long, device=self.device)
        # 0=none, 1=short collision, 2=long support.  Eval uses this to avoid
        # pooling two physically different recovery problems into one number.
        self.dist_event_kind = torch.zeros(
            self.num_envs, dtype=torch.int8, device=self.device)
        self.dist_last_event_kind = torch.zeros(
            self.num_envs, dtype=torch.int8, device=self.device)
        self.dist_event_serial = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device)
        self.dist_last_expected_impulse = torch.zeros(
            self.num_envs, device=self.device)
        self.dist_last_expected_torque_impulse = torch.zeros(
            self.num_envs, device=self.device)
        self.dist_wrench_apply_calls = 0

    # H1/H2 reflect the robot about its local y=0 plane.  Policy observation
    # and actions are handled by GoalPoseV3; the asymmetric critic channels
    # also have to be reflected for transition-level PPO augmentation.
    def mirror_privileged_obs(self, obs):
        out = obs.clone()
        # base COM y; base linear vy; applied force y.  Torque is an axial
        # vector, hence Tx and Tz change sign under a y reflection while Ty does not.
        out[..., [1, 5, 9, 11, 13]] *= -1.0
        return out

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        if len(env_ids) == 0:
            return
        # A force active at a fall must never leak into the freshly spawned
        # episode.  V7 omitted this reset and could continue pushing a reset robot.
        self.pushing_forces[env_ids] = 0.0
        self.pushing_torques[env_ids] = 0.0
        self.dist_steps_left[env_ids] = 0
        self.dist_event_kind[env_ids] = 0
        self.last_feet_contact[env_ids] = False
        self.last_stability_vel[env_ids] = 0.0
        self.stability_accel_filtered[env_ids] = 0.0
        d = self.cfg["randomization"].get("disturbance") or {}
        if d.get("enabled", False):
            lo, hi = d.get("interval_s", [8.0, 14.0])
            self.dist_next[env_ids] = torch.randint(
                max(1, int(lo / self.dt)), max(2, int(hi / self.dt)),
                (len(env_ids),), device=self.device)

    def step(self, actions):
        out = super().step(actions)
        done = out[2]
        self.last_feet_contact[:] = self.feet_contact & ~done.unsqueeze(-1)
        self.last_stability_vel[:] = self.filtered_lin_vel
        return out

    def _push_robots(self):
        d = self.cfg["randomization"].get("disturbance") or {}
        if not d.get("enabled", False):
            return super()._push_robots()

        self.dist_steps_left = (self.dist_steps_left - 1).clamp(min=0)
        expired = self.dist_steps_left == 0
        self.pushing_forces[expired] = 0.0
        self.pushing_torques[expired] = 0.0
        self.dist_event_kind[expired] = 0

        self.dist_next -= 1
        due = (self.dist_next <= 0).nonzero(as_tuple=False).flatten()
        if len(due) > 0:
            lo, hi = d.get("interval_s", [8.0, 14.0])
            self.dist_next[due] = torch.randint(
                max(1, int(lo / self.dt)), max(2, int(hi / self.dt)),
                (len(due),), device=self.device)

            ramp_steps = max(1, int(d.get("ramp_steps", 1)))
            ramp = min(1.0, self.common_step_counter / float(ramp_steps))
            probability = float(d.get("event_probability", 1.0)) * ramp
            event_prob = torch.full((len(due),), probability, device=self.device)
            boost = float(d.get("high_speed_probability_boost", 1.0))
            if boost > 1.0:
                fast_due = (self.is_path_env[due]
                            & (torch.norm(self.filtered_lin_vel[due, :2], dim=-1)
                               >= float(d.get("high_speed_threshold_mps", 0.8))))
                event_prob[fast_due] *= boost
            fire = due[torch.rand(len(due), device=self.device) < event_prob.clamp(max=1.0)]
            if len(fire) > 0:
                k = len(fire)
                # One event owns one body.  Clear any still-active wrench before
                # replacing it so an accidentally short interval cannot leave a
                # force on the previous body and silently accumulate multi-body
                # loads.  Normal H configs already space events beyond their
                # maximum duration; this makes the invariant explicit in code.
                self.pushing_forces[fire] = 0.0
                self.pushing_torques[fire] = 0.0
                body = self.dist_body_indices[
                    torch.randint(0, len(self.dist_body_indices), (k,), device=self.device)]
                self.dist_active_body[fire] = body
                is_collision = torch.rand(k, device=self.device) < float(d.get("collision_share", 0.5))
                self.dist_event_kind[fire] = torch.where(
                    is_collision,
                    torch.ones(k, dtype=torch.int8, device=self.device),
                    torch.full((k,), 2, dtype=torch.int8, device=self.device),
                )
                self.dist_last_event_kind[fire] = self.dist_event_kind[fire]
                self.dist_event_serial[fire] += 1
                collision, support = d.get("collision", {}), d.get("support", {})

                def sample_pair(section, key, default):
                    bounds = section.get(key, default)
                    return torch_rand_float(bounds[0], bounds[1], (k, 1), device=self.device).squeeze(1)

                fmag = torch.where(is_collision,
                                   sample_pair(collision, "force_n", [40.0, 100.0]),
                                   sample_pair(support, "force_n", [3.0, 8.0]))
                tmag = torch.where(is_collision,
                                   sample_pair(collision, "torque_nm", [3.0, 12.0]),
                                   sample_pair(support, "torque_nm", [0.2, 1.0]))
                duration = torch.where(is_collision,
                                       sample_pair(collision, "duration_s", [0.05, 0.10]),
                                       sample_pair(support, "duration_s", [0.5, 1.5]))
                duration_steps = torch.ceil(duration / self.dt).long().clamp(min=1)
                applied_duration = duration_steps.float() * self.dt
                self.dist_last_expected_impulse[fire] = fmag * applied_duration
                self.dist_last_expected_torque_impulse[fire] = (
                    tmag * applied_duration)

                angle = torch_rand_float(-np.pi, np.pi, (k, 1), device=self.device).squeeze(1)
                self.pushing_forces[fire, body, 0] = fmag * torch.cos(angle)
                self.pushing_forces[fire, body, 1] = fmag * torch.sin(angle)
                axis = torch_rand_float(-1.0, 1.0, (k, 3), device=self.device)
                axis /= axis.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                self.pushing_torques[fire, body] = axis * tmag.unsqueeze(-1)
                self.dist_steps_left[fire] = duration_steps

    def _apply_external_wrenches_substep(self):
        """Apply a held control-step wrench on every decimated physics tick.

        ``apply_rigid_body_force_tensors`` is an immediate-timestep API.  The
        event schedule remains at the 50 Hz control rate, while this hook makes
        its configured duration/impulse real at the 500 Hz physics rate.
        """
        d = self.cfg["randomization"].get("disturbance") or {}
        if not d.get("enabled", False):
            return
        self.dist_wrench_apply_calls += 1
        # ENV_SPACE keeps a long support push fixed in the world instead of
        # rotating the force vector with the robot.  It is still a wrench proxy,
        # not a second simulated robot collision.
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.pushing_forces),
            gymtorch.unwrap_tensor(self.pushing_torques),
            gymapi.ENV_SPACE,
        )

    def _reward_high_speed_stability(self):
        """Stabilize only steady high-speed motion; do not punish acceleration lean."""
        c = self.cfg["rewards"].get("high_speed_stability", {}) or {}
        # Use the existing ~0.2 s low-pass velocity, not one-step trunk-link
        # velocity.  The latter contains stride sway and would classify nearly
        # every high-speed step as "accelerating", making this term inert.
        speed = torch.norm(self.filtered_lin_vel[:, :2], dim=-1)
        acc_instant = (self.filtered_lin_vel - self.last_stability_vel) / self.dt
        alpha = float(c.get("accel_filter_alpha", 0.10))
        self.stability_accel_filtered[:] = (
            alpha * acc_instant + (1.0 - alpha) * self.stability_accel_filtered)
        acc_body = self.stability_accel_filtered
        acc_xy = torch.norm(acc_body[:, :2], dim=-1)
        speed_gate = torch.sigmoid((speed - float(c.get("min_speed_mps", 0.8))) /
                                   float(c.get("speed_width_mps", 0.10)))
        # At |a| above the threshold this gate tends to zero, explicitly
        # preserving the useful forward lean used to accelerate.
        steady_gate = torch.sigmoid((float(c.get("max_accel_mps2", 0.3)) - acc_xy) /
                                    float(c.get("accel_width_mps2", 0.08)))
        gx, gy, gz = self.projected_gravity.unbind(dim=-1)
        pitch = torch.atan2(-gx, -gz)
        roll = torch.atan2(gy, -gz)
        angular = torch.square(self.base_ang_vel[:, :2]).sum(dim=-1)
        vertical = torch.square(self.base_lin_vel[:, 2])
        penalty = (torch.square(pitch) + torch.square(roll)
                   + float(c.get("angular_rate_weight", 0.10)) * angular
                   + float(c.get("vertical_velocity_weight", 0.02)) * vertical)
        return speed_gate * steady_gate * penalty

    def _reward_heel_strike_ahead(self):
        """H3-only kinematic touchdown proxy, gated off outside forward walking.

        Isaac Gym exposes a net force per foot body, not the true sole contact
        point.  We therefore use the first-contact transition plus the known
        heel corner.  A smooth capture-point-like target avoids the brittle
        binary rule "heel must be ahead of trunk" and its over-striding failure.
        """
        c = self.cfg["rewards"].get("heel_strike", {}) or {}
        first = self.feet_contact & ~self.last_feet_contact
        forward = self.base_lin_vel[:, 0] > float(c.get("min_forward_speed_mps", 0.6))
        active = first & forward.unsqueeze(-1) & self.is_path_env.unsqueeze(-1)

        heel_local = torch.tensor([-0.1015, 0.0, -0.03], device=self.device)
        heel_local = heel_local.view(1, 1, 3).expand(self.num_envs, len(self.feet_indices), 3)
        heel_world = self.feet_pos + quat_rotate(
            self.feet_quat.reshape(-1, 4), heel_local.reshape(-1, 3)).reshape_as(self.feet_pos)
        rel_world = heel_world - self.base_pos.unsqueeze(1)
        rel_body = quat_rotate_inverse(
            self.base_quat.unsqueeze(1).expand(-1, len(self.feet_indices), -1).reshape(-1, 4),
            rel_world.reshape(-1, 3),
        ).reshape_as(rel_world)
        target = (float(c.get("velocity_gain_s", 0.08)) * self.base_lin_vel[:, 0]).clip(
            min=float(c.get("target_min_m", 0.02)), max=float(c.get("target_max_m", 0.12)))
        sigma = float(c.get("sigma_m", 0.04))
        score = torch.exp(-torch.square(rel_body[:, :, 0] - target.unsqueeze(-1)) /
                          max(sigma * sigma, 1e-8))
        return (score * active.float()).sum(dim=-1)

    def _compute_observations(self):
        super()._compute_observations()
        # Base GoalPose publishes only body index 0's wrench.  HBatch can push
        # five different bodies and stores those wrenches in ENV_SPACE, so give
        # the asymmetric critic the resultant expressed in the robot frame.
        # This makes the privileged mirror map well-defined and prevents an
        # arm/hip hit from becoming invisible to the critic.
        force_body = quat_rotate_inverse(self.base_quat, self.pushing_forces.sum(dim=1))
        torque_body = quat_rotate_inverse(self.base_quat, self.pushing_torques.sum(dim=1))
        self.privileged_obs_buf[:, 8:11] = force_body * self.cfg["normalization"]["push_force"]
        self.privileged_obs_buf[:, 11:14] = torque_body * self.cfg["normalization"]["push_torque"]
        self.extras["privileged_obs"] = self.privileged_obs_buf
        e = self.extras.setdefault("hbatch", {})
        speed = torch.norm(self.base_lin_vel[:, :2], dim=-1)
        fast = speed >= float((self.cfg["rewards"].get("high_speed_stability") or {}).get(
            "min_speed_mps", 0.8))
        if bool(fast.any()):
            gx, gy, gz = self.projected_gravity[fast].unbind(dim=-1)
            e["high_speed_pitch_abs_mean"] = float(torch.atan2(-gx, -gz).abs().mean().item())
            e["high_speed_roll_abs_mean"] = float(torch.atan2(gy, -gz).abs().mean().item())
            e["high_speed_ang_xy_mean"] = float(torch.norm(self.base_ang_vel[fast, :2], dim=-1).mean().item())
        e["disturbance_active_share"] = float((self.dist_steps_left > 0).float().mean().item())
        e["joint_encoder_bias_abs_mean"] = float(self.joint_encoder_bias.abs().mean().item())
        e["joint_target_offset_abs_mean"] = float(self.joint_target_offset.abs().mean().item())
