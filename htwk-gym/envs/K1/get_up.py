"""K1 get-up task (v4) -- learn to stand up from arbitrary fallen poses.

22-DOF K1_serial.urdf (head 2 + arms 8 + legs 12): arms are essential to push
off the ground. Design follows FRASA (arXiv:2410.08655) adapted to K1's 3D
falls (projected gravity instead of sagittal trunk pitch) and to this
framework's vectorized-env conventions:

  - reset: random joint pose + fully random base orientation, released at
    drop_height and settled for a random number of control steps. During
    settling the policy is ignored, rewards are zeroed, and extras["settling"]
    tells the off-policy runner to skip those transitions.
  - actions (22): joint-velocity commands integrated into PD position targets
    (dof_targets += a * target_vel_scale * dt, clamped to soft joint limits).
    Get-up needs +-2 rad excursions and slow deployable motions -- FRASA's
    velocity actions give both.
  - rewards: single posture kernel exp(-(joint error + uprightness + height))
    toward a crouched stand, plus an upright-hold bonus and small smoothness /
    effort penalties. No height termination (the robot starts on the floor).
  - off-policy correctness: observations are computed BEFORE resets and
    published as extras["terminal_obs"] / ["terminal_privileged_obs"], so the
    runner has the true s' for done rows (the returned obs is post-reset).

Trainer: utils/runner_crossq.py via train_v4.py (CrossQ -- user decision).
The env is runner-agnostic: a PPO fallback only needs a config copy.
    python train_v4.py --task K1/Get_Up --headless True --sim_device cuda:0 --rl_device cuda:0
"""

import os

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import (
    get_axis_params,
    to_torch,
    quat_rotate_inverse,
)

assert gymtorch

import torch
import numpy as np

from envs.base_task import BaseTask
from utils.utils import apply_randomization


class GetUp(BaseTask):

    def __init__(self, cfg):
        super().__init__(cfg)
        self._create_envs()
        self.gym.prepare_sim(self.sim)
        self._init_buffers()
        self._prepare_reward_function()

    def _create_envs(self):
        self.num_envs = self.cfg["env"]["num_envs"]
        asset_cfg = self.cfg["asset"]
        asset_root = os.path.dirname(asset_cfg["file"])
        asset_file = os.path.basename(asset_cfg["file"])

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = asset_cfg["default_dof_drive_mode"]
        asset_options.collapse_fixed_joints = asset_cfg["collapse_fixed_joints"]
        asset_options.replace_cylinder_with_capsule = asset_cfg["replace_cylinder_with_capsule"]
        asset_options.flip_visual_attachments = asset_cfg["flip_visual_attachments"]
        asset_options.fix_base_link = asset_cfg["fix_base_link"]
        asset_options.density = asset_cfg["density"]
        asset_options.angular_damping = asset_cfg["angular_damping"]
        asset_options.linear_damping = asset_cfg["linear_damping"]
        asset_options.max_angular_velocity = asset_cfg["max_angular_velocity"]
        asset_options.max_linear_velocity = asset_cfg["max_linear_velocity"]
        asset_options.armature = asset_cfg["armature"]
        asset_options.thickness = asset_cfg["thickness"]
        asset_options.disable_gravity = asset_cfg["disable_gravity"]

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)

        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        self.dof_pos_limits = torch.zeros(self.num_dofs, 2, dtype=torch.float, device=self.device)
        self.dof_vel_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        self.torque_limits = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            self.dof_pos_limits[i, 0] = dof_props_asset["lower"][i].item()
            self.dof_pos_limits[i, 1] = dof_props_asset["upper"][i].item()
            self.dof_vel_limits[i] = dof_props_asset["velocity"][i].item()
            self.torque_limits[i] = dof_props_asset["effort"][i].item()

        self.dof_stiffness = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.dof_damping = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.dof_friction = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            found = False
            for name in self.cfg["control"]["stiffness"].keys():
                if name in self.dof_names[i]:
                    self.dof_stiffness[:, i] = self.cfg["control"]["stiffness"][name]
                    self.dof_damping[:, i] = self.cfg["control"]["damping"][name]
                    found = True
            if not found:
                raise ValueError(f"PD gain of joint {self.dof_names[i]} were not defined")
        self.dof_stiffness = apply_randomization(self.dof_stiffness, self.cfg["randomization"].get("dof_stiffness"))
        self.dof_damping = apply_randomization(self.dof_damping, self.cfg["randomization"].get("dof_damping"))
        self.dof_friction = apply_randomization(self.dof_friction, self.cfg["randomization"].get("dof_friction"))

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.base_indice = self.gym.find_asset_rigid_body_index(robot_asset, asset_cfg["base_name"])
        # bodies whose ground contact is monitored (head protection metric)
        head_names = [s for s in body_names if "Head" in s]
        self.head_indices = torch.zeros(len(head_names), dtype=torch.long, device=self.device)
        for i, name in enumerate(head_names):
            self.head_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, name)

        # randomize contact properties on ALL shapes (whole body touches the
        # ground here, unlike locomotion where only feet matter)
        base_init_state_list = (
            self.cfg["init_state"]["pos"] + self.cfg["init_state"]["rot"] + self.cfg["init_state"]["lin_vel"] + self.cfg["init_state"]["ang_vel"]
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device)
        start_pose = gymapi.Transform()

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.envs = []
        self.actor_handles = []
        self.base_mass_scaled = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[2] += self.base_init_state[2]
            start_pose.p = gymapi.Vec3(*pos)

            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, asset_cfg["name"], i, asset_cfg["self_collisions"], 0)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)
            shape_props = self._process_rigid_shape_props(shape_props)
            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, shape_props)

            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

    def _process_rigid_body_props(self, props, i):
        for j in range(self.num_bodies):
            if j == self.base_indice:
                props[j].com.x, self.base_mass_scaled[i, 0] = apply_randomization(
                    props[j].com.x, self.cfg["randomization"].get("base_com"), return_noise=True
                )
                props[j].com.y, self.base_mass_scaled[i, 1] = apply_randomization(
                    props[j].com.y, self.cfg["randomization"].get("base_com"), return_noise=True
                )
                props[j].com.z, self.base_mass_scaled[i, 2] = apply_randomization(
                    props[j].com.z, self.cfg["randomization"].get("base_com"), return_noise=True
                )
                props[j].mass, self.base_mass_scaled[i, 3] = apply_randomization(
                    props[j].mass, self.cfg["randomization"].get("base_mass"), return_noise=True
                )
            else:
                props[j].com.x = apply_randomization(props[j].com.x, self.cfg["randomization"].get("other_com"))
                props[j].com.y = apply_randomization(props[j].com.y, self.cfg["randomization"].get("other_com"))
                props[j].com.z = apply_randomization(props[j].com.z, self.cfg["randomization"].get("other_com"))
                props[j].mass = apply_randomization(props[j].mass, self.cfg["randomization"].get("other_mass"))
            props[j].invMass = 1.0 / props[j].mass
        return props

    def _process_rigid_shape_props(self, props):
        # every body can touch the ground in this task -> randomize all shapes
        for i in range(len(props)):
            props[i].friction = apply_randomization(0.0, self.cfg["randomization"].get("friction"))
            props[i].compliance = apply_randomization(0.0, self.cfg["randomization"].get("compliance"))
            props[i].restitution = apply_randomization(0.0, self.cfg["randomization"].get("restitution"))
        return props

    def _get_env_origins(self):
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        num_cols = np.floor(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_cols)
        xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="ij")
        spacing = self.cfg["env"]["env_spacing"]
        self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
        self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
        self.env_origins[:, 2] = 0.0

    def _init_buffers(self):
        self.num_obs = self.cfg["env"]["num_observations"]
        self.num_privileged_obs = self.cfg["env"]["num_privileged_obs"]
        self.num_actions = self.cfg["env"]["num_actions"]
        self.dt = self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float, device=self.device)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, dtype=torch.float, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.time_out_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}
        self.extras["rew_terms"] = {}

        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)
        self.base_pos = self.root_states[:, 0:3]
        self.base_quat = self.root_states[:, 3:7]

        self.common_step_counter = 0
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.dof_targets = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        # drop-and-settle machinery
        self.settle_steps_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.upright_hold_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # soft joint limits: both the integrated PD targets and the drop-pose
        # sampler stay inside these
        limit_mid = 0.5 * (self.dof_pos_limits[:, 0] + self.dof_pos_limits[:, 1])
        limit_half = 0.5 * (self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0])
        soft = self.cfg["control"]["soft_target_limit"]
        self.soft_lower = (limit_mid - soft * limit_half).unsqueeze(0)
        self.soft_upper = (limit_mid + soft * limit_half).unsqueeze(0)

        def build_pose(key):
            pose = torch.zeros(1, self.num_dofs, dtype=torch.float, device=self.device)
            angles = self.cfg["init_state"][key]
            for i in range(self.num_dofs):
                found = False
                for name in angles.keys():
                    if name in self.dof_names[i] and name != "default":
                        pose[:, i] = angles[name]
                        found = True
                if not found:
                    pose[:, i] = angles["default"]
            return pose

        self.default_dof_pos = build_pose("default_joint_angles")
        # posture the reward kernel pulls toward (crouched stand by default; set
        # target_joint_angles to the PREP stand for a direct locomotion handoff)
        self.target_dof_pos = build_pose("target_joint_angles")

        # curriculum stubs: unused here but read by runner logging/save
        self.curriculum_prob = torch.zeros(1, dtype=torch.float, device=self.device)
        self.mean_lin_vel_level = 0.0
        self.mean_ang_vel_level = 0.0
        self.max_lin_vel_level = 0.0
        self.max_ang_vel_level = 0.0

    def _prepare_reward_function(self):
        self.reward_scales = self.cfg["rewards"]["scales"].copy()
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            self.reward_names.append(name)
            self.reward_functions.append(getattr(self, "_reward_" + name))

    def reset(self):
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self._compute_observations()
        self.extras["settling"] = self.episode_length_buf < self.settle_steps_buf
        self.extras["terminal_obs"] = self.obs_buf.clone()
        self.extras["terminal_privileged_obs"] = self.privileged_obs_buf.clone()
        return self.obs_buf, self.extras

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        rand_cfg = self.cfg["randomization"]

        # random joint pose inside the soft limits (uniform) or near-default
        if rand_cfg.get("init_pose_mode", "uniform") == "uniform":
            u = torch.rand(n, self.num_dofs, device=self.device)
            self.dof_pos[env_ids] = self.soft_lower + u * (self.soft_upper - self.soft_lower)
        else:
            self.dof_pos[env_ids] = apply_randomization(self.default_dof_pos, rand_cfg.get("init_dof_pos"))
        self.dof_vel[env_ids] = 0.0

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
        )

        # released above the ground in a fully random orientation (uniform quat)
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, 0:2] += self.env_origins[env_ids, 0:2]
        self.root_states[env_ids, 2] = rand_cfg["drop_height"]
        quat = torch.randn(n, 4, device=self.device)
        self.root_states[env_ids, 3:7] = quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)
        self.root_states[env_ids, 7:13] = 0.0
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.root_states), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
        )

        # PD targets frozen at the drop pose until settling ends
        self.dof_targets[env_ids] = self.dof_pos[env_ids]
        self.settle_steps_buf[env_ids] = torch.randint(
            rand_cfg["settle_steps"][0], rand_cfg["settle_steps"][1] + 1, (n,), device=self.device
        )
        self.episode_length_buf[env_ids] = 0
        self.upright_hold_time[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.extras["time_outs"] = self.time_out_buf

    def step(self, actions):
        # pre physics: integrate velocity actions into PD targets (settling envs
        # keep their frozen drop-pose targets and ignore the policy)
        self.actions[:] = torch.clip(actions, -self.cfg["normalization"]["clip_actions"], self.cfg["normalization"]["clip_actions"])
        settling = self.episode_length_buf < self.settle_steps_buf
        active = (~settling).unsqueeze(-1).float()
        self.dof_targets += self.actions * self.cfg["control"]["target_vel_scale"] * self.dt * active
        self.dof_targets.clamp_(self.soft_lower, self.soft_upper)

        self.torques.zero_()
        for _ in range(self.cfg["control"]["decimation"]):
            dof_torques = self.dof_stiffness * (self.dof_targets - self.dof_pos) - self.dof_damping * self.dof_vel
            friction = torch.min(self.dof_friction, dof_torques.abs()) * torch.sign(dof_torques)
            dof_torques = torch.clip(dof_torques - friction, min=-self.torque_limits, max=self.torque_limits)
            self.torques += dof_torques
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(dof_torques))
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.torques /= self.cfg["control"]["decimation"]
        self.render()

        # post physics
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        upright = self._is_upright()
        self.upright_hold_time = torch.where(upright, self.upright_hold_time + self.dt, torch.zeros_like(self.upright_hold_time))

        self._check_termination()
        self._compute_reward()
        # settling envs earn nothing and their transitions are skipped upstream
        self.rew_buf *= (~settling).float()

        # off-policy correctness: publish pre-reset observations as s'
        self._compute_observations()
        self.extras["terminal_obs"] = self.obs_buf.clone()
        self.extras["terminal_privileged_obs"] = self.privileged_obs_buf.clone()
        self.extras["settling"] = settling

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self._reset_idx(env_ids)
            self._compute_observations()

        self.last_actions[:] = self.actions
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _is_upright(self):
        near_vertical = self.projected_gravity[:, 2] < -self.cfg["rewards"]["upright_gravity_z"]
        tall_enough = self.base_pos[:, 2] > self.cfg["rewards"]["upright_height"]
        return near_vertical & tall_enough

    def _check_termination(self):
        # NO height termination -- lying on the floor is the whole point.
        self.reset_buf = torch.norm(self.base_ang_vel, dim=-1) > self.cfg["rewards"]["terminate_ang_vel"]
        self.reset_buf |= self.root_states[:, 7:13].square().sum(dim=-1) > self.cfg["rewards"]["terminate_vel"]
        self.time_out_buf = self.episode_length_buf > np.ceil(self.cfg["rewards"]["episode_length_s"] / self.dt)
        # success exit: held upright long enough -> treated as timeout so the
        # off-policy target still bootstraps (not a "death" state)
        success = self.upright_hold_time >= self.cfg["rewards"]["hold_success_s"]
        self.time_out_buf |= success
        self.reset_buf |= self.time_out_buf
        self.extras["time_outs"] = self.time_out_buf
        # episode-summed tensorboard metrics (fire once per episode at most)
        self.extras["rew_terms"]["getup_success"] = success.float()

    def _compute_reward(self):
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.extras["rew_terms"][name] = rew
        if self.cfg["rewards"]["only_positive_rewards"]:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

    def _compute_observations(self):
        self.obs_buf = torch.cat(
            (
                apply_randomization(self.projected_gravity, self.cfg["noise"].get("gravity")),
                apply_randomization(self.base_ang_vel, self.cfg["noise"].get("ang_vel")) * self.cfg["normalization"]["ang_vel"],
                apply_randomization(self.dof_pos - self.default_dof_pos, self.cfg["noise"].get("dof_pos")) * self.cfg["normalization"]["dof_pos"],
                apply_randomization(self.dof_vel, self.cfg["noise"].get("dof_vel")) * self.cfg["normalization"]["dof_vel"],
                (self.dof_targets - self.default_dof_pos) * self.cfg["normalization"]["dof_pos"],
                self.actions,
            ),
            dim=-1,
        )
        self.privileged_obs_buf = torch.cat(
            (
                self.base_mass_scaled,
                self.base_lin_vel,
                self.base_pos[:, 2].unsqueeze(-1),
            ),
            dim=-1,
        )
        self.extras["privileged_obs"] = self.privileged_obs_buf

    # ------------ reward functions ----------------
    def _reward_posture_kernel(self):
        # FRASA's exp(-||state - target||^2) with K1's 3D uprightness folded in:
        # joint-space distance to the target stand + gravity alignment + height.
        joint_err = torch.sum(torch.square(self.dof_pos - self.target_dof_pos), dim=-1) / self.cfg["rewards"]["posture_joint_sigma"]
        upright_err = torch.sum(
            torch.square(self.projected_gravity - torch.tensor([0.0, 0.0, -1.0], device=self.device)), dim=-1
        ) / self.cfg["rewards"]["posture_gravity_sigma"]
        height_err = torch.square(self.base_pos[:, 2] - self.cfg["rewards"]["posture_height_target"]) / self.cfg["rewards"]["posture_height_sigma"]
        return torch.exp(-(joint_err + upright_err + height_err))

    def _reward_upright_hold(self):
        return self._is_upright().float()

    def _reward_action_smooth(self):
        # FRASA's exp(-||a - a_last||) smoothness bonus
        return torch.exp(-torch.norm(self.actions - self.last_actions, dim=-1))

    def _reward_torques(self):
        return torch.sum(torch.square(self.torques), dim=-1)

    def _reward_dof_vel(self):
        return torch.sum(torch.square(self.dof_vel), dim=-1)

    def _reward_dof_pos_limits(self):
        lower = self.dof_pos_limits[:, 0] + 0.5 * (1 - self.cfg["rewards"]["soft_dof_pos_limit"]) * (
            self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        )
        upper = self.dof_pos_limits[:, 1] - 0.5 * (1 - self.cfg["rewards"]["soft_dof_pos_limit"]) * (
            self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        )
        return torch.sum(((self.dof_pos < lower) | (self.dof_pos > upper)).float(), dim=-1)

    def _reward_head_contact(self):
        # discourage grinding the head into the floor while maneuvering
        return torch.any(torch.norm(self.contact_forces[:, self.head_indices, :], dim=-1) > 5.0, dim=-1).float()
