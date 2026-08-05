import os

from isaacgym import gymtorch, gymapi
from isaacgym.torch_utils import (
    get_axis_params,
    to_torch,
    quat_rotate_inverse,
    quat_from_euler_xyz,
    torch_rand_float,
    get_euler_xyz,
    quat_rotate,
)

assert gymtorch

import torch

import numpy as np
from envs.base_task import BaseTask

from utils.utils import apply_randomization


class GoalPose(BaseTask):

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

        # ---- 발목 전용 이득 오차 ------------------------------------------
        # 2026-08-05, 실기 2.4초와 MuJoCo를 같은 48채널로 겹쳐 보니 **발목만** 어긋난다:
        # 발목 pitch 토크 rms가 실기에서 sim의 0.6-0.7배인데 관절 궤적은 1.3배 더 크다.
        # 각도는 크고 토크는 작다 = 발목이 지면을 못 민다. 그리고 roll 축 궤적만
        # 1.9-4.3배로 튀고(pitch/yaw는 0.6-1.3배) 정책의 roll 출력도 3배다.
        #
        # 원인은 미확정이다. 평행 링크를 의심했지만 (a) SDK가 관절->액추에이터 매핑을
        # 이미 처리하고 있고(아니면 관절각 명령이 아예 안 먹는다), (b) 교차결합 지표는
        # MuJoCo(평행 결합 없음)에서 오히려 더 크게 나와 그 가설을 지지하지 않는다.
        # 백래시, 기어 마찰, 드라이버 게인 차이도 같은 증상을 만든다.
        #
        # 그래서 **메커니즘을 특정하지 않고 결과만 랜덤화한다**: 발목의 유효 이득이
        # 명령의 몇 배인지를 env마다 흔든다. 원인이 무엇이든 "발목이 약할 수 있다"에
        # 강인해지는 것이 목표다.
        #
        # 기본값은 없음(키가 없으면 no-op)이라 기존 arm은 전혀 영향받지 않는다.
        ankle_gain = self.cfg["randomization"].get("ankle_gain")
        if ankle_gain:
            ankle_dofs = [i for i in range(self.num_dofs) if "Ankle" in self.dof_names[i]]
            if not ankle_dofs:
                raise ValueError("randomization.ankle_gain 이 켜졌는데 이름에 'Ankle'이 든 "
                                 "관절이 없다 -- 자산이 바뀌었거나 키가 오타다")
            scale = torch.ones(self.num_envs, len(ankle_dofs),
                               dtype=torch.float, device=self.device)
            scale = apply_randomization(scale, ankle_gain)
            # 강성과 마찰 모두에 건다. 이득이 낮다는 것은 같은 명령에 토크가 덜
            # 나온다는 뜻이고, 그건 강성 저하로 표현된다.
            self.dof_stiffness[:, ankle_dofs] *= scale
            self.dof_damping[:, ankle_dofs] *= scale

        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        # Keep the exact loaded ordering available to subclasses.  HBatch uses
        # it to distribute synthetic collision wrenches over trunk/arm/hip
        # bodies instead of silently applying every event at the trunk COM.
        self.body_names = list(body_names)
        penalized_contact_names = []
        for name in self.cfg["rewards"]["penalize_contacts_on"]:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg["rewards"]["terminate_contacts_on"]:
            termination_contact_names.extend([s for s in body_names if name in s])
        self.base_indice = self.gym.find_asset_rigid_body_index(robot_asset, asset_cfg["base_name"])

        # prepare penalized and termination contact indices
        self.penalized_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(penalized_contact_names)):
            self.penalized_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, penalized_contact_names[i])
        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, termination_contact_names[i])

        rbs_list = self.gym.get_asset_rigid_body_shape_indices(robot_asset)
        self.feet_indices = torch.zeros(len(asset_cfg["foot_names"]), dtype=torch.long, device=self.device)
        self.foot_shape_indices = []
        for i in range(len(asset_cfg["foot_names"])):
            indices = self.gym.find_asset_rigid_body_index(robot_asset, asset_cfg["foot_names"][i])
            self.feet_indices[i] = indices
            self.foot_shape_indices += list(range(rbs_list[indices].start, rbs_list[indices].start + rbs_list[indices].count))

        base_init_state_list = (
            self.cfg["init_state"]["pos"] + self.cfg["init_state"]["rot"] + self.cfg["init_state"]["lin_vel"] + self.cfg["init_state"]["ang_vel"]
        )
        self.base_init_state = to_torch(base_init_state_list, device=self.device)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0.0, 0.0, 0.0)
        env_upper = gymapi.Vec3(0.0, 0.0, 0.0)
        self.envs = []
        self.actor_handles = []
        self.base_mass_scaled = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device)
        for i in range(self.num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            start_pose.p = gymapi.Vec3(*pos)

            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, asset_cfg["name"], i, asset_cfg["self_collisions"], 0)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            shape_props = self.gym.get_actor_rigid_shape_properties(env_handle, actor_handle)
            shape_props = self._process_rigid_shape_props(shape_props)
            self.gym.set_actor_rigid_shape_properties(env_handle, actor_handle, shape_props)
            self.gym.enable_actor_dof_force_sensors(env_handle, actor_handle)
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
        for i in self.foot_shape_indices:
            props[i].friction = apply_randomization(0.0, self.cfg["randomization"].get("friction"))
            props[i].compliance = apply_randomization(0.0, self.cfg["randomization"].get("compliance"))
            props[i].restitution = apply_randomization(0.0, self.cfg["randomization"].get("restitution"))
        return props

    def _get_env_origins(self):
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        if self.cfg["terrain"]["type"] == "plane":
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="ij")
            spacing = self.cfg["env"]["env_spacing"]
            self.env_origins[:, 0] = spacing * xx.flatten()[: self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[: self.num_envs]
            self.env_origins[:, 2] = 0.0
        else:
            num_cols = max(1.0, np.floor(np.sqrt(self.num_envs * self.terrain.env_length / self.terrain.env_width)))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols), indexing="ij")
            self.env_origins[:, 0] = self.terrain.env_width / (num_rows + 1) * (xx.flatten()[: self.num_envs] + 1)
            self.env_origins[:, 1] = self.terrain.env_length / (num_cols + 1) * (yy.flatten()[: self.num_envs] + 1)
            self.env_origins[:, 2] = self.terrain.terrain_heights(self.env_origins)

    def _init_buffers(self):
        self.num_obs = self.cfg["env"]["num_observations"]
        self.num_privileged_obs = self.cfg["env"]["num_privileged_obs"]
        self.num_actions = self.cfg["env"]["num_actions"]
        self.dt = self.cfg["control"]["decimation"] * self.cfg["sim"]["dt"]

        self.obs_buf = torch.zeros(self.num_envs, self.num_obs, dtype=torch.float, device=self.device)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.num_privileged_obs, dtype=torch.float, device=self.device)
        self.rew_buf = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.reset_buf = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.extras = {}
        self.extras["rew_terms"] = {}

        # get gym state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dofs, 2)[..., 1]
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3)  # shape: num_envs, num_bodies, xyz axis
        self.body_states = gymtorch.wrap_tensor(body_state).view(self.num_envs, self.num_bodies, 13)
        self.base_pos = self.root_states[:, 0:3]
        self.base_quat = self.root_states[:, 3:7]
        self.feet_pos = self.body_states[:, self.feet_indices, 0:3]
        self.feet_quat = self.body_states[:, self.feet_indices, 3:7]

        # initialize some data used later on
        self.common_step_counter = 0
        self.gravity_vec = to_torch(get_axis_params(-1.0, self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.last_dof_targets = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        # Episode-constant calibration errors are distinct from both the reset
        # pose spread and the iid sensor noise used below.  Defaults are zero,
        # so every existing task remains bit-for-bit compatible unless a new
        # config explicitly enables these fields.
        self.joint_encoder_bias = torch.zeros_like(self.dof_pos)
        self.joint_target_offset = torch.zeros_like(self.dof_pos)
        self.delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Perceived-goal model (sim2real): the real goal_rel_x/y/heading come from a
        # perception/localization pipeline, not an instantaneous ground-truth read.
        # goal_obs_bias persists per goal segment (systematic localization error between
        # re-detections); goal_obs_hold_counter staggers refresh to emulate a camera
        # running slower than the control loop (K1 manual: ~20fps camera vs 50Hz control).
        self.goal_obs_bias = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.goal_obs_hold_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.goal_obs_cached = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device)
        self.torques = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.commands = torch.zeros(self.num_envs, self.cfg["commands"]["num_commands"], dtype=torch.float, device=self.device)
        self.cmd_resample_time = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.gait_frequency = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.gait_process = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.filtered_lin_vel = self.base_lin_vel.clone()
        self.filtered_ang_vel = self.base_ang_vel.clone()

        # goal-pose state: target sampled in the robot's local frame at resample time,
        # then stored/tracked in world frame so it stays fixed while the robot walks.
        self.goal_pos_world = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.goal_heading_world = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.goal_rel_pos = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.goal_dist = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.heading_error = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.last_goal_dist = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Per-segment provenance, written in _resample_goals and never read by a reward,
        # observation or termination: eval_goal_pose.py needs to know WHICH goal a
        # segment was (category), where it started from and when, to tell "the policy is
        # imprecise" apart from "the goal was unreachable in the time it was given".
        self.goal_category = torch.full((self.num_envs,), -1, dtype=torch.int8, device=self.device)
        self.goal_start_pos = torch.zeros(self.num_envs, 2, dtype=torch.float, device=self.device)
        self.goal_start_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # curriculum machinery kept for compatibility with utils/runner.py's logging
        # (it reads self.env.mean_lin_vel_level etc. unconditionally); unused while
        # cfg["commands"]["curriculum"] is false.
        self.curriculum_prob = torch.zeros(
            1 + 2 * self.cfg["commands"]["lin_vel_levels"],
            1 + 2 * self.cfg["commands"]["ang_vel_levels"],
            dtype=torch.float,
            device=self.device,
        )
        self.curriculum_prob[self.cfg["commands"]["lin_vel_levels"], self.cfg["commands"]["ang_vel_levels"]] = 1.0
        self.env_curriculum_level = torch.zeros(self.num_envs, 2, dtype=torch.long, device=self.device)
        self.mean_lin_vel_level = 0.0
        self.mean_ang_vel_level = 0.0
        self.max_lin_vel_level = 0.0
        self.max_ang_vel_level = 0.0

        self.pushing_forces = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device)
        self.pushing_torques = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device)
        self.feet_roll = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.feet_yaw = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.feet_yaw_rel = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.feet_pitch = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        self.last_feet_pos = torch.zeros_like(self.feet_pos)
        self.feet_contact = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device)
        # How high the foot actually is, measured at its LOWEST corner -- that is
        # the corner that catches on anything.  feet_contact only says whether it
        # is under 1 cm, which is all feet_swing ever asks for; this says by how
        # much, so terrain arms can be judged on the thing they exist to change.
        self.feet_clearance = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)
        # Phase-free swing accounting for _reward_feet_air_time. Self-contained:
        # the reward accumulates, reads at touchdown and clears, so it needs no
        # update site elsewhere and cannot get out of order with the contact
        # refresh.
        self.feet_air_time = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.float, device=self.device)

        self.dof_pos_ref = torch.zeros(self.num_envs, self.num_dofs, dtype=torch.float, device=self.device)
        self.default_dof_pos = torch.zeros(1, self.num_dofs, dtype=torch.float, device=self.device)
        for i in range(self.num_dofs):
            found = False
            for name in self.cfg["init_state"]["default_joint_angles"].keys():
                if name in self.dof_names[i]:
                    self.default_dof_pos[:, i] = self.cfg["init_state"]["default_joint_angles"][name]
                    found = True
            if not found:
                self.default_dof_pos[:, i] = self.cfg["init_state"]["default_joint_angles"]["default"]

    def _prepare_reward_function(self):
        """Prepares a list of reward functions, whcih will be called to compute the total reward.
        Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        self.reward_scales = self.cfg["rewards"]["scales"].copy()
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale == 0:
                self.reward_scales.pop(key)
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            self.reward_names.append(name)
            name = "_reward_" + name
            self.reward_functions.append(getattr(self, name))

    def reset(self):
        """Reset all robots"""
        self._reset_idx(torch.arange(self.num_envs, device=self.device))
        self._resample_goals()
        self._update_goal_state()
        self._compute_observations()
        return self.obs_buf, self.extras

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        self._update_curriculum(env_ids)
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self.last_dof_targets[env_ids] = self.dof_pos[env_ids]
        self.joint_encoder_bias[env_ids] = apply_randomization(
            torch.zeros(len(env_ids), self.num_dofs, device=self.device),
            self.cfg["randomization"].get("joint_encoder_bias"),
        )
        self.joint_target_offset[env_ids] = apply_randomization(
            torch.zeros(len(env_ids), self.num_dofs, device=self.device),
            self.cfg["randomization"].get("joint_target_offset"),
        )
        self.last_root_vel[env_ids] = self.root_states[env_ids, 7:13]
        self.episode_length_buf[env_ids] = 0
        self.filtered_lin_vel[env_ids] = 0.0
        self.filtered_ang_vel[env_ids] = 0.0
        self.cmd_resample_time[env_ids] = 0
        # 리셋은 순간이동이라, 넘겨받은 체공시간은 이번 에피소드의 스윙이 아니다.
        self.feet_air_time[env_ids] = 0.0

        self.delay_steps[env_ids] = torch.randint(0, self.cfg["control"]["decimation"], (len(env_ids),), device=self.device)
        self.extras["time_outs"] = self.time_out_buf

    def _reset_dofs(self, env_ids):
        self.dof_pos[env_ids] = apply_randomization(self.default_dof_pos, self.cfg["randomization"].get("init_dof_pos"))
        self.dof_vel[env_ids] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self.dof_state), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32)
        )

    def _reset_root_states(self, env_ids):
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :2] += self.env_origins[env_ids, :2]
        self.root_states[env_ids, :2] = apply_randomization(self.root_states[env_ids, :2], self.cfg["randomization"].get("init_base_pos_xy"))
        self.root_states[env_ids, 2] += self.terrain.terrain_heights(self.root_states[env_ids, :2])
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(
            torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
            torch.zeros(len(env_ids), dtype=torch.float, device=self.device),
            torch.rand(len(env_ids), device=self.device) * (2 * torch.pi),
        )
        self.root_states[env_ids, 7:9] = apply_randomization(
            torch.zeros(len(env_ids), 2, dtype=torch.float, device=self.device),
            self.cfg["randomization"].get("init_base_lin_vel_xy"),
        )
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _teleport_robot(self):
        if self.terrain.type == "plane":
            return
        out_x_min = self.root_states[:, 0] < -0.75 * self.terrain.border_size
        out_x_max = self.root_states[:, 0] > self.terrain.env_width + 0.75 * self.terrain.border_size
        out_y_min = self.root_states[:, 1] < -0.75 * self.terrain.border_size
        out_y_max = self.root_states[:, 1] > self.terrain.env_length + 0.75 * self.terrain.border_size
        self.root_states[out_x_min, 0] += self.terrain.env_width + self.terrain.border_size
        self.root_states[out_x_max, 0] -= self.terrain.env_width + self.terrain.border_size
        self.root_states[out_y_min, 1] += self.terrain.env_length + self.terrain.border_size
        self.root_states[out_y_max, 1] -= self.terrain.env_length + self.terrain.border_size
        self.body_states[out_x_min, :, 0] += self.terrain.env_width + self.terrain.border_size
        self.body_states[out_x_max, :, 0] -= self.terrain.env_width + self.terrain.border_size
        self.body_states[out_y_min, :, 1] += self.terrain.env_length + self.terrain.border_size
        self.body_states[out_y_max, :, 1] -= self.terrain.env_length + self.terrain.border_size
        if out_x_min.any() or out_x_max.any() or out_y_min.any() or out_y_max.any():
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
            self._refresh_feet_state()

    def _resample_goals(self):
        """Sample a new (Δx, Δy, Δθ) goal in the robot's current local frame, and
        store it as a fixed world-frame target (goal_pos_world/goal_heading_world)."""
        if getattr(self, "manual_control", False):
            return
        env_ids = (self.episode_length_buf == self.cmd_resample_time).nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return

        dx_local = torch_rand_float(
            self.cfg["commands"]["goal_dx"][0], self.cfg["commands"]["goal_dx"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        dy_local = torch_rand_float(
            self.cfg["commands"]["goal_dy"][0], self.cfg["commands"]["goal_dy"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        dtheta_local = torch_rand_float(
            self.cfg["commands"]["goal_dtheta"][0], self.cfg["commands"]["goal_dtheta"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)

        # optional "No More Marching" (arXiv:2508.14098) goal-type mixture:
        # stand / straight / lateral / turn-in-place / combined. Zeroes the unused
        # goal components per category; uniform sampling when disabled.
        cat_cfg = self.cfg["commands"].get("goal_categories")
        stand = torch.zeros(len(env_ids), dtype=torch.bool, device=self.device)
        cat = torch.full((len(env_ids),), -1, dtype=torch.long, device=self.device)  # -1 = uniform (mixture off)
        if cat_cfg and cat_cfg.get("enabled", False):
            probs = torch.tensor(
                [cat_cfg["stand"], cat_cfg["straight"], cat_cfg["lateral"], cat_cfg["turn"], cat_cfg["combined"]],
                dtype=torch.float, device=self.device,
            )
            cat = torch.multinomial(probs, len(env_ids), replacement=True)
            stand = cat == 0
            zero = torch.zeros_like(dx_local)
            dx_local = torch.where(stand | (cat == 2) | (cat == 3), zero, dx_local)
            dy_local = torch.where(stand | (cat == 1) | (cat == 3), zero, dy_local)
            dtheta_local = torch.where(stand | (cat == 1) | (cat == 2), zero, dtheta_local)

        # Evaluation-only abrupt-direction probes.  Lateral uses a signed
        # 1-2 m magnitude rather than a continuous [-2,2] draw, which would
        # pollute the stress set with near-zero goals that require no response.
        pattern = self.cfg.get("evaluation", {}).get("goal_pattern")
        if pattern == "lateral":
            magnitude = torch_rand_float(1.0, 2.0, (len(env_ids), 1), device=self.device).squeeze(1)
            sign = torch.where(torch.rand(len(env_ids), device=self.device) < 0.5,
                               -torch.ones_like(magnitude), torch.ones_like(magnitude))
            dx_local = torch.zeros_like(dx_local)
            dy_local = sign * magnitude
            dtheta_local = torch.zeros_like(dtheta_local)
            cat = torch.full_like(cat, 2)
            stand = torch.zeros_like(stand)
        elif pattern == "reverse":
            dx_local = -torch_rand_float(
                1.0, 2.0, (len(env_ids), 1), device=self.device).squeeze(1)
            dy_local = torch.zeros_like(dy_local)
            dtheta_local = torch.zeros_like(dtheta_local)
            cat = torch.full_like(cat, 1)
            stand = torch.zeros_like(stand)

        _, _, base_yaw = get_euler_xyz(self.base_quat[env_ids])
        base_yaw = (base_yaw + torch.pi) % (2 * torch.pi) - torch.pi
        cos_yaw = torch.cos(base_yaw)
        sin_yaw = torch.sin(base_yaw)
        self.goal_pos_world[env_ids, 0] = self.base_pos[env_ids, 0] + cos_yaw * dx_local - sin_yaw * dy_local
        self.goal_pos_world[env_ids, 1] = self.base_pos[env_ids, 1] + sin_yaw * dx_local + cos_yaw * dy_local
        self.goal_heading_world[env_ids] = (base_yaw + dtheta_local + torch.pi) % (2 * torch.pi) - torch.pi

        # eval-only provenance (see _init_buffers); base_pos here IS the frame the goal
        # was sampled in, so start_dist == sqrt(dx_local^2 + dy_local^2) exactly.
        self.goal_category[env_ids] = cat.to(torch.int8)
        self.goal_start_pos[env_ids] = self.base_pos[env_ids, :2]
        self.goal_start_step[env_ids] = self.episode_length_buf[env_ids]

        self.gait_frequency[env_ids] = torch_rand_float(
            self.cfg["commands"]["gait_frequency"][0], self.cfg["commands"]["gait_frequency"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        # stand-category envs get a zero gait clock, like ParameterWalk's still envs:
        # this disables the feet_swing stepping incentive so standing still is optimal
        self.gait_frequency[env_ids[stand]] = 0.0
        self.commands[env_ids, 3] = self.gait_frequency[env_ids]
        self.commands[env_ids, 4] = torch_rand_float(
            self.cfg["commands"]["foot_yaw_L"][0], self.cfg["commands"]["foot_yaw_L"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        self.commands[env_ids, 5] = torch_rand_float(
            self.cfg["commands"]["foot_yaw_R"][0], self.cfg["commands"]["foot_yaw_R"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        self.commands[env_ids, 6] = torch_rand_float(
            self.cfg["commands"]["body_pitch_target"][0], self.cfg["commands"]["body_pitch_target"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        self.commands[env_ids, 7] = torch_rand_float(
            self.cfg["commands"]["body_roll_target"][0], self.cfg["commands"]["body_roll_target"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        self.commands[env_ids, 8] = torch_rand_float(
            self.cfg["commands"]["feet_offset_x_target"][0], self.cfg["commands"]["feet_offset_x_target"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        self.commands[env_ids, 9] = torch_rand_float(
            self.cfg["commands"]["feet_offset_y_target"][0], self.cfg["commands"]["feet_offset_y_target"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)

        self.cmd_resample_time[env_ids] += torch.randint(
            int(self.cfg["commands"]["resampling_time_s"][0] / self.dt),
            int(self.cfg["commands"]["resampling_time_s"][1] / self.dt),
            (len(env_ids),),
            device=self.device,
        )

        # New goal = a new perception detection: draw a fresh systematic bias for this
        # segment and force an immediate (unstale) perceived-goal refresh next step.
        bias_pos_cfg = self.cfg["noise"].get("goal_pos_bias")
        bias_head_cfg = self.cfg["noise"].get("goal_heading_bias")
        self.goal_obs_bias[env_ids, 0:2] = apply_randomization(torch.zeros(len(env_ids), 2, device=self.device), bias_pos_cfg)
        self.goal_obs_bias[env_ids, 2] = apply_randomization(torch.zeros(len(env_ids), device=self.device), bias_head_cfg)
        self.goal_obs_hold_counter[env_ids] = 0
        # 관측 지연은 그 로봇의 성질이지 매 스텝 흔들리는 값이 아니다. 에피소드마다
        # 다시 뽑되 에피소드 안에서는 고정한다. 히스토리도 같이 지워야 한다 --
        # 안 지우면 리셋된 env가 넘어지기 직전의 관측을 물려받는다(swing_apex를
        # done에서 리셋하지 않았던 것과 같은 부류의 실수다).
        cfg = self.cfg["noise"].get("obs_delay_steps")
        if cfg and int(cfg[1]) > 0 and hasattr(self, "_obs_delay"):
            self._obs_delay[env_ids] = torch.randint(
                int(cfg[0]), int(cfg[1]) + 1, (len(env_ids),), device=self.device)
            self._obs_hist[:, env_ids] = self.obs_buf[env_ids]

    def _update_goal_state(self):
        """Recompute the goal position/heading relative to the robot's current local
        frame and write it into commands[:, 0:3] (the ParameterWalk lin_vel_x/y and
        ang_vel_yaw slots, reused here so warm-starting from a ParameterWalk checkpoint
        works). Must be called whenever base_pos/base_quat changes (every physics step
        and after resets)."""
        to_goal = self.goal_pos_world - self.base_pos[:, :2]
        _, _, base_yaw = get_euler_xyz(self.base_quat)
        base_yaw = (base_yaw + torch.pi) % (2 * torch.pi) - torch.pi
        cos_yaw = torch.cos(base_yaw)
        sin_yaw = torch.sin(base_yaw)
        self.goal_rel_pos[:, 0] = cos_yaw * to_goal[:, 0] + sin_yaw * to_goal[:, 1]
        self.goal_rel_pos[:, 1] = -sin_yaw * to_goal[:, 0] + cos_yaw * to_goal[:, 1]
        self.goal_dist[:] = torch.norm(to_goal, dim=-1)
        self.heading_error[:] = (self.goal_heading_world - base_yaw + torch.pi) % (2 * torch.pi) - torch.pi

        self.commands[:, 0] = self.goal_rel_pos[:, 0]
        self.commands[:, 1] = self.goal_rel_pos[:, 1]
        self.commands[:, 2] = self.heading_error

    def _update_curriculum(self, env_ids):
        if not self.cfg["commands"]["curriculum"]:
            return
        raise NotImplementedError("Goal-position curriculum is not implemented; keep cfg['commands']['curriculum'] = false.")

    def _obs_dofs(self, x):
        """Which DOFs the observation carries. Identity here; v7 overrides it to
        hide scripted (non-learned) joints so the observation width is stable."""
        return x

    def _dof_targets_from_actions(self):
        """PD position targets for every DOF.

        Extracted so a subclass can drive DOFs the policy does not control
        (v7's scripted elbows) without duplicating step().
        """
        return (self.default_dof_pos + self.cfg["control"]["action_scale"] * self.actions
                + self.joint_target_offset)

    def step(self, actions):
        # pre physics step
        self.actions[:] = torch.clip(actions, -self.cfg["normalization"]["clip_actions"], self.cfg["normalization"]["clip_actions"])
        # snapshot for goal_progress: goals only change after reward computation, so this
        # distance and the post-physics one are guaranteed to be against the same goal
        self.last_goal_dist[:] = self.goal_dist
        dof_targets = self._dof_targets_from_actions()

        # perform physics step
        self.torques.zero_()
        for i in range(self.cfg["control"]["decimation"]):
            self.last_dof_targets[self.delay_steps == i] = dof_targets[self.delay_steps == i]
            dof_torques = self.dof_stiffness * (self.last_dof_targets - self.dof_pos) - self.dof_damping * self.dof_vel
            friction = torch.min(self.dof_friction, dof_torques.abs()) * torch.sign(dof_torques)
            dof_torques = torch.clip(dof_torques - friction, min=-self.torque_limits, max=self.torque_limits)
            self.torques += dof_torques
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(dof_torques))
            # Isaac Gym consumes rigid-body force tensors for the immediate
            # physics timestep only.  Tasks with a control-step disturbance
            # schedule therefore have to re-submit the held wrench before each
            # decimation substep; a single call after the loop would deliver
            # only 1/decimation of the configured impulse.
            self._apply_external_wrenches_substep()
            self.gym.simulate(self.sim)
            if self.device == "cpu":
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        self.torques /= self.cfg["control"]["decimation"]
        self.render()

        # post physics step
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.filtered_lin_vel[:] = self.base_lin_vel[:] * self.cfg["normalization"]["filter_weight"] + self.filtered_lin_vel[:] * (
            1.0 - self.cfg["normalization"]["filter_weight"]
        )
        self.filtered_ang_vel[:] = self.base_ang_vel[:] * self.cfg["normalization"]["filter_weight"] + self.filtered_ang_vel[:] * (
            1.0 - self.cfg["normalization"]["filter_weight"]
        )

        self._refresh_feet_state()
        self._update_goal_state()

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.gait_process[:] = torch.fmod(self.gait_process + self.dt * self.gait_frequency, 1.0)

        self._kick_robots()
        self._push_robots()
        self._check_termination()
        self._compute_reward()

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self._reset_idx(env_ids)
        self._teleport_robot()
        self._resample_goals()
        self._update_goal_state()

        self._compute_observations()

        self.last_actions[:] = self.actions
        self.last_dof_vel[:] = self.dof_vel
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_feet_pos[:] = self.feet_pos

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def _apply_external_wrenches_substep(self):
        """Optional task hook, called immediately before every physics tick."""
        return

    def _kick_robots(self):
        """Random kick the robots. Emulates an impulse by setting a randomized base velocity."""
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["kick_interval_s"] / self.dt) == 0:
            self.root_states[:, 7:10] = apply_randomization(self.root_states[:, 7:10], self.cfg["randomization"].get("kick_lin_vel"))
            self.root_states[:, 10:13] = apply_randomization(self.root_states[:, 10:13], self.cfg["randomization"].get("kick_ang_vel"))
            self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _push_robots(self):
        """Random push the robots. Emulates an impulse by setting a randomized force."""
        if self.common_step_counter % np.ceil(self.cfg["randomization"]["push_interval_s"] / self.dt) == 0:
            self.pushing_forces[:, self.base_indice, :] = apply_randomization(
                torch.zeros_like(self.pushing_forces[:, 0, :]),
                self.cfg["randomization"].get("push_force"),
            )
            self.pushing_torques[:, self.base_indice, :] = apply_randomization(
                torch.zeros_like(self.pushing_torques[:, 0, :]),
                self.cfg["randomization"].get("push_torque"),
            )
        elif self.common_step_counter % np.ceil(self.cfg["randomization"]["push_interval_s"] / self.dt) == np.ceil(
            self.cfg["randomization"]["push_duration_s"] / self.dt
        ):
            self.pushing_forces[:, self.base_indice, :].zero_()
            self.pushing_torques[:, self.base_indice, :].zero_()
        self.gym.apply_rigid_body_force_tensors(
            self.sim,
            gymtorch.unwrap_tensor(self.pushing_forces),
            gymtorch.unwrap_tensor(self.pushing_torques),
            gymapi.LOCAL_SPACE,
        )

    def _refresh_feet_state(self):
        self.feet_pos[:] = self.body_states[:, self.feet_indices, 0:3]
        self.feet_quat[:] = self.body_states[:, self.feet_indices, 3:7]
        roll, _, yaw = get_euler_xyz(self.feet_quat.reshape(-1, 4))
        self.feet_roll[:] = (roll.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        self.feet_yaw[:] = (yaw.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi
        _, pitch, _ = get_euler_xyz(self.feet_quat.reshape(-1, 4))
        self.feet_pitch[:] = (pitch.reshape(self.num_envs, len(self.feet_indices)) + torch.pi) % (2 * torch.pi) - torch.pi

        # Compute relative yaw to trunk
        _, _, base_yaw = get_euler_xyz(self.base_quat)
        self.feet_yaw_rel = (self.feet_yaw - base_yaw.unsqueeze(-1) + torch.pi) % (2 * torch.pi) - torch.pi

        feet_edge_relative_pos = (
            to_torch(self.cfg["asset"]["feet_edge_pos"], device=self.device)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(self.num_envs, len(self.feet_indices), -1, -1)
        )
        expanded_feet_pos = self.feet_pos.unsqueeze(2).expand(-1, -1, feet_edge_relative_pos.shape[2], -1).reshape(-1, 3)
        expanded_feet_quat = self.feet_quat.unsqueeze(2).expand(-1, -1, feet_edge_relative_pos.shape[2], -1).reshape(-1, 4)
        feet_edge_pos = expanded_feet_pos + quat_rotate(expanded_feet_quat, feet_edge_relative_pos.reshape(-1, 3))
        edge_height = (
            feet_edge_pos[:, 2] - self.terrain.terrain_heights(feet_edge_pos)
        ).reshape(self.num_envs, len(self.feet_indices), feet_edge_relative_pos.shape[2])
        self.feet_contact[:] = torch.any(edge_height < 0.01, dim=2)
        self.feet_clearance[:] = edge_height.min(dim=2).values

    def _check_termination(self):
        """Check if environments need to be reset"""
        physical = torch.any(
            torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0,
            dim=1,
        )
        physical |= (self.root_states[:, 7:13].square().sum(dim=-1)
                     > self.cfg["rewards"]["terminate_vel"])
        physical |= (self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos)
                     < self.cfg["rewards"]["terminate_height"])
        episode_timeout = self.episode_length_buf > np.ceil(
            self.cfg["rewards"]["episode_length_s"] / self.dt)
        segment_boundary = self.episode_length_buf == self.cmd_resample_time
        self.reset_buf = physical | episode_timeout
        # Segment boundaries are marked for PPO bootstrapping without resetting
        # the simulator.  A physical fall on that same step must take priority;
        # otherwise eval hides the fall and PPO bootstraps through a terminal
        # state merely because the goal happened to resample simultaneously.
        self.time_out_buf = (episode_timeout | segment_boundary) & ~physical
        # _reset_idx() is not guaranteed to run with a non-empty id set, and
        # this method replaces time_out_buf with a new tensor every step.  Keep
        # extras synchronized here; otherwise PPO/eval can receive a stale mask
        # from an earlier step and bootstrap the wrong transitions.
        self.extras["time_outs"] = self.time_out_buf
        self.extras["episode_time_outs"] = episode_timeout
        self.extras["physical_failures"] = physical

    def _compute_reward(self):
        """Compute rewards
        Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
        adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.0
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.extras["rew_terms"][name] = rew
        if self.cfg["rewards"]["only_positive_rewards"]:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.0)

    def _update_perceived_goal(self):
        """Refresh the perceived (goal_rel_x, goal_rel_y, heading_error) triplet that
        actually reaches the observation, modeling a perception pipeline instead of an
        instant ground-truth read:
          - staleness: refreshed only every goal_obs_hold_steps control steps (camera
            runs slower than the 50Hz control loop -- K1 manual specs the RGB/depth
            camera at 20fps)
          - persistent per-segment bias (goal_obs_bias, resampled in _resample_goals):
            systematic localization/detection error that holds between re-detections
          - per-step jitter on top (noise.goal_pos / noise.goal_heading): high-frequency
            measurement noise
        Off by default (goal_obs_hold_steps [0,0], bias ranges [0,0]) so v1 behavior
        (instant noiseless read) is unchanged unless these are turned on.
        """
        hold_cfg = self.cfg["noise"].get("goal_obs_hold_steps")
        refresh = self.goal_obs_hold_counter <= 0
        if hold_cfg and hold_cfg[1] > 0:
            new_hold = torch.randint(hold_cfg[0], hold_cfg[1] + 1, (self.num_envs,), device=self.device)
        else:
            new_hold = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.goal_obs_hold_counter = torch.where(refresh, new_hold, self.goal_obs_hold_counter - 1)

        fresh = self.commands[:, 0:3].clone()
        fresh[:, 0:2] = apply_randomization(fresh[:, 0:2], self.cfg["noise"].get("goal_pos")) + self.goal_obs_bias[:, 0:2]
        fresh[:, 2] = apply_randomization(fresh[:, 2], self.cfg["noise"].get("goal_heading")) + self.goal_obs_bias[:, 2]
        self.goal_obs_cached = torch.where(refresh.unsqueeze(-1), fresh, self.goal_obs_cached)

    def _compute_observations(self):
        """Computes observations"""
        commands_scale = torch.tensor(
            [
                self.cfg["normalization"]["goal_pos"],           # 0: goal_rel_x
                self.cfg["normalization"]["goal_pos"],           # 1: goal_rel_y
                self.cfg["normalization"]["goal_heading"],       # 2: heading_error
                self.cfg["normalization"]["gait_frequency"],     # 3: gait_frequency
                self.cfg["normalization"]["foot_yaw"],           # 4: foot_yaw_L
                self.cfg["normalization"]["foot_yaw"],           # 5: foot_yaw_R
                self.cfg["normalization"]["body_pitch_target"],  # 6: body_pitch_target
                self.cfg["normalization"]["body_roll_target"],   # 7: body_roll_target
                self.cfg["normalization"]["feet_offset_x_target"],  # 8: feet_offset_x_target
                self.cfg["normalization"]["feet_offset_y_target"],  # 9: feet_offset_y_target
            ],
            device=self.device,
        )
        # Goal-channel observation noise (sim2real: on the real robot goal_rel_x/y and
        # heading_error come from perception/localization/odometry, not an instant
        # ground-truth read; the reward stays clean because it reads goal_dist/
        # heading_error, not commands). See _update_perceived_goal for the model.
        self._update_perceived_goal()
        noisy_commands = self.commands[:, :10].clone()
        noisy_commands[:, 0:3] = self.goal_obs_cached
        self.obs_buf = torch.cat(
            (
                apply_randomization(self.projected_gravity, self.cfg["noise"].get("gravity")) * self.cfg["normalization"]["gravity"],
                apply_randomization(self.base_ang_vel, self.cfg["noise"].get("ang_vel")) * self.cfg["normalization"]["ang_vel"],
                noisy_commands * commands_scale,
                (torch.cos(2 * torch.pi * self.gait_process)).unsqueeze(-1),
                (torch.sin(2 * torch.pi * self.gait_process)).unsqueeze(-1),
                self._obs_dofs(apply_randomization(
                    self.dof_pos + self.joint_encoder_bias - self.default_dof_pos,
                    self.cfg["noise"].get("dof_pos"),
                ) * self.cfg["normalization"]["dof_pos"]),
                self._obs_dofs(apply_randomization(self.dof_vel, self.cfg["noise"].get("dof_vel")) * self.cfg["normalization"]["dof_vel"]),
                self.actions,
            ),
            dim=-1,
        )
        self.privileged_obs_buf = torch.cat(
            (
                self.base_mass_scaled,
                apply_randomization(self.base_lin_vel, self.cfg["noise"].get("lin_vel")) * self.cfg["normalization"]["lin_vel"],
                apply_randomization(self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos), self.cfg["noise"].get("height")).unsqueeze(-1),
                self.pushing_forces[:, 0, :] * self.cfg["normalization"]["push_force"],
                self.pushing_torques[:, 0, :] * self.cfg["normalization"]["push_torque"],
            ),
            dim=-1,
        )
        self._apply_obs_delay()
        self.extras["privileged_obs"] = self.privileged_obs_buf

    def _apply_obs_delay(self):
        """관측을 per-env로 지연시킨다. 학습이 유일하게 모델링하지 않던 축이다.

        왜: `delay_steps`(goal_pose.py의 step)는 **액션** 쪽 0-18 ms 순수지연이고,
        관측 쪽 지연은 0이다. 실기 대조에서 이 축만 증상을 재현했다 --
        2026-08-05 MuJoCo signature 탐색에서 obs 지연 20 ms가

            signature 점수  7.86(기본) -> 1.34   (최솟값)
            다리 교차       2.3%(기본) -> 9.2%   (실기 9.9%)

        로 실기의 "다리가 모이면서 발끼리 부딪혀 넘어짐"을 재현했다. 다른 어떤
        레버도(IMU 바이어스, 마찰, 발 관성, 토크 상한, 정책 주기, 관절 영점)
        교차를 재현하지 못했다 -- IMU 바이어스는 점수는 좋았지만 교차를 0.4%로
        **줄였다.**

        실기 `low_state_age`는 median 1 ms지만 그것은 **전송 구간만**이다.
        센서 -> IMU 필터 -> SDK 발행 구간은 안 잡히고, 상보/칼만 필터의 실효
        지연은 통상 10-40 ms다. 즉 1 ms 측정은 이 가설을 배제하지 못한다.

        구현: obs_buf의 링 버퍼. per-env 지연은 리셋마다 뽑아 고정한다 --
        실제 파이프라인 지연은 매 스텝 흔들리는 값이 아니라 그 로봇의 성질이다.
        기본은 [0, 0]이라 키가 없거나 0이면 정확한 no-op이고 기존 arm은 무관하다.
        """
        cfg = self.cfg["noise"].get("obs_delay_steps")
        if not cfg or int(cfg[1]) <= 0:
            return
        max_d = int(cfg[1])
        if not hasattr(self, "_obs_hist") or self._obs_hist.shape[0] != max_d + 1:
            # (지연+1, envs, obs) 링 버퍼. 처음에는 현재 관측으로 채운다 --
            # 0으로 채우면 에피소드 시작마다 정책이 "중력 0"을 보게 된다.
            self._obs_hist = self.obs_buf.unsqueeze(0).repeat(max_d + 1, 1, 1).clone()
            self._obs_ptr = 0
            self._obs_delay = torch.randint(int(cfg[0]), max_d + 1,
                                            (self.num_envs,), device=self.device)
        self._obs_ptr = (self._obs_ptr + 1) % (max_d + 1)
        self._obs_hist[self._obs_ptr] = self.obs_buf
        idx = (self._obs_ptr - self._obs_delay) % (max_d + 1)
        self.obs_buf = self._obs_hist[idx, torch.arange(self.num_envs, device=self.device)]

    # ------------ reward functions----------------
    def _reward_survival(self):
        # Reward survival
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)

    def _reward_goal_position(self):
        # Reward for closing the distance to the target position
        return torch.exp(-torch.square(self.goal_dist) / self.cfg["rewards"]["goal_position_sigma"])

    def _reward_goal_heading(self):
        # Reward for facing the target heading
        return torch.exp(-torch.square(self.heading_error) / self.cfg["rewards"]["goal_heading_sigma"])

    def _reward_goal_stop(self):
        # Penalize residual base velocity once inside the goal radius, to encourage stopping there
        close = (self.goal_dist < self.cfg["rewards"]["goal_reach_radius"]).float()
        vel_error = torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=-1) + torch.square(self.base_ang_vel[:, 2])
        return close * vel_error

    # --- modular alternatives, all disabled by default (scale 0 in yaml).
    # Swap in/out per experiment by changing only reward scales; see MASTERPLAN.

    def _reward_constellation(self):
        # "No More Marching" (arXiv:2508.14098) constellation reward: N points on a
        # circle of radius r rigidly attached to the base frame, compared with the
        # same circle placed at the goal pose. The mean squared point distance
        # decomposes exactly into ||Δc||^2 + 2r^2(1 - cos θ) for a circle, coupling
        # position and heading in ONE kernel (both must be good for high reward,
        # unlike additive goal_position + goal_heading). 2(1-cosθ) ≈ θ^2 for small
        # errors (the paper's I_c·θ^2 form) but stays smooth at ±π.
        w = self.cfg["rewards"]["constellation_weight"]
        r = self.cfg["rewards"]["constellation_radius"]
        d_con = torch.square(self.goal_dist) + 2.0 * r * r * (1.0 - torch.cos(self.heading_error))
        return torch.exp(-w * d_con)

    def _reward_goal_progress(self):
        # Potential-based progress toward the goal [m/s]: positive while closing distance.
        # Alternative/addition to goal_position: dense gradient even far from the goal,
        # where exp(-dist^2/sigma) is nearly flat. Clipped for robustness to kicks.
        progress = (self.last_goal_dist - self.goal_dist) / self.dt
        clip = self.cfg["rewards"]["goal_progress_clip"]
        return progress.clip(min=-clip, max=clip) * (self.episode_length_buf > 1).float()

    def _reward_goal_reached(self):
        # Sparse bonus: 1 per step while stopped inside the goal radius.
        # Directly rewards the actual task success condition (arrive AND stop).
        stopped = torch.norm(self.root_states[:, 7:9], dim=-1) < self.cfg["rewards"]["stop_speed_threshold"]
        return ((self.goal_dist < self.cfg["rewards"]["goal_reach_radius"]) & stopped).float()

    def _reward_heading_near_goal(self):
        # Heading tracking gated to the near-goal region. Alternative to goal_heading:
        # lets the robot face its direction of travel while walking, only demanding the
        # target heading once it is close to the goal position.
        heading = torch.exp(-torch.square(self.heading_error) / self.cfg["rewards"]["goal_heading_sigma"])
        gate = torch.exp(-torch.square(self.goal_dist) / self.cfg["rewards"]["heading_gate_sigma"])
        return heading * gate

    def _reward_stand_posture(self):
        # Near the goal, settle into the default standing posture rather than a
        # mid-stride crouch: the deceleration/arrival pose should look like PREP-mode
        # standing so the RLKick handoff starts from a clean, repeatable stance.
        close = (self.goal_dist < self.cfg["rewards"]["stand_posture_radius"]).float()
        return close * torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=-1)

    def _reward_base_height(self):
        # Tracking of base height
        base_height = self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos)
        return torch.square(base_height - self.cfg["rewards"]["base_height_target"])

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(torch.norm(self.contact_forces[:, self.penalized_contact_indices, :], dim=-1) > 1.0, dim=-1)

    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.filtered_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=-1)

    def _reward_orientation(self):
        """
        Reward for tracking body pitch and roll targets.
        Computes reward based on:
         - normalized roll angle (minus body_roll_target)
         - normalized pitch angle (minus body_pitch_target)
        Result: orient_reward = roll_error^2 + pitch_error^2
        """
        # Get all Euler angles (roll, pitch, yaw) from base_quat
        roll_all, pitch_all, _ = get_euler_xyz(self.base_quat)

        # Normalize to [-π, +π]
        roll_norm = (roll_all + torch.pi) % (2 * torch.pi) - torch.pi
        pitch_norm = (pitch_all + torch.pi) % (2 * torch.pi) - torch.pi

        # Get target values from commands
        target_pitch = self.commands[:, 6]  # body_pitch_target
        target_roll = self.commands[:, 7]   # body_roll_target

        # Calculate errors
        roll_error = roll_norm - target_roll
        pitch_error = pitch_norm - target_pitch

        # Return quadratic reward (smaller is better)
        orient_reward = torch.square(roll_error) + torch.square(pitch_error)
        return orient_reward

    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=-1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=-1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=-1)

    def _reward_root_acc(self):
        # Penalize root accelerations
        return torch.sum(torch.square((self.last_root_vel - self.root_states[:, 7:13]) / self.dt), dim=-1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=-1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        lower = self.dof_pos_limits[:, 0] + 0.5 * (1 - self.cfg["rewards"]["soft_dof_pos_limit"]) * (
            self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        )
        upper = self.dof_pos_limits[:, 1] - 0.5 * (1 - self.cfg["rewards"]["soft_dof_pos_limit"]) * (
            self.dof_pos_limits[:, 1] - self.dof_pos_limits[:, 0]
        )
        return torch.sum(((self.dof_pos < lower) | (self.dof_pos > upper)).float(), dim=-1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum(
            (torch.abs(self.dof_vel) - self.dof_vel_limits * self.cfg["rewards"]["soft_dof_vel_limit"]).clip(min=0.0, max=1.0),
            dim=-1,
        )

    def _reward_torque_limits(self):
        # Penalize torques too close to the limit
        return torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self.cfg["rewards"]["soft_torque_limit"]).clip(min=0.0),
            dim=-1,
        )

    def _reward_torque_tiredness(self):
        # Penalize torque tiredness
        return torch.sum(torch.square(self.torques / self.torque_limits).clip(max=1.0), dim=-1)

    def _reward_power(self):
        # Penalize power
        return torch.sum((self.torques * self.dof_vel).clip(min=0.0), dim=-1)

    def _reward_feet_slip(self):
        # Penalize feet velocities when contact
        return (
            torch.sum(
                torch.square((self.last_feet_pos - self.feet_pos) / self.dt).sum(dim=-1) * self.feet_contact.float(),
                dim=-1,
            )
            * (self.episode_length_buf > 1).float()
        )

    def _reward_feet_vel_z(self):
        return torch.sum(torch.square((self.last_feet_pos - self.feet_pos) / self.dt)[:, :, 2], dim=-1)

    def _reward_feet_roll(self):
        return torch.sum(torch.square(self.feet_roll), dim=-1)

    def _reward_feet_pitch(self):
        return torch.sum(torch.square(self.feet_pitch), dim=-1)

    def _reward_feet_yaw_diff(self):
        """
        Reward for tracking the commanded difference between left and right foot yaw angles.
        """
        commanded_diff = self.commands[:, 5] - self.commands[:, 4]  # foot_yaw_R - foot_yaw_L
        actual_diff = self.feet_yaw_rel[:, 1] - self.feet_yaw_rel[:, 0]  # right - left
        diff_error = (actual_diff - commanded_diff + torch.pi) % (2 * torch.pi) - torch.pi
        return torch.square(diff_error)

    def _reward_feet_yaw_mean(self):
        """
        Reward for tracking the commanded mean foot yaw angle.
        """
        commanded_mean = (self.commands[:, 5] + self.commands[:, 4]) * 0.5  # (foot_yaw_R + foot_yaw_L) / 2
        actual_mean = self.feet_yaw_rel.mean(dim=-1)
        mean_error = (actual_mean - commanded_mean + torch.pi) % (2 * torch.pi) - torch.pi
        return torch.square(mean_error)

    def _reward_feet_offset_x(self):
        """Reward for tracking feet x-offset target, scaled down at higher forward speed"""
        feet_x_offset, _ = self.get_feet_offset()
        target_x_offset = self.commands[:, 8]  # feet_offset_x_target
        x_error = feet_x_offset - target_x_offset
        x_reward = torch.clip(torch.abs(x_error), min=0.0, max=0.1)

        forward_vel = self.filtered_lin_vel[:, 0]
        max_forward_vel = self.cfg["rewards"]["feet_offset_vel_scale_x"]
        vel_scale = torch.clamp((1.0 - torch.abs(forward_vel) / max_forward_vel) ** 2, min=0.0, max=1.0)
        return x_reward * vel_scale

    def _reward_feet_offset_y(self):
        """Reward for tracking feet y-offset target, scaled down at higher lateral speed"""
        _, feet_y_offset = self.get_feet_offset()
        target_y_offset = self.commands[:, 9]  # feet_offset_y_target
        y_error = feet_y_offset - target_y_offset
        y_reward = torch.clip(torch.abs(y_error), min=0.0, max=0.1)

        lateral_vel = self.filtered_lin_vel[:, 1]
        max_lateral_vel = self.cfg["rewards"]["feet_offset_vel_scale_y"]
        vel_scale = torch.clamp((1.0 - torch.abs(lateral_vel) / max_lateral_vel) ** 2, min=0.0, max=1.0)
        return y_reward * vel_scale

    def _reward_feet_cross(self):
        """좌우 발 간격이 안전 구간을 벗어나는 것을 벌한다(양쪽 다).

        왜 `feet_offset_y`로 부족한가: 그것은 **명목 보폭**을 추종하는 보상이라
        평균을 맞추고, 순간적으로 다리가 교차하는 것은 벌하지 않는다. 실기 증언은
        "다리가 모이면서 발끼리 부딪혀서 넘어져"이고, 실기 로그를 MuJoCo 운동학에
        재생하니 좌우 발이 **9.9 % 구간에서 겹치고 p1이 −16.35 cm**다(MuJoCo 2.3 %,
        p1 −0.41 cm). 평균 보폭은 정상인데 꼬리가 교차한다.

        ⛔ 그리고 sim은 그 교차를 벌하지 않는다. 2026-08-06 확인: MuJoCo에서 좌우
        발을 인위적으로 겹치면 −4.7 cm에서 접촉 6건, −12.8 cm에서 2건, **−20.6 cm
        이상에서 0건**이다 -- 깊게 교차하면 그냥 통과한다. 실기의 p1(−16.4 cm)은
        접촉이 거의 사라지는 구간이다. 즉 정책은 교차를 피할 이유를 학습한 적이 없고,
        sim에서는 공짜인 행동이 실기에서는 넘어지는 원인이 된다.

        ⛔ 그리고 **양쪽을 벌해야 한다.** 2026-08-06 확인: 증상이 체크포인트마다
        반대다. i2a는 "다리가 모이면서 발끼리 부딪혀" 넘어졌고, `feet_offset_y -10`을
        넣은 stance10은 반대로 **너무 벌어진다** -- 좌우 발 간격 median이 실기
        14.06 cm 대 MuJoCo 7.47 cm로 2배이고, 벌어짐(L-R)이 median +5.9도에 폭
        56.9도(MuJoCo -0.4도 / 22.6도)다. 한쪽만 벌하면 반대쪽으로 밀려난다.

        `get_feet_offset`의 y는 이미 `feet_distance_ref`를 뺀 상대 오프셋이므로,
        절대 간격은 `|y + ref|`다. 좁아지는 쪽은 발 너비(`feet_min_gap`)를, 넓어지는
        쪽은 `feet_max_gap`을 넘은 만큼만 벌한다. 그 사이에서는 정확히 0이라 기존
        보상과 충돌하지 않는다 -- 자세 목표가 아니라 **안전 구간 제약**이다.
        """
        _, feet_y_offset = self.get_feet_offset()
        gap = torch.abs(feet_y_offset + self.cfg["rewards"]["feet_distance_ref"])
        lo = self.cfg["rewards"].get("feet_min_gap", 0.07)   # 발 너비 = 0.07 m
        hi = self.cfg["rewards"].get("feet_max_gap", 0.26)   # 실기 median 0.14 + 여유
        return torch.clip(lo - gap, min=0.0) + torch.clip(gap - hi, min=0.0)

    def _reward_feet_air_time(self):
        """Reward the swing duration that actually occurred, at touchdown.

        feet_swing is defined ON the gait phase: it pays when a foot is not in
        contact during the window the clock scheduled for it. That makes the
        clock load-bearing -- remove it and nothing rewards lifting a foot at
        all, so the robot stops stepping. This is the standard phase-free
        replacement: measure how long the foot was airborne and pay for it when
        it lands.

        Two guards matter. The reward is gated on the OTHER foot being down,
        which restores the alternation the phase used to enforce for free --
        without it, hopping on both feet scores the same as walking. And it is
        gated on a live gait clock, so stand goals (which freeze the clock at 0)
        still get no stepping incentive; that keeps the existing standing
        mechanism working even though the clock no longer drives the reward.
        """
        contact = self.feet_contact
        # 착지 순간: 공중에 있던 시간이 쌓여 있고 지금 닿았다.
        first_contact = (self.feet_air_time > 0.0) & contact
        self.feet_air_time = self.feet_air_time + self.dt
        other_down = (contact[:, [1, 0]] if len(self.feet_indices) == 2
                      else torch.ones_like(contact))
        target = float(self.cfg["rewards"].get("feet_air_time_target", 0.10))
        clip = float(self.cfg["rewards"].get("feet_air_time_clip", 0.30))
        credit = (self.feet_air_time - target).clamp(max=clip)
        rew = torch.sum(credit * (first_contact & other_down).float(), dim=-1)
        # 닿은 발의 누적을 지운다. 보상을 읽은 뒤여야 한다.
        self.feet_air_time = self.feet_air_time * (~contact).float()
        live = self.gait_frequency > 1.0e-8
        if live.dim() > 1:
            live = live.squeeze(-1)
        return rew * live.float()

    def _reward_feet_swing(self):
        left_swing = (torch.abs(self.gait_process - 0.25) < 0.5 * self.cfg["rewards"]["swing_period"]) & (self.gait_frequency > 1.0e-8)
        right_swing = (torch.abs(self.gait_process - 0.75) < 0.5 * self.cfg["rewards"]["swing_period"]) & (self.gait_frequency > 1.0e-8)
        return (left_swing & ~self.feet_contact[:, 0]).float() + (right_swing & ~self.feet_contact[:, 1]).float()

    def _reward_foot_yaw_L(self):
        """Reward for tracking left foot yaw angle"""
        error = self.feet_yaw_rel[:, 0] - self.commands[:, 4]
        return torch.square(error)

    def _reward_foot_yaw_R(self):
        """Reward for tracking right foot yaw angle"""
        error = self.feet_yaw_rel[:, 1] - self.commands[:, 5]
        return torch.square(error)

    def get_feet_offset(self):
        """
        Helper function to calculate feet offsets in robot coordinates.
        Returns both x and y offsets between right and left feet (right - left).
        For y-offset, the feet_distance_ref is subtracted to get the relative offset.
        """
        # Get base yaw to transform to robot coordinates
        _, _, base_yaw = get_euler_xyz(self.base_quat)

        # Calculate feet positions in robot coordinates
        # Transform from world to robot coordinates
        feet_x_offset = (
            torch.cos(base_yaw) * (self.feet_pos[:, 0, 0] - self.feet_pos[:, 1, 0]) +
            torch.sin(base_yaw) * (self.feet_pos[:, 0, 1] - self.feet_pos[:, 1, 1])
        )

        feet_y_offset = (
            -torch.sin(base_yaw) * (self.feet_pos[:, 0, 0] - self.feet_pos[:, 1, 0]) +
            torch.cos(base_yaw) * (self.feet_pos[:, 0, 1] - self.feet_pos[:, 1, 1])
        )

        # Subtract feet_distance_ref from y-offset to get relative offset
        feet_y_offset = feet_y_offset - self.cfg["rewards"]["feet_distance_ref"]

        return feet_x_offset, feet_y_offset

    def get_feet_x_offset(self):
        """
        Helper function to calculate feet x-offset in robot coordinates.
        Returns the x-offset between right and left feet (right - left).
        """
        feet_x_offset, _ = self.get_feet_offset()
        return feet_x_offset
