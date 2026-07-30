import numpy as np
import torch


class GoalPosePolicy:
    """E0 GoalPose position-policy deploy wrapper.

    Unlike the ParameterWalk wrapper (utils/policy.py), which puts (vx, vy, vyaw)
    in obs[6:8], E0 consumes a 10-wide ``commands`` block at obs[6:16] whose first
    three entries are the goal-pose error in the ROBOT LOCAL frame:

        commands[0] = goal_rel_x     [m]   clamped to the trained range
        commands[1] = goal_rel_y     [m]   clamped to the trained range
        commands[2] = heading_error  [rad] wrapped to [-pi, pi]
        commands[3] = gait_frequency [Hz]
        commands[4:10] = style slots (0.0; fixed to neutral at train time)

    Observation layout MUST match envs/K1/goal_pose.py::_compute_observations
    (via GoalPoseV7 -> ... -> GoalPose):

        0:3    projected_gravity           * norm.gravity
        3:6    base_ang_vel                * norm.ang_vel
        6:16   commands(10)                * commands_scale
        16     cos(2*pi*gait_process)
        17     sin(2*pi*gait_process)
        18:30  (dof_pos - default)[legs]   * norm.dof_pos
        30:42  dof_vel[legs]               * norm.dof_vel
        42:54  previous action

    NOTE: E0 does NOT gate the gait clock / commands by gait_frequency the way
    utils/policy.py does for ParameterWalk. The env always writes the goal
    command and the gait clock, so we replicate that here (constant gait_freq).
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.policy = torch.jit.load(cfg["policy"]["policy_path"])
        self.policy.eval()
        self._init_inference_variables()

    def get_policy_interval(self):
        return self.policy_interval

    def _init_inference_variables(self):
        c = self.cfg["policy"]
        norm = c["normalization"]

        self.default_dof_pos = np.array(self.cfg["common"]["default_qpos"], dtype=np.float32)

        # commands scale for the 10-wide E0 command block, in command_order:
        # [goal_rel_x, goal_rel_y, heading_error, gait_frequency,
        #  foot_yaw_L, foot_yaw_R, body_pitch, body_roll, feet_off_x, feet_off_y]
        self.commands_scale = np.array(
            [
                norm["goal_pos"], norm["goal_pos"], norm["goal_heading"],
                norm["gait_frequency"],
                norm["foot_yaw"], norm["foot_yaw"],
                norm["body_pitch_target"], norm["body_roll_target"],
                norm["feet_offset_x_target"], norm["feet_offset_y_target"],
            ],
            dtype=np.float32,
        )

        gc = c["goal_clamp"]
        self.goal_x_clamp = float(gc["x_m"])
        self.goal_y_clamp = float(gc["y_m"])

        self.gait_frequency = float(c["gait_frequency"])
        self.gait_process = 0.0

        self.num_obs = int(c["num_observations"])
        self.num_act = int(c["num_actions"])
        self.leg_start = int(c.get("leg_dof_start", 11))
        self.action_scale = float(c["control"]["action_scale"])
        self.clip_actions = float(norm["clip_actions"])
        self.policy_interval = self.cfg["common"]["dt"] * c["control"]["decimation"]

        self.commands = np.zeros(10, dtype=np.float32)
        self.commands[3] = self.gait_frequency  # style slots [4:10] stay 0.0
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.actions = np.zeros(self.num_act, dtype=np.float32)
        self.dof_targets = np.copy(self.default_dof_pos)

    @staticmethod
    def _wrap_pi(a):
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    def inference(self, time_now, dof_pos, dof_vel, base_ang_vel, projected_gravity,
                  goal_rel_x, goal_rel_y, heading_error):
        # Gait clock advances continuously at a constant frequency (no gating).
        self.gait_process = np.fmod(time_now * self.gait_frequency, 1.0)

        # Goal command, robot-local frame, clamped to the E0-trained goal range.
        self.commands[0] = float(np.clip(goal_rel_x, -self.goal_x_clamp, self.goal_x_clamp))
        self.commands[1] = float(np.clip(goal_rel_y, -self.goal_y_clamp, self.goal_y_clamp))
        self.commands[2] = float(self._wrap_pi(heading_error))
        self.commands[3] = self.gait_frequency

        n = self.cfg["policy"]["normalization"]
        ls = self.leg_start
        self.obs[0:3] = projected_gravity * n["gravity"]
        self.obs[3:6] = base_ang_vel * n["ang_vel"]
        self.obs[6:16] = self.commands * self.commands_scale
        self.obs[16] = np.cos(2.0 * np.pi * self.gait_process)
        self.obs[17] = np.sin(2.0 * np.pi * self.gait_process)
        self.obs[18:30] = (dof_pos - self.default_dof_pos)[ls:] * n["dof_pos"]
        self.obs[30:42] = dof_vel[ls:] * n["dof_vel"]
        self.obs[42:54] = self.actions

        self.actions[:] = self.policy(torch.from_numpy(self.obs).unsqueeze(0)).detach().numpy()
        self.actions[:] = np.clip(self.actions, -self.clip_actions, self.clip_actions)

        self.dof_targets[:] = self.default_dof_pos
        self.dof_targets[ls:] += self.action_scale * self.actions
        return self.dof_targets
