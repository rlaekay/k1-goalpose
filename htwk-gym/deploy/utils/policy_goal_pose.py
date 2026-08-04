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

    Two things about the gait clock that this wrapper got wrong once each, so
    they are written down rather than left to be rediscovered:

    - ``gait_frequency`` IS gated. Training samples it as 0.0 for stand goals
      ("No More Marching"), so holding a walking frequency at the goal is off
      distribution, and ``feet_swing`` is itself gated on ``gait_frequency > 0``
      -- a non-zero clock at the goal sits on a stepping incentive. The caller
      owns that gate; see deploy_goal_pose._update_arrival_gait.
    - ``gait_process`` is an INTEGRATOR, not ``fmod(t * freq, 1)``
      (goal_pose.py:621). The difference is invisible at constant frequency but
      not at the gate: integrating freezes the phase while stopped and resumes
      from it, whereas the closed form teleports to 0 on arrival and to an
      arbitrary phase on departure.

    Unlike utils/policy.py for ParameterWalk, the clock channels are NOT zeroed
    when the frequency is zero: goal_pose.py:802 writes raw cos/sin.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.policy = torch.jit.load(cfg["policy"]["policy_path"])
        self.policy.eval()
        self._init_inference_variables()
        self._validate_policy_contract()

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
        self._last_time = None

        self.num_obs = int(c["num_observations"])
        self.num_act = int(c["num_actions"])
        self.leg_start = int(c.get("leg_dof_start", 10))  # 10, not 11: this K1 has no waist
        self.action_scale = float(c["control"]["action_scale"])
        self.clip_actions = float(norm["clip_actions"])
        self.policy_interval = self.cfg["common"]["dt"] * c["control"]["decimation"]

        self.commands = np.zeros(10, dtype=np.float32)
        self.commands[3] = self.gait_frequency  # style slots [4:10] stay 0.0
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        self.actions = np.zeros(self.num_act, dtype=np.float32)
        self.dof_targets = np.copy(self.default_dof_pos)

        # Do not pin the joint count to 23. The SDK constant B1JointCnt is 23 and
        # its B1JointIndex map places a waist at index 10, but this K1 has no
        # waist: low_state carries 22 joints with the legs at 10..21. Hardcoding
        # 23 here rejected the layout that actually matches the robot. The count
        # comes from the config; deploy_goal_pose verifies it against low_state
        # before publishing anything.
        expected = int(self.cfg["common"].get("joint_cnt", self.default_dof_pos.size))
        if self.default_dof_pos.shape != (expected,):
            raise ValueError(
                f"common.default_qpos has {self.default_dof_pos.shape[0]} entries but "
                f"common.joint_cnt is {expected}"
            )
        if self.leg_start + self.num_act != self.default_dof_pos.size:
            raise ValueError(
                "policy leg slice must cover exactly the final joints of the vector: "
                f"leg_start={self.leg_start}, num_actions={self.num_act}, "
                f"joints={self.default_dof_pos.size}. For this K1 that means "
                "leg_dof_start=10 with 22 joints."
            )

    def _validate_policy_contract(self):
        """Fail before CUSTOM mode if the copied TorchScript is the wrong actor."""
        with torch.inference_mode():
            output = self.policy(torch.zeros((1, self.num_obs), dtype=torch.float32))
        if not isinstance(output, torch.Tensor):
            raise TypeError(f"policy must return a Tensor, got {type(output)!r}")
        if tuple(output.shape) != (1, self.num_act):
            raise ValueError(
                f"policy contract mismatch: expected (1,{self.num_act}), got {tuple(output.shape)}"
            )
        if not torch.isfinite(output).all():
            raise ValueError("policy produced non-finite output for a zero-observation smoke test")

    @staticmethod
    def _wrap_pi(a):
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    def advance_gait_clock(self, time_now):
        """Integrate the gait phase, as goal_pose.py:621 does.

        Must be an integrator, not ``fmod(time_now * gait_frequency, 1)``. The
        two agree while the frequency is constant, so the closed form survived a
        long time, but the caller now gates the frequency to 0 at the goal and
        there they diverge badly: integrating freezes the phase and resumes from
        it, while the closed form teleports to 0 on arrival and to an arbitrary
        phase on departure -- a (cos, sin) step averaging 1.27 and reaching 1.996
        out of a possible 2.0, against 0.251 for one ordinary 50 Hz walking step.

        Elapsed time is measured rather than assumed, so 2 Hz is 2 Hz in wall
        clock under loop jitter, and clamped so a long stall (fall recovery, a
        blocked publish) cannot wind the phase forward by the whole gap.
        """
        if self._last_time is None:
            dt = self.policy_interval
        else:
            dt = min(max(time_now - self._last_time, 0.0), 4.0 * self.policy_interval)
        self._last_time = time_now
        self.gait_process = float(np.fmod(self.gait_process + dt * self.gait_frequency, 1.0))
        return self.gait_process

    def inference(self, time_now, dof_pos, dof_vel, base_ang_vel, projected_gravity,
                  goal_rel_x, goal_rel_y, heading_error):
        self.advance_gait_clock(time_now)

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

        with torch.inference_mode():
            output = self.policy(torch.from_numpy(self.obs).unsqueeze(0))
        self.actions[:] = output.squeeze(0).cpu().numpy()
        self.actions[:] = np.clip(self.actions, -self.clip_actions, self.clip_actions)

        self.dof_targets[:] = self.default_dof_pos
        self.dof_targets[ls:] += self.action_scale * self.actions
        return self.dof_targets
