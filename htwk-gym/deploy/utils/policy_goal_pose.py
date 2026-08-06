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

        # 관측 확장. 학습 config 와 **같은 값**이어야 하고, 여기서 폭을 계산해
        # num_observations 와 대조한다 -- 어긋나면 정책을 올리기 전에 죽는다.
        # 순서가 어긋난 관측은 예외를 내지 않고 그냥 이상하게 걷기 때문에,
        # 하드웨어에 나가기 전 정적 검사로 막는 것이 유일하게 확실한 방법이다.
        self.history_steps = int(c.get("history_steps", 1) or 1)
        self.frame_width = (54
                            + (2 if c.get("extra_foot_offset") else 0)
                            + (self.num_act if c.get("extra_dof_tau") else 0))
        expected_obs = self.frame_width * self.history_steps
        if expected_obs != self.num_obs:
            raise ValueError(
                "관측 폭 불일치: config 조합이 {} (프레임 {} x 이력 {})인데 "
                "num_observations 는 {}. 학습 config 의 observation 블록과 "
                "env.num_observations 를 그대로 옮겨 적어라.".format(
                    expected_obs, self.frame_width, self.history_steps, self.num_obs))
        if c.get("extra_dof_tau") and len(c.get("torque_limits", [])) != self.num_act:
            raise ValueError(
                "extra_dof_tau 에는 policy.torque_limits 가 다리 관절 수({})만큼 "
                "필요하다. 학습 자산의 관절 한계와 같은 값이어야 한다.".format(self.num_act))
        self._frame = np.zeros(self.frame_width, dtype=np.float32)
        self._hist = np.zeros((self.history_steps, self.frame_width), dtype=np.float32)
        self._hist_primed = False

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

    # ---- 관측 확장 (envs/K1/goal_pose.py::_obs_extra_channels 와 짝) --------
    #
    # 학습 쪽 레이아웃 규약을 그대로 따른다:
    #     한 프레임 = [legacy 54][foot_offset 2][dof_tau 12]
    #     전체      = [frame_t][frame_{t-1}] ... [frame_{t-k+1}]
    #
    # 어느 것이 켜졌는지는 config 가 정한다. num_observations 로 역산하지 않는다 --
    # 폭이 같은 조합이 여럿이라 역산은 조용히 틀릴 수 있고, 관측 순서가 어긋난
    # 정책은 죽지 않고 그냥 이상하게 걷는다(가장 잡기 어려운 실패다).
    def _extra_frame(self, dof_pos, dof_vel, dof_tau, foot_offset):
        c = self.cfg["policy"]
        parts = []
        if c.get("extra_foot_offset"):
            if foot_offset is None:
                raise ValueError(
                    "policy.extra_foot_offset 이 켜져 있는데 inference 가 "
                    "foot_offset 을 받지 못했다. 호출자가 FK 로 (x, gap) 을 줘야 한다.")
            parts.append(np.asarray(foot_offset, dtype=np.float32)
                         * c["normalization"].get("foot_offset", 5.0))
        if c.get("extra_dof_tau"):
            if dof_tau is None:
                raise ValueError(
                    "policy.extra_dof_tau 가 켜져 있는데 inference 가 dof_tau 를 "
                    "받지 못했다. deploy 는 motor.tau_est 를 이미 읽고 있다.")
            lim = np.asarray(c["torque_limits"], dtype=np.float32)
            parts.append(np.asarray(dof_tau, dtype=np.float32)[self.leg_start:]
                         / np.maximum(lim, 1e-6))
        return np.concatenate(parts).astype(np.float32) if parts else None

    def inference(self, time_now, dof_pos, dof_vel, base_ang_vel, projected_gravity,
                  goal_rel_x, goal_rel_y, heading_error,
                  dof_tau=None, foot_offset=None):
        self.advance_gait_clock(time_now)

        # Goal command, robot-local frame, clamped to the E0-trained goal range.
        self.commands[0] = float(np.clip(goal_rel_x, -self.goal_x_clamp, self.goal_x_clamp))
        self.commands[1] = float(np.clip(goal_rel_y, -self.goal_y_clamp, self.goal_y_clamp))
        self.commands[2] = float(self._wrap_pi(heading_error))
        self.commands[3] = self.gait_frequency

        n = self.cfg["policy"]["normalization"]
        ls = self.leg_start
        frame = self._frame
        frame[0:3] = projected_gravity * n["gravity"]
        frame[3:6] = base_ang_vel * n["ang_vel"]
        frame[6:16] = self.commands * self.commands_scale
        frame[16] = np.cos(2.0 * np.pi * self.gait_process)
        frame[17] = np.sin(2.0 * np.pi * self.gait_process)
        frame[18:30] = (dof_pos - self.default_dof_pos)[ls:] * n["dof_pos"]
        frame[30:42] = dof_vel[ls:] * n["dof_vel"]
        frame[42:54] = self.actions
        extra = self._extra_frame(dof_pos, dof_vel, dof_tau, foot_offset)
        if extra is not None:
            frame[54:] = extra

        if self.history_steps <= 1:
            self.obs[:] = frame
        else:
            # 첫 호출에서는 이력이 없다. 0 으로 채우면 정책이 "중력 0 이었던 과거"를
            # 보게 되므로, 학습 쪽 _apply_obs_history 와 같이 **현재 프레임으로**
            # 채운다. 그래야 sim 의 에피소드 시작과 실기의 정책 시작이 같은 모양이다.
            if not self._hist_primed:
                for i in range(self.history_steps):
                    self._hist[i] = frame
                self._hist_primed = True
            else:
                self._hist[1:] = self._hist[:-1]
                self._hist[0] = frame
            self.obs[:] = self._hist.reshape(-1)

        with torch.inference_mode():
            output = self.policy(torch.from_numpy(self.obs).unsqueeze(0))
        self.actions[:] = output.squeeze(0).cpu().numpy()
        self.actions[:] = np.clip(self.actions, -self.clip_actions, self.clip_actions)

        self.dof_targets[:] = self.default_dof_pos
        self.dof_targets[ls:] += self.action_scale * self.actions
        return self.dof_targets
