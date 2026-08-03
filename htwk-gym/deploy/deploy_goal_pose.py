"""E0 GoalPose position-policy real-robot deploy loop.

Mirrors deploy_parameter_walk.py but feeds the E0 policy a goal-pose error
(goal_rel_x, goal_rel_y, heading_error) in the robot-local frame instead of a
velocity command. See ROBOT_DEPLOY_E0_GUIDE.md.

Goal source (``--goal-source``):
  * ``ros``    : MISSION mode (default). Subscribe INHA-Player
                 /locomotion_test/goal_rel -- the robot-local goal error the
                 Brain recomputes every BT tick from camera-PF localization.
                 Run the Brain (tree:=locomotion_test) alongside this, on the
                 same ROS_DOMAIN_ID. rclpy must be importable here.
  * ``fixed``  : constant goal from config deploy_goal or --goal "x,y,theta".
  * ``stdin``  : type "x y theta" lines to change the goal live.

NOTE: this touches the real robot and has NOT been run here. Validate the
guide's TorchScript smoke + a LowState replay before ground contact.
"""

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time

import numpy as np
import yaml

from booster_robotics_sdk_python import (
    ChannelFactory,
    B1LocoApiId,
    B1LocoClient,
    B1LowCmdPublisher,
    B1LowStateSubscriber,
    LowCmd,
    LowState,
    B1JointCnt,
    RobotMode,
)

from utils.command import create_prepare_cmd, create_first_frame_rl_cmd, init_Cmd_T1
from utils.remote_control_service import RemoteControlService
from utils.rotate import rotate_vector_inverse_rpy
from utils.timer import TimerConfig, Timer
from utils.policy_goal_pose import GoalPosePolicy


class GoalSource:
    """Thread-safe robot-local goal (goal_rel_x, goal_rel_y, heading_error)."""

    def __init__(self, cfg, initial=None):
        dg = cfg.get("deploy_goal", {})
        x = float(dg.get("goal_rel_x", 0.0))
        y = float(dg.get("goal_rel_y", 0.0))
        h = float(dg.get("heading_error", 0.0))
        if initial is not None:
            x, y, h = initial
        self._goal = np.array([x, y, h], dtype=np.float32)
        self._lock = threading.Lock()
        self._stale_timeout = float(dg.get("stale_timeout_s", 0.5))
        self._last_update = time.monotonic()
        self._use_timeout = False  # fixed source never goes stale

    def set(self, x, y, h):
        with self._lock:
            self._goal[:] = (x, y, h)
            self._last_update = time.monotonic()

    def get(self):
        with self._lock:
            if self._use_timeout and (time.monotonic() - self._last_update) > self._stale_timeout:
                return 0.0, 0.0, 0.0  # stale -> stand
            return float(self._goal[0]), float(self._goal[1]), float(self._goal[2])

    def enable_timeout(self):
        self._use_timeout = True

    def start_stdin_reader(self, logger):
        self.enable_timeout()

        def _reader():
            logger.info("stdin goal source: type 'x y theta' (m m rad), or 'stop'")
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                if line.lower() in ("stop", "0"):
                    self.set(0.0, 0.0, 0.0)
                    continue
                try:
                    parts = line.replace(",", " ").split()
                    x, y, h = float(parts[0]), float(parts[1]), float(parts[2])
                    self.set(x, y, h)
                    logger.info(f"goal <- ({x:.3f}, {y:.3f}, {h:.3f})")
                except (ValueError, IndexError):
                    logger.warning(f"bad goal line: {line!r}")

        t = threading.Thread(target=_reader, daemon=True)
        t.start()
        return t


class RosGoalSource:
    """Mission goal source: subscribe INHA-Player /locomotion_test/goal_rel.

    The Brain publishes geometry_msgs/Vector3Stamped where
        vector.x = goal_rel_x [m] (forward), vector.y = goal_rel_y [m] (left),
        vector.z = heading_error [rad], already in the robot-local frame and
    recomputed every BT tick from camera-PF localization. We just hold the
    latest; a stale message (no update within stale_timeout_s) replaces the
    command with (0,0,0). That requests E0's learned zero-goal behavior; it is
    not a hardware stop. Ctrl-C/fault cleanup separately requests DAMPING.

    Requires rclpy importable in this process: run with ROS2 sourced and on the
    same ROS_DOMAIN_ID as the Brain (e.g. both on the robot board).
    """

    def __init__(self, cfg, topic=None, debug_topic=None):
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import Vector3Stamped
        from std_msgs.msg import String

        self._rclpy = rclpy
        dg = cfg.get("deploy_goal", {})
        topic = topic or str(dg.get("topic", "/locomotion_test/goal_rel"))
        debug_topic = debug_topic or str(
            dg.get("debug_topic", "/locomotion_test/policy_debug")
        )
        self._stale_timeout = float(dg.get("stale_timeout_s", 0.5))
        self._goal = np.zeros(3, dtype=np.float32)
        self._last = 0.0
        self._lock = threading.Lock()
        self._received = 0
        self._rejected = 0

        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = Node("e0_goal_rel_sub")
        self._node.create_subscription(Vector3Stamped, topic, self._cb, 10)
        self._debug_pub = self._node.create_publisher(String, debug_topic, 10)
        self.topic = topic
        self.debug_topic = debug_topic
        self._spin = threading.Thread(target=lambda: rclpy.spin(self._node), daemon=True)
        self._spin.start()

    def _cb(self, msg):
        goal = np.array((msg.vector.x, msg.vector.y, msg.vector.z), dtype=np.float32)
        with self._lock:
            if not np.all(np.isfinite(goal)):
                self._rejected += 1
                return
            self._goal[:] = goal
            self._last = time.monotonic()
            self._received += 1

    def get_with_status(self):
        with self._lock:
            age = float("inf") if self._last <= 0.0 else time.monotonic() - self._last
            stale = age > self._stale_timeout
            goal = (0.0, 0.0, 0.0) if stale else tuple(float(v) for v in self._goal)
            status = {
                "goal_age_sec": None if not np.isfinite(age) else age,
                "goal_stale": stale,
                "goal_messages_received": self._received,
                "goal_messages_rejected": self._rejected,
            }
            return goal, status

    def publish_debug(self, payload):
        from std_msgs.msg import String

        msg = String()
        msg.data = json.dumps(payload, allow_nan=False, separators=(",", ":"))
        self._debug_pub.publish(msg)

    def close(self):
        if self._rclpy.ok():
            self._node.destroy_node()
            self._rclpy.shutdown()
        if self._spin.is_alive():
            self._spin.join(timeout=1.0)


# LocoApiId values the Python binding does not expose by name. B1LocoApiId(int)
# constructs fine, so these are reachable even though only kChangeMode/kMove/
# kRotateHead appear in the enum.
API_ID_GET_UP = 2008
API_ID_GET_UP_WITH_MODE = 2025


class FallState:
    """Mirrors booster_interface FallDownStateType."""
    IS_READY = 0
    IS_FALLING = 1
    HAS_FALLEN = 2
    IS_GETTING_UP = 3

    NAMES = {0: "IS_READY", 1: "IS_FALLING", 2: "HAS_FALLEN", 3: "IS_GETTING_UP"}


class FallMonitor:
    """Authoritative fall state from the SDK, over ROS.

    /fall_down_recovery_state is booster_msgs/RawBytesMsg carrying a 3-byte
    struct (see brain types.h RobotRecoveryStateData):
        uint8 state, uint8 is_recovery_available, uint8 current_planner_index

    Measured at ~1 Hz on hardware, which is far too slow to *trigger* a stop --
    the IMU watchdog at 500 Hz does that. This exists for the part the IMU
    cannot tell us: whether the robot has settled into HAS_FALLEN, and whether
    the SDK is willing to run its get-up (is_recovery_available). Calling GetUp
    when it is not available just fails.
    """

    def __init__(self, topic="/fall_down", raw_topic="/fall_down_recovery_state",
                 node=None, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.available = False
        self.state = None
        self.is_recovery_available = False
        self.planner_index = None
        self.last_update = 0.0
        self._lock = threading.Lock()
        self._own_node = None
        self._spin = None

        try:
            import rclpy
            from rclpy.node import Node
            # Both live in booster_interface (not booster_msgs, which only has
            # the RPC types). /fall_down is a typed FallDownState; the raw
            # variant carries the same thing as 3 bytes and is subscribed too so
            # a single dead publisher cannot blind the recovery path.
            from booster_interface.msg import FallDownState, RawBytesMsg
        except ImportError as exc:
            self.logger.warning(
                "FallMonitor disabled (%s). Recovery will rely on the IMU "
                "watchdog alone and will not know when get-up is permitted.", exc)
            return

        if node is None:
            if not rclpy.ok():
                rclpy.init(args=None)
            node = Node("e0_fall_monitor")
            self._own_node = node
        node.create_subscription(FallDownState, topic, self._cb_typed, 10)
        node.create_subscription(RawBytesMsg, raw_topic, self._cb_raw, 10)
        if self._own_node is not None:
            self._spin = threading.Thread(
                target=lambda: rclpy.spin(self._own_node), daemon=True)
            self._spin.start()
        self.available = True
        self.logger.info("FallMonitor subscribed to %s (typed) and %s (raw)",
                         topic, raw_topic)

    def _store(self, state, recov, planner=None):
        with self._lock:
            self.state = int(state)
            self.is_recovery_available = bool(recov)
            if planner is not None:
                self.planner_index = planner
            self.last_update = time.monotonic()

    def _cb_typed(self, msg):
        self._store(msg.fall_down_state, msg.is_recovery_available)

    def _cb_raw(self, msg):
        raw = bytes(bytearray(msg.msg))
        if len(raw) < 2:
            return
        self._store(raw[0], raw[1], raw[2] if len(raw) > 2 else None)

    def snapshot(self):
        with self._lock:
            age = float("inf") if self.last_update <= 0 else time.monotonic() - self.last_update
            return self.state, self.is_recovery_available, age

    def close(self):
        if self._own_node is not None:
            try:
                self._own_node.destroy_node()
            except Exception:
                pass


class Controller:
    def __init__(self, cfg_file, goal_source_mode="ros", initial_goal=None,
                 goal_topic=None, debug_topic=None,
                 hold_prepare=False, prepare_settle_log_s=1.0) -> None:
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Mode-transition instrumentation/behaviour knobs. See --hold-prepare.
        self.hold_prepare = bool(hold_prepare)
        self.prepare_settle_log_s = float(prepare_settle_log_s)
        self._custom_mode_entered_monotonic = 0.0
        self._prepare_q = None
        self._joint_layout_checked = False

        # --- fall recovery ---------------------------------------------------
        # RECOVER_NONE while the policy is driving; anything else means the
        # policy is suspended and the recovery sequence owns the robot.
        self._recovery_phase = "none"
        self._recovery_reason = ""
        self._recovery_t0 = 0.0
        self._recovery_count = 0
        self._fall_events = []
        self._latest_rpy = np.zeros(3, dtype=np.float32)

        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

        # Joint vector length for THIS robot, from the config rather than the
        # SDK's B1JointCnt. See _init_low_state_values / _verify_joint_layout.
        self.joint_cnt = int(self.cfg["common"].get(
            "joint_cnt", len(self.cfg["common"]["default_qpos"])))

        # Load and contract-check the actor before creating SDK/remote-control
        # services. A missing or wrong model must fail without touching robot I/O.
        self.policy = GoalPosePolicy(cfg=self.cfg)

        self.running_policy = True
        rec_cfg = self.cfg.get("safety", {}).get("recovery", {})
        self.fall_monitor = (FallMonitor(
            topic=str(rec_cfg.get("fall_topic", "/fall_down")),
            raw_topic=str(rec_cfg.get("fall_raw_topic", "/fall_down_recovery_state")),
            logger=self.logger) if bool(rec_cfg.get("enable", True)) else None)

        self.goal_source_mode = goal_source_mode
        if goal_source_mode == "ros":
            self.goal_source = RosGoalSource(
                self.cfg, topic=goal_topic, debug_topic=debug_topic
            )  # mission mode: live BT bridge
        else:
            self.goal_source = GoalSource(self.cfg, initial=initial_goal)  # fixed/stdin

        self.remoteControlService = RemoteControlService()

        # Initialize every field touched by an SDK callback before opening the
        # channels: InitChannel may deliver LowState immediately.
        self.publish_runner = None
        self.running = True
        self.publish_lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._cleaned_up = False
        self._custom_mode_started = False
        self._last_low_state_monotonic = 0.0
        self._last_debug_monotonic = 0.0
        self._last_goal = (0.0, 0.0, 0.0)
        self._latest_rpy = np.zeros(3, dtype=np.float32)

        self._init_timer()
        self._init_low_state_values()
        self._init_communication()

    def _init_timer(self):
        self.timer = Timer(TimerConfig(time_step=self.cfg["common"]["dt"]))
        self.next_publish_time = self.timer.get_time()
        self.next_inference_time = self.timer.get_time()

    def _init_low_state_values(self):
        # Joint count comes from the config, NOT from B1JointCnt.
        #
        # B1JointCnt is 23 and the SDK's B1JointIndex map has a waist at index
        # 10 with legs at 11..22. This K1 has no waist: low_state carries 22
        # entries and the legs start at index 10. Sizing these arrays from the
        # SDK constant silently shifts every leg command by one joint -- knee
        # targets land on the ankle -- with no error anywhere. Verified on
        # hardware: the parallel-mechanism joints (serial != parallel) sit at
        # 14,15,20,21, not the 15,16,21,22 a 23-joint layout implies.
        n = self.joint_cnt
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.zeros(3, dtype=np.float32)
        self.dof_pos = np.zeros(n, dtype=np.float32)
        self.dof_vel = np.zeros(n, dtype=np.float32)

        self.dof_target = np.zeros(n, dtype=np.float32)
        self.filtered_dof_target = np.zeros(n, dtype=np.float32)
        self.dof_pos_latest = np.zeros(n, dtype=np.float32)

    def _init_communication(self) -> None:
        try:
            self.low_cmd = LowCmd()
            self.low_state_subscriber = B1LowStateSubscriber(self._low_state_handler)
            self.low_cmd_publisher = B1LowCmdPublisher()
            self.client = B1LocoClient()

            self.low_state_subscriber.InitChannel()
            self.low_cmd_publisher.InitChannel()
            self.client.Init()
        except Exception as e:
            self.logger.error(f"Failed to initialize communication: {e}")
            raise

    def _verify_joint_layout(self, low_state_msg):
        """Fail loudly if the robot's joint vector is not the one we configured.

        A mismatch here is silent and dangerous: numpy would simply write fewer
        joints than we think, or leave the last one at zero, and every leg
        command shifts. Checked once, on the first low_state, before any LowCmd
        is published.
        """
        if self._joint_layout_checked:
            return
        self._joint_layout_checked = True
        actual = len(low_state_msg.motor_state_serial)
        if actual != self.joint_cnt:
            raise RuntimeError(
                "Joint count mismatch: robot reports %d joints, config says %d "
                "(common.joint_cnt, or len(common.default_qpos)). "
                "This K1 has no waist joint, so the correct layout is 22 joints "
                "with policy.leg_dof_start=10. Running with the wrong layout "
                "shifts every leg command by one joint."
                % (actual, self.joint_cnt)
            )
        # The parallel-mechanism indices are a second, independent fingerprint of
        # the layout: those are the only joints where serial and parallel differ.
        try:
            observed = {
                i for i in range(actual)
                if abs(low_state_msg.motor_state_serial[i].q
                       - low_state_msg.motor_state_parallel[i].q) > 1e-4
            }
        except Exception:
            return
        configured = set(self.cfg.get("mech", {}).get("parallel_mech_indexes", []))
        if observed and configured and observed != configured:
            raise RuntimeError(
                "Parallel-mechanism indices disagree with the robot: observed %s, "
                "config mech.parallel_mech_indexes=%s. The configured joint layout "
                "does not match this hardware."
                % (sorted(observed), sorted(configured))
            )
        self.logger.info(
            "[joint-layout] %d joints, legs %d..%d, parallel %s -- matches hardware",
            actual, self.policy.leg_start,
            self.policy.leg_start + self.policy.num_act - 1, sorted(configured))

    def _low_state_handler(self, low_state_msg: LowState):
        # Safety watchdog: a large base roll/pitch means the robot is going over.
        self._verify_joint_layout(low_state_msg)
        self._last_low_state_monotonic = time.monotonic()
        self._latest_rpy[:] = low_state_msg.imu_state.rpy
        rpy_limit = float(self.cfg.get("safety", {}).get("roll_pitch_limit_rad", 1.0))
        if (abs(low_state_msg.imu_state.rpy[0]) > rpy_limit or
                abs(low_state_msg.imu_state.rpy[1]) > rpy_limit):
            # This is the fast path: low_state runs at ~500 Hz, while the SDK's
            # own fall topic publishes at 1 Hz. Waiting for that would mean up
            # to a second of the policy still driving a robot that is going
            # over. Previously this killed the process; now it hands off to the
            # recovery sequence so the run can continue after a get-up.
            self._request_recovery(
                "imu_rpy roll=%.2f pitch=%.2f > %.2f rad"
                % (low_state_msg.imu_state.rpy[0], low_state_msg.imu_state.rpy[1],
                   rpy_limit))
        self.timer.tick_timer_if_sim()
        time_now = self.timer.get_time()
        for i, motor in enumerate(low_state_msg.motor_state_serial):
            self.dof_pos_latest[i] = motor.q
        if time_now >= self.next_inference_time:
            self.projected_gravity[:] = rotate_vector_inverse_rpy(
                low_state_msg.imu_state.rpy[0],
                low_state_msg.imu_state.rpy[1],
                low_state_msg.imu_state.rpy[2],
                np.array([0.0, 0.0, -1.0]),
            )
            self.base_ang_vel[:] = low_state_msg.imu_state.gyro
            for i, motor in enumerate(low_state_msg.motor_state_serial):
                self.dof_pos[i] = motor.q
                self.dof_vel[i] = motor.dq

    def _send_cmd(self, cmd: LowCmd):
        self.low_cmd_publisher.Write(cmd)

    def cleanup(self) -> None:
        with self._cleanup_lock:
            if self._cleaned_up:
                return
            self._cleaned_up = True
            self.running = False

            if self.publish_runner is not None and self.publish_runner.is_alive():
                self.publish_runner.join(timeout=1.0)

            # Ctrl-C/SystemExit also comes through __exit__. Always leave CUSTOM
            # mode before closing the command channel once we entered it.
            if self._custom_mode_started and hasattr(self, "client"):
                try:
                    self.client.ChangeMode(RobotMode.kDamping)
                except Exception as exc:
                    self.logger.error("Failed to request DAMPING during cleanup: %s", exc)

            if self.fall_monitor is not None:
                self.fall_monitor.close()
            if hasattr(self.goal_source, "close"):
                self.goal_source.close()
            self.remoteControlService.close()
            if hasattr(self, "low_cmd_publisher"):
                self.low_cmd_publisher.CloseChannel()
            if hasattr(self, "low_state_subscriber"):
                self.low_state_subscriber.CloseChannel()

    def _log_joint_deviation(self, tag, target_q):
        """Report how far the measured legs are from a commanded pose.

        The mode transition is not instantaneous, and the only way to see what
        the robot is actually doing during it is the measured-vs-commanded gap on
        the 12 policy joints.  Everything else (SDK internals, the mode state
        machine) is opaque from here.
        """
        legs = slice(self.policy.leg_start, self.policy.leg_start + self.policy.num_act)
        err = np.asarray(self.dof_pos[legs], dtype=np.float64) - np.asarray(target_q[legs], dtype=np.float64)
        worst = int(np.argmax(np.abs(err)))
        self.logger.info(
            "[mode-timing] %-22s max|q_meas-q_cmd|=%.4f rad at leg idx %d  rms=%.4f rad",
            tag, float(np.max(np.abs(err))), self.policy.leg_start + worst,
            float(np.sqrt(np.mean(err ** 2))),
        )

    def start_custom_mode_conditionally(self):
        self._require_fresh_low_state("before waiting for CUSTOM mode")
        print(f"{self.remoteControlService.get_custom_mode_operation_hint()}")
        while True:
            if self.remoteControlService.start_custom_mode():
                break
            time.sleep(0.1)

        # The operator may spend an arbitrary amount of time at the remote-control
        # prompt.  Re-check immediately before the first LowCmd and mode change;
        # the earlier check alone cannot make that transition safe.
        self._require_fresh_low_state("immediately before CUSTOM mode")
        create_prepare_cmd(self.low_cmd, self.cfg)
        prepare_q = np.array([self.low_cmd.motor_cmd[i].q for i in range(self.joint_cnt)],
                             dtype=np.float64)
        for i in range(self.joint_cnt):
            self.dof_target[i] = self.low_cmd.motor_cmd[i].q
            self.filtered_dof_target[i] = self.low_cmd.motor_cmd[i].q

        # The `prepare` pose and gains are NOT the RL ones: prepare holds hips at
        # -0.1 / knees 0.2 with stiffness 350-450, while the policy runs at
        # -0.2 / 0.4 with stiffness 100/50.  Entering CUSTOM therefore snaps the
        # legs to a different, much stiffer posture before the policy ever runs,
        # and that snap is the visible part of the "mode change delay".
        self._log_joint_deviation("before prepare cmd", prepare_q)
        self._send_cmd(self.low_cmd)

        # Mark the transition as attempted before the SDK call: if ChangeMode
        # raises after the robot accepted it, cleanup must still request DAMPING.
        self._custom_mode_started = True
        t0 = time.monotonic()
        self.client.ChangeMode(RobotMode.kCustom)
        self._custom_mode_entered_monotonic = time.monotonic()
        self.logger.info("[mode-timing] ChangeMode(kCustom) returned in %.3f s",
                         self._custom_mode_entered_monotonic - t0)

        # Watch the legs settle onto the prepare pose. Nothing publishes LowCmd
        # between here and the RL-gait prompt unless --hold-prepare is given, so
        # without this the entire transition is invisible.
        settle_deadline = time.monotonic() + max(0.0, self.prepare_settle_log_s)
        while time.monotonic() < settle_deadline:
            if self.hold_prepare:
                self._send_cmd(self.low_cmd)
            time.sleep(0.1)
            self._log_joint_deviation(
                "settling +%.1fs" % (time.monotonic() - self._custom_mode_entered_monotonic),
                prepare_q)
        self._prepare_q = prepare_q

    def _require_fresh_low_state(self, context):
        low_state_timeout = float(self.cfg.get("safety", {}).get("low_state_timeout_s", 0.2))
        low_state_age = time.monotonic() - self._last_low_state_monotonic
        if self._last_low_state_monotonic <= 0.0 or low_state_age > low_state_timeout:
            raise RuntimeError(
                f"No fresh LowState {context} (age={low_state_age:.3f}s, "
                f"limit={low_state_timeout:.3f}s)"
            )
        return low_state_age

    # ------------------------------------------------------------------ entry --
    def _enter_custom_latched(self, ramp_s=None):
        """Enter CUSTOM commanding the pose the robot is already in, then ramp.

        Measured cost of the old path (prepare pose + prepare gains, snap):
        0.53 rad of joint travel and 2.8 s to settle. Almost all of that is the
        gap between what the robot happens to be holding and what we command at
        the instant CUSTOM engages.

        Latching the *measured* pose makes commanded == measured at t=0, so the
        step is structurally zero no matter what pose we come from -- standing,
        or straight out of a get-up. The move to the RL default pose and RL
        gains then happens as a controlled ramp instead of a snap. Aligning the
        static `prepare` block to the RL block would only help when the robot
        happens to be in exactly that pose; this always helps.
        """
        rec = self.cfg.get("safety", {}).get("recovery", {})
        ramp_s = float(rec.get("custom_entry_ramp_s", 0.6)) if ramp_s is None else ramp_s

        self._require_fresh_low_state("before latched CUSTOM entry")
        latched = np.array(self.dof_pos_latest[:self.joint_cnt], dtype=np.float32)

        prep_stiff = np.asarray(self.cfg["prepare"]["stiffness"], dtype=np.float32)
        prep_damp = np.asarray(self.cfg["prepare"]["damping"], dtype=np.float32)
        rl_stiff = np.asarray(self.cfg["common"]["stiffness"], dtype=np.float32)
        rl_damp = np.asarray(self.cfg["common"]["damping"], dtype=np.float32)
        rl_q = np.asarray(self.cfg["common"]["default_qpos"], dtype=np.float32)

        init_Cmd_T1(self.low_cmd)
        for i in range(self.joint_cnt):
            self.low_cmd.motor_cmd[i].q = float(latched[i])
            self.low_cmd.motor_cmd[i].kp = float(prep_stiff[i])
            self.low_cmd.motor_cmd[i].kd = float(prep_damp[i])
        self.dof_target[:] = latched
        self.filtered_dof_target[:] = latched
        self._send_cmd(self.low_cmd)

        self._custom_mode_started = True
        t0 = time.monotonic()
        # Do not block on the return value. Measured: 0.8 ms when the RPC
        # round-trips, but exactly 1.000 s of dead timeout (rc=100) when it does
        # not -- and the mode change still took effect every time. Completion is
        # judged from joint behaviour below, not from this code.
        try:
            rc = self.client.ChangeMode(RobotMode.kCustom)
        except Exception as exc:
            rc = "exception: %s" % exc
        self._custom_mode_entered_monotonic = time.monotonic()
        self.logger.info("[mode-timing] ChangeMode(kCustom) rc=%s in %.3f s",
                         rc, self._custom_mode_entered_monotonic - t0)

        # Ramp pose and gains together. Keeping the command streaming through
        # the transition is what makes it smooth; a single frame followed by
        # silence leaves the robot on a stale target.
        dt = float(self.cfg["common"]["dt"])
        steps = max(1, int(ramp_s / dt))
        for s in range(1, steps + 1):
            a = s / steps
            for i in range(self.joint_cnt):
                self.low_cmd.motor_cmd[i].q = float((1 - a) * latched[i] + a * rl_q[i])
                self.low_cmd.motor_cmd[i].kp = float((1 - a) * prep_stiff[i] + a * rl_stiff[i])
                self.low_cmd.motor_cmd[i].kd = float((1 - a) * prep_damp[i] + a * rl_damp[i])
            self._send_cmd(self.low_cmd)
            time.sleep(dt)

        self.dof_target[:] = rl_q
        self.filtered_dof_target[:] = rl_q
        self._prepare_q = latched
        moved = float(np.max(np.abs(
            np.asarray(self.dof_pos_latest[:self.joint_cnt]) - latched)))
        self.logger.info(
            "[mode-timing] latched CUSTOM entry done in %.2f s, joints moved %.4f rad",
            time.monotonic() - t0, moved)

    # --------------------------------------------------------------- recovery --
    def _request_recovery(self, reason):
        if self._recovery_phase != "none":
            return
        rec = self.cfg.get("safety", {}).get("recovery", {})
        if not bool(rec.get("enable", True)):
            self.logger.error("fall detected (%s) and recovery is disabled; stopping.",
                              reason)
            self.running = False
            return
        self._recovery_phase = "stopping"
        self._recovery_reason = reason
        self._recovery_t0 = time.monotonic()
        self._recovery_count += 1
        self._fall_events.append({"reason": reason, "t": time.time()})
        self.logger.warning("FALL DETECTED (%s) -- suspending policy, recovery #%d",
                            reason, self._recovery_count)

    def in_recovery(self):
        return self._recovery_phase != "none"

    def _step_recovery(self):
        """One tick of the fall-recovery sequence.

        CUSTOM -> DAMPING -> GetUp -> (SDK leaves us in walking) -> CUSTOM.
        GetUpWithMode is documented as landing in kWalking or kSoccer only, so
        there is no way to get up straight back into CUSTOM; the re-entry is a
        separate step.
        """
        rec = self.cfg.get("safety", {}).get("recovery", {})
        phase = self._recovery_phase
        elapsed = time.monotonic() - self._recovery_t0
        state, recov_ok, fall_age = (
            self.fall_monitor.snapshot() if self.fall_monitor else (None, False, float("inf")))

        if phase == "stopping":
            # Stop driving joints first, before anything else.
            self.running_policy = False
            self.logger.warning("[recovery] damping")
            try:
                self.client.ChangeMode(RobotMode.kDamping)
            except Exception as exc:
                self.logger.error("[recovery] ChangeMode(kDamping) failed: %s", exc)
            self._custom_mode_started = False
            self._recovery_phase = "damping"
            self._recovery_t0 = time.monotonic()
            return

        if phase == "damping":
            settle = float(rec.get("damping_settle_s", 1.5))
            if elapsed < settle:
                return
            # Prefer the SDK's own judgement of whether a get-up will work.
            # Without the monitor we can only wait out the settle time and try.
            if self.fall_monitor and self.fall_monitor.available and fall_age < 5.0:
                if not recov_ok:
                    if elapsed > float(rec.get("recovery_wait_timeout_s", 10.0)):
                        self.logger.error(
                            "[recovery] get-up never became available (state=%s); stopping.",
                            FallState.NAMES.get(state, state))
                        self.running = False
                        self._recovery_phase = "none"
                    return
            self.logger.warning("[recovery] calling GetUp (state=%s, available=%s)",
                                FallState.NAMES.get(state, state), recov_ok)
            try:
                # The Python binding names only kChangeMode/kMove/kRotateHead,
                # but B1LocoApiId accepts the raw id, so GetUp (2008) is
                # reachable from here after all.
                self.client.SendApiRequest(B1LocoApiId(API_ID_GET_UP), "")
            except Exception as exc:
                self.logger.error("[recovery] GetUp call failed: %s", exc)
                self.running = False
                self._recovery_phase = "none"
                return
            self._recovery_phase = "getup"
            self._recovery_t0 = time.monotonic()
            return

        if phase == "getup":
            timeout = float(rec.get("getup_timeout_s", 20.0))
            standing = False
            if self.fall_monitor and self.fall_monitor.available and fall_age < 5.0:
                standing = state == FallState.IS_READY and elapsed > 2.0
            else:
                # No fall topic: fall back to the IMU. Upright and quiet is the
                # best signal available.
                upright = (abs(self._latest_rpy[0]) < 0.3 and abs(self._latest_rpy[1]) < 0.3)
                standing = upright and elapsed > float(rec.get("getup_blind_wait_s", 8.0))
            if standing:
                self.logger.warning("[recovery] standing again after %.1f s; re-entering CUSTOM",
                                    elapsed)
                self._recovery_phase = "reenter"
                self._recovery_t0 = time.monotonic()
                return
            if elapsed > timeout:
                self.logger.error("[recovery] get-up did not complete in %.0f s; stopping.",
                                  timeout)
                self.running = False
                self._recovery_phase = "none"
            return

        if phase == "reenter":
            try:
                self._enter_custom_latched()
            except Exception as exc:
                self.logger.error("[recovery] CUSTOM re-entry failed: %s", exc)
                self.running = False
                self._recovery_phase = "none"
                return
            self.next_inference_time = self.timer.get_time()
            self.next_publish_time = self.timer.get_time()
            self.policy.reset() if hasattr(self.policy, "reset") else None
            self.running_policy = True
            self._recovery_phase = "none"
            self.logger.warning(
                "[recovery] complete (#%d, %.1f s total). Mission continues; the "
                "elapsed time for this run is no longer a clean measurement.",
                self._recovery_count, time.monotonic() - self._recovery_t0)
            return

    def start_rl_gait_conditionally(self):
        print(f"{self.remoteControlService.get_rl_gait_operation_hint()}")
        wait_t0 = time.monotonic()
        while True:
            if self.remoteControlService.start_rl_gait():
                break
            # Between CUSTOM entry and this prompt nothing publishes LowCmd
            # unless --hold-prepare is set: the publish thread only starts below.
            # The robot simply holds the single prepare frame it was given.
            if self.hold_prepare:
                self._send_cmd(self.low_cmd)
            time.sleep(0.1)
        self.logger.info(
            "[mode-timing] operator wait at RL-gait prompt: %.2f s "
            "(CUSTOM entered %.2f s ago)",
            time.monotonic() - wait_t0,
            time.monotonic() - getattr(self, "_custom_mode_entered_monotonic", wait_t0))

        if getattr(self, "_prepare_q", None) is not None:
            self._log_joint_deviation("at RL-gait start (vs prepare)", self._prepare_q)
        create_first_frame_rl_cmd(self.low_cmd, self.cfg)
        rl_q = np.array([self.low_cmd.motor_cmd[i].q for i in range(self.joint_cnt)],
                        dtype=np.float64)
        # This is the second posture change of the sequence: prepare pose/gains
        # -> RL pose/gains. Its size is the jump the policy has to start from.
        self._log_joint_deviation("at RL-gait start (vs rl pose)", rl_q)
        self._send_cmd(self.low_cmd)
        self.next_inference_time = self.timer.get_time()
        self.next_publish_time = self.timer.get_time()
        if self.goal_source_mode == "stdin":
            self.goal_source.start_stdin_reader(self.logger)
        self.publish_runner = threading.Thread(target=self._publish_cmd)
        self.publish_runner.daemon = True
        self.publish_runner.start()
        print(f"{self.remoteControlService.get_operation_hint()}")

    def run(self):
        # While recovering, the policy does not drive. The publish thread keeps
        # streaming the last dof_target so the joints are never left commandless.
        if self.in_recovery():
            self._step_recovery()
            time.sleep(0.01)
            return

        time_now = self.timer.get_time()
        if time_now < self.next_inference_time:
            time.sleep(0.001)
            return
        self.next_inference_time += self.policy.get_policy_interval()

        # The SDK's own fall state is the slower, authoritative channel; the IMU
        # watchdog in _low_state_handler is the fast one. Either can start
        # recovery.
        if self.fall_monitor is not None:
            fstate, _avail, fage = self.fall_monitor.snapshot()
            if fage < 5.0 and fstate in (FallState.HAS_FALLEN, FallState.IS_FALLING):
                self._request_recovery("fall_down=%s" % FallState.NAMES.get(fstate, fstate))
                return

        low_state_timeout = float(self.cfg.get("safety", {}).get("low_state_timeout_s", 0.2))
        low_state_age = time.monotonic() - self._last_low_state_monotonic
        if self._last_low_state_monotonic <= 0.0 or low_state_age > low_state_timeout:
            self.logger.error(
                "LowState stale (age=%.3fs, limit=%.3fs); stopping LowCmd.",
                low_state_age, low_state_timeout,
            )
            self.running = False
            return

        if self.goal_source_mode == "ros":
            goal, goal_status = self.goal_source.get_with_status()
        else:
            goal = self.goal_source.get()
            goal_status = {
                "goal_age_sec": 0.0,
                "goal_stale": False,
                "goal_messages_received": None,
                "goal_messages_rejected": None,
            }
        goal_rel_x, goal_rel_y, heading_error = goal
        self._last_goal = goal
        dof_target = self.policy.inference(
            time_now=time_now,
            dof_pos=self.dof_pos,
            dof_vel=self.dof_vel,
            base_ang_vel=self.base_ang_vel,
            projected_gravity=self.projected_gravity,
            goal_rel_x=goal_rel_x,
            goal_rel_y=goal_rel_y,
            heading_error=heading_error,
        )

        # Safety watchdog: never publish a non-finite target.
        if not np.all(np.isfinite(dof_target)):
            self.logger.error("Non-finite dof target from policy; stopping.")
            self.running = False
            return
        self.dof_target[:] = dof_target
        self._publish_policy_debug(low_state_age, goal_status)
        time.sleep(0.001)

    def _publish_policy_debug(self, low_state_age, goal_status):
        if self.goal_source_mode != "ros":
            return
        now = time.monotonic()
        period = float(self.cfg.get("deploy_goal", {}).get("debug_period_s", 0.10))
        if now - self._last_debug_monotonic < max(0.02, period):
            return
        self._last_debug_monotonic = now
        payload = {
            "schema": "locomotion_test.policy_debug.v1",
            "recovery": {
                "phase": self._recovery_phase,
                "count": self._recovery_count,
                "reason": self._recovery_reason or None,
            },
            "goal_source": "ros",
            "goal_topic": self.goal_source.topic,
            "goal_rel": {
                "x_m": self._last_goal[0],
                "y_m": self._last_goal[1],
                "heading_error_rad": self._last_goal[2],
            },
            **goal_status,
            "low_state_age_sec": low_state_age,
            "rpy_rad": [float(v) for v in self._latest_rpy],
            "action_min": float(np.min(self.policy.actions)),
            "action_max": float(np.max(self.policy.actions)),
            "running": self.running,
        }
        try:
            self.goal_source.publish_debug(payload)
        except Exception as exc:
            self.logger.warning("Failed to publish policy debug: %s", exc)

    def _publish_cmd(self):
        while self.running:
            time_now = self.timer.get_time()
            if time_now < self.next_publish_time:
                time.sleep(0.001)
                continue
            self.next_publish_time += self.cfg["common"]["dt"]

            self.filtered_dof_target = self.filtered_dof_target * 0.8 + self.dof_target * 0.2

            for i in range(self.joint_cnt):
                self.low_cmd.motor_cmd[i].q = self.filtered_dof_target[i]

            # Series-parallel conversion for the parallel-mechanism joints.
            for i in self.cfg["mech"]["parallel_mech_indexes"]:
                self.low_cmd.motor_cmd[i].q = self.dof_pos_latest[i]
                self.low_cmd.motor_cmd[i].tau = np.clip(
                    (self.filtered_dof_target[i] - self.dof_pos_latest[i]) * self.cfg["common"]["stiffness"][i],
                    -self.cfg["common"]["torque_limit"][i],
                    self.cfg["common"]["torque_limit"][i],
                )
                self.low_cmd.motor_cmd[i].kp = 0.0

            self._send_cmd(self.low_cmd)
            time.sleep(0.001)

    def __enter__(self) -> "Controller":
        return self

    def __exit__(self, *args) -> None:
        self.cleanup()


if __name__ == "__main__":
    def signal_handler(sig, frame):
        print("\nShutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Goal_Pose_E0.yaml", type=str,
                        help="Config file name under configs/.")
    parser.add_argument("--net", type=str, default="127.0.0.1",
                        help="Network interface for SDK communication.")
    parser.add_argument("--goal-source", choices=["ros", "fixed", "stdin"], default="ros",
                        help="ros = live /locomotion_test/goal_rel from Brain (MISSION mode); "
                             "fixed/stdin = manual constant goal.")
    parser.add_argument("--goal", type=str, default=None,
                        help='Initial robot-local goal "x,y,theta" (m,m,rad). Overrides config.')
    parser.add_argument("--goal-topic", type=str, default=None,
                        help="Override deploy_goal.topic for ROS mission mode.")
    parser.add_argument("--debug-topic", type=str, default=None,
                        help="Override deploy_goal.debug_topic for ROS mission mode.")
    parser.add_argument("--hold-prepare", action="store_true",
                        help="Keep republishing the prepare frame from CUSTOM entry "
                             "until RL gait starts. By default a single frame is sent "
                             "and then nothing publishes until the operator starts the "
                             "gait, which makes the transition timing depend on how "
                             "long the prompt is left waiting.")
    parser.add_argument("--prepare-settle-log-s", type=float, default=1.0,
                        help="Seconds to log measured-vs-commanded joint error right "
                             "after entering CUSTOM (0 disables).")
    args = parser.parse_args()
    cfg_file = os.path.join("configs", args.config)

    initial_goal = None
    if args.goal is not None:
        gx, gy, gh = (float(v) for v in args.goal.replace(",", " ").split())
        initial_goal = (gx, gy, gh)

    print(f"Starting E0 GoalPose controller, connecting to {args.net} ...")
    ChannelFactory.Instance().Init(0, args.net)

    with Controller(
        cfg_file,
        goal_source_mode=args.goal_source,
        initial_goal=initial_goal,
        goal_topic=args.goal_topic,
        debug_topic=args.debug_topic,
        hold_prepare=args.hold_prepare,
        prepare_settle_log_s=args.prepare_settle_log_s,
    ) as controller:
        time.sleep(2)  # wait for channels
        print("Initialization complete.")
        controller.start_custom_mode_conditionally()
        controller.start_rl_gait_conditionally()

        try:
            while controller.running:
                controller.run()
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Cleaning up...")
