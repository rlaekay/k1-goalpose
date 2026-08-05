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
import shutil
import signal
import subprocess
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


def _restore_terminal():
    """Put the tty back into a sane state.

    RemoteControlService starts an sshkeyboard listener that switches the
    terminal to raw mode. If the process exits without unwinding that, the
    shell keeps eating and reordering characters -- typed commands come out as
    "fpython3", and paths land in the wrong argument. Cheap to do, and it makes
    the difference between a usable terminal and one that has to be reset by
    hand.
    """
    if not sys.stdin.isatty():
        return
    stty = shutil.which("stty")
    if not stty:
        return
    try:
        subprocess.run([stty, "sane"], stdin=sys.stdin, timeout=2, check=False)
    except Exception:
        pass


def _spin_node_isolated(node):
    """Spin one node on its own executor, in its own thread.

    rclpy.spin() drives the global default executor. This process has three
    nodes (goal source, fall monitor, mode monitor), and spinning them all that
    way makes their spin threads collide with "generator already executing";
    the losers stop delivering messages. On the first hardware run that silently
    disabled the mode monitor, so the standing gate never actually ran.
    """
    from rclpy.executors import SingleThreadedExecutor
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    t = threading.Thread(target=executor.spin, daemon=True)
    t.start()
    return executor, t


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
        self._executor, self._spin = _spin_node_isolated(self._node)

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


class RobotModeId:
    """Mirrors booster::robot::RobotMode. kPrepare IS the standing mode:
    "the robot keeps standing on both feet and can switch to walking mode"."""
    UNKNOWN = -1
    DAMPING = 0
    PREPARE = 1
    WALKING = 2
    CUSTOM = 3
    SOCCER = 4

    NAMES = {-1: "UNKNOWN", 0: "DAMPING", 1: "PREPARE", 2: "WALKING",
             3: "CUSTOM", 4: "SOCCER"}


class ModeMonitor:
    """The robot's actual mode, from /robot_states.

    ChangeMode's return value cannot be trusted for this: measured rc=100 after
    a full 1.000 s timeout on transitions that had in fact taken effect. This
    topic reports what the robot really is, which is what the CUSTOM entry gate
    needs.
    """

    def __init__(self, topic="/robot_states", logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.available = False
        self.mode = RobotModeId.UNKNOWN
        self.body_control = None
        self.last_update = 0.0
        self._lock = threading.Lock()
        self._node = None
        try:
            import rclpy
            from rclpy.node import Node
            from booster_interface.msg import RobotStatesMsg
        except ImportError as exc:
            self.logger.warning("ModeMonitor disabled (%s); CUSTOM entry cannot "
                                "verify the robot is standing first.", exc)
            return
        if not rclpy.ok():
            rclpy.init(args=None)
        self._node = Node("e0_mode_monitor")
        self._node.create_subscription(RobotStatesMsg, topic, self._cb, 10)
        _spin_node_isolated(self._node)
        self.available = True

    def _cb(self, msg):
        with self._lock:
            self.mode = int(msg.current_mode)
            self.body_control = int(msg.current_body_control)
            self.last_update = time.monotonic()

    def snapshot(self):
        with self._lock:
            age = float("inf") if self.last_update <= 0 else time.monotonic() - self.last_update
            return self.mode, age

    def name(self):
        return RobotModeId.NAMES.get(self.snapshot()[0], "?")

    def close(self):
        if self._node is not None:
            try:
                self._node.destroy_node()
            except Exception:
                pass


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
            self._executor, self._spin = _spin_node_isolated(self._own_node)
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
                 hold_prepare=False, prepare_settle_log_s=1.0,
                 log_timing=None, abort_file=None,
                 parallel_torque=False) -> None:
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

        # 관측 지연 계측. policy_debug에도 low_state_age_sec가 실리지만
        # _publish_policy_debug의 첫 줄이 goal_source_mode != "ros"면 즉시 반환이라
        # fixed 모드에서는 아무것도 안 나온다 -- 지금까지의 실기 시험이 전부 fixed였다.
        # 이 CSV는 goal source와 무관하게 남는다. 걷지 않아도 된다: 서 있기만 해도
        # 루프는 50 Hz로 돌고 LowState는 계속 들어온다.
        # 중단 파일. 키보드 리스너와 시그널 전달을 모두 우회하는 정지 경로다.
        # 기본값을 켜 둔다 -- 정지 수단이 옵션이면 필요한 순간에 꺼져 있다.
        self._parallel_torque = bool(parallel_torque)
        self._pub_last = 0.0
        self._pub_dt = []
        self._abort_file = abort_file or "/tmp/e0_abort"
        # 지난 실행이 남긴 파일이 있으면 시작하자마자 중단된다. 시작 시 지운다.
        if self._abort_file and os.path.exists(self._abort_file):
            try:
                os.remove(self._abort_file)
                self.logger.info("[abort] 지난 실행의 %s 를 지웠다", self._abort_file)
            except OSError:
                pass

        self._timing_fp = None
        self._timing_t0 = 0.0
        self._timing_last_tick = 0.0
        if log_timing:
            self._timing_fp = open(log_timing, "w", buffering=1, encoding="utf-8")
            ls = int(self.cfg["policy"].get("leg_dof_start", 10))
            cols = (["t_s", "low_state_age_s", "tick_dt_s", "tilt_deg",
                     "roll", "pitch", "gx", "gy", "gz",
                     "walking", "gait_freq", "gait_process", "pub_hz",
                     "goal_x", "goal_y", "heading_err"]
                    + ["q%d" % i for i in range(12)]
                    + ["dq%d" % i for i in range(12)]
                    + ["tau%d" % i for i in range(12)]
                    + ["act%d" % i for i in range(12)])
            self._timing_fp.write(",".join(cols) + "\n")
            self._timing_t0 = time.monotonic()
            self.logger.info("[timing] logging to %s", log_timing)

        # Joint vector length for THIS robot, from the config rather than the
        # SDK's B1JointCnt. See _init_low_state_values / _verify_joint_layout.
        self.joint_cnt = int(self.cfg["common"].get(
            "joint_cnt", len(self.cfg["common"]["default_qpos"])))

        # Load and contract-check the actor before creating SDK/remote-control
        # services. A missing or wrong model must fail without touching robot I/O.
        self.policy = GoalPosePolicy(cfg=self.cfg)

        self.running_policy = True
        rec_cfg = self.cfg.get("safety", {}).get("recovery", {})

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
        self._last_pre_custom_log = -1e9
        self._latest_tilt = 0.0
        self._moving_gait_frequency = float(self.cfg["policy"]["gait_frequency"])
        self._walking = False

        # SDK channels FIRST, while this is still a single-threaded process.
        #
        # B1LowStateSubscriber.InitChannel() blocks in C++ while the SDK starts
        # delivering into a Python callback, which needs the GIL. With rclpy
        # executor threads already spinning, that deadlocked every thread in
        # futex_wait and the run never reached the operator prompt. Adding the
        # second monitor is what tipped it over; the ordering was luck before.
        self._init_timer()
        self._init_low_state_values()
        self._init_communication()

        # Only now bring up the ROS side.
        self.mode_monitor = ModeMonitor(logger=self.logger)
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
        # tau_est는 원래 안 받았다. 덜덜 떠는 것이 토크 포화인지 진동인지
        # 가르려면 이게 있어야 한다.
        self.dof_tau = np.zeros(n, dtype=np.float32)

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
        # How far gravity has moved from where it sits when the robot is upright.
        #
        # Not raw roll/pitch: rpy is ZYX Euler, so roll is degenerate near
        # pitch = +-90 deg. A robot lying face down at pitch 78.8 deg reported
        # roll 172.7 deg, which reads as "flipped onto its back" and is not --
        # cos(pitch) is 0.19 there, so the roll term swings freely. Reading the
        # gravity direction in the body frame has no such singularity, and it is
        # the same quantity the policy already consumes as obs[0:3], so the two
        # cannot drift apart.
        #
        # Upright, gravity in the body frame is [0, 0, -1]. The angle away from
        # that is the tilt: 0 upright, 90 deg horizontal.
        roll, pitch, yaw = (float(v) for v in low_state_msg.imu_state.rpy)
        g_body = rotate_vector_inverse_rpy(roll, pitch, yaw, np.array([0.0, 0.0, -1.0]))
        tilt = float(np.arccos(np.clip(-g_body[2], -1.0, 1.0)))
        self._latest_tilt = tilt
        safety = self.cfg.get("safety", {})
        rpy_limit = float(safety.get("fall_tilt_limit_rad",
                                     safety.get("roll_pitch_limit_rad", 1.0)))
        if tilt > rpy_limit:
            # This is the fast path: low_state runs at ~500 Hz, while the SDK's
            # own fall topic publishes at 1 Hz. Waiting for that would mean up
            # to a second of the policy still driving a robot that is going
            # over. Previously this killed the process; now it hands off to the
            # recovery sequence so the run can continue after a get-up.
            self._request_recovery(
                "tilt=%.0fdeg > %.0fdeg (roll=%.0f pitch=%.0f)"
                % (np.degrees(tilt), np.degrees(rpy_limit),
                   np.degrees(roll), np.degrees(pitch)))
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
                self.dof_tau[i] = motor.tau_est

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
            if getattr(self, "mode_monitor", None) is not None:
                self.mode_monitor.close()
            if hasattr(self.goal_source, "close"):
                self.goal_source.close()
            self.remoteControlService.close()
            if hasattr(self, "low_cmd_publisher"):
                self.low_cmd_publisher.CloseChannel()
            if hasattr(self, "low_state_subscriber"):
                self.low_state_subscriber.CloseChannel()
            if self._timing_fp is not None:
                try:
                    self._timing_fp.close()
                    self.logger.info("[timing] log closed")
                except Exception:
                    pass
                self._timing_fp = None
            _restore_terminal()

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

    def _abort_requested(self, where):
        """중단 파일이 생겼는가.

        ⛔ 왜 필요한가: `sshkeyboard.listen_keyboard`가 터미널을 잡고 있으면
        Ctrl-C가 파이썬 시그널 핸들러에 닿지 않는다. 2026-08-05 실기에서 `r`
        프롬프트 대기 중 로봇이 넘어졌는데 **Ctrl-C, SIGINT, SIGTERM이 전부
        안 먹혀** SIGKILL로 죽여야 했고, 그러면 cleanup()의
        ChangeMode(kDamping)에 도달하지 못한다 -- 즉 넘어진 로봇의 관절을
        소프트웨어로 놓아줄 방법이 없었다.

        파일 감시는 키보드 리스너도 시그널 전달도 거치지 않는다. 원격에서
        `touch <abort_file>` 하나로 정상 종료 경로(DAMPING 전환 포함)를 탄다.
        """
        if not self._abort_file:
            return False
        if os.path.exists(self._abort_file):
            self.logger.warning("[abort] %s 감지 -- %s 에서 중단하고 DAMPING으로 나간다",
                                self._abort_file, where)
            try:
                os.remove(self._abort_file)
            except OSError:
                pass
            return True
        return False

    def start_custom_mode_conditionally(self):
        self._require_fresh_low_state("before waiting for CUSTOM mode")
        print(f"{self.remoteControlService.get_custom_mode_operation_hint()}")
        while True:
            if self.remoteControlService.start_custom_mode():
                break
            if self._abort_requested("CUSTOM prompt"):
                raise KeyboardInterrupt("abort file")
            time.sleep(0.1)

        # Same path the recovery re-entry uses: verify the robot is standing,
        # latch the measured pose, then ramp to the RL pose. Startup and
        # recovery must not diverge -- a difference between them is a difference
        # nobody would notice until the robot behaves oddly after a fall.
        self._enter_custom_latched()

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
    def _fill_low_cmd(self, q_target, kp, kd):
        """Write one LowCmd frame: plain position control on every joint.

        No parallel-mechanism torque conversion. The verified E1 wrapper
        (deploy_choon.py) commands all 22 joints by position and uses
        mech.parallel_mech_indexes only to range-check the config.

        What made the robot ring on the first attempt was the gain, not the
        control mode: codex's prepare block had ankle kp=450 with kd=0.5, a
        damping ratio near zero. E1 uses kp=250 with kd=5. Replacing position
        control with a torque feedforward instead just traded ringing for an
        ankle that folded slowly under load.
        """
        for i in range(self.joint_cnt):
            self.low_cmd.motor_cmd[i].q = float(q_target[i])
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].tau = 0.0
            self.low_cmd.motor_cmd[i].kp = float(kp[i])
            self.low_cmd.motor_cmd[i].kd = float(kd[i])

    def _wait_for_low_state_after_custom(self, hold_target, kp, kd):
        """Hold the entry pose until LowState resumes after the mode change.

        ChangeMode can interrupt LowState. Ramping across that gap would compute
        targets from a stale measurement, so hold still until fresh state is
        back and refuse to continue if it never is. From the E1 wrapper.
        """
        timeout_s = 3.0
        deadline = time.monotonic() + timeout_s
        low_state_timeout = float(self.cfg.get("safety", {}).get("low_state_timeout_s", 0.2))
        while time.monotonic() < deadline:
            age = time.monotonic() - self._last_low_state_monotonic
            if self._last_low_state_monotonic > 0.0 and age <= low_state_timeout:
                return
            self._fill_low_cmd(hold_target, kp, kd)
            self._send_cmd(self.low_cmd)
            time.sleep(0.02)
        raise RuntimeError(
            "LowState did not resume within %.1fs after entering CUSTOM; "
            "stopped while holding the measured entry pose" % timeout_s)


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

        self._ensure_standing_before_custom()
        self._require_fresh_low_state("before latched CUSTOM entry")
        latched = np.array(self.dof_pos_latest[:self.joint_cnt], dtype=np.float32)

        prep_stiff = np.asarray(self.cfg["prepare"]["stiffness"], dtype=np.float32)
        prep_damp = np.asarray(self.cfg["prepare"]["damping"], dtype=np.float32)
        rl_stiff = np.asarray(self.cfg["common"]["stiffness"], dtype=np.float32)
        rl_damp = np.asarray(self.cfg["common"]["damping"], dtype=np.float32)
        rl_q = np.asarray(self.cfg["common"]["default_qpos"], dtype=np.float32)

        init_Cmd_T1(self.low_cmd, self.joint_cnt)
        self._fill_low_cmd(latched, prep_stiff, prep_damp)
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
        self._wait_for_low_state_after_custom(latched, prep_stiff, prep_damp)

        # Ramp pose and gains together. Keeping the command streaming through
        # the transition is what makes it smooth; a single frame followed by
        # silence leaves the robot on a stale target.
        # Blend measured pose -> policy default. Duration, shape and 20 ms
        # cadence follow the E1 wrapper. Smoothstep has zero slope at both ends,
        # so there is no velocity step on entry or exit; the linear ramp applied
        # its full rate instantly, which is what threw the ankles once the ramp
        # was shortened from 7 s to 0.6 s.
        step_dt = 0.02
        ramp_start = time.monotonic()
        next_send = ramp_start
        while True:
            self._require_fresh_low_state("during CUSTOM entry ramp")
            alpha = min(1.0, (time.monotonic() - ramp_start) / max(1e-6, ramp_s))
            blend = alpha * alpha * (3.0 - 2.0 * alpha)
            # Position only. The gains stay at prepare (legs 350/250) until the
            # gait starts, exactly as the E1 wrapper does. Ramping them down to
            # the policy values here left the robot holding a crouch on 100/50,
            # which are the gains the policy closes its own loop with -- with
            # nobody closing it, the ankles could not hold and it went over
            # forwards.
            self._fill_low_cmd((1 - blend) * latched + blend * rl_q,
                               prep_stiff, prep_damp)
            self._send_cmd(self.low_cmd)
            if alpha >= 1.0:
                break
            next_send += step_dt
            time.sleep(max(0.0, next_send - time.monotonic()))

        self.dof_target[:] = rl_q
        self.filtered_dof_target[:] = rl_q
        self._prepare_q = latched
        # Gains are still the prepare ones; start_rl_gait_conditionally swaps
        # them for the policy gains at the same moment inference begins.

        # Start streaming immediately. The operator prompt that follows can sit
        # for minutes -- it sat for 124 s on the first hardware start -- and
        # until now nothing published during that window, so the robot held one
        # stale frame the whole time. The publish loop is also the only place
        # that keeps joint targets streaming while the prompt waits.
        if self.publish_runner is None or not self.publish_runner.is_alive():
            self.publish_runner = threading.Thread(target=self._publish_cmd, daemon=True)
            self.publish_runner.start()
        moved = float(np.max(np.abs(
            np.asarray(self.dof_pos_latest[:self.joint_cnt]) - latched)))
        self.logger.info(
            "[mode-timing] latched CUSTOM entry done in %.2f s, joints moved %.4f rad",
            time.monotonic() - t0, moved)

    def _ensure_standing_before_custom(self):
        """Refuse to take the joints unless the robot is standing (kPrepare).

        kPrepare is the standing mode -- the SDK describes it as "the robot keeps
        standing on both feet and can switch to walking mode". The working order
        has always been damping -> stand -> walk, and taking over from a
        collapsed robot means the policy starts from a heap on the floor.

        Nothing enforced this before: ChangeMode(kCustom) was issued from
        whatever mode happened to be active, including DAMPING. ChangeMode's
        return code cannot be used to check either -- measured rc=100 after a
        full 1.000 s timeout on transitions that had actually taken effect -- so
        the current mode is read from /robot_states instead.
        """
        rec = self.cfg.get("safety", {}).get("recovery", {})
        if not bool(rec.get("require_standing_before_custom", True)):
            return
        if self.mode_monitor is None or not self.mode_monitor.available:
            self.logger.warning(
                "cannot read /robot_states; entering CUSTOM without confirming the "
                "robot is standing")
            return

        mode, age = self.mode_monitor.snapshot()
        if age > 5.0:
            raise RuntimeError("/robot_states is stale (%.1fs); refusing to enter "
                               "CUSTOM without knowing the current mode" % age)
        if mode == RobotModeId.PREPARE:
            return

        # A robot that is down cannot be stood up with ChangeMode(kPrepare) --
        # that is not the get-up routine, so the request would simply time out.
        # Get it up first. Measured on hardware: HAS_FALLEN -> IS_READY in 8.0 s.
        if self.fall_monitor is not None and self.fall_monitor.available:
            fstate, recov_ok, fage = self.fall_monitor.snapshot()
            if fage < 5.0 and fstate == FallState.HAS_FALLEN:
                if not recov_ok:
                    raise RuntimeError(
                        "robot has fallen and the SDK reports get-up unavailable; "
                        "stand it up manually before starting")
                self.logger.warning("[mode-gate] robot has fallen -- calling GetUp "
                                    "before CUSTOM entry")
                try:
                    self.client.SendApiRequest(B1LocoApiId(API_ID_GET_UP), "")
                except Exception as exc:
                    raise RuntimeError("GetUp call failed: %s" % exc)
                up_deadline = time.monotonic() + float(rec.get("getup_timeout_s", 20.0))
                while time.monotonic() < up_deadline:
                    time.sleep(0.3)
                    fstate, _ok, fage = self.fall_monitor.snapshot()
                    if fage < 5.0 and fstate == FallState.IS_READY:
                        self.logger.info("[mode-gate] get-up complete")
                        break
                else:
                    raise RuntimeError("robot did not stand up within the get-up "
                                       "timeout; refusing to enter CUSTOM")
                time.sleep(float(rec.get("prepare_settle_s", 1.0)))
                mode, age = self.mode_monitor.snapshot()
                if age < 5.0 and mode == RobotModeId.PREPARE:
                    return
        if mode == RobotModeId.CUSTOM:
            self.logger.info("[mode-gate] already in CUSTOM")
            return

        self.logger.warning("[mode-gate] robot is %s, not PREPARE -- requesting "
                            "PREPARE before taking the joints",
                            RobotModeId.NAMES.get(mode, mode))
        try:
            self.client.ChangeMode(RobotMode.kPrepare)
        except Exception as exc:
            raise RuntimeError("ChangeMode(kPrepare) failed: %s" % exc)

        timeout = float(rec.get("prepare_timeout_s", 15.0))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            time.sleep(0.2)
            mode, age = self.mode_monitor.snapshot()
            if age < 5.0 and mode == RobotModeId.PREPARE:
                self.logger.info("[mode-gate] PREPARE reached; robot is standing")
                time.sleep(float(rec.get("prepare_settle_s", 1.0)))
                return
        raise RuntimeError(
            "robot did not reach PREPARE within %.0fs (still %s). Refusing to "
            "enter CUSTOM from a non-standing state." % (
                timeout, RobotModeId.NAMES.get(self.mode_monitor.snapshot()[0], "?")))

    def log_hold_diagnostic(self, seconds):
        """Record what the ankles do while holding a static pose in CUSTOM.

        Two failure modes look similar from across the room but are opposite
        problems: a joint with no position servo drifts steadily away from its
        target, while an under-damped one oscillates around it. This logs target
        vs measured on the parallel-mechanism joints so the difference is a
        number rather than a guess.
        """
        parallel = list(self.cfg.get("mech", {}).get("parallel_mech_indexes", []))
        t0 = time.monotonic()
        rows = []
        while time.monotonic() - t0 < seconds:
            time.sleep(0.05)
            t = time.monotonic() - t0
            rows.append((t, [float(self.filtered_dof_target[i]) for i in parallel],
                         [float(self.dof_pos_latest[i]) for i in parallel],
                         float(np.degrees(self._latest_tilt))))
        self.logger.info("[hold-diag] %d samples over %.1fs, joints %s",
                         len(rows), seconds, parallel)
        for t, tgt, meas, tilt in rows[::max(1, len(rows)//20)]:
            errs = " ".join("%+.4f" % (m - g) for g, m in zip(tgt, meas))
            self.logger.info("[hold-diag] t=%5.2fs tilt=%5.1fdeg  err(meas-target)= %s",
                             t, tilt, errs)
        if rows:
            first = np.array(rows[0][2]) - np.array(rows[0][1])
            last = np.array(rows[-1][2]) - np.array(rows[-1][1])
            drift = last - first
            allerr = np.array([np.array(m) - np.array(g) for _, g, m, _ in rows])
            spread = allerr.max(axis=0) - allerr.min(axis=0)
            self.logger.info("[hold-diag] net drift over window: %s",
                             " ".join("%+.4f" % v for v in drift))
            self.logger.info("[hold-diag] peak-to-peak:          %s",
                             " ".join("%.4f" % v for v in spread))
            self.logger.info("[hold-diag] drift >> spread means no position servo; "
                             "spread >> drift means under-damped")

    # --------------------------------------------------------------- recovery --
    def _request_recovery(self, reason):
        if self._recovery_phase != "none":
            return
        # Do not arm recovery before we own the joints. low_state arrives as soon
        # as the subscriber attaches, so a robot that is already lying down would
        # otherwise trip the IMU watchdog during construction -- queuing a
        # damping/get-up sequence that fires the instant the operator starts the
        # run. Until CUSTOM is entered the SDK is in charge and it is not our
        # place to intervene.
        if not self._custom_mode_started:
            # Throttled: this is evaluated on every low_state message (~500 Hz),
            # so logging per call floods the console and starves the process.
            now = time.monotonic()
            if now - self._last_pre_custom_log > 5.0:
                self._last_pre_custom_log = now
                self.logger.info("ignoring fall trigger before CUSTOM entry (%s); "
                                 "the robot is not under our control yet", reason)
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
            if self._abort_requested("RL-gait prompt"):
                raise KeyboardInterrupt("abort file")
            # The publish thread has been streaming since CUSTOM entry, so the
            # robot is held properly for however long this prompt waits.
            time.sleep(0.1)
        self.logger.info(
            "[mode-timing] operator wait at RL-gait prompt: %.2f s "
            "(CUSTOM entered %.2f s ago)",
            time.monotonic() - wait_t0,
            time.monotonic() - getattr(self, "_custom_mode_entered_monotonic", wait_t0))

        # Swap prepare gains for the policy gains, holding the current position.
        # The pose is already the policy default from the entry ramp; only the
        # gains change, and they change exactly when the policy starts closing
        # the loop that those gains assume.
        self._fill_low_cmd(self.filtered_dof_target,
                           np.asarray(self.cfg["common"]["stiffness"], dtype=np.float32),
                           np.asarray(self.cfg["common"]["damping"], dtype=np.float32))
        self._send_cmd(self.low_cmd)
        rl_q = np.asarray(self.cfg["common"]["default_qpos"], dtype=np.float64)
        self._log_joint_deviation("at RL-gait start (vs rl pose)", rl_q)
        self.next_inference_time = self.timer.get_time()
        self.next_publish_time = self.timer.get_time()
        if self.goal_source_mode == "stdin":
            self.goal_source.start_stdin_reader(self.logger)
        # The publisher already started at CUSTOM entry; do not start a second one.
        print(f"{self.remoteControlService.get_operation_hint()}")

    def run(self):
        # While recovering, the policy does not drive. The publish thread keeps
        # streaming the last dof_target so the joints are never left commandless.
        if self.in_recovery():
            self._step_recovery()
            time.sleep(0.01)
            return

        if self._abort_requested("run loop"):
            self.running = False
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
            # IS_GETTING_UP is deliberately not a trigger, and IS_FALLING is not
            # relied on: a measured get-up went HAS_FALLEN -> IS_READY without
            # ever reporting IS_GETTING_UP, so the intermediate states of this
            # 1 Hz topic cannot be assumed to appear at all. The IMU watchdog is
            # what actually catches a fall in progress.
            if fage < 5.0 and fstate == FallState.HAS_FALLEN:
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
        self._update_arrival_gait(goal_rel_x, goal_rel_y, heading_error)
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
        self._log_timing_row(low_state_age, goal_rel_x, goal_rel_y, heading_error)
        time.sleep(0.001)

    def _log_timing_row(self, low_state_age, gx, gy, herr):
        """관측 지연 한 줄. MuJoCo 스윕(§8-40)에서 이 값만 여유가 0이었다 --
        10 ms 무사, 20 ms에서 흔들리고, 30-35 ms면 2-3 걸음마다 넘어진다.
        실기가 그 구간에 있는지가 질문이고, 걷지 않아도 답이 나온다."""
        if self._timing_fp is None:
            return
        now = time.monotonic()
        dt_tick = (now - self._timing_last_tick) if self._timing_last_tick else 0.0
        self._timing_last_tick = now
        ls = int(self.cfg["policy"].get("leg_dof_start", 10))
        try:
            head = [now - self._timing_t0, low_state_age, dt_tick,
                    float(np.degrees(self._latest_tilt)),
                    float(self._latest_rpy[0]), float(self._latest_rpy[1]),
                    float(self.base_ang_vel[0]), float(self.base_ang_vel[1]),
                    float(self.base_ang_vel[2]),
                    float(self._walking), float(self.policy.gait_frequency),
                    float(self.policy.gait_process),
                    (1.0 / (sum(self._pub_dt) / len(self._pub_dt)))
                    if self._pub_dt else 0.0,
                    gx, gy, herr]
            body = (list(self.dof_pos[ls:ls + 12])
                    + list(self.dof_vel[ls:ls + 12])
                    + list(self.dof_tau[ls:ls + 12])
                    + list(self.policy.actions[:12]))
            self._timing_fp.write(
                ",".join("%.5g" % float(v) for v in head + body) + "\n")
        except Exception:
            # 계측이 로봇을 멈추게 하면 안 된다.
            self._timing_fp = None

    def _update_arrival_gait(self, goal_rel_x, goal_rel_y, heading_error):
        """Zero the gait clock once the goal is reached, with hysteresis.

        Training assigns gait_frequency = 0 to stand-category goals
        (goal_pose.py: "stand-category envs get a zero gait clock ... so
        standing still is optimal"), and the goal mixture is named "No More
        Marching" for that reason. Feeding a constant 2 Hz at (0,0,0) asks for
        behaviour the stand portion of the training set never contained, and the
        robot marches in place.

        Setting policy.gait_frequency to 0 also freezes gait_process, so both the
        command channel and the clock inputs match the stand case. That holds
        only because the phase is integrated, as training does; the closed form
        this wrapper used first would have snapped it to 0 here instead. See
        GoalPosePolicy.advance_gait_clock.

        feet_swing is itself gated on gait_frequency > 0 (goal_pose.py:1048), so
        a non-zero clock at the goal also sits on a stepping incentive, and
        goal_reached pays only while the base is below stop_speed_threshold.

        Separate stop/start thresholds keep it from chattering at the boundary.
        """
        dg = self.cfg.get("deploy_goal", {})
        stop_pos = float(dg.get("stop_radius_m", 0.06))
        start_pos = float(dg.get("start_radius_m", 0.10))
        stop_heading = float(dg.get("stop_heading_rad", 0.06))
        start_heading = float(dg.get("start_heading_rad", 0.10))
        distance = float(np.hypot(goal_rel_x, goal_rel_y))
        heading = abs(float(heading_error))

        if self._walking:
            self._walking = distance > stop_pos or heading > stop_heading
        else:
            self._walking = distance > start_pos or heading > start_heading

        self.policy.gait_frequency = (
            self._moving_gait_frequency if self._walking else 0.0)

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
            "tilt_deg": float(np.degrees(self._latest_tilt)),
            "walking": self._walking,
            "gait_frequency": float(self.policy.gait_frequency),
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

            # ⛔ 이 필터의 계수(0.8/0.2)는 **500 Hz 발행을 가정**하고 고른 값이다.
            # 이 루프는 파이썬이고 time.sleep(0.001)이라 실제 발행률이 그보다
            # 훨씬 낮을 수 있다. 낮으면 차단주파수가 같이 내려가 보행 자체를 깎는다:
            #   500 Hz -> fc 17.8 Hz, 2 Hz 보행 감쇠 0.99
            #   100 Hz -> fc  3.6 Hz,               0.87
            #    50 Hz -> fc  1.8 Hz,               0.66
            # 2026-08-05 실기 로그의 추종률(명령 대비 실제 도달)이 median 0.61인데
            # MuJoCo는 0.93이다. 발행률이 50 Hz 근처면 그 차이가 필터만으로 설명된다.
            # 그래서 **실제 발행률을 잰다.** 추론 루프도 설계 20 ms인데 실측
            # 25.3 ms였으므로, 이 루프가 설계대로 돈다고 가정하면 안 된다.
            _pub_now = time.monotonic()
            if self._pub_last:
                self._pub_dt.append(_pub_now - self._pub_last)
                if len(self._pub_dt) > 2000:
                    self._pub_dt.pop(0)
            self._pub_last = _pub_now
            self.filtered_dof_target = self.filtered_dof_target * 0.8 + self.dof_target * 0.2

            for i in range(self.joint_cnt):
                self.low_cmd.motor_cmd[i].q = self.filtered_dof_target[i]

            # 기본은 위치 제어다. 검증된 E1 wrapper가 22관절 전부를 위치로 명령했고,
            # codex의 토크 피드포워드 변형은 이 로봇에서 **하중에 천천히 접히는
            # 발목**을 만들었다. 그런데 2026-08-05 실기 대조에서 나온 증상이
            # 정확히 그것이다 -- 발목 토크가 sim의 0.6-0.7배인데 궤적은 1.3배다.
            # 즉 지금의 위치 제어도 같은 문제를 겪고 있을 수 있다는 뜻이라,
            # 되돌려 비교할 수 있게 스위치로 남긴다. 기본은 꺼짐.
            #
            # 켜면 base_walk와 같은 처리를 한다(deploy_base_walk.py:181):
            # P 항을 드라이버에서 소프트웨어로 옮기고(kp=0), 관절 공간에서 계산한
            # 토크를 피드포워드로 보낸다. 이것은 액추에이터 공간 변환이 아니다 --
            # 관절->액추에이터 매핑은 SDK가 처리한다(아니면 관절각 명령이 안 먹는다).
            if self._parallel_torque:
                mech = self.cfg.get("mech", {}).get("parallel_mech_indexes", [])
                stiff = self.cfg["common"]["stiffness"]
                tlim = self.cfg["common"]["torque_limit"]
                for i in mech:
                    self.low_cmd.motor_cmd[i].q = self.dof_pos_latest[i]
                    self.low_cmd.motor_cmd[i].tau = float(np.clip(
                        (self.filtered_dof_target[i] - self.dof_pos_latest[i]) * stiff[i],
                        -tlim[i], tlim[i]))
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
    parser.add_argument("--hold-diag", type=float, default=0.0,
                        help="After CUSTOM entry, hold and log ankle target-vs-measured "
                             "for this many seconds before the gait prompt. Tells drift "
                             "(no position servo) apart from oscillation (under-damped).")
    parser.add_argument("--parallel-torque", action="store_true",
                        help="발목 4관절(mech.parallel_mech_indexes)을 base_walk와 같은 "
                             "방식으로 보낸다: kp=0 + 관절공간 토크 피드포워드. "
                             "주석에 따르면 예전에 하중에 접히는 발목을 만들었는데, "
                             "2026-08-05 대조에서 나온 증상이 정확히 그것이라 "
                             "되돌려 비교할 수 있게 스위치로 둔다. 기본 꺼짐.")
    parser.add_argument("--abort-file", type=str, default="/tmp/e0_abort",
                        help="이 파일이 생기면 즉시 중단하고 DAMPING으로 나간다. "
                             "sshkeyboard가 터미널을 잡고 있으면 Ctrl-C/SIGINT/SIGTERM이 "
                             "모두 안 먹는다(2026-08-05 실기에서 SIGKILL로 죽여야 했고, "
                             "그러면 DAMPING 전환에 도달하지 못한다). "
                             "원격에서 touch 한 번으로 정상 종료 경로를 탄다.")
    parser.add_argument("--log-timing", type=str, default=None,
                        help="관측 지연을 이 CSV에 50 Hz로 남긴다 (goal source 무관). "
                             "MuJoCo 스윕에서 관측 지연만 여유가 0이었다 -- 10 ms 무사, "
                             "30-35 ms면 2-3 걸음마다 낙상. 걷지 않아도 된다: 서 있기만 "
                             "해도 루프는 50 Hz로 돌고 LowState는 계속 들어온다.")
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
        log_timing=args.log_timing,
        abort_file=args.abort_file,
        parallel_torque=args.parallel_torque,
    ) as controller:
        time.sleep(2)  # wait for channels
        print("Initialization complete.")
        controller.start_custom_mode_conditionally()
        if args.hold_diag > 0:
            controller.log_hold_diagnostic(args.hold_diag)
        controller.start_rl_gait_conditionally()

        try:
            while controller.running:
                controller.run()
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Cleaning up...")
