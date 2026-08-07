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
import math
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
        except ImportError as exc:
            self.logger.warning("ModeMonitor disabled (%s); CUSTOM entry cannot "
                                "verify the robot is standing first.", exc)
            return
        # ⛔ 이 빌드의 `booster_interface/msg/__init__.py` 는 RobotStatesMsg 를
        # export 하지 않는다. 그런데 `ros2 topic info /robot_states` 는 타입이
        # 정확히 `booster_interface/msg/RobotStatesMsg` 라고 답한다 -- 메시지는
        # 생성돼 있고 패키지의 __init__ 만 그것을 빼먹은 것이다.
        #
        # 2026-08-07 실기에서 이 import 실패 하나가 서 있는지 확인하는 게이트를
        # 통째로 무력화했고(경고만 찍고 통과한다), 그 상태로 get-up 중간에 CUSTOM
        # 에 재진입해 펌웨어 관절 보호가 걸렸다. 그래서 공개 경로가 실패하면
        # 생성된 모듈을 직접 집는다.
        RobotStatesMsg = None
        for mod, name in (("booster_interface.msg", "RobotStatesMsg"),
                          ("booster_interface.msg._robot_states_msg", "RobotStatesMsg")):
            try:
                RobotStatesMsg = getattr(__import__(mod, fromlist=[name]), name)
                break
            except (ImportError, AttributeError):
                continue
        if RobotStatesMsg is None:
            self.logger.warning(
                "ModeMonitor disabled: booster_interface 에서 RobotStatesMsg 를 "
                "찾지 못했다 (__init__ 도 _robot_states_msg 도 실패). CUSTOM 진입이 "
                "로봇이 서 있는지 확인하지 못한다 -- 복구 재진입은 관절 정지 조건에 "
                "의존한다.")
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
                 parallel_torque=False, rate_fixed_filter=False,
                 filter_tau_s=0.010) -> None:
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
        # get-up 이 정말 끝났는지의 신호. IS_READY 는 이르다 -- 다리를 오므려
        # 서기까지가 남아 있다. 그 동작이 끝나면 다리 관절 속도가 0 으로 간다.
        self._legs_quiet_since = 0.0
        self._last_getup_wait_log = 0.0
        # GetUp 을 실제로 불렀는가. 이미 서 있어서 건너뛴 경우에는 최소 대기를
        # 걸 이유가 없다 -- 기다릴 동작 자체가 없다.
        self._getup_called = False

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
        self._rate_fixed_filter = bool(rate_fixed_filter)
        # 설계 의도: 500 Hz에서 계수 0.2 = 시정수 10 ms.
        self._filter_tau_s = float(filter_tau_s)
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
        # ⛔ `low_cmd` 를 만지는 **모든** 경로가 이 락 뒤에 있어야 한다.
        #
        # 2026-08-07 감사: 이 락은 여기서 만들어지고 **어디에서도 쓰이지 않았다.**
        # 그 사이 진입 램프(50 Hz)와 발행 스레드(LowState 1건당 1회, ~500 Hz)가
        # 같은 `low_cmd.motor_cmd[i].q` 를 동시에 쓰고 둘 다 `_send_cmd` 를 불렀다.
        # 최초 진입에서는 발행 스레드가 아직 없어서 드러나지 않았고,
        # **낙상 복구 재진입에서만** 터진다 -- 그 구간의 실측 이동량이
        # 1.5934 / 1.6773 rad 이므로 램프 자세와 시작 자세가 번갈아 나가면
        # 진폭 ~96도의 사각파가 모터로 간다. 균형을 닫는 주체가 없는 바로 그
        # 구간이고, 실제로 펌웨어 관절 보호(빨간불)가 걸린 구간이다
        # (HANDOFF_DEPLOY_ENTRY_20260807.md §3).
        self.publish_lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._cleaned_up = False
        self._custom_mode_started = False
        # 지금 `low_cmd` 에 실려 있는 게인이 **정책 게인**인가(common), 아니면
        # **prepare 게인**인가. `--parallel-torque` 의 토크 변환이
        # `common.stiffness`(발목 50)를 쓰므로, prepare 게인(발목 250)으로
        # 붙잡는 진입·복구 구간에 그것을 적용하면 게인 집합이 섞인다.
        self._policy_gains_active = False
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

    # InitChannel 데드락을 잘라내는 시간(초). 정상 초기화는 1초 안에 끝나므로
    # 8초면 오검출이 없다.
    INIT_WATCHDOG_S = 8.0
    # 워치독이 발동하면 이 파일에 스택이 남는다. faulthandler는 종료코드를 1로
    # 고정하므로(커스텀 불가), run_e0.sh는 **이 파일이 비어 있지 않은지**로
    # 데드락을 판별하고 재시도한다.
    INIT_DEADLOCK_LOG = "/tmp/e0_init_deadlock.log"

    def _init_communication(self) -> None:
        """SDK 채널을 연다. InitChannel의 GIL 데드락에 하드 타임아웃을 건다.

        무엇이 일어나는가
        ------------------
        `B1LowStateSubscriber(handler)`는 **파이썬 콜백**을 등록하고, 그 다음
        `InitChannel()`이 C++에서 블록한다. 그 창(window) 안에 LowState가 한 개라도
        배달되면 DDS 스레드가 콜백을 부르려고 GIL을 요구하는데, GIL은 InitChannel을
        부른 스레드가 쥐고 있다. 순환 대기 -- 전 스레드가 futex_wait로 들어간다.

        왜 스레드로 못 푸는가
        ---------------------
        **GIL은 프로세스 전역이다.** InitChannel을 별도 스레드에서 돌려도 그 스레드가
        GIL을 쥔 채 블록하므로 메인도 DDS도 못 돈다. 콜백을 나중에 등록하는 방법도
        없다 -- SDK API가 `__init__(handler)` 하나뿐이다.

        그래서 무엇을 하는가
        --------------------
        경합이므로 **대개는 통과한다**(창이 좁다). 통과 못 한 실행만 잘라내고 다시
        띄우면 된다. 자르는 수단으로 `faulthandler`를 쓴다 -- 이것의 워치독은
        **C 스레드**라 GIL 없이 발동해서 `_exit()`를 부른다. `signal.alarm`은 안 된다:
        파이썬 시그널 핸들러는 바이트코드 사이에서만 돌고, 그 인터프리터가 멈춰 있다.
        `threading.Timer`도 안 된다 -- 깨어날 때 GIL이 필요하다.

        걸린 실행은 INIT_DEADLOCK_LOG 에 스택을 남기고 죽는다. faulthandler는
        종료코드를 1로 고정하므로 run_e0.sh는 그 파일이 비어 있지 않은지로 판별한다.
        """
        import faulthandler
        try:
            self.low_cmd = LowCmd()
            self.low_state_subscriber = B1LowStateSubscriber(self._low_state_handler)
            self.low_cmd_publisher = B1LowCmdPublisher()
            self.client = B1LocoClient()

            # 데드락은 여기서만 난다. 창을 정확히 이 호출로 좁힌다.
            # 파일은 열어둔 채로 둔다 -- _exit()는 버퍼를 안 비우므로 워치독이
            # 직접 쓸 수 있게 살아 있어야 한다. 통과하면 아래에서 지운다.
            wd = open(self.INIT_DEADLOCK_LOG, "w")
            faulthandler.dump_traceback_later(self.INIT_WATCHDOG_S, file=wd, exit=True)
            try:
                self.low_state_subscriber.InitChannel()
            finally:
                faulthandler.cancel_dump_traceback_later()
            wd.close()
            os.remove(self.INIT_DEADLOCK_LOG)   # 통과 -- 흔적을 남기지 않는다

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

    def _publish_thread_alive(self):
        """발행 스레드가 이미 `low_cmd` 를 소유하고 있는가.

        최초 진입에서는 False 다 -- 스레드는 `_enter_custom_latched` **끝**에서
        시작된다. 복구 재진입에서는 True 다 -- 스레드는 `cleanup()` 에서만
        join 되므로 낙상·DAMPING·get-up 내내 살아 있다.
        """
        return self.publish_runner is not None and self.publish_runner.is_alive()

    def _emit_low_cmd(self, q_target, kp, kd):
        """한 프레임을 채워서 보낸다 -- 발행 스레드와 **원자적으로** 배타.

        채우기와 보내기가 갈라지면 발행 스레드가 그 사이에 끼어들어 절반만 바뀐
        프레임을 내보낸다. 락을 프레임 단위로 잡아 그것을 막는다.
        ⚠️ 이 안에서 절대 sleep 하지 마라 -- 발행 스레드가 그만큼 굶는다.
        """
        with self.publish_lock:
            self._fill_low_cmd(q_target, kp, kd)
            self._send_cmd(self.low_cmd)

    def _wait_for_low_state_after_custom(self, hold_target, kp, kd):
        """Hold the entry pose until LowState resumes after the mode change.

        ChangeMode can interrupt LowState. Ramping across that gap would compute
        targets from a stale measurement, so hold still until fresh state is
        back and refuse to continue if it never is. From the E1 wrapper.

        ⛔ 여기만은 `dof_target` 경로로 못 바꾼다. 발행 스레드의 데드라인이
        `Timer` 카운터인데 그 카운터는 **LowState 콜백에서만** 증가한다
        (`utils/timer.py`, `_low_state_handler`). 즉 LowState 가 끊긴 동안
        발행 스레드는 아무것도 안 보낸다 -- 이 구간을 메우는 것이 이 루프의
        존재 이유다. 그래서 직접 보내되 **락 안에서** 보낸다.
        """
        timeout_s = 3.0
        deadline = time.monotonic() + timeout_s
        low_state_timeout = float(self.cfg.get("safety", {}).get("low_state_timeout_s", 0.2))
        while time.monotonic() < deadline:
            age = time.monotonic() - self._last_low_state_monotonic
            if self._last_low_state_monotonic > 0.0 and age <= low_state_timeout:
                return
            self._emit_low_cmd(hold_target, kp, kd)
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
        or straight out of a get-up. The move to the RL default pose then
        happens as a controlled ramp instead of a snap. Aligning the static
        `prepare` block to the RL block would only help when the robot happens
        to be in exactly that pose; this always helps.

        게인은 램프 내내 prepare 값이다 -- 정책 게인으로의 교체는
        `start_rl_gait_conditionally` 가 추론을 시작하는 그 순간에 한다.
        (아래 램프 루프의 주석 참조: 여기서 게인을 내렸더니 아무도 루프를 닫지
        않는 상태로 100/50 에 웅크리고 있다가 앞으로 넘어갔다.)

        램프 시간은 고정이 아니라 **이동량에 비례**한다. 이유는 아래 참조.
        """
        rec = self.cfg.get("safety", {}).get("recovery", {})
        requested_ramp_s = ramp_s

        self._ensure_standing_before_custom()
        self._require_fresh_low_state("before latched CUSTOM entry")
        latched = np.array(self.dof_pos_latest[:self.joint_cnt], dtype=np.float32)

        prep_stiff = np.asarray(self.cfg["prepare"]["stiffness"], dtype=np.float32)
        prep_damp = np.asarray(self.cfg["prepare"]["damping"], dtype=np.float32)
        # ⛔ 진입 램프의 목표는 **prepare 자세**이지 RL 자세가 아니다.
        #
        # 여기서 `r` 을 누를 때까지는 균형을 닫는 주체가 없다 -- 정책이 균형
        # 제어기인데 아직 안 돈다. 그래서 이 구간이 붙잡는 자세는 개루프로 설 수
        # 있어야 하고, 그 조건이 hip+knee+ankle = 0 이다.
        # prepare 는 -0.1/0.2/-0.1 로 합이 정확히 0, RL 은 -0.2/0.4/-0.25 로 -0.05.
        #
        # 2026-08-07 실기에서 RL 자세를 붙잡았더니 관절 추종은 0.6도인데 몸통이
        # 4.9-13.8도 기울어 15초 내내 흔들렸다. `r` 을 누르면 멀쩡해지는 이유가
        # 이것이다 -- 그때 비로소 균형 루프가 닫힌다.
        # RL 자세로는 `start_rl_gait_conditionally` 가 추론 직전에 옮긴다.
        hold_q = np.asarray(self.cfg["prepare"]["default_qpos"], dtype=np.float32)

        # ⛔ 램프 시간을 **이동량에 비례**시킨다. 고정 시간은 거리를 무시한다:
        # 2026-08-07 실기에서 최초 진입은 0.4894 rad, get-up 뒤 재진입은 1.5934 /
        # 1.6773 rad 이었는데 셋 다 같은 2 s 였다. 마지막 둘은 0.8 rad/s -- 균형을
        # 닫는 주체가 없는 구간에서 96 도를 2 초에 끈 것이고, 거기서 펌웨어 관절
        # 보호(빨간불)가 걸렸다.
        #
        # `custom_entry_ramp_s` 는 이제 **하한**이다(짧은 이동도 그보다 빨리 가지
        # 않는다). 상한은 별도로 둔다 -- 비례식만 두면 아주 큰 이동에서 램프가
        # 무한정 길어져 그동안 로봇이 무방비로 서 있게 된다.
        legs = slice(self.policy.leg_start,
                     self.policy.leg_start + self.policy.num_act)
        travel = float(np.max(np.abs(hold_q[legs] - latched[legs])))
        if requested_ramp_s is None:
            floor_s = float(rec.get("custom_entry_ramp_s", 0.6))
            max_rate = float(rec.get("custom_entry_max_rate_rps", 0.3))
            cap_s = float(rec.get("custom_entry_ramp_max_s", 8.0))
            ramp_s = min(cap_s, max(floor_s, travel / max(max_rate, 1e-6)))
        else:
            ramp_s = float(requested_ramp_s)
        self.logger.info(
            "[mode-timing] entry ramp: 다리 최대 이동 %.4f rad -> %.2f s (%.2f rad/s)",
            travel, ramp_s, travel / max(ramp_s, 1e-6))

        # ⛔ 최초 진입과 복구 재진입은 **동시성 조건이 다르다.**
        #
        #   최초 진입  : 발행 스레드가 아직 없다(이 함수 끝에서 시작한다).
        #                이 함수가 유일한 기록자다.
        #   복구 재진입: 발행 스레드가 **이미 살아 있다**(`cleanup()` 에서만 join).
        #                아래 램프와 500 Hz 로 같은 `low_cmd` 를 두고 경합한다.
        #
        # 재진입에서 그 경합이 만드는 것: 램프는 blend 된 자세를 쓰고 발행
        # 스레드는 `filtered_dof_target`(= 램프가 끝날 때까지 `latched` 에 고정)
        # 을 쓴다. 두 프레임이 번갈아 나가므로 로봇은 **부드러운 램프가 아니라
        # 램프 자세와 시작 자세를 오가는 사각파**를 받는다. 실측 재진입 이동량이
        # 1.5934 / 1.6773 rad 이므로 진폭이 ~96도다.
        streaming = self._publish_thread_alive()

        # 이 구간의 게인은 **prepare** 다. `--parallel-torque` 는 `common.stiffness`
        # 를 쓰므로 여기서 적용되면 게인 집합이 섞인다(발목 250 대 50).
        self._policy_gains_active = False

        with self.publish_lock:
            # ⛔ `init_Cmd_T1` 은 `low_cmd.motor_cmd` **벡터를 통째로 교체**한다.
            # 발행 스레드가 그 안으로 인덱싱하는 중이면 프레임이 찢어진다.
            # 재진입에서는 필요하지도 않다 -- `cmd_type` 과 `weight` 는 최초
            # 진입에서 설정된 뒤 아무도 안 바꾸고, 나머지 필드는 바로 아래
            # `_fill_low_cmd` 가 전부 덮는다.
            if not streaming:
                init_Cmd_T1(self.low_cmd, self.joint_cnt)
            # 발행 스레드가 읽는 두 배열을 먼저 맞춘다. 이걸 프레임보다 먼저
            # 해야 스레드가 끼어들어도 같은 값(latched)을 내보낸다.
            self.dof_target[:] = latched
            self.filtered_dof_target[:] = latched
            self._fill_low_cmd(latched, prep_stiff, prep_damp)
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
            q_cmd = (1 - blend) * latched + blend * hold_q
            if streaming:
                # ⛔ 기록자를 **하나로** 유지한다. 발행 스레드가 이미 500 Hz 로
                # `low_cmd` 를 쓰고 있으므로, 여기서는 그것이 읽는 `dof_target`
                # 만 움직이고 프레임은 그쪽이 낸다.
                # `start_rl_gait_conditionally` 의 램프가 쓰는 방식과 같다
                # (그 함수 주석: "램프는 dof_target 만 움직인다 ... 경합이 없다").
                #
                # 저역통과를 한 번 더 지나지만 무해하다: 500 Hz 에서 시정수가
                # 8.96 ms 이고 램프는 최소 0.6 s -- 지연이 램프의 1.5 % 미만이다.
                # 그리고 위에서 `filtered_dof_target` 을 `latched` 로 맞춰 뒀으므로
                # 시작 계단이 없다.
                # 게인은 진입 프레임에서 이미 prepare 로 실렸고 발행 스레드는
                # `q` 만 쓰므로 램프 내내 유지된다.
                #
                # LowState 가 끊기면 발행 스레드도 같이 멈추지만(카운터 Timer),
                # 그 경우 루프 첫 줄의 `_require_fresh_low_state` 가 먼저 raise 한다.
                self.dof_target[:] = q_cmd
            else:
                self._emit_low_cmd(q_cmd, prep_stiff, prep_damp)
            if alpha >= 1.0:
                break
            next_send += step_dt
            time.sleep(max(0.0, next_send - time.monotonic()))

        with self.publish_lock:
            self.dof_target[:] = hold_q
            self.filtered_dof_target[:] = hold_q
        self._prepare_q = latched
        # 이 단계의 기준은 prepare 자세다. RL 자세 기준 편차는 `r` 뒤에 따로 찍는다
        # -- 단계마다 그 단계가 도달했어야 할 자세로 재야 경고가 경고로 남는다.
        self._log_joint_deviation("at CUSTOM entry (vs prepare pose)", hold_q)
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

    def _legs_quiet_for(self, thresh_rps, need_s):
        """다리 12관절이 임계 아래로 연속 `need_s` 초 동안 조용했는가.

        ⛔ 왜 시간이 아니라 이것인가: SDK 의 `IS_READY` 는 get-up 의 **완료가
        아니다.** 2026-08-07 실기에서 IS_READY 를 보고 4.9 s 에 재진입했는데,
        로봇은 아직 다리를 오므려 서기를 하는 중이었다. 그 중간 자세를 래치하고
        RL 자세로 1.6 rad 을 끌다가 펌웨어 관절 보호가 걸렸다.

        오므리는 동작이 끝나면 다리 관절 속도가 0 으로 간다. 그것은 상태 토픽이
        아니라 LowState 에서 직접 읽히므로, `/robot_states` 가 없는 빌드에서도
        쓸 수 있다. 실측 get-up 은 8.0 s 였고 우리는 4.9 s 에 들어갔다 --
        시간 상수를 추측하는 대신 로봇이 멈추는 것을 본다.
        """
        legs = slice(self.policy.leg_start,
                     self.policy.leg_start + self.policy.num_act)
        peak = float(np.max(np.abs(self.dof_vel[legs])))
        now = time.monotonic()
        if peak > thresh_rps:
            self._legs_quiet_since = 0.0
            return False, peak
        if not self._legs_quiet_since:
            self._legs_quiet_since = now
        return (now - self._legs_quiet_since) >= need_s, peak

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
            # 정책 게인은 여기서 내려온다. 재진입은 prepare 게인으로 시작하고
            # `_start_policy_control()` 이 다시 세운다 -- 그 호출이 "reenter"
            # 분기에 **있다**. (2026-08-08 이전 이 주석은 `start_rl_gait_conditionally`
            # 를 가리켰는데, 그 함수는 `__main__` 에서 1회만 불려 재진입 경로를
            # 지나지 않았다. 주석이 버그를 사양으로 적고 있었다.)
            self._policy_gains_active = False
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
                # ⛔ IS_READY 는 "일으킬 수 없다"가 아니라 "일으킬 것이 없다"이다.
                # 이미 서 있으면 SDK 가 is_recovery_available=False 로 답하는 것이
                # 정상인데, 예전 코드는 그것을 실패로 읽고 10 초 뒤 중단했다
                # (2026-08-07: "get-up never became available (state=IS_READY)").
                # GetUp 을 건너뛰고 완료 판정으로 바로 넘어간다 -- 거기서 다리가
                # 조용해질 때까지 기다리므로 이르게 진입하지 않는다.
                if state == FallState.IS_READY:
                    self.logger.warning(
                        "[recovery] state=IS_READY -- 이미 서 있다. GetUp 을 건너뛰고 "
                        "관절 정지 확인으로 넘어간다")
                    self._recovery_phase = "getup"
                    self._recovery_t0 = time.monotonic()
                    self._legs_quiet_since = 0.0
                    self._getup_called = False
                    return
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
            self._legs_quiet_since = 0.0
            self._getup_called = True
            return

        if phase == "getup":
            timeout = float(rec.get("getup_timeout_s", 20.0))
            # ⛔ 완료 판정은 **두 조건이 모두** 맞아야 한다.
            #
            #   (1) 상태가 IS_READY 이고 최소 시간이 지났다
            #   (2) 다리 12관절이 멈췄다
            #
            # (1)만 보던 것이 2026-08-07 사고였다. 실측 get-up 은 HAS_FALLEN ->
            # IS_READY 8.0 s 인데 하드코딩된 `elapsed > 2.0` 때문에 4.9 s 에
            # 재진입했고, 다리를 오므리는 중간 자세에서 관절을 가져갔다.
            # (2)가 본질이다 -- 오므리는 동작이 끝나야 속도가 0 이 된다.
            min_wait = (float(rec.get("getup_min_wait_s", 8.0))
                        if self._getup_called else 0.0)
            quiet_rps = float(rec.get("getup_quiet_dof_vel_rps", 0.15))
            quiet_hold = float(rec.get("getup_quiet_hold_s", 0.5))
            quiet, peak = self._legs_quiet_for(quiet_rps, quiet_hold)

            # ⛔ 직립은 **어느 경로에서도** 요구한다. /fall_down 은 ~1 Hz 라
            # 방금 DAMPING 으로 관절을 놓은 직후에도 낡은 IS_READY 를 들고 있을 수
            # 있고, 그때 다리는 힘이 빠져서 오히려 조용하다. 즉 "IS_READY + 조용함"
            # 만으로는 무너지는 중인 로봇을 서 있다고 읽을 수 있다. tilt 는
            # LowState 에서 500 Hz 로 오므로 그 창이 없다.
            upright_lim = float(rec.get("getup_upright_tilt_rad", 0.35))
            upright = self._latest_tilt < upright_lim
            if self.fall_monitor and self.fall_monitor.available and fall_age < 5.0:
                state_ok = (state == FallState.IS_READY and elapsed > min_wait
                            and upright)
            else:
                # No fall topic: fall back to the IMU alone.
                state_ok = upright and elapsed > max(
                    min_wait, float(rec.get("getup_blind_wait_s", 8.0)))

            if state_ok and quiet:
                self.logger.warning(
                    "[recovery] standing again after %.1f s (다리 정지 %.1f s, "
                    "peak |dq| %.3f rad/s); re-entering CUSTOM",
                    elapsed, quiet_hold, peak)
                self._recovery_phase = "reenter"
                self._recovery_t0 = time.monotonic()
                return
            # 상태는 됐는데 아직 움직이는 중이면 그 사실을 남긴다. 예전에는 여기서
            # 그냥 들어갔다.
            now = time.monotonic()
            if state_ok and not quiet and now - self._last_getup_wait_log > 1.0:
                self._last_getup_wait_log = now
                self.logger.info(
                    "[recovery] 상태는 준비됐지만 다리가 아직 움직인다 "
                    "(peak |dq| %.3f > %.3f rad/s) -- 기다린다 (%.1f s)",
                    peak, quiet_rps, elapsed)
            if elapsed > timeout:
                self.logger.error(
                    "[recovery] get-up did not complete in %.0f s "
                    "(state_ok=%s, upright=%s tilt=%.0fdeg, legs_quiet=%s, "
                    "peak |dq| %.3f rad/s); stopping.",
                    timeout, state_ok, upright, np.degrees(self._latest_tilt),
                    quiet, peak)
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
            # ⛔ 진입과 **정확히 같은 경로**로 정책 구동을 시작한다. 예전에는
            # 여기서 타이머만 되감고 곧바로 추론을 켰다 -- prepare 게인 위에서,
            # 자세 램프 없이. 독립 감사 3회가 전부 이것을 1순위로 짚었다.
            try:
                self._start_policy_control(reason="recovery re-entry")
            except Exception as exc:
                self.logger.error(
                    "[recovery] policy control restart failed: %s", exc)
                self.running = False
                self._recovery_phase = "none"
                return
            # 정책 내부 상태(actions / gait_process / 관측 이력)는 낙상을 건너
            # 살아 있으면 안 된다. `hasattr` 로 감싸면 메서드가 없을 때 **조용히**
            # 아무 일도 안 하므로, 있는지를 여기서 명시적으로 본다.
            reset = getattr(self.policy, "reset", None)
            if callable(reset):
                reset()
            else:
                self.logger.warning(
                    "[recovery] policy has no reset(); actions/gait_process/"
                    "observation history carry across the fall")
            self.running_policy = True
            self._recovery_phase = "none"
            self.logger.warning(
                "[recovery] complete (#%d, %.1f s total). Mission continues; the "
                "elapsed time for this run is no longer a clean measurement.",
                self._recovery_count, time.monotonic() - self._recovery_t0)
            return

    def _start_policy_control(self, reason):
        """prepare 자세/게인 -> RL 자세/게인. **정책이 루프를 닫기 직전에** 부른다.

        ⛔ 2026-08-08. 이 블록은 원래 `start_rl_gait_conditionally` 안에 인라인으로
        있었고, 그 함수의 호출자는 `__main__` **하나**였다. 그래서 낙상 복구
        재진입(`_step_recovery` 의 "reenter")은 여기를 **한 번도 지나지 않았다**:

          * 게인이 prepare 인 채로 정책이 돈다. hip/knee kp 350 (학습 100),
            ankle kp 250 (학습 50) = **5배**. 정책은 자기가 가정한 임피던스가
            아닌 곳에서 구동된다 -- 첫 낙상 이후 **영구히**.
          * prepare->RL 자세 램프가 없어 첫 추론이 자세 계단을 만든다.
          * `_policy_gains_active` 가 True 로 돌아올 길이 없어 `--parallel-torque`
            가 첫 낙상 이후 조용히 자기무력화된다. (이건 `8d763fb` 가 만든 회귀다 --
            그 이전 판에는 게이트 자체가 없어 복구 뒤에도 계속 작동했다.)

        독립 감사 3회가 전부 이 결함을 1순위로 지목했다(⛔1/⛔-1/⛔1).

        진입(`b`)이 붙잡는 것은 hip+knee+ankle = 0 인 prepare 자세다. 그 구간에는
        균형을 닫는 주체가 없어서 개루프로 설 수 있어야 하기 때문이다. RL 자세는
        합이 -0.05 라 개루프로는 앞으로 기운다(2026-08-07 실기 tilt 4.9-13.8도).
        그래서 그 이동을 여기까지 미룬다.

        램프는 `dof_target` 만 움직인다 -- 발행 스레드가 500 Hz 로 그것을 읽어
        내보내므로 low_cmd 를 여기서 직접 쓰지 않는다(경합이 없다).
        """
        rec = self.cfg.get("safety", {}).get("recovery", {})
        hold_q = np.asarray(self.cfg["prepare"]["default_qpos"], dtype=np.float32)
        rl_q = np.asarray(self.cfg["common"]["default_qpos"], dtype=np.float32)
        legs = slice(self.policy.leg_start,
                     self.policy.leg_start + self.policy.num_act)
        travel = float(np.max(np.abs(rl_q[legs] - hold_q[legs])))
        gait_ramp_s = float(rec.get("gait_entry_ramp_s", 1.0))
        self.logger.info(
            "[mode-timing] prepare -> RL 자세 (%s): 다리 최대 이동 %.4f rad, %.2f s",
            reason, travel, gait_ramp_s)
        step_dt = 0.02
        ramp_start = time.monotonic()
        while True:
            alpha = min(1.0, (time.monotonic() - ramp_start) / max(1e-6, gait_ramp_s))
            blend = alpha * alpha * (3.0 - 2.0 * alpha)
            self.dof_target[:] = (1 - blend) * hold_q + blend * rl_q
            if alpha >= 1.0:
                break
            time.sleep(step_dt)
        self.dof_target[:] = rl_q

        # Swap prepare gains for the policy gains, holding the current position.
        # The pose is now the policy default from the ramp above; only the gains
        # change, and they change exactly when the policy starts closing the loop
        # that those gains assume.
        #
        # 발행 스레드가 여기서도 살아 있으므로 프레임을 락 안에서 만든다.
        # `_policy_gains_active` 도 **같은 락 안에서** 세운다 -- 그래야
        # `--parallel-torque` 의 kp=0 이 정책 게인이 실린 뒤에만 적용된다.
        with self.publish_lock:
            self._fill_low_cmd(
                self.dof_target,
                np.asarray(self.cfg["common"]["stiffness"], dtype=np.float32),
                np.asarray(self.cfg["common"]["damping"], dtype=np.float32))
            self._policy_gains_active = True
            self._send_cmd(self.low_cmd)
        self._log_joint_deviation("at %s (vs rl pose)" % reason, rl_q)
        self.next_inference_time = self.timer.get_time()
        self.next_publish_time = self.timer.get_time()

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

        self._start_policy_control(reason="RL-gait start")
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
            _dt = (_pub_now - self._pub_last) if self._pub_last else self.cfg["common"]["dt"]
            if self._pub_last:
                self._pub_dt.append(_dt)
                if len(self._pub_dt) > 2000:
                    self._pub_dt.pop(0)
            self._pub_last = _pub_now

            # ⛔ 여기부터 `_send_cmd` 까지가 임계 구역이다. 진입·복구 램프가
            # 같은 `low_cmd` 를 만지므로, 프레임을 절반만 바꾼 채 내보내면
            # 관절마다 다른 시점의 목표가 섞여 나간다. sleep 은 락 **밖**이다.
            with self.publish_lock:
                self._publish_one_frame(_dt)
            time.sleep(0.001)

    def _publish_one_frame(self, _dt):
        """저역통과 한 스텝 + `low_cmd` 채우기 + 발행.

        ⛔ **`self.publish_lock` 을 쥔 채로만 부른다.** 여기서 sleep 하지 마라.
        """
        if self._rate_fixed_filter:
            # ⛔ 근본 수정: 필터를 **루프 속도와 무관**하게 만든다.
            #
            # 고정 계수 0.2는 500 Hz 발행을 가정하고 고른 값이라 시정수 10 ms를
            # 의도한 것이다. 루프가 느려지면 같은 계수가 훨씬 긴 시정수를 만들고,
            # 그러면 필터가 보행 자체를 깎는다:
            #     500 Hz -> tau  10 ms, 2 Hz 감쇠 0.99
            #      50 Hz -> tau 100 ms,          0.66
            # 2026-08-06 MuJoCo에서 이것을 재현했다 -- 필터를 50 Hz로 돌리면
            # 추종률이 0.93 -> 0.64로 떨어지고(실기 0.61) 낙상이 0 -> 60/분이 된다.
            #
            # 실측 dt로 계수를 매번 다시 계산하면 시정수가 의도대로 고정된다.
            # dt가 튀는 순간에도 alpha가 1을 넘지 않으므로 발산하지 않는다.
            alpha = 1.0 - math.exp(-max(_dt, 1e-6) / self._filter_tau_s)
            alpha = min(alpha, 1.0)
            self.filtered_dof_target = (self.filtered_dof_target * (1.0 - alpha)
                                        + self.dof_target * alpha)
        else:
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
        #
        # ⛔ `_policy_gains_active` 로 게이트한다(2026-08-07 감사).
        # 이 블록은 `common.stiffness`(발목 50)로 토크를 만들고 `kp` 를 0 으로
        # 내린다. 그런데 `b`~`r` 구간과 복구 재진입은 **prepare 게인**(발목 250)
        # 으로 붙잡는 구간이다. 게이트가 없으면 그 구간에서 위치 서보가
        # 사라지고 5배 약한 피드포워드만 남는다 -- 균형을 닫는 주체가 없는
        # 바로 그 구간에서. 게다가 이 코드는 `kp` 를 **되돌리지 않으므로**
        # 진입 램프가 매 프레임 250 으로 복구하고 여기가 매 프레임 0 으로
        # 내리는 왕복이 됐다.
        if self._parallel_torque and self._policy_gains_active:
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

    def __enter__(self) -> "Controller":
        return self

    def __exit__(self, *args) -> None:
        self.cleanup()


if __name__ == "__main__":
    def signal_handler(sig, frame):
        print("\nShutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    # ⛔ 2026-08-08. SIGTERM 에 핸들러가 없었다. `kill <pid>` 는 기본 처분이
    # 즉시 종료라 `__exit__` -> `cleanup()` 이 안 돌고, 그 안의
    # `ChangeMode(kDamping)` 도 안 나간다 -- **CUSTOM 인 채로 프로세스만 사라지고
    # 마지막 LowCmd 프레임이 모터에 걸린 채 남는다.** 이 파일의 주석은 그 증상을
    # sshkeyboard 탓으로 적어 뒀지만 원인은 핸들러 부재였다.
    # `sys.exit` 는 SystemExit 를 올리므로 `with` 의 `__exit__` 가 돈다.
    signal.signal(signal.SIGTERM, signal_handler)

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
    parser.add_argument("--rate-fixed-filter", action="store_true",
                        help="관절 목표 필터를 루프 속도와 무관하게 만든다. 고정 계수 0.2는 "
                             "500 Hz 발행을 가정한 값이라 시정수 10 ms를 의도한 것인데, "
                             "루프가 느리면 같은 계수가 훨씬 긴 시정수를 만들어 보행을 깎는다 "
                             "(50 Hz면 2 Hz 신호가 0.66으로). 실측 dt로 계수를 매번 다시 "
                             "계산해 시정수를 고정한다. 기본 꺼짐 -- 실기에서 A/B 하라.")
    parser.add_argument("--filter-tau-ms", type=float, default=10.0,
                        help="--rate-fixed-filter 의 목표 시정수(ms). 기본 10 = 설계 의도.")
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
        rate_fixed_filter=args.rate_fixed_filter,
        filter_tau_s=args.filter_tau_ms / 1000.0,
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
