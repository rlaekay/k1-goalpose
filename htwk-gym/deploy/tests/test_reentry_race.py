"""복구 재진입 경합 회귀 검사 — 실제 `_enter_custom_latched` 를 돌린다.

로직을 복사해 오지 않는다. SDK/ROS 만 스텁으로 갈아끼우고 **저장소의 진짜 메서드**를
호출한다(그러지 않으면 검사 대상이 배포가 쓰는 코드가 아니게 된다).

검사:
  T1 재진입: 발행 스레드가 살아 있는 상태에서 진입 램프를 돌린다.
             발행된 모든 프레임의 q 가 단조 smoothstep 을 따라야 한다.
             (수정 전에는 램프 자세와 latched 를 오가는 사각파가 나왔다.)
  T2 최초진입: 발행 스레드 없이. 프레임이 램프와 정확히 일치해야 한다.
  T3 init_Cmd_T1: 재진입에서는 호출되지 않아야 한다(벡터 교체 = 프레임 찢김).
  T4 게인: 램프 내내 prepare 게인이 유지돼야 한다.
  T5 --parallel-torque: prepare 구간에서 kp 가 0 으로 내려가지 않아야 한다.
  T6 락: low_cmd 를 만지는 동안 두 스레드가 겹치지 않아야 한다(계측으로 확인).

돌리는 법 (numpy 필요 -- Mac 기본 python3 에는 없다. 서버 conda 에서 돌려라):
    cd htwk-gym/deploy && python tests/test_reentry_race.py

⛔ 음성 대조를 반드시 같이 봐라. 수정 전 코드(`git show <pre-fix>:...`)에 이 검사를
걸면 T1 이 **램프 이탈 1.6000 rad (91.7 deg)** 로 실패한다. 그것이 이 검사가
무의미하지 않다는 유일한 증거다 -- 이 저장소는 "검사 도구가 먼저 틀린" 전례가 셋 있다.

이 파일을 만들면서도 세 번 틀렸다(기록):
  1. `init_Cmd_T1` 호출 계수의 기준선 -- `make_ctl()` 자신이 한 번 부른다
  2. smoothstep 대조를 **송신 시각**으로 잼 -- 계산 시각과 ~200 us 어긋나 9.9e-05 rad
  3. 단조성 문턱 1e-9 -- float32 eps(0.2 근처 1.5e-08)보다 빡빡했다
"""
import sys, os, types, threading, time, math
import numpy as np

DEPLOY = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ---- SDK / ROS 스텁 --------------------------------------------------------
class MotorCmd:
    __slots__ = ("q", "dq", "tau", "kp", "kd", "weight")
    def __init__(self):
        self.q = self.dq = self.tau = self.kp = self.kd = self.weight = 0.0

class LowCmd:
    def __init__(self):
        self.cmd_type = None
        self.motor_cmd = []

class _Enum:
    SERIAL = "SERIAL"; PARALLEL = "PARALLEL"

sdk = types.ModuleType("booster_robotics_sdk_python")
for _n in ("ChannelFactory", "B1LocoApiId", "B1LocoClient", "B1LowCmdPublisher",
           "B1LowStateSubscriber", "LowState", "RobotMode"):
    setattr(sdk, _n, type(_n, (), {}))
sdk.LowCmd = LowCmd
sdk.MotorCmd = MotorCmd
sdk.LowCmdType = _Enum
sdk.B1JointCnt = 23
sys.modules["booster_robotics_sdk_python"] = sdk

for name in ("utils.remote_control_service", "utils.rotate", "utils.policy_goal_pose"):
    m = types.ModuleType(name); sys.modules[name] = m
sys.modules["utils.remote_control_service"].RemoteControlService = type("R", (), {})
sys.modules["utils.rotate"].rotate_vector_inverse_rpy = lambda *a: np.zeros(3)
sys.modules["utils.policy_goal_pose"].GoalPosePolicy = type("P", (), {})

sys.path.insert(0, DEPLOY)
os.chdir(DEPLOY)
import deploy_goal_pose as D          # utils.command / utils.timer 는 진짜를 쓴다

JN = 22
LEGS = slice(10, 22)
PREP_Q = np.array([0, 0, 0.0, -1.3, 0, 0.0, 0.0, 1.3, 0, 0.0,
                   -0.1, 0, 0, 0.2, -0.1, 0, -0.1, 0, 0, 0.2, -0.1, 0], dtype=np.float32)
PREP_KP = np.full(JN, 250.0, dtype=np.float32)
PREP_KD = np.full(JN, 5.0, dtype=np.float32)


def make_ctl(parallel_torque=False):
    c = D.Controller.__new__(D.Controller)
    import logging
    c.logger = logging.getLogger("t"); c.logger.setLevel(logging.CRITICAL)
    c.joint_cnt = JN
    c.cfg = {"common": {"dt": 0.002, "stiffness": [50.0]*JN, "damping": [1.0]*JN,
                        "torque_limit": [20.0]*JN, "default_qpos": list(PREP_Q)},
             "prepare": {"stiffness": list(PREP_KP), "damping": list(PREP_KD),
                         "default_qpos": list(PREP_Q)},
             "mech": {"parallel_mech_indexes": [14, 15, 20, 21]},
             "safety": {"low_state_timeout_s": 0.2,
                        "recovery": {"custom_entry_ramp_s": 0.6,
                                     "custom_entry_max_rate_rps": 0.3,
                                     "custom_entry_ramp_max_s": 8.0}}}
    c.publish_lock = threading.Lock()
    c.running = True
    c.publish_runner = None
    c._policy_gains_active = False
    c._parallel_torque = parallel_torque
    c._rate_fixed_filter = False
    c._filter_tau_s = 0.010
    c._pub_last = 0.0; c._pub_dt = []
    c._custom_mode_started = False
    c._custom_mode_entered_monotonic = 0.0
    c._prepare_q = None
    c._latest_tilt = 0.0
    c.low_cmd = LowCmd()
    D.init_Cmd_T1(c.low_cmd, JN)
    c.dof_target = np.zeros(JN, dtype=np.float32)
    c.filtered_dof_target = np.zeros(JN, dtype=np.float32)
    c.dof_pos = np.zeros(JN, dtype=np.float32)
    c.dof_vel = np.zeros(JN, dtype=np.float32)
    c.dof_pos_latest = np.zeros(JN, dtype=np.float32)
    c.policy = types.SimpleNamespace(leg_start=10, num_act=12)
    c.timer = D.Timer(D.TimerConfig(time_step=0.002))
    c.next_publish_time = c.timer.get_time()
    c.client = types.SimpleNamespace(ChangeMode=lambda m: 0)
    c._last_low_state_monotonic = time.monotonic()

    # 계측: 발행된 프레임을 전부 기록 + 락 밖 접근 탐지
    c._frames = []
    c._init_calls = []
    c._in_critical = [0]
    c._overlap = [0]
    def _send(cmd):
        c._frames.append((time.monotonic(),
                          np.array([m.q for m in cmd.motor_cmd], dtype=np.float64),
                          np.array([m.kp for m in cmd.motor_cmd], dtype=np.float64)))
    c._send_cmd = _send
    c._ensure_standing_before_custom = lambda: None
    c._require_fresh_low_state = lambda ctx: 0.0
    c._wait_for_low_state_after_custom = lambda *a: None
    c._log_joint_deviation = lambda *a: None
    return c


def run_publish_thread(c, hz=500.0, seconds=None, stop_evt=None):
    """LowState 콜백이 하는 일(카운터 tick)까지 흉내내어 발행 스레드를 돌린다."""
    def ticker():
        while not stop_evt.is_set():
            c.timer.tick_timer_if_sim()
            time.sleep(1.0 / hz)
    def pub():
        while not stop_evt.is_set():
            now = c.timer.get_time()
            if now < c.next_publish_time:
                time.sleep(0.0005); continue
            c.next_publish_time += c.cfg["common"]["dt"]
            _dt = 1.0 / hz
            with c.publish_lock:
                c._in_critical[0] += 1
                if c._in_critical[0] > 1: c._overlap[0] += 1
                c._publish_one_frame(_dt)
                c._in_critical[0] -= 1
            time.sleep(0.0005)
    t1 = threading.Thread(target=ticker, daemon=True); t1.start()
    t2 = threading.Thread(target=pub, daemon=True); t2.start()
    return t1, t2


def smoothstep_track(latched, hold_q, ramp_s, t):
    a = min(1.0, max(0.0, t / ramp_s)); b = a * a * (3.0 - 2.0 * a)
    return (1 - b) * latched + b * hold_q


FAIL = []
def check(name, ok, detail=""):
    print(("  ✅ " if ok else "  ❌ ") + name + (("  -- " + detail) if detail else ""))
    if not ok: FAIL.append(name)


# =========================== T1: 재진입 ====================================
print("T1  복구 재진입 (발행 스레드 살아 있음)")
c = make_ctl()
latched = PREP_Q.copy(); latched[LEGS] += 1.60      # 실측 재진입 이동량 1.5934 rad
c.dof_pos_latest[:] = latched
stop = threading.Event()
# 발행 스레드가 이미 도는 상태를 만든다 (재진입 조건)
c.publish_runner = threading.Thread(target=lambda: stop.wait(), daemon=True)
c.publish_runner.start()
run_publish_thread(c, stop_evt=stop)
time.sleep(0.05)
c._frames.clear()
t0 = time.monotonic()
D.Controller._enter_custom_latched(c)
ramp_s = min(8.0, max(0.6, 1.60 / 0.3))
stop.set(); time.sleep(0.05)

frames = [f for f in c._frames if f[0] >= t0]
dev = []
for ts, q, kp in frames:
    want = smoothstep_track(latched, PREP_Q, ramp_s, ts - t0)
    dev.append(float(np.max(np.abs(q[LEGS] - want[LEGS]))))
worst = max(dev) if dev else float("inf")
check("발행 프레임이 램프를 따른다 (최대 이탈 < 0.05 rad)", worst < 0.05,
      "frames=%d  worst=%.4f rad (%.1f deg)" % (len(frames), worst, math.degrees(worst)))
# 사각파 탐지: 연속 프레임 간 점프
jumps = [float(np.max(np.abs(frames[i][1][LEGS] - frames[i-1][1][LEGS])))
         for i in range(1, len(frames))]
check("연속 프레임 간 점프 없음 (< 0.05 rad)", (max(jumps) if jumps else 0) < 0.05,
      "max jump=%.4f rad (%.1f deg)" % (max(jumps), math.degrees(max(jumps))) if jumps else "")
check("락 겹침 0", c._overlap[0] == 0, "overlap=%d" % c._overlap[0])
check("게인이 램프 내내 prepare 유지", all(abs(kp[10] - 250.0) < 1e-6 for _, _, kp in frames))

# =========================== T2: 최초 진입 =================================
print("\nT2  최초 진입 (발행 스레드 없음) -- 동작이 바뀌면 안 된다")
c2 = make_ctl()
c2.dof_pos_latest[:] = latched
t0 = time.monotonic()
D.Controller._enter_custom_latched(c2)
frames2 = [f for f in c2._frames if f[0] >= t0]
dev2 = [float(np.max(np.abs(q[LEGS] - smoothstep_track(latched, PREP_Q, ramp_s, ts - t0)[LEGS])))
        for ts, q, kp in frames2[1:]]
check("램프 프레임이 smoothstep 을 따른다 (< 1e-3 rad; 잔차는 송신-계산 시각 skew)",
      (max(dev2) if dev2 else 1) < 1e-3,
      "frames=%d  worst=%.2e rad" % (len(frames2), max(dev2) if dev2 else -1))
_q2 = [q[13] for _, q, _ in frames2[1:]]          # Knee_Pitch: latched 1.8 -> hold 0.2
# 문턱은 float32 eps 기준이다(0.2 근처 ~1.5e-08). 그보다 빡빡하게 잡으면
# 저역통과의 0.2*0.8+0.2*0.2 반올림이 "위반"으로 잡힌다 -- 8.5e-7 도짜리 유령이다.
check("램프가 단조 감소 (float32 eps 허용)",
      all(_q2[i] <= _q2[i-1] + 1e-6 for i in range(1, len(_q2))),
      "max 증가 %.2e rad" % max([_q2[i]-_q2[i-1] for i in range(1,len(_q2))] or [0]))
check("끝점이 정확히 prepare 자세", abs(frames2[-1][1][13] - PREP_Q[13]) < 1e-6,
      "last=%.6f want=%.6f" % (frames2[-1][1][13], PREP_Q[13]))
check("최초 진입은 발행 스레드를 새로 띄운다",
      c2.publish_runner is not None and c2.publish_runner.is_alive())
c2.running = False

# =========================== T3: init_Cmd_T1 ==============================
print("\nT3  init_Cmd_T1 은 재진입에서 호출되지 않는다")
calls = {"n": 0}
_orig = D.init_Cmd_T1
D.init_Cmd_T1 = lambda lc, n: (calls.__setitem__("n", calls["n"] + 1), _orig(lc, n))[1]
c3 = make_ctl(); c3.dof_pos_latest[:] = latched
calls["n"] = 0            # ⛔ make_ctl() 자신이 한 번 부른다. 기준선을 지운다.
stop3 = threading.Event()
c3.publish_runner = threading.Thread(target=lambda: stop3.wait(), daemon=True); c3.publish_runner.start()
D.Controller._enter_custom_latched(c3); stop3.set()
check("재진입: init_Cmd_T1 호출 0회", calls["n"] == 0, "calls=%d" % calls["n"])
c4 = make_ctl(); c4.dof_pos_latest[:] = latched
calls["n"] = 0            # 〃
D.Controller._enter_custom_latched(c4); c4.running = False
check("최초진입: init_Cmd_T1 호출 1회", calls["n"] == 1, "calls=%d" % calls["n"])
D.init_Cmd_T1 = _orig

# =========================== T5: parallel-torque ==========================
print("\nT5  --parallel-torque 가 prepare 구간에서 kp 를 내리지 않는다")
c5 = make_ctl(parallel_torque=True)
c5.dof_pos_latest[:] = latched
stop5 = threading.Event()
c5.publish_runner = threading.Thread(target=lambda: stop5.wait(), daemon=True); c5.publish_runner.start()
run_publish_thread(c5, stop_evt=stop5)
time.sleep(0.05); c5._frames.clear()
D.Controller._enter_custom_latched(c5)
stop5.set(); time.sleep(0.05)
ank_kp = [kp[14] for _, _, kp in c5._frames]
check("발목 kp 가 prepare(250) 로 유지", all(abs(v - 250.0) < 1e-6 for v in ank_kp),
      "min=%.1f max=%.1f over %d frames" % (min(ank_kp), max(ank_kp), len(ank_kp)))
c5._policy_gains_active = True
with c5.publish_lock:
    c5._publish_one_frame(0.002)
check("정책 게인 활성 뒤에는 kp 가 0 으로 내려간다 (기능 보존)",
      abs(c5._frames[-1][2][14]) < 1e-9, "kp=%.3f" % c5._frames[-1][2][14])

print("\n" + ("=" * 62))
print("실패 %d 건" % len(FAIL) + ((": " + ", ".join(FAIL)) if FAIL else " — 전부 통과"))
sys.exit(1 if FAIL else 0)
