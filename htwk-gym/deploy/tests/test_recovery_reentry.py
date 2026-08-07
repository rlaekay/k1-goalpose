"""낙상 복구 **재진입** 경로 회귀 테스트.

⛔ 왜 별도 파일인가
`test_reentry_race.py` 는 `_enter_custom_latched` 의 경합만 본다. 복구 상태기계
(`_step_recovery`)를 **한 번도 돌지 않아서**, 독립 감사 3회가 만장일치로 1순위로
지목한 결함을 원리적으로 잡을 수 없었다:

    복구 재진입은 `_start_policy_control()`(구 `start_rl_gait_conditionally` 본문)을
    지나지 않았다. 그 함수의 호출자가 `__main__` 하나였기 때문이다. 결과:
      1. 정책이 **prepare 게인** 위에서 돈다 -- ankle kp 250 대 학습 50 = 5배
      2. prepare -> RL 자세 램프가 없어 첫 추론이 자세 계단을 만든다
      3. `_policy_gains_active` 가 True 로 못 돌아와 `--parallel-torque` 가
         첫 낙상 이후 영구히 자기무력화된다 (= 커밋 8d763fb 의 회귀)

⚠️ 게인은 **비균일**로 만든다. 기존 테스트가 `PREP_KP` 를 균일 250 으로 둬서
   게인 인덱스가 밀려도 통과했다(감사 지적). 관절마다 다른 값을 넣으면 밀림이 걸린다.

실행: `python tests/test_recovery_reentry.py` (deploy/ 에서, numpy 만 필요)
"""
import sys, os, ast, types, threading, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEPLOY = os.path.dirname(HERE)


# ---- SDK 스텁 (test_reentry_race.py 와 같은 방식) ---------------------------
class MotorCmd:
    def __init__(self):
        self.q = 0.0; self.dq = 0.0; self.kp = 0.0; self.kd = 0.0; self.tau = 0.0
        self.weight = 0.0


class LowCmd:
    def __init__(self):
        self.motor_cmd = [MotorCmd() for _ in range(22)]
        self.cmd_type = None


class _Enum:
    def __getattr__(self, k):
        return k


sdk = types.ModuleType("booster_robotics_sdk_python")
# PEP 562: 이름을 열거하지 않는다. SDK 심볼이 하나 늘어날 때마다 테스트가
# ImportError 로 죽는 것은 이 테스트가 재려는 것과 아무 상관이 없다.
sdk.__getattr__ = lambda name: _Enum()
sdk.LowCmd = LowCmd
sdk.MotorCmd = MotorCmd
sdk.B1JointCnt = 23
sys.modules["booster_robotics_sdk_python"] = sdk
for name in ("utils.remote_control_service", "utils.rotate", "utils.policy_goal_pose"):
    sys.modules[name] = types.ModuleType(name)
sys.modules["utils.remote_control_service"].RemoteControlService = type("R", (), {})
sys.modules["utils.rotate"].rotate_vector_inverse_rpy = lambda *a: np.zeros(3)
sys.modules["utils.policy_goal_pose"].GoalPosePolicy = type("P", (), {})

sys.path.insert(0, DEPLOY)
os.chdir(DEPLOY)
import deploy_goal_pose as D                                       # noqa: E402

JN = 22
LEGS = slice(10, 22)
PREP_Q = np.array([0, 0, 0.0, -1.3, 0, 0.0, 0.0, 1.3, 0, 0.0,
                   0.0, 0, 0, 0.1, -0.1, 0, 0.0, 0, 0, 0.1, -0.1, 0], dtype=np.float32)
# RL 자세: 합 -0.05 (hip -0.2 / knee 0.4 / ankle -0.25). prepare 는 합 0.
RL_Q = np.array([0, 0, 0.0, -1.3, 0, 0.0, 0.0, 1.3, 0, 0.0,
                 -0.2, 0, 0, 0.4, -0.25, 0, -0.2, 0, 0, 0.4, -0.25, 0], dtype=np.float32)
# ⚠️ 비균일. 인덱스가 밀리면 값이 안 맞는다.
PREP_KP = np.array([250.0 + i for i in range(JN)], dtype=np.float32)
PREP_KD = np.array([5.0 + 0.1 * i for i in range(JN)], dtype=np.float32)
RL_KP = np.array([50.0 + 2 * i for i in range(JN)], dtype=np.float32)
RL_KD = np.array([1.0 + 0.05 * i for i in range(JN)], dtype=np.float32)


class FakePolicy:
    def __init__(self):
        self.leg_start = 10
        self.num_act = 12
        self.reset_calls = 0
        self.actions = np.ones(12, dtype=np.float32)
        self.gait_process = 0.77

    def reset(self):
        self.reset_calls += 1
        self.actions[:] = 0.0
        self.gait_process = 0.0


def make_ctl(parallel_torque=False, policy=None):
    c = D.Controller.__new__(D.Controller)
    import logging
    c.logger = logging.getLogger("t"); c.logger.setLevel(logging.CRITICAL)
    c.joint_cnt = JN
    c.cfg = {"common": {"dt": 0.002, "stiffness": list(RL_KP), "damping": list(RL_KD),
                        "torque_limit": [20.0] * JN, "default_qpos": list(RL_Q)},
             "prepare": {"stiffness": list(PREP_KP), "damping": list(PREP_KD),
                         "default_qpos": list(PREP_Q)},
             "mech": {"parallel_mech_indexes": [14, 15, 20, 21]},
             "safety": {"low_state_timeout_s": 0.2,
                        "recovery": {"custom_entry_ramp_s": 0.05,
                                     "custom_entry_max_rate_rps": 100.0,
                                     "custom_entry_ramp_max_s": 8.0,
                                     "gait_entry_ramp_s": 0.10}}}
    c.publish_lock = threading.Lock()
    c.running = True
    c.running_policy = False
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
    c._recovery_phase = "reenter"
    c._recovery_count = 1
    c._recovery_t0 = time.monotonic()
    c.fall_monitor = None
    # ⚠️ 메서드를 스텁하지 않는다. `_ensure_standing_before_custom` 은 진짜로
    # 지나가게 두고, ModeMonitor 만 없는 빌드(경고 후 통과) 상태로 만든다 --
    # 감사가 지적한 대로 기존 테스트는 이 메서드를 통째로 무력화해서
    # 그 경로의 결함을 원리적으로 못 봤다.
    c.mode_monitor = None
    c._getup_called = False
    c._legs_quiet_since = 0.0
    c._fall_events = []
    c._recovery_reason = "test"
    c.low_cmd = LowCmd()
    D.init_Cmd_T1(c.low_cmd, JN)
    c.dof_target = np.array(PREP_Q, dtype=np.float32)
    c.filtered_dof_target = np.array(PREP_Q, dtype=np.float32)
    c.dof_pos = np.array(PREP_Q, dtype=np.float32)
    c.dof_vel = np.zeros(JN, dtype=np.float32)
    c.dof_pos_latest = np.array(PREP_Q, dtype=np.float32)
    c.dof_tau = np.zeros(JN, dtype=np.float32)
    c.policy = policy if policy is not None else FakePolicy()
    c.timer = D.Timer(D.TimerConfig(time_step=0.002))
    c.next_publish_time = c.timer.get_time()
    c.next_inference_time = c.timer.get_time()
    c.client = types.SimpleNamespace(ChangeMode=lambda m: 0)
    c._last_low_state_monotonic = time.monotonic()

    c._frames = []
    c._targets = []

    def _send(cmd):
        c._frames.append((np.array([m.q for m in cmd.motor_cmd]),
                          np.array([m.kp for m in cmd.motor_cmd]),
                          np.array([m.kd for m in cmd.motor_cmd]),
                          np.array([m.tau for m in cmd.motor_cmd])))
    c._send_cmd = _send
    c._require_fresh_low_state = lambda ctx: 0.0
    c._wait_for_low_state_after_custom = lambda *a: None
    c._log_joint_deviation = lambda *a: None
    return c


FAILED = []


def check(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + (("   " + detail) if detail else ""))
    if not ok:
        FAILED.append(name)


print("=" * 72)
print("복구 재진입: 정책 게인 · 자세 램프 · parallel torque · policy.reset")
print("=" * 72)

def live_publisher(c):
    """진짜 발행 스레드를 띄운다.

    ⛔ 스레드 없이 재진입을 돌리면 안 된다. `_start_policy_control` 의 자세 램프는
    `dof_target` 만 쓰고 **발행은 스레드가 한다**. 스레드가 없으면 램프가 한 줄도
    안 나가고 게인 교체 프레임 하나만 남아서, 계단인지 램프인지 구별이 안 된다
    (첫 작성 때 실제로 그 상태로 T6 이 실패했다). 그리고 재진입 시점의 실기
    구성이 바로 "스레드가 살아 있는 상태"다 -- 이쪽이 충실도도 높다.
    """
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            c.timer.tick_timer_if_sim()
            with c.publish_lock:
                c._publish_one_frame(0.002)
            time.sleep(0.002)
    th = threading.Thread(target=loop, daemon=True)
    c.publish_runner = th
    th.start()
    return stop, th


# ---- T1~T7: 재진입이 정책 게인과 RL 자세로 끝나는가 ------------------------
c = make_ctl()
_stop, _th = live_publisher(c)
time.sleep(0.02)
c._frames.clear()                     # 재진입 이전 프레임은 판정에서 뺀다
c._step_recovery()
_stop.set(); _th.join(timeout=1.0)

legs_kp = np.array([m.kp for m in c.low_cmd.motor_cmd])[LEGS]
legs_kd = np.array([m.kd for m in c.low_cmd.motor_cmd])[LEGS]
check("T1 재진입 뒤 다리 kp = common.stiffness (prepare 아님)",
      np.allclose(legs_kp, RL_KP[LEGS]),
      "kp[10:22]=%s" % np.array2string(legs_kp, precision=0))
check("T2 재진입 뒤 다리 kd = common.damping",
      np.allclose(legs_kd, RL_KD[LEGS]))
check("T3 `_policy_gains_active` 가 True 로 돌아온다", c._policy_gains_active is True)
check("T4 `dof_target` 이 RL 자세에서 끝난다",
      np.allclose(c.dof_target, RL_Q, atol=1e-6))
check("T5 `running_policy` 가 켜지고 복구 phase 가 해제된다",
      c.running_policy is True and c._recovery_phase == "none")

# ---- T6: 자세가 계단이 아니라 램프로 움직였는가 ----------------------------
# 발행 프레임의 다리 목표가 prepare -> RL 로 단조 이동하고, 프레임 간 최대
# 점프가 전체 이동량보다 뚜렷하게 작아야 한다(= 계단이 아니다).
qs = np.array([f[0][LEGS] for f in c._frames])
travel = float(np.max(np.abs(RL_Q[LEGS] - PREP_Q[LEGS])))
if len(qs) >= 3:
    jumps = np.max(np.abs(np.diff(qs, axis=0)), axis=1)
    biggest = float(np.max(jumps))
    # ⚠️ "점프가 작다" 만 보면 **공허하게 통과한다** -- 자세가 아예 안 움직이면
    # 점프도 0 이다. 수정 전 코드에서 실제로 그랬다(T4 는 실패하는데 T6 은 통과).
    # 그래서 도착까지 같이 요구한다: 움직였고, 그런데 계단이 아니었다.
    arrived = bool(np.allclose(qs[-1], RL_Q[LEGS], atol=2e-2))
    check("T6 자세가 RL 자세까지 **램프로** 이동한다 (도착 + 점프 < 이동량 50 %)",
          arrived and biggest < 0.5 * travel,
          "도착=%s, 점프 %.4f rad / 이동 %.4f rad, 프레임 %d"
          % (arrived, biggest, travel, len(qs)))
else:
    check("T6 자세가 램프로 이동한다", False,
          "발행 프레임이 %d 개뿐이라 판정 불가" % len(qs))

# ---- T7: policy.reset 이 실제로 불린다 -------------------------------------
check("T7 `policy.reset()` 이 재진입에서 정확히 1회 불린다",
      c.policy.reset_calls == 1, "calls=%d" % c.policy.reset_calls)

# ---- T8: reset 이 없는 정책이면 조용히 넘어가지 않고 경고한다 --------------
class NoResetPolicy:
    leg_start = 10
    num_act = 12


import logging                                                     # noqa: E402


class Grab(logging.Handler):
    def __init__(self):
        super().__init__(); self.msgs = []

    def emit(self, r):
        self.msgs.append(r.getMessage())


c2 = make_ctl(policy=NoResetPolicy())
h = Grab(); c2.logger.addHandler(h); c2.logger.setLevel(logging.WARNING)
_s2, _t2 = live_publisher(c2)
c2._step_recovery()
_s2.set(); _t2.join(timeout=1.0)
check("T8 reset 없는 정책이면 경고를 남긴다 (조용한 no-op 금지)",
      any("no reset()" in m for m in h.msgs),
      "로그 %d 줄" % len(h.msgs))

# ---- T9: --parallel-torque 가 복구 뒤에도 다시 작동한다 --------------------
# 8d763fb 가 만든 회귀의 회귀 테스트. 게이트가 True 로 못 돌아오면 발목 kp 가
# 0 으로 안 내려가고, `--parallel-torque` 가 조용히 자기무력화된 것이다.
c3 = make_ctl(parallel_torque=True)
_s3, _t3 = live_publisher(c3)
c3._step_recovery()
_s3.set(); _t3.join(timeout=1.0)
# ⚠️ 먼저 발목 kp 가 **0 이 아님**을 확인한다. 안 그러면 "게이트가 꺼져서 아무것도
# 안 했다" 와 "게이트가 켜져서 0 으로 내렸다" 가 구별되지 않는다 -- 초기 low_cmd 의
# kp 가 0 이라 이 검사를 빠뜨리면 고장난 코드도 통과한다(첫 작성 때 실제로 그랬다).
ank_before = np.array([m.kp for m in c3.low_cmd.motor_cmd])[[14, 15, 20, 21]]
check("T9a 재진입 직후 발목 kp 가 정책 게인(0 아님)이다",
      np.allclose(ank_before, RL_KP[[14, 15, 20, 21]]),
      "kp=%s" % np.array2string(ank_before, precision=1))
c3.dof_target[:] = RL_Q
c3._publish_one_frame(0.002)
ank = np.array([m.kp for m in c3.low_cmd.motor_cmd])[[14, 15, 20, 21]]
check("T9b 복구 뒤 `--parallel-torque` 가 다시 발목 kp 를 0 으로 내린다",
      np.allclose(ank, 0.0), "ankle kp=%s" % np.array2string(ank, precision=1))

# ---- T10: 호출 구조 — 두 진입 경로가 같은 함수를 쓴다 (정적) ---------------
src = open(os.path.join(DEPLOY, "deploy_goal_pose.py"), encoding="utf-8").read()
tree = ast.parse(src)
callers = set()
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Call) and isinstance(sub.func, ast.Attribute)
                    and sub.func.attr == "_start_policy_control"):
                callers.add(node.name)
check("T10 `_start_policy_control` 을 진입과 재진입 **양쪽**이 부른다",
      callers == {"start_rl_gait_conditionally", "_step_recovery"},
      "호출자=%s" % sorted(callers))

# 게인을 싣는 곳이 그 함수 하나인지도 본다 -- 두 군데가 되면 다시 갈라진다.
gain_writers = set()
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        for sub in ast.walk(node):
            if (isinstance(sub, ast.Attribute) and sub.attr == "_policy_gains_active"
                    and isinstance(sub.ctx, ast.Store)):
                gain_writers.add(node.name)
check("T11 `_policy_gains_active` 를 True 로 세우는 곳이 하나뿐이다",
      "_start_policy_control" in gain_writers,
      "쓰는 함수=%s" % sorted(gain_writers))

# ---- T12: GoalPosePolicy.reset 이 실재하고 상태를 지운다 (정적) ------------
# 실제 모듈은 torch 를 요구해서 여기서 import 하지 않는다. AST 로 본다 --
# 애초의 결함이 "메서드가 아예 없었다" 였으므로 이 검사가 정확히 그것을 잡는다.
psrc = open(os.path.join(DEPLOY, "utils", "policy_goal_pose.py"), encoding="utf-8").read()
ptree = ast.parse(psrc)
reset_fn = None
for node in ast.walk(ptree):
    if isinstance(node, ast.ClassDef) and node.name == "GoalPosePolicy":
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "reset":
                reset_fn = item
check("T12 `GoalPosePolicy.reset` 이 실재한다", reset_fn is not None)
if reset_fn is not None:
    cleared = set()
    for sub in ast.walk(reset_fn):
        if isinstance(sub, ast.Attribute) and isinstance(sub.ctx, ast.Store):
            cleared.add(sub.attr)
        if isinstance(sub, ast.Subscript) and isinstance(sub.value, ast.Attribute):
            cleared.add(sub.value.attr)
    need = {"actions", "gait_process", "_last_time", "_hist", "_hist_primed"}
    check("T13 reset 이 낙상을 건너던 상태 5개를 전부 지운다",
          need <= cleared, "빠진 것=%s" % sorted(need - cleared))

print()
if FAILED:
    print("실패 %d: %s" % (len(FAILED), ", ".join(FAILED)))
    sys.exit(1)
print("전부 통과.")
