"""`_gate_joint_sample` 회귀 테스트 -- 오염된 발목 roll 표본을 버리는가.

근거는 ibatch §8-69. 이 게이트는 **실기 로그에서 발견된 것**을 막는 것이므로
합성 스파이크만으로 통과시키지 않는다. 뒤쪽 T4/T5 는 `realdata/` 의 실제 CSV 를
그대로 흘려서 **양성(오염을 잡는다) + 음성(조용할 때 안 잡는다)** 을 같이 본다.

실행: `python tests/test_dof_gate.py` (deploy/ 에서)
"""
import csv
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
DEPLOY = os.path.dirname(HERE)
sys.path.insert(0, DEPLOY)
sys.path.insert(0, HERE)

# SDK 스텁을 sys.modules 에 심어 주는 기존 테스트를 먼저 읽는다. 스텁을 두 벌
# 두면 갈라지므로 재사용한다.
import test_recovery_reentry as _stub  # noqa: F401
import numpy as np
from deploy_goal_pose import Controller as GoalPoseController

REAL = os.path.join(DEPLOY, "..", "..", "realdata")
FAILS = []


def check(name, ok, detail=""):
    print(("  ✅ " if ok else "  ⛔ ") + name + ("  " + detail if detail else ""))
    if not ok:
        FAILS.append(name)


def make_gate(n=22, rate=30.0, consec=5, step=0.20):
    s = type("S", (), {})()
    s._dof_gate_max_rate = rate
    s._dof_gate_max_consec = consec
    s._dof_gate_max_step = step
    s._dof_gate_prev = None
    s._dof_gate_prev_t = None
    s._dof_gate_consec = np.zeros(n, dtype=np.int32)
    s._dof_gate_rejected = np.zeros(n, dtype=np.int64)
    s._dof_gate_samples = 0
    return s


def feed(s, q, t):
    """monotonic 을 고정 dt 로 흉내낸다."""
    import deploy_goal_pose as m
    real = m.time.monotonic
    m.time.monotonic = lambda: t
    try:
        return GoalPoseController._gate_joint_sample(s, list(q))
    finally:
        m.time.monotonic = real


print("T1 첫 표본은 무조건 통과한다 (직전 값이 없다)")
s = make_gate()
out = feed(s, [0.5] * 22, 0.0)
check("T1", out[0] == 0.5)

print("T2 LowState dt 2.6 ms 에서 0.72 rad 도약(=277 rad/s)을 버리고 직전 값을 유지")
s = make_gate()
feed(s, [0.0] * 22, 0.0)
out = feed(s, [0.72] + [0.0] * 21, 0.0026)
check("T2", out[0] == 0.0 and s._dof_gate_rejected[0] == 1,
      "out=%.3f rejected=%d" % (out[0], s._dof_gate_rejected[0]))

print("T3 진짜 운동(3 rad/s)은 통과시킨다 -- 음성 대조")
s = make_gate()
feed(s, [0.0] * 22, 0.0)
out = feed(s, [3.0 * 0.0026] + [0.0] * 21, 0.0026)
check("T3", abs(out[0] - 0.0078) < 1e-9 and s._dof_gate_rejected[0] == 0)

print("T4 연속 거부에 상한이 있다 -- 옛 값을 영원히 붙들지 않는다")
s = make_gate(consec=3)
feed(s, [0.0] * 22, 0.0)
for k in range(5):
    out = feed(s, [5.0] + [0.0] * 21, 0.0026 * (k + 1))
check("T4", out[0] == 5.0 and s._dof_gate_rejected[0] == 3,
      "consec 상한 3 이후 채택, rejected=%d" % s._dof_gate_rejected[0])

print("T5 rate<=0 이면 게이트가 꺼진다 (음성 대조군용)")
s = make_gate(rate=0.0)
feed(s, [0.0] * 22, 0.0)
out = feed(s, [99.0] + [0.0] * 21, 0.0026)
check("T5", out[0] == 99.0)

print("T5b ⛔ 예산 구멍(분석 세션): 콜백 24 ms 지연 + 0.72 도약")
# 상한 없으면 budget = 30×0.024 = 0.72 → 통과(구멍 실증). 상한 0.20 이면 거부.
s = make_gate(step=0.0)
feed(s, [0.0] * 22, 0.0)
out = feed(s, [0.72] + [0.0] * 21, 0.024)
check("T5b-구멍 실증(상한 OFF 면 통과해 버린다)", out[0] == 0.72)
s = make_gate(step=0.20)
feed(s, [0.0] * 22, 0.0)
out = feed(s, [0.72] + [0.0] * 21, 0.024)
check("T5b-상한이 막는다", out[0] == 0.0 and s._dof_gate_rejected[0] == 1)
# 지연 콜백에서도 정당한 큰 움직임(3 rad/s × 24 ms = 0.072)은 통과
s = make_gate(step=0.20)
feed(s, [0.0] * 22, 0.0)
out = feed(s, [0.072] + [0.0] * 21, 0.024)
check("T5b-음성 대조(지연 틱의 진짜 운동 통과)", out[0] == 0.072)


def replay(path, qcols):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    # step=0: 절대 상한은 재생에서 끈다. 상한은 dt > 6.7 ms 에서만 걸리는데
    # 이 로그의 모든 행이 25 ms 라 -- 실기에서는 예외인 지연 콜백이 재생에서는
    # 전부가 된다. ⛔ 따라서 **재생 테스트는 상한 경로를 전혀 검사하지 않는다.**
    # "재생 전부 통과"를 상한 검증으로 읽지 마라. 상한은 T5b 가 합성으로 검사한다.
    s = make_gate(n=12, step=0.0)
    kept = []
    t = 0.0
    for r in rows:
        t += float(r["tick_dt_s"] or 0.0) or 1e-4
        kept.append(feed(s, [float(r[c]) for c in qcols], t))
    return rows, kept, s


QC = ["q%d" % i for i in range(12)]

# ⛔⛔ 재생 테스트가 무엇을 **못** 하는지 먼저 못 박는다.
#
# 게이트는 LowState 콜백(≈380 Hz, dt 2.6 ms)에서 돈다. 거기서 예산은
# 30 × 0.0026 = **0.078 rad** 이다. 그런데 `realdata/` CSV 는 **정책 주기
# (25 ms)로 내려찍은 것**이라 재생하면 예산이 30 × 0.025 = **0.75 rad** 이 된다.
# 즉 재생은 게이트를 **실제보다 10배 느슨한 조건**에서 돌린다 -- 관측된 도약
# 0.2~0.8 rad 의 대부분이 통과해 버린다. 이것은 게이트의 결함이 아니라
# **우리에게 LowState 율 관절 로그가 없다**는 사실이다(MJC 가 요청한 그 측정).
#
# 그래서 재생으로는 "얼마나 잡나"를 주장하지 않는다. 재생이 **정말로** 검사할
# 수 있는 것은 불변식 둘뿐이고, 그 둘만 본다:
#   T6 통과한 출력에는 예산을 넘는 표본이 남지 않는다 (연속거부 상한 탈출 제외)
#   T7 거부가 발목 roll 두 채널에 **국소적**이다 -- 다른 10채널을 안 건드린다
# 진짜 효과 검증은 LowState 율 로그가 생긴 뒤에 한다. 그 전까지 미완이다.
print("⚠️ 재생은 정책주기(25 ms) 로그라 게이트를 실제(2.6 ms)보다 10배 느슨하게 건다.")
print("   '얼마나 잡나'는 이 테스트로 주장하지 않는다. 불변식 둘만 본다.")

for label, fn in (("서기 184 s", "2026-08-09_t_stand_i3b.csv"),
                  ("보행 run2", "2026-08-09_t_walk_i3b_run2.csv")):
    p = os.path.join(REAL, fn)
    if not os.path.exists(p):
        print("  ⚠️ %s 없음 -- 건너뜀" % fn)
        continue
    rows, kept, s = replay(p, QC)
    # T6 불변식: 출력의 함의속도가 예산을 넘는 표본은 연속거부 상한을 탈출한
    # 것뿐이어야 한다. 그 수는 거부 수보다 많을 수 없다.
    viol = 0
    for i in range(1, len(kept)):
        dt = float(rows[i]["tick_dt_s"] or 0.0) or 1e-4
        for j in range(12):
            if abs(kept[i][j] - kept[i - 1][j]) > 30.0 * dt + 1e-9:
                viol += 1
    rej_roll = int(s._dof_gate_rejected[5] + s._dof_gate_rejected[11])
    rej_other = int(sum(s._dof_gate_rejected)) - rej_roll
    check("T6 [%s] 출력에 남은 예산초과는 연속거부 탈출뿐" % label,
          viol <= int(sum(s._dof_gate_rejected)),
          "잔여 %d ≤ 거부 %d" % (viol, int(sum(s._dof_gate_rejected))))
    check("T7 [%s] 거부가 발목 roll 에 국소적" % label,
          rej_other <= rej_roll,
          "발목 %d / 다른10채널 %d" % (rej_roll, rej_other))


# Wall-clock scheduling regression (2026-08-09): `_low_state_handler` used the
# main thread's `next_inference_time` as its own sampling gate.  The main thread
# advanced that deadline before the callback could observe it, so a nominal
# 50 Hz policy consumed only 22 distinct q snapshots in 50 ticks and sometimes
# held q/dq/tau for 218 ms.  A future deadline must never suppress LowState.
print("T8 LowState producer는 next_inference_time과 무관하게 최신 상태를 갱신한다")


class _Motor:
    def __init__(self, q, dq, tau):
        self.q = q
        self.dq = dq
        self.tau_est = tau


class _Imu:
    rpy = [0.0, 0.0, 0.0]
    gyro = [0.1, 0.2, 0.3]


class _LowState:
    imu_state = _Imu()

    def __init__(self, q):
        self.motor_state_serial = [_Motor(q + i * 0.01, 2.0, 3.0)
                                   for i in range(22)]


c = GoalPoseController.__new__(GoalPoseController)
c._verify_joint_layout = lambda msg: None
c._gate_joint_sample = lambda q: list(q)
c._request_recovery = lambda why: None
c._last_low_state_monotonic = 0.0
c._latest_rpy = np.zeros(3, dtype=np.float32)
c._latest_tilt = 0.0
c._test_tilt_abort_rad = 0.0
c._policy_authorized = True
c.running_policy = True
c.running = True
c.cfg = {"safety": {"fall_tilt_limit_rad": 1.0}}
c.next_inference_time = 1e100       # the old bug suppressed this callback
c._state_lock = threading.Lock()
c.dof_pos_latest = np.zeros(22, dtype=np.float32)
c.projected_gravity = np.zeros(3, dtype=np.float32)
c.base_ang_vel = np.zeros(3, dtype=np.float32)
c.dof_pos = np.zeros(22, dtype=np.float32)
c.dof_vel = np.zeros(22, dtype=np.float32)
c.dof_tau = np.zeros(22, dtype=np.float32)
c._dof_gate_consec = np.zeros(22, dtype=np.int32)
c._low_state_seq = 0
c._low_state_sample_monotonic = 0.0
GoalPoseController._low_state_handler(c, _LowState(0.5))
snap = GoalPoseController._snapshot_policy_state(c)
check("T8", c._low_state_seq == 1 and np.allclose(snap[0][0], 0.5)
      and np.allclose(snap[1][0], 2.0) and np.allclose(snap[2][0], 3.0),
      "seq=%d q0=%.2f dq0=%.2f tau0=%.2f" %
      (c._low_state_seq, snap[0][0], snap[1][0], snap[2][0]))

print("T9 정책 snapshot은 다음 콜백이 와도 원자적으로 고정된다")
GoalPoseController._low_state_handler(c, _LowState(1.0))
check("T9", np.allclose(snap[0][0], 0.5) and np.allclose(c.dof_pos[0], 1.0),
      "snapshot q0=%.2f latest q0=%.2f" % (snap[0][0], c.dof_pos[0]))

print("\n%s  (실패 %d)"
      % ("전부 통과" if not FAILS else "실패: " + ", ".join(FAILS), len(FAILS)))
sys.exit(1 if FAILS else 0)
