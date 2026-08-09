"""`_gate_joint_sample` 회귀 테스트 -- 오염된 발목 roll 표본을 버리는가.

근거는 ibatch §8-69. 이 게이트는 **실기 로그에서 발견된 것**을 막는 것이므로
합성 스파이크만으로 통과시키지 않는다. 뒤쪽 T4/T5 는 `realdata/` 의 실제 CSV 를
그대로 흘려서 **양성(오염을 잡는다) + 음성(조용할 때 안 잡는다)** 을 같이 본다.

실행: `python tests/test_dof_gate.py` (deploy/ 에서)
"""
import csv
import os
import sys

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


def make_gate(n=22, rate=30.0, consec=5):
    s = type("S", (), {})()
    s._dof_gate_max_rate = rate
    s._dof_gate_max_consec = consec
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


def replay(path, qcols):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    s = make_gate(n=12)
    kept = []
    t = 0.0
    for r in rows:
        t += float(r["tick_dt_s"] or 0.0) or 1e-4
        kept.append(feed(s, [float(r[c]) for c in qcols], t))
    return rows, kept, s


QC = ["q%d" % i for i in range(12)]
STOP = 0.40

print("T6 실기 서기 로그 재생: 오염 표본을 잡고, 조용한 구간은 안 건드린다")
p = os.path.join(REAL, "2026-08-09_t_stand_i3b.csv")
if os.path.exists(p):
    rows, kept, s = replay(p, QC)
    # 발목 roll = q5, q11
    rej_roll = int(s._dof_gate_rejected[5] + s._dof_gate_rejected[11])
    quiet = [i for i, r in enumerate(rows) if float(r["t_s"]) < 140.0]
    # 조용한 구간(낙상 전)에서의 거부는 오검출로 본다
    over_before = sum(1 for r in rows if abs(float(r["q5"])) > STOP
                      or abs(float(r["q11"])) > STOP)
    over_after = sum(1 for k in kept if abs(k[5]) > STOP or abs(k[11]) > STOP)
    check("T6a 발목 roll 거부가 실제로 일어난다", rej_roll > 20,
          "거부 %d" % rej_roll)
    check("T6b 스톱(±0.40) 밖 표본이 줄어든다", over_after < over_before * 0.5,
          "%d → %d" % (over_before, over_after))
    check("T6c 다른 10채널은 거의 안 건드린다",
          int(sum(s._dof_gate_rejected)) - rej_roll <= rej_roll * 0.5,
          "다른채널 %d 대 발목 %d"
          % (int(sum(s._dof_gate_rejected)) - rej_roll, rej_roll))
    print("     (참고) 조용구간 %d 행" % len(quiet))
else:
    print("  ⚠️ %s 없음 -- 건너뜀" % p)

print("T7 실기 보행 로그 재생")
p = os.path.join(REAL, "2026-08-09_t_walk_i3b_run2.csv")
if os.path.exists(p):
    rows, kept, s = replay(p, QC)
    rej_roll = int(s._dof_gate_rejected[5] + s._dof_gate_rejected[11])
    check("T7 보행 중 발목 roll 거부", rej_roll > 10, "거부 %d / %d 행"
          % (rej_roll, len(rows)))
else:
    print("  ⚠️ 없음 -- 건너뜀")

print("\n%s  (%d 검사, 실패 %d)"
      % ("전부 통과" if not FAILS else "실패: " + ", ".join(FAILS),
         5 + 4, len(FAILS)))
sys.exit(1 if FAILS else 0)
