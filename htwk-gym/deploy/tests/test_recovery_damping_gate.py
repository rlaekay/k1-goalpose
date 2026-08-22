"""damping 단계의 **IS_READY 오독** 회귀 테스트 (2026-08-23).

⛔ 무엇을 막는가

custom 플래너로 넘어지면 펌웨어가 `HAS_FALLEN` 을 아예 보고하지 않는다.
INHA-Player 팀 실측(2026-08-22): **tilt 86도로 누운 채 rs=IS_READY / planner=WALKING**
이 끝까지 유지됐다. 우리 `_step_recovery` 의 damping 분기는 `state == IS_READY`
**하나만** 보고 GetUp 을 건너뛰었으므로, 그 상태에서:

    damping -> (IS_READY 니까 GetUp 생략) -> getup 단계
    getup 단계의 upright(20도) 조건이 영원히 거짓 -> 20 s 타임아웃 -> running=False

즉 **GetUp 을 한 발도 안 쏘고** 끝난다. 2026-08-07 사고(IS_READY 오독)와 같은 부류다.

⚠️ 구멍이 둘이었다. tilt 교차검증만 넣으면 아래로 떨어지는데, 거기 `is_recovery_available`
게이트가 또 막는다 -- 펌웨어가 "낙상 아님"으로 보고 있으니 그 플래그는 정의상 False 라
10 s 뒤 종료로 끝나고 GetUp 은 역시 안 나간다. 그래서 T2 가 따로 있다.

**음성 대조 필수**: 수정을 되돌리면 T1/T2 가 실패해야 한다(아래 주석에 되돌리는 법).

실행: `python tests/test_recovery_damping_gate.py` (deploy/ 에서, numpy 만 필요)
"""
import sys, os, types, threading, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEPLOY = os.path.dirname(HERE)


# ---- SDK 스텁 (test_recovery_reentry.py 와 같은 방식) -----------------------
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

    # ⚠️ 호출 가능해야 한다. 코드가 `B1LocoApiId(API_ID_GET_UP)` 로 raw id 를 감싼다
    # (deploy_goal_pose.py:1609,1848). 호출 불가 스텁이면 TypeError 가 나고 코드는
    # 그것을 "GetUp call failed" 로 잡아 running=False 로 간다 -- 그러면 **수정하지
    # 않은 경로(T4)까지 실패**해서 테스트가 무엇을 재는지 알 수 없게 된다.
    def __call__(self, value=None):
        return value


sdk = types.ModuleType("booster_robotics_sdk_python")
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
UPRIGHT_LIM = 0.35          # rad = 20.05도. config 의 getup_upright_tilt_rad 와 같은 값
TILT_FALLEN = np.radians(86.0)   # 실측 지문 (2026-08-22)
TILT_STANDING = np.radians(1.4)  # 실측 서기 median


class FakeFallMonitor:
    """`snapshot()` 이 (state, is_recovery_available, age) 를 준다."""

    def __init__(self, state, recov_ok, age=0.1):
        self.available = True
        self._snap = (state, recov_ok, age)

    def snapshot(self):
        return self._snap


def make_ctl(tilt, state, recov_ok, elapsed_s=2.0):
    """damping 단계 한복판에 있는 컨트롤러를 만든다.

    `elapsed_s` 는 damping_settle_s(1.5) 보다 크게 둬서 판정이 실제로 돌게 한다.
    """
    c = D.Controller.__new__(D.Controller)
    import logging
    c.logger = logging.getLogger("t_damp"); c.logger.setLevel(logging.CRITICAL)
    c.joint_cnt = JN
    c.cfg = {"safety": {"recovery": {
        "damping_settle_s": 1.5,
        "recovery_wait_timeout_s": 10.0,
        "getup_timeout_s": 20.0,
        "getup_min_wait_s": 8.0,
        "getup_upright_tilt_rad": UPRIGHT_LIM,
    }}}
    c.running = True
    c.running_policy = False
    c._recovery_phase = "damping"
    c._recovery_t0 = time.monotonic() - elapsed_s
    c._latest_tilt = float(tilt)
    c._getup_called = False
    c._legs_quiet_since = 0.0
    c.fall_monitor = FakeFallMonitor(state, recov_ok)
    c.publish_lock = threading.Lock()

    # GetUp 호출을 잡는다. 이것이 이 테스트의 종점이다.
    c._getup_calls = []
    c.client = types.SimpleNamespace(
        ChangeMode=lambda m: 0,
        SendApiRequest=lambda api_id, payload: c._getup_calls.append(api_id) or 0,
    )
    return c


FAILED = []


def check(name, ok, detail=""):
    print(("  PASS  " if ok else "  FAIL  ") + name + (("   " + detail) if detail else ""))
    if not ok:
        FAILED.append(name)


print("=" * 72)
print("damping 단계 IS_READY 오독 — custom 낙상에서 GetUp 이 나가는가")
print("=" * 72)

# ── T1: 누웠는데 IS_READY (2026-08-22 실측 지문) → GetUp 이 나가야 한다 ────────
# 되돌리기(음성 대조): damping 분기의 `and really_upright` 를 지우면 여기서 실패한다.
c = make_ctl(tilt=TILT_FALLEN, state=D.FallState.IS_READY, recov_ok=True)
c._step_recovery()
check("T1 누운 IS_READY(tilt 86도)에서 GetUp 을 쏜다",
      c._getup_calls == [D.API_ID_GET_UP],
      f"calls={c._getup_calls} phase={c._recovery_phase}")
check("T1b 그 경로는 _getup_called=True 로 표시된다 (8s 최소대기가 살아야 한다)",
      c._getup_called is True, f"_getup_called={c._getup_called}")

# ── T2: 같은 상황 + available=False → 그래도 GetUp 이 나가야 한다 ─────────────
# 펌웨어가 "낙상 아님"으로 보므로 available 은 정의상 False 다. 이 게이트에 막히면
# 10 s 뒤 종료로 끝나고 구멍이 그대로 남는다.
# 되돌리기(음성 대조): `and not fallen_but_ready` 를 지우면 여기서 실패한다.
c = make_ctl(tilt=TILT_FALLEN, state=D.FallState.IS_READY, recov_ok=False)
c._step_recovery()
check("T2 available=False 여도 GetUp 을 쏜다 (게이트가 막지 않는다)",
      c._getup_calls == [D.API_ID_GET_UP],
      f"calls={c._getup_calls} phase={c._recovery_phase} running={c.running}")
check("T2b 프로세스를 죽이지 않는다",
      c.running is True, f"running={c.running}")

# ── T3: 진짜로 서 있는 IS_READY → GetUp 을 쏘지 않는다 (기존 동작 보존) ───────
# 2026-08-07 사고의 반대 방향. 이 갈래가 죽으면 서 있는 로봇에 GetUp 을 쏘게 된다.
c = make_ctl(tilt=TILT_STANDING, state=D.FallState.IS_READY, recov_ok=False)
c._step_recovery()
check("T3 진짜 서 있는 IS_READY 는 GetUp 을 건너뛴다",
      c._getup_calls == [] and c._recovery_phase == "getup",
      f"calls={c._getup_calls} phase={c._recovery_phase}")
check("T3b 건너뛴 경로는 _getup_called=False (8s 최소대기를 생략한다)",
      c._getup_called is False, f"_getup_called={c._getup_called}")

# ── T4: 정상 HAS_FALLEN 경로가 그대로 산다 ────────────────────────────────────
c = make_ctl(tilt=TILT_FALLEN, state=D.FallState.HAS_FALLEN, recov_ok=True)
c._step_recovery()
check("T4 HAS_FALLEN + available=True 는 종전대로 GetUp",
      c._getup_calls == [D.API_ID_GET_UP] and c._recovery_phase == "getup",
      f"calls={c._getup_calls} phase={c._recovery_phase}")

# ── T5: HAS_FALLEN + available=False 는 종전대로 기다린다 (동작 변경 없음) ────
# 이 갈래까지 바꾸면 검증된 경로를 건드리는 것이다. 최소 수정임을 고정한다.
c = make_ctl(tilt=TILT_FALLEN, state=D.FallState.HAS_FALLEN, recov_ok=False,
             elapsed_s=2.0)
c._step_recovery()
check("T5 HAS_FALLEN + available=False 는 아직 기다린다 (10s 전)",
      c._getup_calls == [] and c.running is True and c._recovery_phase == "damping",
      f"calls={c._getup_calls} phase={c._recovery_phase} running={c.running}")

# ── T6: 경계값 — tilt 가 판정선 바로 아래/위 ──────────────────────────────────
c = make_ctl(tilt=UPRIGHT_LIM - 0.01, state=D.FallState.IS_READY, recov_ok=False)
c._step_recovery()
below = (c._getup_calls == [])
c = make_ctl(tilt=UPRIGHT_LIM + 0.01, state=D.FallState.IS_READY, recov_ok=False)
c._step_recovery()
above = (c._getup_calls == [D.API_ID_GET_UP])
check("T6 판정선(20도) 양쪽에서 갈린다", below and above,
      f"below_skips={below} above_fires={above}")

# ── T7: tilt 가 NaN 이면 안전한 쪽(GetUp 발사)으로 간다 ───────────────────────
# `np.isfinite` 가 False 면 really_upright=False 이므로 발사된다. 누웠는데 안 쏘는
# 것보다 서 있는데 쏘는 쪽이 안전하다(GetUp 은 실패해도 해가 없다).
c = make_ctl(tilt=float("nan"), state=D.FallState.IS_READY, recov_ok=False)
c._step_recovery()
check("T7 tilt=NaN 이면 GetUp 을 쏜다 (안전 쪽으로 실패)",
      c._getup_calls == [D.API_ID_GET_UP], f"calls={c._getup_calls}")

print("-" * 72)
if FAILED:
    print(f"FAILED {len(FAILED)}: " + ", ".join(FAILED))
    sys.exit(1)
print("모두 통과")
