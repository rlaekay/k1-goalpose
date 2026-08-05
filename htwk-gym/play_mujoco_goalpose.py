"""Isaac Gym에서 학습한 GoalPose 정책을 MuJoCo에서 돌린다 (sim-to-sim 교차검증).

왜 필요한가
-----------
sim에서 낙상률 0.06 %인데 실기에서 세 걸음이다. 두 가지가 가능하다:

  (a) 정책이 Isaac Gym 특유의 물리(접촉 모델, solver, 관성)를 이용하고 있다
  (b) 갭이 물리가 아니라 다른 데(하드웨어 지연, 영점, 평행 발목)에 있다

**다른 엔진에서 돌려보면 갈린다.** MuJoCo에서도 잘 걸으면 (a)가 배제되고 남은 용의자가
줄어든다. MuJoCo에서 무너지면 Isaac에서만 되는 정책을 학습해 온 것이고, 그건 실기에
올리기 전에 sim 쪽에서 고쳐야 하는 문제다. Booster도 같은 이유로 `play_mujoco.py`를
파이프라인에 두고 Isaac -> MuJoCo -> 실기 순으로 검증한다.

관측을 다시 유도하지 않는다
---------------------------
관측 54칸을 여기서 새로 조립하면 그게 틀렸을 때 "MuJoCo에서 못 걷는다"가 정책 탓인지
내 조립 탓인지 갈리지 않는다. 그래서 **하드웨어에서 이미 검증된**
`deploy/utils/policy_goal_pose.py::GoalPosePolicy`를 그대로 쓴다. 배포와 같은 코드가
같은 체크포인트를 같은 규약으로 읽는다.

자산에 대해 알고 쓰는 것
------------------------
* `K1_serial.xml`은 팔이 **자유 관절**이다(학습 URDF는 팔이 고정). 여기서는 배포와
  똑같이 22관절 전부를 PD로 잡는다 -- 즉 이 실행은 학습 sim보다 **실기에 가깝다.**
* MJCF의 다리 `forcerange`는 **45/30/30/45/20/20**이고 이는 deploy config의
  `common.torque_limit`과 일치한다. 학습이 쓰는 URDF의 `effort`는
  **30/20/20/40/20/15**로 더 낮다. 즉 이 스크립트를 기본값으로 돌리면 정책은
  **학습 때보다 33-50 % 관대한 토크 상한**에서 걷는다. 그것 자체가 실기 조건이므로
  기본값으로 두되, `--torque-limits urdf`로 학습 조건도 잴 수 있게 한다.
  둘의 차이가 크면 "토크 상한이 sim2real 갭"이라는 가설이 그 자리에서 증명된다.

사용:
    python play_mujoco_goalpose.py --duration 180 --video mj.mp4
    python play_mujoco_goalpose.py --torque-limits urdf   # 학습과 같은 상한
"""

import os
import sys
import json
import math
import argparse

import numpy as np

# 헤드리스 서버에서 오프스크린 렌더링. mujoco를 import 하기 전에 정해야 한다.
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import yaml  # noqa: E402

# `deploy/utils/`를 sys.path로 붙이면 안 된다 -- 저장소 루트에도 `utils/` 패키지가
# 있어서(학습 쪽) `utils.policy_goal_pose`가 그쪽으로 해석돼 ModuleNotFoundError가 난다.
# 파일 경로로 직접 적재해서 이름 충돌 자체를 없앤다.
def _load_deploy_policy_class():
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "deploy", "utils", "policy_goal_pose.py")
    spec = importlib.util.spec_from_file_location("_deploy_policy_goal_pose", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GoalPosePolicy


GoalPosePolicy = _load_deploy_policy_class()

MJCF = "resources/K1/K1_serial.xml"
DEPLOY_CFG = "deploy/configs/Goal_Pose_E0.yaml"

# 학습이 실제로 쓰는 상한(K1_locomotion_armsdown.urdf의 effort). 다리 12개, 좌우 동일.
URDF_LEG_EFFORT = [30.0, 20.0, 20.0, 40.0, 20.0, 15.0] * 2


def quat_to_mat(q):
    """MuJoCo qpos[3:7]은 (w, x, y, z)다."""
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def wrap_pi(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


# 학습이 쓰는 K1_locomotion_armsdown.urdf의 발 <inertial>. 링크 프레임 기준 전체 텐서.
# 주모멘트로 풀면 (0.000783, 0.002022, 0.007410)이고 삼각부등식을 2.64배 위반한다 --
# 어떤 실제 강체도 가질 수 없는 값이다. 벤더 MJCF는 같은 발을 (0.000197, 0.000741,
# 0.000790)으로 적어 두었다. 둘 다 벤더가 준 파일이고 서로 2.7-9.4배 다르다.
URDF_FOOT = {
    "mass": "0.3831",
    # MJCF fullinertia 순서: ixx iyy izz ixy ixz iyz
    "fullinertia": "0.00202 0.00741 0.000785 0.0 -0.000053 0.0",
}


def _write_urdf_foot_variant(src):
    """발 관성만 학습 URDF 값으로 바꾼 MJCF를 옆에 쓴다.

    meshdir이 상대경로("meshes/")라 같은 디렉토리에 써야 메시가 풀린다.
    원본은 건드리지 않는다 -- 벤더 자산을 덮어쓰면 다음 사람이 무엇을 보고 있는지
    알 수 없게 된다(3-1이 그렇게 일어났다).
    """
    import re as _re
    txt = open(src, encoding="utf-8").read()
    n = 0

    def sub(m):
        nonlocal n
        n += 1
        return ('<inertial pos="%s" mass="%s" fullinertia="%s"/>'
                % (m.group("pos"), URDF_FOOT["mass"], URDF_FOOT["fullinertia"]))

    # foot_link 바디 안의 inertial만 바꾼다. quat/diaginertia는 fullinertia와
    # 공존할 수 없으므로 통째로 교체한다.
    for side in ("left", "right"):
        pat = (r'(?s)(<body name="%s_foot_link".*?)<inertial\s+pos="(?P<pos>[^"]+)"'
               r'[^/]*?/>' % side)
        txt = _re.sub(pat, lambda m: m.group(1) + sub(m), txt, count=1)
    if n != 2:
        raise SystemExit("발 inertial 치환 실패: %d/2 -- 자산 구조가 바뀌었다" % n)
    dst = os.path.join(os.path.dirname(src), "_k1_serial_urdffoot.xml")
    open(dst, "w", encoding="utf-8").write(txt)
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=180.0, help="초")
    ap.add_argument("--video", default=None, help="mp4 경로 (생략하면 렌더링 안 함)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default=None, help="TorchScript 경로 (기본: deploy config)")
    ap.add_argument("--torque-limits", choices=["mjcf", "urdf"], default="mjcf",
                    help="mjcf=벤더/배포 값(45/30/30/45/20/20), urdf=학습 값(30/20/20/40/20/15)")
    ap.add_argument("--goal-hold", action="store_true",
                    help="목표를 로컬 2 m 앞에 고정한다(forward_hold). 도착하지 않으므로 계속 걷는다")
    ap.add_argument("--stand", action="store_true",
                    help="배포의 도착 상태를 그대로 재현한다: 목표 (0,0,0) + gait_frequency 0. "
                         "이 조건은 학습에 있다 -- goal_categories.stand 10 %, 목표 거리 정확히 0, "
                         "_reward_stand_posture 보상, 그리고 stand 목표는 클럭을 얼린다. "
                         "deploy_goal_pose._update_arrival_gait가 실기에서 같은 상태를 만든다.")
    ap.add_argument("--foot-inertia", choices=["mjcf", "urdf"], default="mjcf",
                    help="발 관성을 어느 자산에서 가져올지. 같은 벤더의 두 파일이 다르다 -- "
                         "MJCF는 주모멘트 (0.000197, 0.000741, 0.000790)로 물리적으로 유효하고, "
                         "학습이 쓰는 URDF는 (0.000783, 0.002022, 0.007410)로 삼각부등식을 "
                         "2.64배 위반한다(2.7-9.4배 과대). 기본 mjcf로 돌리면 MuJoCo는 "
                         "**학습과 다른 발**을 신는다. urdf를 주면 학습과 같아진다 -- "
                         "MuJoCo가 그 텐서를 받아주기만 한다면.")
    # ---- 감지 열화 ---------------------------------------------------------
    # 두 시뮬레이터 모두 obs[0:6](projected_gravity, base_ang_vel)을 **정확히** 준다.
    # 실기에서는 그게 IMU와 상태추정에서 온다. 균형에 가장 중요한 6채널인데 한 번도
    # 열화시켜 본 적이 없다. 강체 동역학은 MuJoCo가 이미 용의선상에서 지웠으므로
    # (엔진/접촉/발관성/팔자유도를 다 바꿔도 낙상 0) 남은 것이 여기다.
    ap.add_argument("--imu-noise-deg", type=float, default=0.0,
                    help="중력벡터에 매 스텝 가우시안 기울기 잡음 (std, 도)")
    ap.add_argument("--imu-bias-deg", type=float, default=0.0,
                    help="중력벡터에 실행 내내 고정된 기울기 바이어스 (도). "
                         "학습의 noise.gravity는 평균 0이라 바이어스를 본 적이 없다")
    ap.add_argument("--gyro-noise", type=float, default=0.0,
                    help="base_ang_vel 가우시안 잡음 (std, rad/s)")
    ap.add_argument("--dofvel-noise", type=float, default=0.0,
                    help="dof_vel 가우시안 잡음 (std, rad/s). 실기의 dof_vel은 "
                         "인코더 미분이라 sim의 0.1보다 훨씬 거칠 수 있다")
    ap.add_argument("--sense-lag-ms", type=float, default=0.0,
                    help="obs[0:6]에 순수 지연 (ms). 학습은 액션 쪽 0-18 ms만 모델링하고 "
                         "관측 쪽 지연은 전혀 없다")
    ap.add_argument("--period-ms", type=float, default=None,
                    help="정책 주기(ms). 기본은 config의 dt*decimation = 20 ms(50 Hz). "
                         "실기 실측은 median 25.3 ms / p99 48 ms였다 -- 학습이 본 적 없는 값이다. "
                         "액션 유지 시간이 길어지면 위상 지연이 생기고, 그것이 고주파 진동을 만든다.")
    ap.add_argument("--no-torque-clamp", action="store_true",
                    help="토크 클램프를 푼다. sim은 URDF effort로 하드 클램프하는데 실기는 "
                         "deploy가 tau=0으로 온보드 PD에 맡겨 막지 않는다. 실기 2.4초에서 "
                         "Hip_Roll이 표본의 7-8%에서 학습 한계 20 N*m을 넘고 max 34였다.")
    # ---- 관절 영점 오차 ----------------------------------------------------
    # 학습에서 randomization.joint_encoder_bias / joint_target_offset은 선언만 돼
    # 있고 [0, 0]으로 꺼져 있다. 배포 후보(stance10)도 그 상태로 학습됐다.
    # 실제 영점 드리프트는 둘 다 일으킨다 -- 정책이 **보는** 값과 PD가 **겨누는**
    # 값이 함께 틀어진다. 그래서 같은 크기로 같이 넣는다(학습의 _jointcal과 동일).
    #
    # 실기 2.4초의 특징: Hip_Roll이 13.5도 벌어지고 안 돌아오며, 토크 p99가 sim의
    # 1.6배인데 **부호전환 주파수는 sim과 같다.** 진동이 아니라 "계속 세게 미는데
    # 안 돌아온다"의 모양이고, 영점이 틀어졌을 때 정확히 그렇게 된다.
    ap.add_argument("--joint-bias-deg", type=float, default=0.0,
                    help="12개 다리 관절 전부에 독립 균일 [-v,+v] 영점 오차 (도). "
                         "encoder(정책이 보는 값)와 target(PD가 겨누는 값)에 같이 넣는다.")
    ap.add_argument("--hiproll-bias-deg", type=float, default=0.0,
                    help="Hip_Roll 두 개에만 영점 오차 (도). 실기 증상이 그 축이다.")
    ap.add_argument("--deploy-filter", action="store_true",
                    help="배포의 관절 목표 저역통과 필터를 재현한다"
                         " (deploy_goal_pose.py:1273, 500 Hz에서 y=0.8y+0.2x)."
                         " 학습 sim에는 이 필터가 없다 -- tau 8.9 ms, fc 17.8 Hz,"
                         " 정책 1 tick(20 ms) 뒤 도달률 89.3 %.")
    ap.add_argument("--ankle-gain", type=float, default=1.0,
                    help="발목 4관절의 kp/kd를 이 배수로. 실기 대조에서 발목 pitch 토크가 "
                         "sim의 0.6-0.7배였다(궤적은 1.3배). 원인 미상이므로 결과만 흔든다.")
    ap.add_argument("--ankle-damp", type=float, default=1.0,
                    help="발목 4관절의 **kd만** 이 배수로. 실기 대조에서 발목 roll의 "
                         "관절속도 rms가 sim의 2.9-4.5배인데 토크는 1.1-1.3배로 정상이다. "
                         "속도는 높고 저항은 없다 = 감쇠 부족의 모양이고, kp/kd를 함께 "
                         "흔드는 --ankle-gain으로는 이 축을 분리할 수 없다.")
    ap.add_argument("--ankle-roll-damp", type=float, default=1.0,
                    help="Ankle_Roll 두 관절만 kd 배수. 신호가 roll에 편중돼 있다.")
    ap.add_argument("--hip-roll-gain", type=float, default=1.0,
                    help="Hip_Roll 두 관절의 kp/kd 배수.")
    ap.add_argument("--lat-friction", type=float, default=None,
                    help="지면 마찰계수(기본 MuJoCo 1.0). 횡방향 미끄러짐 가설용.")
    ap.add_argument("--dump-csv", default=None,
                    help="실기 deploy --log-timing과 **동일한 컬럼**으로 매 정책 tick을 "
                         "남긴다. 가설 없이 두 로그를 같은 축에 겹쳐 보기 위한 것이다.")
    ap.add_argument("--out", default="logs/mujoco/result.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cfg = yaml.safe_load(open(DEPLOY_CFG, encoding="utf-8"))
    if args.policy:
        cfg["policy"]["policy_path"] = args.policy
    # 배포 설정은 로봇 위의 상대 경로를 담고 있다. 서버에서는 저장소 기준으로 푼다.
    pp = cfg["policy"]["policy_path"]
    if not os.path.isabs(pp):
        cand = os.path.join("deploy", pp.lstrip("./"))
        cfg["policy"]["policy_path"] = cand if os.path.exists(cand) else pp
    print("정책: %s" % cfg["policy"]["policy_path"])

    policy = GoalPosePolicy(cfg)          # 하드웨어에서 검증된 관측 규약 그대로
    if args.stand:
        # 배포가 도착에서 하는 것과 같다(deploy_goal_pose._update_arrival_gait).
        # 클럭을 얼리지 않으면 정책은 제자리 행진을 한다 -- 학습에서 stand 목표는
        # gait_frequency가 0이고, 0이 아닌 클럭은 feet_swing의 스텝 유인 위에 앉는다.
        policy.gait_frequency = 0.0
        print("stand 모드: gait_frequency = 0 (배포의 도착 상태와 동일)")
    dt = float(cfg["common"]["dt"])       # 0.002
    decim = int(cfg["policy"]["control"]["decimation"])  # 10 -> 50 Hz
    kp = np.array(cfg["common"]["stiffness"], dtype=np.float64)
    kd = np.array(cfg["common"]["damping"], dtype=np.float64)
    # 관절 인덱스: 다리 10..21 = [HipP,HipR,HipY,Knee,AnkP,AnkR] x 2
    if args.ankle_gain != 1.0:
        for i in (14, 15, 20, 21):
            kp[i] *= args.ankle_gain; kd[i] *= args.ankle_gain
        print("발목 이득 x%.2f" % args.ankle_gain)
    if args.ankle_damp != 1.0:
        for i in (14, 15, 20, 21):
            kd[i] *= args.ankle_damp
        print("발목 감쇠 x%.2f" % args.ankle_damp)
    if args.ankle_roll_damp != 1.0:
        for i in (15, 21):
            kd[i] *= args.ankle_roll_damp
        print("발목 roll 감쇠 x%.2f" % args.ankle_roll_damp)
    if args.hip_roll_gain != 1.0:
        for i in (11, 17):
            kp[i] *= args.hip_roll_gain; kd[i] *= args.hip_roll_gain
        print("Hip_Roll 이득 x%.2f" % args.hip_roll_gain)
    default_q = np.array(cfg["common"]["default_qpos"], dtype=np.float64)
    nj = default_q.size                   # 22

    mjcf_path = MJCF
    if args.foot_inertia == "urdf":
        mjcf_path = _write_urdf_foot_variant(MJCF)
        print("발 관성: 학습 URDF 값을 강제 (%s)" % mjcf_path)
    else:
        print("발 관성: 벤더 MJCF 값 (학습 URDF와 다르다 -- 2.7-9.4배 작고 물리적으로 유효)")

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    model.opt.timestep = dt
    if args.lat_friction is not None:
        model.geom_friction[:, 0] = args.lat_friction
        print("지면/전체 마찰 -> %.2f" % args.lat_friction)
    data = mujoco.MjData(model)

    assert model.nu == nj, "actuator %d != joints %d" % (model.nu, nj)

    # 토크 상한. MJCF는 이미 벤더 값을 담고 있으므로 mjcf 모드에서는 건드리지 않는다.
    lim = model.actuator_forcerange[:, 1].copy()
    if args.no_torque_clamp:
        model.actuator_forcerange[:, 0] = -1e4
        model.actuator_forcerange[:, 1] = 1e4
        print('토크 클램프 해제: MuJoCo actuator 한계도 함께 품')
    if args.torque_limits == "urdf":
        lim[10:22] = URDF_LEG_EFFORT
        print("토크 상한: 학습(URDF effort) %s" % lim[10:16])
    else:
        print("토크 상한: 벤더 MJCF/배포 %s" % lim[10:16])

    # 초기 자세: 기본 관절각으로 세우고, 발이 지면에 닿을 높이에서 떨어뜨린다.
    data.qpos[:] = 0.0
    data.qpos[2] = 0.60
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:7 + nj] = default_q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    renderer = None
    frames = []
    if args.video:
        renderer = mujoco.Renderer(model, height=480, width=640)
    frame_every = max(1, int(round(1.0 / (args.fps * dt))))

    # ---- 목표 관리 ---------------------------------------------------------
    gx, gy = float(cfg["policy"]["goal_clamp"]["x_m"]), float(cfg["policy"]["goal_clamp"]["y_m"])

    def new_goal(px, py, yaw):
        """학습 범위 안에서 로컬 목표를 뽑아 월드로 옮긴다."""
        lx = rng.uniform(-gx, gx)
        ly = rng.uniform(-gy, gy)
        lh = rng.uniform(-np.pi, np.pi)
        c, s = math.cos(yaw), math.sin(yaw)
        return px + c * lx - s * ly, py + s * lx + c * ly, wrap_pi(yaw + lh)

    yaw0 = 0.0
    goal = new_goal(0.0, 0.0, yaw0)
    seg_t0 = 0.0
    arrivals, falls, segments = [], 0, 0
    fall_tilt = float(cfg["safety"]["fall_tilt_limit_rad"])
    stop_r = float(cfg["deploy_goal"]["stop_radius_m"])
    stop_h = float(cfg["deploy_goal"]["stop_heading_rad"])

    # 감지 열화 상태. 바이어스는 실행 내내 고정된 축을 중심으로 한 기울기다 --
    # IMU 정렬 오차나 추정기 드리프트가 그 모양이고, 학습의 noise.gravity는 평균 0의
    # 매 스텝 잡음이라 이 형태를 한 번도 본 적이 없다.
    bias_axis = rng.normal(size=3)
    bias_axis /= np.linalg.norm(bias_axis)
    bias_rad = math.radians(args.imu_bias_deg)
    lag_steps = int(round(args.sense_lag_ms / 1000.0 / dt))
    sense_buf = []          # (proj_g, ang_vel) 물리 스텝마다 append

    def tilt(v, rad, axis):
        """v를 axis 둘레로 rad만큼 회전 (Rodrigues)."""
        if rad == 0.0:
            return v
        k = axis
        return (v * math.cos(rad) + np.cross(k, v) * math.sin(rad)
                + k * np.dot(k, v) * (1 - math.cos(rad)))

    # 영점 오차: 다리 12관절에 대해 한 번 뽑아 실행 내내 고정한다(드리프트는 상수다).
    joint_bias = np.zeros(nj)
    if args.joint_bias_deg > 0:
        b = math.radians(args.joint_bias_deg)
        joint_bias[10:22] = rng.uniform(-b, b, 12)
    if args.hiproll_bias_deg != 0.0:
        b = math.radians(args.hiproll_bias_deg)
        joint_bias[11] = b           # Left_Hip_Roll
        joint_bias[17] = b           # Right_Hip_Roll
    if np.any(joint_bias):
        print("관절 영점 오차(도): %s" % np.round(np.degrees(joint_bias[10:22]), 2))

    period_s = (args.period_ms / 1000.0) if args.period_ms else (dt * decim)
    next_infer = 0.0
    # 실기와 같은 지표를 낸다: 관절별 |tau| 분위수와 부호전환 횟수.
    tau_hist = [[] for _ in range(nj)]
    hiproll_hist = []
    dump_fp = None
    if args.dump_csv:
        os.makedirs(os.path.dirname(args.dump_csv) or ".", exist_ok=True)
        dump_fp = open(args.dump_csv, "w", buffering=1)
        dump_fp.write(",".join(
            ["t_s", "low_state_age_s", "tick_dt_s", "tilt_deg", "roll", "pitch",
             "gx", "gy", "gz", "walking", "gait_freq", "gait_process",
             "goal_x", "goal_y", "heading_err"]
            + ["q%d" % i for i in range(12)] + ["dq%d" % i for i in range(12)]
            + ["tau%d" % i for i in range(12)] + ["act%d" % i for i in range(12)]) + "\n")
    tau_prev = np.zeros(nj)
    applied_prev = np.zeros(nj)
    tilt_rad_prev = 0.0
    tau_flips = np.zeros(nj, dtype=int)
    nsteps = int(args.duration / dt)
    targets = np.copy(default_q)
    # 배포는 정책의 목표를 그대로 보내지 않는다. 500 Hz 발행 루프가
    # filtered = 0.8*filtered + 0.2*target 로 걸러서 보낸다(deploy_goal_pose.py:1273).
    # 학습 sim에는 이 필터가 없으므로, 이 플래그가 켜진 실행과 꺼진 실행의 차이가
    # 곧 "학습이 모르는 배포 지연"의 비용이다.
    filtered = np.copy(default_q)
    stand_tilt, stand_drift = [], []
    px0 = py0 = 0.0          # stand 모드의 기준점: 첫 스텝의 위치
    t = 0.0

    for it in range(nsteps):
        q = data.qpos[7:7 + nj].copy()
        dq = data.qvel[6:6 + nj].copy()
        R = quat_to_mat(data.qpos[3:7])
        proj_g = R.T @ np.array([0.0, 0.0, -1.0])
        ang_vel = data.qvel[3:6].copy()           # free joint: 각속도는 body frame
        px, py = data.qpos[0], data.qpos[1]
        yaw = math.atan2(R[1, 0], R[0, 0])
        if it == 0:
            px0, py0 = px, py

        # 감지 열화는 정책이 읽는 값에만 건다. 물리와 낙상 판정은 참값을 쓴다 --
        # 그래야 "정책이 속아서 넘어졌다"와 "실제로 기울었다"가 섞이지 않는다.
        sense_buf.append((proj_g.copy(), ang_vel.copy()))
        if len(sense_buf) > lag_steps + 1:
            sense_buf.pop(0)
        s_g, s_w = sense_buf[0] if lag_steps > 0 else (proj_g, ang_vel)
        obs_g = tilt(s_g, bias_rad, bias_axis)
        if args.imu_noise_deg > 0.0:
            n_ax = rng.normal(size=3)
            n_ax /= max(np.linalg.norm(n_ax), 1e-9)
            obs_g = tilt(obs_g, math.radians(rng.normal(0.0, args.imu_noise_deg)), n_ax)
        obs_w = s_w + (rng.normal(0.0, args.gyro_noise, 3) if args.gyro_noise > 0 else 0.0)
        obs_dq = dq + (rng.normal(0.0, args.dofvel_noise, nj) if args.dofvel_noise > 0 else 0.0)

        if t >= next_infer - 1e-9:
            next_infer = t + period_s
            if args.stand:
                # 배포의 도착 상태. _update_arrival_gait가 stop_radius 안에서
                # gait_frequency를 0으로 내리고, 정책은 그 조건을 학습에서 봤다.
                grx, gry, herr = 0.0, 0.0, 0.0
            elif args.goal_hold:
                grx, gry, herr = 2.0, 0.0, 0.0    # 로컬 2 m 앞 고정 -> 도달하지 않는다
            else:
                dx, dy = goal[0] - px, goal[1] - py
                c, s = math.cos(-yaw), math.sin(-yaw)
                grx, gry = c * dx - s * dy, s * dx + c * dy
                herr = wrap_pi(goal[2] - yaw)
            targets = policy.inference(
                t, (q + joint_bias).astype(np.float32), obs_dq.astype(np.float32),
                obs_w.astype(np.float32), obs_g.astype(np.float32),
                grx, gry, herr)
            if dump_fp is not None:
                rr = math.atan2(R[2, 1], R[2, 2]); pp = math.asin(-max(-1.0, min(1.0, R[2, 0])))
                head = [t, 0.001, period_s, math.degrees(tilt_rad_prev), rr, pp,
                        ang_vel[0], ang_vel[1], ang_vel[2],
                        1.0, policy.gait_frequency, policy.gait_process, grx, gry, herr]
                body = (list(q[10:22]) + list(dq[10:22])
                        + list(applied_prev[10:22]) + list(policy.actions[:12]))
                dump_fp.write(",".join("%.5g" % float(v) for v in head + body) + "\n")

        # 필터는 정책 tick이 아니라 발행 루프(=물리 스텝)마다 돈다. 배포가 그렇다.
        if args.deploy_filter:
            filtered[:] = filtered * 0.8 + targets * 0.2
            cmd_q = filtered
        else:
            cmd_q = targets
        tau = kp * ((cmd_q + joint_bias) - q) - kd * dq
        applied = tau if args.no_torque_clamp else np.clip(tau, -lim, lim)
        # MuJoCo의 actuator forcerange가 여전히 자르므로, 클램프를 정말 풀려면
        # 모델 쪽 한계도 같이 올려야 한다. 아래 model.actuator_forcerange에서 처리.
        data.ctrl[:] = applied
        applied_prev = applied.copy()
        hiproll_hist.append((float(q[11]), float(q[17])))
        for k in range(nj):
            a = float(applied[k])
            tau_hist[k].append(abs(a))
            if a * tau_prev[k] < 0:
                tau_flips[k] += 1
            tau_prev[k] = a
        mujoco.mj_step(model, data)
        t += dt

        # 낙상: 배포와 같은 판정(중력 벡터와 직립 사이 각). raw roll/pitch가 아니다.
        # 낙상 판정은 **참값** proj_g로 한다. 정책이 속은 것과 실제로 기운 것을
        # 섞지 않기 위해서다. 이름을 tilt로 쓰면 위의 tilt() 함수를 덮어쓴다.
        tilt_rad = math.acos(np.clip(-proj_g[2], -1.0, 1.0))
        tilt_rad_prev = tilt_rad
        fallen = tilt_rad > fall_tilt
        if args.stand:
            # 서 있는 과제에는 도착도 구간도 없다. 재는 것은 두 가지다 --
            # 넘어지는가, 그리고 제자리에 있는가(표류).
            stand_tilt.append(math.degrees(tilt_rad))
            stand_drift.append(math.hypot(px - px0, py - py0))
            if fallen:
                falls += 1
        elif not args.goal_hold:
            dist = math.hypot(goal[0] - px, goal[1] - py)
            reached = dist < stop_r and abs(wrap_pi(goal[2] - yaw)) < stop_h
            timeout = (t - seg_t0) > 8.0
            if reached or timeout or fallen:
                segments += 1
                if fallen:
                    falls += 1
                else:
                    arrivals.append((dist, abs(wrap_pi(goal[2] - yaw))))
                goal = new_goal(px, py, yaw)
                seg_t0 = t
        elif fallen:
            falls += 1

        if fallen:
            # 넘어지면 다시 세운다. 그래야 3분 영상이 첫 낙상에서 끝나지 않는다.
            data.qpos[:] = 0.0
            data.qpos[2] = 0.60
            data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            data.qpos[7:7 + nj] = default_q
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            # 필터도 같이 리셋한다. 안 하면 리셋된 로봇이 넘어지기 직전의 목표를
            # 물려받아, 스폰 직후 몇 십 ms를 엉뚱한 자세로 시작한다.
            filtered[:] = default_q
            targets = np.copy(default_q)

        if renderer is not None and it % frame_every == 0:
            cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(cam)
            cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.35]
            cam.distance, cam.azimuth, cam.elevation = 3.0, 130.0, -15.0
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())

    # ---- 결과 --------------------------------------------------------------
    res = {
        "policy": cfg["policy"]["policy_path"],
        "torque_limits": args.torque_limits,
        "foot_inertia": args.foot_inertia,
        "deploy_filter": bool(args.deploy_filter),
        "sense": {
            "imu_noise_deg": args.imu_noise_deg,
            "imu_bias_deg": args.imu_bias_deg,
            "gyro_noise": args.gyro_noise,
            "dofvel_noise": args.dofvel_noise,
            "sense_lag_ms": args.sense_lag_ms,
        },
        "duration_s": args.duration,
        "mode": "goal_hold" if args.goal_hold else "goal_reach",
        "segments": segments,
        "falls": falls,
        # goal_hold에는 구간이 없다. 연속보행에서는 분당 낙상이 비교 단위다.
        "falls_per_min": round(falls / (args.duration / 60.0), 3),
        "period_ms": round(period_s * 1000.0, 2),
        "torque_clamped": not args.no_torque_clamp,
    }
    LEGN = ["L_HipP","L_HipR","L_HipY","L_Knee","L_AnkP","L_AnkR",
            "R_HipP","R_HipR","R_HipY","R_Knee","R_AnkP","R_AnkR"]
    URDF_LIM = [30,20,20,40,20,15]*2
    tq = {}
    for k in range(12):
        v = tau_hist[10 + k]
        if not v: continue
        a = np.array(v)
        tq[LEGN[k]] = {
            "p50": round(float(np.percentile(a,50)),2),
            "p99": round(float(np.percentile(a,99)),2),
            "max": round(float(a.max()),2),
            "over_limit_pct": round(100.0*float((a > URDF_LIM[k]).mean()),1),
            "flips_per_s": round(tau_flips[10+k]/max(args.duration,1e-9),1),
        }
    res["torque"] = tq
    if hiproll_hist:
        hl = np.array([h[0] for h in hiproll_hist]); hr = np.array([h[1] for h in hiproll_hist])
        res["hip_roll_deg"] = {
            "L_median": round(float(np.degrees(np.median(hl))), 2),
            "R_median": round(float(np.degrees(np.median(hr))), 2),
            "L_range": round(float(np.degrees(hl.max() - hl.min())), 1),
            "R_range": round(float(np.degrees(hr.max() - hr.min())), 1)}
    if args.stand:
        res["mode"] = "stand"
        res.pop("segments", None)
        if stand_tilt:
            res["tilt_deg"] = {"median": float(np.median(stand_tilt)),
                               "p99": float(np.percentile(stand_tilt, 99)),
                               "max": float(np.max(stand_tilt))}
            res["drift_m"] = {"median": float(np.median(stand_drift)),
                              "final": float(stand_drift[-1]),
                              "max": float(np.max(stand_drift))}
    if arrivals:
        d = np.array([a[0] for a in arrivals])
        h = np.array([a[1] for a in arrivals])
        te = np.hypot(d, h) * 100.0
        res.update({
            "arrivals": len(arrivals),
            "pos_err_cm_median": float(np.median(d) * 100),
            "heading_err_deg_median": float(np.degrees(np.median(h))),
            "task_err_cm_median": float(np.median(te)),
        })
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2)
    print(json.dumps(res, indent=2, ensure_ascii=False))

    if frames:
        import imageio
        imageio.mimsave(args.video, frames, fps=args.fps, macro_block_size=1)
        print("영상: %s (%d 프레임)" % (args.video, len(frames)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
