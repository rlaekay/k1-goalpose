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

# ---- armature (기어박스 뒤 로터 관성) ------------------------------------
# ⛔ 세 자산이 서로 다르고, **어느 것도 벤더 값이 아니었다.**
#   Isaac `Goal_Pose_V7.yaml`  : asset.armature 0.0  -> 전 관절 0
#   MuJoCo `K1_serial.xml`     : 발목 0.05, **힙·무릎은 속성 없음 = 0**
#   벤더 공식(booster_train)   : 무릎 0.0956 이 최대값이다
# 즉 `--leg-armature 0.05` 같은 스칼라로는 벤더 분포를 만들 수 없고, MJCF 기본값은
# "armature 를 켰다"가 아니라 **"발목에만 켰다"** 이다.
# 출처: VENDOR_ACTUATOR_SPEC_K1.md §1 (booster_train actuator.py / booster.py).
# 발목은 평행기구 래퍼가 armature_ratio 2.0 을 곱한 뒤의 값이다.
VENDOR_ARMATURE = {
    "Hip_Pitch": 0.047813, "Hip_Roll": 0.033955, "Hip_Yaw": 0.028253,
    "Knee_Pitch": 0.095625, "Ankle_Pitch": 0.056506, "Ankle_Roll": 0.056506,
}
# ⚠️ 벤더는 게인을 armature 에서 **유도**한다: kp = armature*(2*pi*4Hz)^2,
# kd = 2*zeta*armature*(2*pi*4Hz), zeta 1.5(무릎만 1.0). armature 만 옮기고 kp 를
# 그대로 두면 벤더 관점에서 짝이 안 맞는 조합이다 -- `--vendor-gains` 로 같이 옮긴다.
VENDOR_FN_HZ = 4.0
VENDOR_ZETA = {"Knee_Pitch": 1.0}
VENDOR_ZETA_DEFAULT = 1.5


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
                         "실기 실측은 median 25.24 / **mean 26.14** ms다. "
                         "⛔ 케이던스를 맞추려면 mean 을 써라 -- 위상은 적분량이라 "
                         "창 전체 케이던스 = 평균율 x 증분이고, median 은 오른쪽 꼬리를 "
                         "버려서 케이던스를 과대평가한다(median 25.24 -> 1.585 Hz, "
                         "실측 케이던스는 1.512 Hz).")
    # ---- 시간 팽창 ---------------------------------------------------------
    # `deploy/utils/timer.py` 의 Timer 는 벽시계가 아니라 **LowState 콜백에서만
    # 증가하는 카운터**다(증가 지점은 deploy_goal_pose.py:871 하나뿐). 콜백이
    # 설계 500 Hz 보다 느리게 실행되면 정책이 보는 시간축 전체가 같이 느려진다.
    #
    # 실기 로그(realdata/2026-08-0x_real_walk_i3b.csv, 91틱)가 이 기전의 지문을
    # 그대로 담고 있다 -- gait_process 증분 90개가 **전부 0.004 의 정수배**이고
    # (0.004 = time_step 0.002 x gait_freq 2.0 = 콜백 1회분 위상), 값이 7개뿐이며
    # 잔차가 정확히 0 이다. 벽시계였다면 연속분포에 sd 0.0167 이 나와야 한다.
    #
    # ⛔ MuJoCo 는 물리 스텝(dt=0.002)이 콜백과 1:1 이라 두 시계가 자동으로 같다.
    # 팽창을 재현하려면 **명시적으로 갈라야** 한다: 물리는 --period-ms 마다 정책을
    # 부르고, 정책에 넘기는 시계는 nominal(dt*decimation) 만큼만 전진시킨다.
    ap.add_argument("--clock-mode", choices=["sim", "counter"], default="sim",
                    help="정책에 넘기는 시간축. sim=물리 시간(=완전한 벽시계, 팽창 없음). "
                         "counter=틱당 nominal dt*decimation 만 전진하는 카운터"
                         "(=실기 Timer 재현). --period-ms 가 nominal 과 같으면 둘은 "
                         "동일해야 한다 -- 그 널 셀이 이 플래그의 자체 검증이다.")
    ap.add_argument("--tick-replay", default=None,
                    help="틱 주기를 CSV 에서 재생한다. 헤더 wall_dt_s,counter_dt_s. "
                         "실기 로그에서 뽑은 (벽시계 주기, 카운터 증분) 쌍을 순환 재생해 "
                         "지터까지 포함한 조건을 만든다. tools/make_tick_replay.py 가 만든다.")
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
    ap.add_argument("--fix-filter", action="store_true",
                    help="--filter-hz 로 느린 루프를 흉내내되, 계수를 실측 dt에서 다시 "
                         "계산해 시정수를 10 ms로 고정한다(deploy의 --rate-fixed-filter와 "
                         "같은 수정). 이것이 증상을 없애면 그 수정이 옳다는 증거다.")
    ap.add_argument("--real-asset", action="store_true",
                    help="로봇 실물 질량·관성으로 돌린다(K1_serial_realmass.xml). "
                         "⛔ 기본 K1_serial.xml은 총 질량 18.7142 kg으로 **학습 자산과 같다** -- "
                         "즉 지금까지의 Isaac 대 MuJoCo 교차검증은 같은 잘못된 로봇 두 개를 "
                         "비교한 것이고 실기 실패가 재현될 리가 없었다. 실물은 19.666 kg, "
                         "Ixx +14.2 %, Izz yaw +19.3 %, CoM -10.8 mm, 발 질량 +29 %.")
    ap.add_argument("--contact-stiff", type=float, default=None,
                    help="접촉 solref의 시간상수(초). 작을수록 딱딱하다. MuJoCo 기본 0.02. "
                         "실기 발목 roll은 평균 -5 W로 **역구동**된다(MuJoCo -0.55/+1.17). "
                         "모터가 미는 게 아니라 지면이 발목을 흔들고 모터가 저항한다. "
                         "무른 접촉이면 그 충격이 안 생긴다.")
    ap.add_argument("--foot-slip", type=float, default=None,
                    help="발 geom의 미끄럼 마찰만 따로. 착지 순간의 횡방향 반응을 본다.")
    ap.add_argument("--filter-hz", type=float, default=None,
                    help="배포 필터를 이 주기로만 적용한다(기본은 물리 스텝마다 = 500 Hz). "
                         "deploy_goal_pose.py:1273의 filtered=0.8*filtered+0.2*target은 "
                         "500 Hz 발행을 가정하고 고른 계수인데, _publish_cmd는 파이썬 루프에 "
                         "time.sleep(0.001)이라 실제로는 훨씬 느릴 수 있다. 루프가 50 Hz면 "
                         "차단주파수가 17.8 -> 1.78 Hz로 내려가 2 Hz 보행을 0.66으로 깎는다. "
                         "실기 추종률 median이 0.61이고 MuJoCo는 0.93이다.")
    ap.add_argument("--leg-gain", type=float, default=1.0,
                    help="다리 12관절 전체의 kp/kd 배수. 실기의 명령 대비 실제 도달 비율"
                         "(추종률)이 median 0.61인데 MuJoCo는 0.93이다 -- 특히 HipP/Knee가"
                         " 실기 0.57-0.76 대 MuJoCo 0.98-1.01이다. 전역 구동 부족이"
                         " 정책을 off-distribution으로 밀어내는지 본다: 정책은"
                         " dof_pos-default를 관측하므로 '덜 움직였다'를 보고 더 크게"
                         " 명령하며, 그 액션이 obs[42:54]로 되먹임된다.")
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
    ap.add_argument("--leg-armature", type=float, default=None,
                    help="다리 12관절의 armature 를 이 값으로 덮는다. MJCF 는 발목에만 "
                         "0.05 를 주고 힙·무릎은 0 인데, Isaac(Goal_Pose_V7.yaml) 은 "
                         "asset.armature=0.0 으로 **전 관절 0** 이다. 그래서 두 "
                         "시뮬레이터가 발목만 다른 로봇을 돌렸다. 0 으로 맞추면 Isaac 의 "
                         "발목 roll 분포(평균 7.5-12.8 rad/s, 한계 초과 27-55 %%)가 "
                         "재현되는지가 이 플래그로 갈린다.")
    # ---- 배포 세션 요청 셀용 레버 (MJC_TEST_REQUEST_20260809.md §3) --------
    ap.add_argument("--act-lag-ms", type=float, default=0.0,
                    help="**액션** 경로의 순수 전달 지연(ms). --sense-lag-ms 는 관측 쪽이라 "
                         "다른 축이다. 배포 실효 구동 지연 추정 = tick/2 12.2 ms + "
                         "EMA(fc 6.57 Hz) 위상지연 23.5 ms ~= 35.6 ms. 학습 모델은 0~18 ms 다.")
    ap.add_argument("--body-force-n", type=float, default=0.0,
                    help="Trunk 에 거는 지속 외력(N). 실기에서 사용자가 옆에서 잡아준 상태를 "
                         "흉내낸다 -- 정책은 tilt~0 을 보면서 정체불명의 힘을 받는다.")
    ap.add_argument("--body-force-dir", choices=["lateral", "forward"], default="lateral",
                    help="외력 방향(월드 y = 측방 / x = 전방).")
    ap.add_argument("--armature-preset", choices=["asset", "vendor", "zero", "ankle"], default="asset",
                    help="다리 armature 를 관절별로 정한다. asset=MJCF 그대로"
                         "(발목 0.05, **힙·무릎 0**), vendor=벤더 공식값"
                         "(무릎 0.0956 이 최대), zero=Isaac 과 같은 전 관절 0. "
                         "⛔ --leg-armature 는 스칼라라 벤더 분포를 만들 수 없다. "
                         "학습 세션 측정: 채점 물리만 armature 로 바꿔도 낙상간격이 "
                         "1.5 s 와 3,740 s 로 갈린다 -- 이 축을 고정하지 않은 대조는 "
                         "그 차이를 통째로 물려받는다.")
    ap.add_argument("--ankle-armature", type=float, default=None,
                    help="발목 4관절 armature 만 이 값으로 (프리셋 **뒤**에 적용). "
                         "발목 roll |dq| 는 이 값이 단독으로 지배한다 -- 실측: "
                         "0.0 -> p90 88.8 rad/s, 0.0565 -> 4.4, 실기 22.2 는 그 사이다. "
                         "⚠️ 벤더 0.056506 은 평행기구 래퍼가 armature_ratio 2.0 을 곱한 "
                         "뒤의 값인데 K1_serial.xml 은 **직렬 근사**다. 직렬 모델에 그 "
                         "배율을 쓰는 것이 맞는지는 확인된 적이 없다 -- 아니면 0.028253 이다.")
    ap.add_argument("--vendor-gains", action="store_true",
                    help="벤더 관계식으로 다리 kp/kd 를 armature 에서 유도한다"
                         " (kp = J*(2*pi*4)^2, kd = 2*zeta*J*(2*pi*4), zeta 1.5/무릎 1.0)."
                         " armature 만 옮기고 게인을 두면 벤더 관점에서 짝이 안 맞는다.")
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
    LEG_ORDER = ["Hip_Pitch", "Hip_Roll", "Hip_Yaw",
                 "Knee_Pitch", "Ankle_Pitch", "Ankle_Roll"] * 2
    if args.vendor_gains:
        # kp = J*(2*pi*f_n)^2, kd = 2*zeta*J*(2*pi*f_n). 다른 배수 플래그보다 **먼저**
        # 적용해서, --ankle-gain 같은 레버가 이 위에 얹히게 한다(순서를 바꾸면
        # 그 레버들이 조용히 무시된다).
        w = 2.0 * math.pi * VENDOR_FN_HZ
        for k, name in enumerate(LEG_ORDER):
            J = VENDOR_ARMATURE[name]
            z = VENDOR_ZETA.get(name, VENDOR_ZETA_DEFAULT)
            kp[10 + k] = J * w * w
            kd[10 + k] = 2.0 * z * J * w
        print("벤더 유도 게인: kp %s / kd %s"
              % (np.round(kp[10:16], 1), np.round(kd[10:16], 2)))
    if args.ankle_gain != 1.0:
        for i in (14, 15, 20, 21):
            kp[i] *= args.ankle_gain; kd[i] *= args.ankle_gain
        print("발목 이득 x%.2f" % args.ankle_gain)
    if args.leg_gain != 1.0:
        for i in range(10, 22):
            kp[i] *= args.leg_gain; kd[i] *= args.leg_gain
        print("다리 전역 이득 x%.2f" % args.leg_gain)
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

    mjcf_path = ("resources/K1/K1_serial_realmass.xml" if args.real_asset else MJCF)
    if args.real_asset:
        print("자산: 로봇 실물 (19.666 kg)")
    if args.foot_inertia == "urdf":
        mjcf_path = _write_urdf_foot_variant(MJCF)
        print("발 관성: 학습 URDF 값을 강제 (%s)" % mjcf_path)
    else:
        print("발 관성: 벤더 MJCF 값 (학습 URDF와 다르다 -- 2.7-9.4배 작고 물리적으로 유효)")

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    if args.armature_preset != "asset":
        shown = []
        for j in range(model.njnt):
            jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            key = next((k for k in VENDOR_ARMATURE if k in jn), None)
            if key is None:
                continue
            if args.armature_preset == "zero":
                v = 0.0
            elif args.armature_preset == "ankle":
                # H3 셀: 발목만 벤더값, 힙·무릎은 0(=학습 조건 그대로).
                v = VENDOR_ARMATURE[key] if key.startswith("Ankle") else 0.0
            else:
                v = VENDOR_ARMATURE[key]
            old = float(model.dof_armature[model.jnt_dofadr[j]])
            model.dof_armature[model.jnt_dofadr[j]] = v
            if jn.startswith("Left"):
                shown.append("%s %.4f->%.4f" % (key, old, v))
        if len(shown) != 6:
            raise SystemExit("armature 프리셋: 왼다리 6관절을 못 찾았다 (%d개) -- "
                             "자산의 관절 이름이 바뀌었다" % len(shown))
        print("armature 프리셋 %s: %s" % (args.armature_preset, " | ".join(shown)))
    if args.ankle_armature is not None:
        n_ank = 0
        for j in range(model.njnt):
            jn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if "Ankle" not in jn:
                continue
            model.dof_armature[model.jnt_dofadr[j]] = args.ankle_armature
            n_ank += 1
        if n_ank != 4:
            raise SystemExit("발목 관절 4개를 못 찾았다 (%d개)" % n_ank)
        print("발목 armature -> %.6f (4관절)" % args.ankle_armature)
    if args.leg_armature is not None:
        n_changed = 0
        for j in range(model.njnt):
            jn = mujoco.mj_name2id and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if not jn or not any(k in jn for k in ("Hip", "Knee", "Ankle")):
                continue
            model.dof_armature[model.jnt_dofadr[j]] = args.leg_armature
            n_changed += 1
        print("다리 armature -> %.4f (관절 %d개)" % (args.leg_armature, n_changed))
    model.opt.timestep = dt
    if args.contact_stiff is not None:
        model.geom_solref[:, 0] = args.contact_stiff
        print("접촉 시간상수 -> %.4f s (기본 0.02, 작을수록 딱딱)" % args.contact_stiff)
    if args.foot_slip is not None:
        for side in ("left", "right"):
            b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "%s_foot_link" % side)
            for g in range(model.ngeom):
                if model.geom_bodyid[g] == b:
                    model.geom_friction[g, 0] = args.foot_slip
        print("발 마찰 -> %.2f" % args.foot_slip)
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
    scene_opt = None
    frames = []
    if args.video:
        renderer = mujoco.Renderer(model, height=480, width=640)
        # The feet carry TWO geoms: the detailed mesh (group 1, contype=0 --
        # visual only) and the collision box (no group -> group 0, opaque grey,
        # same rgba).  MuJoCo draws both, so the box is painted over the mesh and
        # the foot renders as a bare slab -- which reads as the wrong foot
        # entirely.  The geometry is right (box 16.0 x 7.0 cm, mesh 16.5 x 7.8,
        # feet_edge_pos -0.066..0.094 x +-0.035); only the picture was wrong.
        # Hide group 0 so the video shows the shape, not the collider.
        scene_opt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(scene_opt)
        scene_opt.geomgroup[0] = 0
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
    nominal_period_s = dt * decim         # 카운터 시계가 틱마다 전진하는 양(=0.020)
    tick_replay = None
    if args.tick_replay:
        import csv as _csv
        with open(args.tick_replay, newline="", encoding="utf-8") as f:
            rd = _csv.DictReader(f)
            tick_replay = [(float(r["wall_dt_s"]), float(r["counter_dt_s"])) for r in rd]
        if not tick_replay:
            raise SystemExit("--tick-replay 파일이 비었다: %s" % args.tick_replay)
        print("틱 재생: %d 쌍, 벽시계 mean %.5f s / 카운터 mean %.5f s (비 %.4f)"
              % (len(tick_replay),
                 sum(w for w, _ in tick_replay) / len(tick_replay),
                 sum(c for _, c in tick_replay) / len(tick_replay),
                 sum(c for _, c in tick_replay) / max(sum(w for w, _ in tick_replay), 1e-9)))
    if args.clock_mode == "counter":
        print("시계: 카운터 (틱당 %.4f s 전진) -- 실기 Timer 재현" % nominal_period_s)
    next_infer = 0.0
    clk = 0.0                             # 정책에 넘기는 카운터 시계
    clk_elapsed = 0.0                     # 카운터가 실제로 전진한 총량
    last_tick_t = 0.0
    tick_dt_s = period_s
    replay_i = 0
    gp_prev = 0.0
    phase_total = 0.0                     # 보행 위상 누적(랩 복원) -- 실기와 같은 통계
    phase_incr = []
    next_filt = 0.0
    # 실기와 같은 지표를 낸다: 관절별 |tau| 분위수와 부호전환 횟수.
    tau_hist = [[] for _ in range(nj)]
    hiproll_hist = []
    foot_sep = []
    foot_bid = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "%s_foot_link" % s)
                for s in ("left", "right")]
    if any(b < 0 for b in foot_bid):
        raise SystemExit("발 링크를 못 찾았다: left/right_foot_link")
    # ---- 측방 capture point (MJC_TEST_REQUEST_20260809.md §4 의 판정 축) ----
    # capture = roll + gx*tau, tau = 0.214 s (LIP 시상수, CoM 높이 0.449 m).
    # 한계: 단일지지 +-4.4도 = atan(발반폭 0.035 / 0.449), 양발 +-15.2도.
    # ⚠️ 실기 로그에는 접촉 센서가 없다. 배포 세션이 "단일지지 표본"을 무엇으로
    # 갈랐는지 문서에 없으므로, 비교 가능하도록 **세 가지 분모를 전부** 낸다.
    # ⚠️ 그리고 MuJoCo 는 낙상하고 리셋되는데 실기는 잡아준 상태라 낙상이 0 이다.
    # 낙상 중/직후 표본은 capture 를 자동으로 위반하므로, `upright`(tilt<15도)
    # 분모를 같이 낸다 -- 그것이 실기가 있던 상태다.
    CAP_TAU = 0.214
    CAP_SINGLE_DEG, CAP_DOUBLE_DEG, UPRIGHT_DEG = 4.4, 15.2, 15.0
    cap_rows = []            # (cap_deg, n_support, tilt_deg)
    ankroll_dq = []          # |dq| 발목 roll 좌/우
    hiproll_q = []           # 힙롤 좌/우 (rad)
    lhip_pitch = []          # L_Hip_Pitch 드리프트용

    def _support_count():
        """발 링크가 참여한 접촉이 있는 발의 수 (0/1/2)."""
        hit = [False, False]
        for c in range(data.ncon):
            g1, g2 = data.contact[c].geom1, data.contact[c].geom2
            for g in (g1, g2):
                b = model.geom_bodyid[g]
                for k in (0, 1):
                    if b == foot_bid[k]:
                        hit[k] = True
        return int(hit[0]) + int(hit[1])
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
    act_lag_steps = int(round(args.act_lag_ms / 1000.0 / dt))
    act_buf = []
    if act_lag_steps > 0:
        print("액션 전달 지연 %.1f ms (%d 물리 스텝)" % (args.act_lag_ms, act_lag_steps))
    trunk_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Trunk")
    if args.body_force_n != 0.0:
        if trunk_bid < 0:
            raise SystemExit("Trunk 바디를 못 찾았다 -- 외력을 걸 곳이 없다")
        _ax = 1 if args.body_force_dir == "lateral" else 0
        data.xfrc_applied[trunk_bid, _ax] = args.body_force_n
        print("Trunk 지속 외력 %.1f N (%s, 월드축 %d)"
              % (args.body_force_n, args.body_force_dir, _ax))
    FALL_BUF_N = 1200                 # 2.4 s @ dt 0.002
    FF_WIN = 250                      # 0.5 s -- 낙상 직전에 발이 걸렸는지 보는 창
    fall_buf, fall_events = [], []
    ff_steps = ff_episodes = 0
    ff_prev = False
    ff_depth_min = 0.0
    ff_recent = []
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
            # 게이트는 **누산**이다(deploy_goal_pose.py:1753 `+= policy_interval`).
            # `= t + period` 로 리셋하면 period 가 dt 의 배수가 아닐 때 매 틱 올림이
            # 누적돼 실효 주기가 최대 dt 만큼 길어진다(26.14 ms -> 27 ms, -3 %).
            # 실기 데이터가 이 규칙을 확증한다: 틱당 콜백수 mean 9.989 ~= decimation 10.
            if tick_replay is not None:
                wall_dt, cnt_dt = tick_replay[replay_i % len(tick_replay)]
                replay_i += 1
            else:
                wall_dt, cnt_dt = period_s, nominal_period_s
            next_infer += wall_dt
            tick_dt_s = t - last_tick_t
            last_tick_t = t
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
            # 정책은 `advance_gait_clock` 안에서 이 값의 **차분**으로 위상을 적분한다.
            # sim 모드면 물리 시간이 그대로 가고(팽창 없음), counter 모드면 틱당
            # nominal 만 가므로 벽시계 케이던스가 nominal/wall_dt 배로 줄어든다.
            clock_arg = clk if args.clock_mode == "counter" else t
            targets = policy.inference(
                clock_arg, (q + joint_bias).astype(np.float32), obs_dq.astype(np.float32),
                obs_w.astype(np.float32), obs_g.astype(np.float32),
                grx, gry, herr)
            clk += cnt_dt
            clk_elapsed += cnt_dt
            # 실기 로그와 **같은 통계**로 잰다: 랩을 복원한 위상 증분.
            _x = policy.gait_process - gp_prev
            if _x < -0.5:
                _x += 1.0
            if abs(_x) > 1e-12:
                phase_incr.append(_x)
            phase_total += _x
            gp_prev = policy.gait_process
            if dump_fp is not None:
                rr = math.atan2(R[2, 1], R[2, 2]); pp = math.asin(-max(-1.0, min(1.0, R[2, 0])))
                head = [t, 0.001, tick_dt_s, math.degrees(tilt_rad_prev), rr, pp,
                        ang_vel[0], ang_vel[1], ang_vel[2],
                        1.0, policy.gait_frequency, policy.gait_process, grx, gry, herr]
                body = (list(q[10:22]) + list(dq[10:22])
                        + list(applied_prev[10:22]) + list(policy.actions[:12]))
                dump_fp.write(",".join("%.5g" % float(v) for v in head + body) + "\n")

        # 필터는 정책 tick이 아니라 발행 루프(=물리 스텝)마다 돈다. 배포가 그렇다.
        if args.deploy_filter:
            # --filter-hz 가 주어지면 그 주기로만 갱신한다. 물리 스텝마다 돌리는 것이
            # 500 Hz 발행에 해당하고, 느린 루프를 흉내내려면 갱신을 띄엄띄엄 해야 한다.
            if args.filter_hz is None or t >= next_filt - 1e-9:
                step = dt if args.filter_hz is None else 1.0 / args.filter_hz
                if args.filter_hz is not None:
                    next_filt = t + step
                if args.fix_filter:
                    # 실측 주기로 계수를 다시 계산 -> 시정수 10 ms 고정
                    a = min(1.0, 1.0 - math.exp(-step / 0.010))
                else:
                    a = 0.2
                filtered[:] = filtered * (1.0 - a) + targets * a
            cmd_q = filtered
        else:
            cmd_q = targets
        # 액션 경로의 순수 전달 지연. 필터 **뒤**에 건다 -- 배포에서도 지연은
        # 발행 이후(직렬 버스 + 펌웨어)에 생기므로 필터가 먼저다.
        if act_lag_steps > 0:
            act_buf.append(np.array(cmd_q, dtype=np.float64))
            if len(act_buf) > act_lag_steps:
                cmd_q = act_buf.pop(0)
            else:
                cmd_q = np.array(default_q)
        tau = kp * ((cmd_q + joint_bias) - q) - kd * dq
        applied = tau if args.no_torque_clamp else np.clip(tau, -lim, lim)
        # MuJoCo의 actuator forcerange가 여전히 자르므로, 클램프를 정말 풀려면
        # 모델 쪽 한계도 같이 올려야 한다. 아래 model.actuator_forcerange에서 처리.
        data.ctrl[:] = applied
        applied_prev = applied.copy()
        hiproll_hist.append((float(q[11]), float(q[17])))
        # ---- 부호 있는 좌우 발 간격 (학습 세션 요청) -----------------------
        # ⛔ 절대값을 쓰면 다리 교차가 **원리적으로 안 보인다**, 그리고 평균으로도
        # 안 보인다(교차는 스윙 한순간이라 평균 18~21 cm 에 묻힌다). 그래서 부호를
        # 유지한 채 **롤아웃 전체**를 누산하고 최소값과 초과 체류율로 읽는다.
        # 몸통 yaw 프레임의 y 성분: 양수=정상, 음수=교차.
        if it % 5 == 0:
            _d = data.xpos[foot_bid[0]][:2] - data.xpos[foot_bid[1]][:2]
            _c, _s = math.cos(-yaw), math.sin(-yaw)
            foot_sep.append(float(_s * _d[0] + _c * _d[1]))
            # 실기와 같은 정의: roll(트렁크) + gx(롤 각속도)*tau
            _roll = math.atan2(R[2, 1], R[2, 2])
            cap_rows.append((math.degrees(_roll + ang_vel[0] * CAP_TAU),
                             _support_count(),
                             math.degrees(math.acos(np.clip(-proj_g[2], -1.0, 1.0)))))
            ankroll_dq.append((abs(float(dq[15])), abs(float(dq[21]))))
            hiproll_q.append((float(q[11]), float(q[17])))
            lhip_pitch.append(float(q[10]))
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
        # ---- 낙상 사건 기록 (물리 스텝 해상도) ------------------------------
        # ⛔ --dump-csv 로는 이것을 못 잰다: 덤프는 정책 tick(50 Hz)에만 쓰는데
        # 낙상 판정은 물리 스텝(500 Hz)에서 돌고 **즉시 리셋**된다. 실제로 75회 중
        # 2회만 잡혔다. 방향(측방/전방)은 낙상의 판별 축인데 그 표본으로는 못 센다.
        # 그래서 링버퍼를 두고 사건이 나면 그 자리에서 되짚는다.
        # ---- 발끼리 실제 충돌 (간격은 대리지표일 뿐이다) --------------------
        # ⛔ 방향(측방/전방) 분류로는 이 기전을 못 잡는다. **남의 발에 걸리면 앞으로
        # 고꾸라진다** -- 전방 우세는 발 걸림과 모순이 아니라 그 예상 결과다.
        # 그래서 두 발 geom 사이의 접촉을 직접 센다.
        _ff = False
        _ffdepth = 0.0
        for _c in range(data.ncon):
            _b1 = model.geom_bodyid[data.contact[_c].geom1]
            _b2 = model.geom_bodyid[data.contact[_c].geom2]
            if (_b1 == foot_bid[0] and _b2 == foot_bid[1]) or \
               (_b1 == foot_bid[1] and _b2 == foot_bid[0]):
                _ff = True
                _ffdepth = min(_ffdepth, float(data.contact[_c].dist))
        if _ff:
            ff_steps += 1
            ff_depth_min = min(ff_depth_min, _ffdepth)
            if not ff_prev:
                ff_episodes += 1
        ff_prev = _ff
        ff_recent.append(_ff)
        if len(ff_recent) > FF_WIN:
            del ff_recent[0]
        _rr = math.atan2(R[2, 1], R[2, 2])
        _pp = math.asin(-max(-1.0, min(1.0, R[2, 0])))
        _fd = data.xpos[foot_bid[0]][:2] - data.xpos[foot_bid[1]][:2]
        _cc, _ss = math.cos(-yaw), math.sin(-yaw)
        fall_buf.append((t, math.degrees(tilt_rad), math.degrees(_rr),
                         math.degrees(_pp), float(_ss * _fd[0] + _cc * _fd[1])))
        if len(fall_buf) > FALL_BUF_N:
            del fall_buf[0]
        if fallen and len(fall_buf) > 2:
            def _back(thr):
                """뒤에서 앞으로 가며 tilt 가 thr 아래였던 마지막 표본."""
                for j in range(len(fall_buf) - 1, -1, -1):
                    if fall_buf[j][1] < thr:
                        return j
                return 0
            j20, j10 = _back(20.0), _back(10.0)
            fall_events.append({
                "t": round(t, 3),
                # 20도를 넘은 시점의 자세. 이때 어느 축이 앞서 있는지가 방향이다.
                "roll20": round(fall_buf[j20][2], 2),
                "pitch20": round(fall_buf[j20][3], 2),
                "sep20": round(fall_buf[j20][4], 4),
                "lead_10_to_fall_s": round(t - fall_buf[j10][0], 3),
                "sep_min_1s": round(min(r[4] for r in fall_buf[-500:]), 4),
                "lateral": abs(fall_buf[j20][2]) > abs(fall_buf[j20][3]),
                # ⭐ 낙상 직전 0.5 s 안에 두 발이 실제로 부딪혔는가
                "foot_foot_before": bool(any(ff_recent)),
            })
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
            renderer.update_scene(data, camera=cam, scene_option=scene_opt)
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
    # ---- 시계 팽창 지표 ----------------------------------------------------
    # 실기 로그와 **같은 통계**다: 랩 복원한 위상 증분의 분포와, 총위상/총시간.
    # 실기 실측(91틱): cadence 1.5116 Hz, 증분 median 0.040 / sd 0.00424 / 고유 7값.
    if phase_incr:
        _pi = np.array(phase_incr)
        res["gait"] = {
            "cadence_hz": round(float(phase_total / max(args.duration, 1e-9)), 4),
            "commanded_hz": round(float(policy.gait_frequency), 3),
            "ticks": len(_pi),
            "incr_median": round(float(np.median(_pi)), 6),
            "incr_sd": round(float(np.std(_pi)), 6),
            "incr_unique": int(len(set(np.round(_pi, 9)))),
            "clock_ratio": round(float(clk_elapsed / max(args.duration, 1e-9)), 4),
            "clock_mode": args.clock_mode,
            "tick_replay": os.path.basename(args.tick_replay) if args.tick_replay else None,
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
    if foot_sep:
        fs = np.array(foot_sep)
        # 발 폭 7.0 cm = 충돌 box 의 폭(§8-45 부수 발견). 그 아래면 두 발이 겹친다.
        res["foot_sep_m"] = {
            "min": round(float(fs.min()), 4),
            "p1": round(float(np.percentile(fs, 1)), 4),
            "median": round(float(np.median(fs)), 4),
            "mean": round(float(fs.mean()), 4),
            "share_below_0p07": round(float((fs < 0.07).mean()), 6),
            "share_negative": round(float((fs < 0.0).mean()), 6),
            "n": int(fs.size),
        }
    if cap_rows:
        cap = np.array([r[0] for r in cap_rows])
        sup = np.array([r[1] for r in cap_rows])
        tlt = np.array([r[2] for r in cap_rows])
        up = tlt < UPRIGHT_DEG          # 실기(잡아준 상태)가 있던 영역
        single = sup == 1

        def _sh(mask, thr):
            return (round(float((np.abs(cap[mask]) > thr).mean()), 4)
                    if mask.sum() else None)
        res["capture"] = {
            "tau_s": CAP_TAU,
            "n": int(cap.size),
            "abs_max_deg": round(float(np.abs(cap).max()), 2),
            "p90_abs_deg": round(float(np.percentile(np.abs(cap), 90)), 2),
            # 판정 축. 분모를 셋 다 낸다 -- 실기 쪽 분모 정의가 문서에 없다.
            "viol_4p4_single": _sh(single, CAP_SINGLE_DEG),
            "viol_4p4_single_upright": _sh(single & up, CAP_SINGLE_DEG),
            "viol_4p4_all": _sh(np.ones_like(up), CAP_SINGLE_DEG),
            "viol_4p4_all_upright": _sh(up, CAP_SINGLE_DEG),
            "viol_15p2_all": _sh(np.ones_like(up), CAP_DOUBLE_DEG),
            "viol_15p2_all_upright": _sh(up, CAP_DOUBLE_DEG),
            "share_single_support": round(float(single.mean()), 4),
            "share_flight": round(float((sup == 0).mean()), 4),
            "share_upright": round(float(up.mean()), 4),
        }
    if ankroll_dq:
        a = np.array(ankroll_dq)
        res["ankle_roll_dq"] = {
            "L_p90": round(float(np.percentile(a[:, 0], 90)), 2),
            "L_max": round(float(a[:, 0].max()), 2),
            "R_p90": round(float(np.percentile(a[:, 1], 90)), 2),
            "R_max": round(float(a[:, 1].max()), 2),
        }
    if hiproll_q:
        h = np.array(hiproll_q)
        res["hip_roll_p2p_rad"] = {
            "L": round(float(h[:, 0].max() - h[:, 0].min()), 4),
            "R": round(float(h[:, 1].max() - h[:, 1].min()), 4),
        }
    if len(lhip_pitch) >= 8:
        lp = np.array(lhip_pitch)
        k = len(lp) // 4
        res["l_hip_pitch_drift"] = {
            "first_quarter_median": round(float(np.median(lp[:k])), 4),
            "last_quarter_median": round(float(np.median(lp[-k:])), 4),
            "final": round(float(lp[-1]), 4),
            "delta": round(float(np.median(lp[-k:]) - np.median(lp[:k])), 4),
        }
    res["foot_foot"] = {
        "episodes": ff_episodes,
        "per_min": round(ff_episodes / max(args.duration / 60.0, 1e-9), 1),
        "time_share": round(ff_steps / max(nsteps, 1), 5),
        "max_penetration_m": round(-ff_depth_min, 4),
    }
    if fall_events:
        lat = [e for e in fall_events if e["lateral"]]
        res["foot_foot"]["falls_preceded"] = round(float(np.mean(
            [e["foot_foot_before"] for e in fall_events])), 3)
        res["fall_mode"] = {
            "n": len(fall_events),
            "lateral": len(lat),
            "sagittal": len(fall_events) - len(lat),
            "lateral_share": round(len(lat) / len(fall_events), 3),
            "roll20_median": round(float(np.median([e["roll20"] for e in fall_events])), 2),
            "pitch20_median": round(float(np.median([e["pitch20"] for e in fall_events])), 2),
            "abs_roll20_median": round(float(np.median([abs(e["roll20"]) for e in fall_events])), 2),
            "abs_pitch20_median": round(float(np.median([abs(e["pitch20"]) for e in fall_events])), 2),
            "lead_10_to_fall_median_s": round(float(np.median(
                [e["lead_10_to_fall_s"] for e in fall_events])), 3),
            "sep20_median": round(float(np.median([e["sep20"] for e in fall_events])), 4),
            "sep_crossed_before_fall": round(float(np.mean(
                [e["sep_min_1s"] < 0.0 for e in fall_events])), 3),
        }
        res["fall_events"] = fall_events[:40]      # 전수는 크다 -- 앞 40건만
    res["armature_preset"] = args.armature_preset
    res["vendor_gains"] = bool(args.vendor_gains)
    res["act_lag_ms"] = args.act_lag_ms
    res["body_force_n"] = args.body_force_n
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
