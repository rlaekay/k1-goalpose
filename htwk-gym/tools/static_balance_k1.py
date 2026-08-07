#!/usr/bin/env python3
"""자세 하나가 **개루프로 설 수 있는가**를 URDF 로 답한다.

⛔ 왜 만들었나 (2026-08-08)

사용자 실측이 `--hold-diag` 의 두 tilt 값이 서로 **다른 조건**이었음을 밝혔다:

    발목을 손으로 바닥에 눌러 잡음 -> tilt 4.9°
    손을 뗌                        -> tilt 13.8°
    (같은 구간의 관절 추종 오차는 0.6° -- 서보는 명령을 따르고 있다)

정적 예측은 4.5° 였다(로봇 자체 standing 1.6° + pitch 사슬 합 -0.05 rad = 2.87°).
잡고 있을 때가 0.4° 차이로 맞는다 ⇒ **발바닥이 평평하면 정적 자세가 전부 설명한다.**
놓으면 9° 가 더 간다 ⇒ **발바닥이 안 평평하다. 로봇이 발끝을 축으로 넘어간다.**

그러면 질문이 정량으로 바뀐다: **그 자세에서 무게중심이 지지다각형 안에 있는가,
있다면 여유가 얼마인가.** 이 도구가 그것을 답한다.

읽는 법
-------
`여유(앞)` 가 음수면 CoM 이 발끝 밖이다 = **개루프로는 반드시 넘어간다.**
양수여도 작으면 발목 토크가 그 여유를 지켜야 하고, 필요 토크가 정격을 넘으면
결과는 같다. 두 숫자를 같이 본다.

⚠️ 지지다각형의 앞뒤 경계는 **발 충돌 지오메트리**가 정한다. URDF 는 box 하나지만
MJCF 는 geom 이 3개이고 box 가 메시 안에 묻혀 있다(RETRACTIONS C25). 이 도구는
URDF box 를 쓰므로 **MJCF 와 다른 답을 낼 수 있다** -- 그 차이가 나면 그것 자체가
발견이다. `--toe` / `--heel` 로 경계를 직접 넣어 대조할 수 있다.

⚠️ 질량은 URDF 그대로다. 이 URDF 는 총 18.714 kg 인데 **실물은 19.666 kg**
(발 0.38305 대 0.4940). 틀린 발 리비전이다(HANDOFF_TO_TRAINING). `--foot-mass` 로
실물 값을 넣어 감도를 볼 수 있다.

사용
----
    python tools/static_balance_k1.py                      # 두 표준 자세 비교
    python tools/static_balance_k1.py --hip -0.2 --knee 0.4 --ankle -0.25
    python tools/static_balance_k1.py --foot-mass 0.4940   # 실물 발 질량으로
"""
import os
import re
import sys
import math
import argparse

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URDF = os.path.join(REPO, "resources", "K1", "K1_locomotion.urdf")

# 발 메시가 box 보다 3 mm 아래까지 간다(kinematics_k1.py: SOLE_MESH_Z -0.026896
# 대 SOLE_BOX_Z -0.024). 접촉하는 것은 아래쪽이므로 메시를 기준으로 둔다.
SOLE_MESH_Z = -0.026896

# 표준 자세. pitch 사슬 = hip_pitch + knee_pitch + ankle_pitch.
POSES = {
    "vendor_standing": (0.00, 0.10, -0.10),   # 로봇 자체 standing, 실측 tilt 1.6°
    "prepare_fixed":   (-0.10, 0.20, -0.10),  # 수정된 진입 자세 (합 0)
    "rl_default":      (-0.20, 0.40, -0.25),  # 학습 default_qpos (합 -0.05)
}


# ---- 최소 선형대수 (numpy 없이 -- 맥 로컬에 없다) ---------------------------
def mm(A, B):
    return [[sum(A[i][k] * B[k][j] for k in range(3)) for j in range(3)]
            for i in range(3)]


def mv(A, v):
    return [sum(A[i][k] * v[k] for k in range(3)) for i in range(3)]


def vadd(a, b):
    return [a[i] + b[i] for i in range(3)]


def vsub(a, b):
    return [a[i] - b[i] for i in range(3)]


def transpose(A):
    return [[A[j][i] for j in range(3)] for i in range(3)]


def rpy_to_R(r, p, y):
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp,     cp * sr,                cp * cr]]


def axis_angle_to_R(axis, q):
    x, y, z = axis
    n = math.sqrt(x * x + y * y + z * z)
    if n < 1e-12:
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    x, y, z = x / n, y / n, z / n
    c, s, C = math.cos(q), math.sin(q), 1 - math.cos(q)
    return [[x * x * C + c,     x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c,     y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c]]


# ---- URDF ------------------------------------------------------------------
def _xyz(s, default=(0.0, 0.0, 0.0)):
    if not s:
        return list(default)
    return [float(v) for v in s.split()]


def load_urdf(path):
    """⛔ XML 은 XML 파서로 읽는다.

    첫 판은 정규식으로 읽었고 **조인트 하나를 조용히 놓쳤다**(`AAHead_yaw`).
    URDF 가 속성을 여러 줄에 걸쳐 쓰기 때문이다. 결과로 Head_1/Head_2 가 트리에서
    떨어져 나가 총질량이 18.714 대신 **17.803** 으로 나왔다 -- 0.911 kg 이 조용히
    사라졌는데 표에는 아무 표시도 없었다. 그런 도구는 쓰면 안 된다.
    아래 `_check_mass` 가 이제 그것을 잡는다.
    """
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()

    def org(el):
        o = el.find("origin") if el is not None else None
        if o is None:
            return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        return _xyz(o.get("xyz")), _xyz(o.get("rpy"))

    links = {}
    foot_box = {}
    for L in root.findall("link"):
        name = L.get("name")
        inertial = L.find("inertial")
        if inertial is None or inertial.find("mass") is None:
            continue
        com, _ = org(inertial)
        links[name] = {"mass": float(inertial.find("mass").get("value")), "com": com}
        if "foot" in name.lower():
            for c in L.findall("collision"):
                b = c.find("geometry/box")
                if b is None:
                    continue
                o, _ = org(c)
                foot_box[name] = {"size": _xyz(b.get("size")), "origin": o}

    joints = []
    for J in root.findall("joint"):
        par, chi = J.find("parent"), J.find("child")
        if par is None or chi is None:
            continue
        xyz, rpy = org(J)
        ax = J.find("axis")
        joints.append({"name": J.get("name"), "type": J.get("type") or "fixed",
                       "parent": par.get("link"), "child": chi.get("link"),
                       "xyz": xyz, "rpy": rpy,
                       "axis": _xyz(ax.get("xyz")) if ax is not None else [1.0, 0.0, 0.0]})
    return links, joints, foot_box


def _check_mass(links, pose):
    """FK 가 못 닿은 링크가 있으면 **죽는다.** 조용히 빠지는 질량을 금지한다."""
    missing = [(k, links[k]["mass"]) for k in links if k not in pose]
    if missing:
        raise SystemExit(
            "⛔ FK 가 링크 %d 개에 못 닿았다 (질량 %.4f kg 누락): %s\n"
            "   URDF 트리가 끊겼거나 파서가 조인트를 놓쳤다. 무게중심을 그대로\n"
            "   쓰면 틀린 답이 나온다." % (len(missing), sum(m for _, m in missing),
                                          ", ".join(k for k, _ in missing)))


def fk(links, joints, q, root="Trunk"):
    """트렁크 기준으로 모든 링크의 (R, p) 를 낸다. `q` 는 관절이름 -> 각도."""
    children = {}
    for j in joints:
        children.setdefault(j["parent"], []).append(j)
    pose = {root: ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0.0, 0.0, 0.0])}
    stack = [root]
    while stack:
        p = stack.pop()
        Rp, pp = pose[p]
        for j in children.get(p, []):
            Rj = mm(rpy_to_R(*j["rpy"]), axis_angle_to_R(j["axis"], q.get(j["name"], 0.0))
                    if j["type"] in ("revolute", "continuous") else
                    [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            R = mm(Rp, Rj)
            pos = vadd(pp, mv(Rp, j["xyz"]))
            pose[j["child"]] = (R, pos)
            stack.append(j["child"])
    return pose


def analyse(links, joints, foot_box, hip, knee, ankle, foot_mass=None,
            toe=None, heel=None):
    q = {}
    for side in ("Left", "Right"):
        q["%s_Hip_Pitch" % side] = hip
        q["%s_Knee_Pitch" % side] = knee
        q["%s_Ankle_Pitch" % side] = ankle
    pose = fk(links, joints, q)
    _check_mass(links, pose)

    masses = dict((k, v["mass"]) for k, v in links.items())
    if foot_mass is not None:
        for k in masses:
            if "foot" in k.lower():
                masses[k] = foot_mass

    total = 0.0
    com = [0.0, 0.0, 0.0]
    for name, L in links.items():
        if name not in pose:
            continue
        R, p = pose[name]
        c = vadd(p, mv(R, L["com"]))
        m = masses[name]
        total += m
        com = [com[i] + m * c[i] for i in range(3)]
    com = [com[i] / total for i in range(3)]

    # 왼발 sole 프레임. 발바닥이 평평하다는 조건 = 발 링크의 자세로 세계를 맞춘다.
    Rf, pf = pose["left_foot_link"]
    sole_p = vadd(pf, mv(Rf, [0.0, 0.0, SOLE_MESH_Z]))
    Rw = transpose(Rf)                      # 트렁크 -> 세계 (발바닥 수평)
    com_sole = mv(Rw, vsub(com, sole_p))

    box = foot_box.get("left_foot_link")
    if toe is None or heel is None:
        ox, sx = box["origin"][0], box["size"][0]
        toe_x = ox + sx / 2.0 if toe is None else toe
        heel_x = ox - sx / 2.0 if heel is None else heel
    else:
        toe_x, heel_x = toe, heel

    # 발목 pitch 축의 x (sole 프레임). 필요 토크 = 무게 x (CoM_x - ankle_x).
    ankle_p = None
    for j in joints:
        if j["name"] == "Left_Ankle_Pitch":
            Rap, pap = pose[j["child"]]
            ankle_p = mv(Rw, vsub(pap, sole_p))
    # 몸통 기울기: 발바닥이 수평일 때 트렁크 z 축이 수직에서 얼마나 벗어나는가
    trunk_z_world = [Rw[i][2] for i in range(3)]
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, trunk_z_world[2]))))

    # ---- 안정성. **평형이 있는 것과 안정한 것은 다르다.** -------------------
    #
    # 발바닥이 평평하면 몸통이 발목을 축으로 도는 역진자다. 다리 관절이 전부 PD 로
    # 잡혀 있으므로 다리는 강체처럼 굴고, 몸이 θ 만큼 기울면 **발목 관절각만** θ 만큼
    # 바뀐다. 그래서:
    #
    #     복원 강성  = 2 * kp_ankle          (양발)
    #     중력 강성  = m * g * h             (h = 발목 축 위 CoM 높이)  -- 부호가 반대다
    #
    # 2*kp_ankle < m*g*h 이면 **직립 자세가 불안정하다.** CoM 이 지지다각형 한가운데에
    # 있어도 발산한다 -- 되돌릴 힘이 없기 때문이다. 지지다각형 여유는 "얼마나 기울 수
    # 있나"를 말할 뿐 "스스로 서 있나"를 말하지 않는다.
    h = com_sole[2] - (ankle_p[2] if ankle_p else 0.0)
    k_gravity = total * 9.81 * h
    return {
        "total_mass": total, "com_sole": com_sole, "tilt_deg": tilt,
        "toe_x": toe_x, "heel_x": heel_x,
        "margin_front": toe_x - com_sole[0], "margin_rear": com_sole[0] - heel_x,
        "ankle_x": ankle_p[0] if ankle_p else float("nan"),
        "ankle_torque_total": total * 9.81 * (com_sole[0] - (ankle_p[0] if ankle_p else 0.0)),
        "sum_rad": hip + knee + ankle,
        "com_height_above_ankle": h, "k_gravity": k_gravity,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--urdf", default=URDF)
    ap.add_argument("--hip", type=float)
    ap.add_argument("--knee", type=float)
    ap.add_argument("--ankle", type=float)
    ap.add_argument("--foot-mass", type=float, default=None,
                    help="발 링크 질량 덮어쓰기 (실물 0.4940, URDF 0.38305)")
    ap.add_argument("--toe", type=float, help="발끝 경계 x (sole 프레임, m)")
    ap.add_argument("--heel", type=float, help="뒤꿈치 경계 x")
    a = ap.parse_args()

    links, joints, foot_box = load_urdf(a.urdf)
    box = foot_box.get("left_foot_link")
    print("URDF: %s" % a.urdf)
    print("발 충돌 박스: size=%s origin=%s  =>  발끝 %+.3f m / 뒤꿈치 %+.3f m"
          % (box["size"], box["origin"],
             box["origin"][0] + box["size"][0] / 2, box["origin"][0] - box["size"][0] / 2))
    print()
    if a.hip is not None:
        poses = {"cli": (a.hip, a.knee, a.ankle)}
    else:
        poses = POSES
    hdr = ("자세", "합(rad)", "tilt(도)", "CoM_x(mm)", "여유앞(mm)", "여유뒤(mm)",
           "발목토크(Nm)")
    print("%-17s %8s %8s %10s %10s %10s %12s" % hdr)
    print("-" * 82)
    for name, (h, k, an) in poses.items():
        r = analyse(links, joints, foot_box, h, k, an,
                    foot_mass=a.foot_mass, toe=a.toe, heel=a.heel)
        print("%-17s %8.3f %8.2f %10.1f %10.1f %10.1f %12.2f"
              % (name, r["sum_rad"], r["tilt_deg"], 1000 * r["com_sole"][0],
                 1000 * r["margin_front"], 1000 * r["margin_rear"],
                 r["ankle_torque_total"]))
    print()
    print("총질량 %.4f kg" % r["total_mass"])
    print("여유앞 < 0 이면 CoM 이 발끝 밖 = 개루프로는 반드시 앞으로 넘어간다.")
    print("발목토크는 **양발 합**이다(한 발당 절반). 정격 20 N*m 와 대조해라.")

    print()
    print("=" * 82)
    print("직립 자세의 **안정성** -- 지지다각형 여유와 다른 질문이다")
    print("=" * 82)
    print("발목 축 위 CoM 높이 h = %.4f m,  중력 강성 m*g*h = %.1f N*m/rad"
          % (r["com_height_above_ankle"], r["k_gravity"]))
    print()
    print("%-22s %14s %14s %10s" % ("발목 게인(한쪽)", "복원 2*kp", "중력 m*g*h", "판정"))
    print("-" * 66)
    for label, kp in (("prepare (250)", 250.0), ("common/RL (50)", 50.0)):
        rest = 2.0 * kp
        ratio = rest / r["k_gravity"]
        print("%-22s %14.1f %14.1f %10s"
              % (label, rest, r["k_gravity"],
                 "안정 x%.2f" % ratio if ratio > 1.0 else "⛔ 불안정 x%.2f" % ratio))
    print()
    print("⚠️ 이 URDF 는 총 %.3f kg 인데 **실물은 19.666 kg** 이다(틀린 발 리비전)."
          % r["total_mass"])
    print("   실물 질량으로는 중력 강성이 %.1f N*m/rad 가 된다."
          % (r["k_gravity"] * 19.666 / r["total_mass"]))
    print("⚠️ 감쇠(kd)와 지연은 안 넣었다. 넣으면 안정 여유는 **더 줄어든다.**")
    print("   여기서 'x1.1 안정' 은 실제로는 안정하지 않다는 뜻으로 읽어라.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
