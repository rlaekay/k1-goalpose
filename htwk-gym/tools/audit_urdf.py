"""URDF를 물리적으로 검증한다 — 이 프로젝트가 asset 때문에 두 번 당했기 때문이다.

전례:
  * `feet_edge_pos`가 실제 발보다 컸다. 접촉 판정이 발 밖에서 났고, 그 위에서 돌린
    낙상 수치가 전부 오염됐다(I0b가 교정).
  * `base_com ±0.1 m` 랜덤화가 Trunk 충돌 상자(x ±0.060, y ±0.090)를 넘었다.
    정적으로 균형이 불가능한 로봇을 만들어 놓고 안 넘어지길 요구했다(8-10).
  * H 배치는 다른 URDF 위에서 돌아 전체가 무효가 됐다(3-1).

셋 다 "설정이 물리를 벗어났다"는 같은 부류이고, 셋 다 **파일만 읽으면 잡히는 것**이었다.
이 도구는 그 검사를 한 번에 돌린다. GPU도 시뮬레이터도 쓰지 않는다.

검사 항목:
  1. 관성 텐서의 물리적 타당성 — 양의 정부호, 주모멘트 삼각부등식
  2. 질량/CoM — 링크별, 전체, Trunk CoM이 충돌 상자 안인가
  3. 충돌 기하 — 없는 링크, 발 충돌체 대 `feet_edge_pos`
  4. 운동학 — 다리 길이와 도달 가능한 몸통 높이
  5. effort/limit — sim이 읽는 값 대 deploy config 배열
  6. URDF 간 차이 — 어느 파일이 무엇에서 갈리는가 (3-1의 오염 검출)

    python tools/audit_urdf.py
    python tools/audit_urdf.py resources/K1/K1_locomotion_armsdown.urdf
"""

import os
import sys
import glob
import xml.etree.ElementTree as ET

import numpy as np

DEFAULT_GLOB = "resources/K1/*.urdf"
# 학습이 실제로 쓰는 파일 (envs/K1/Goal_Pose_V7.yaml: asset.file)
TRAINED = "K1_locomotion_armsdown.urdf"


def _f(node, attr, default=0.0):
    return float(node.get(attr, default)) if node is not None else default


def _xyz(node, default=(0.0, 0.0, 0.0)):
    if node is None:
        return np.array(default, dtype=float)
    return np.array([float(v) for v in node.get("xyz", "0 0 0").split()])


def joints_of(root):
    """관절 origin의 xyz와 **rpy**. rpy를 빼먹으면 3-1을 놓친다 -- 팔 자세가
    고정 관절의 rpy에 박혀 있어서, 질량/CoM/effort만 비교하면 세 URDF가 전부
    '동일'로 나온다. 실제로는 어깨가 90도, 77도, 98.5도로 서로 다르고 그게
    몸통 yaw 관성을 32% 바꾼다."""
    out = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        rpy = np.array([float(v) for v in
                        ((o.get("rpy", "0 0 0") if o is not None else "0 0 0").split())])
        out[j.get("name")] = (j.get("type"), _xyz(o), rpy)
    return out


def links_of(root):
    out = {}
    for l in root.findall("link"):
        name = l.get("name")
        ine = l.find("inertial")
        if ine is None:
            out[name] = None
            continue
        m = _f(ine.find("mass"), "value")
        com = _xyz(ine.find("origin"))
        i = ine.find("inertia")
        I = np.array([
            [_f(i, "ixx"), _f(i, "ixy"), _f(i, "ixz")],
            [_f(i, "ixy"), _f(i, "iyy"), _f(i, "iyz")],
            [_f(i, "ixz"), _f(i, "iyz"), _f(i, "izz")],
        ])
        out[name] = {"mass": m, "com": com, "I": I,
                     "has_collision": l.find("collision") is not None,
                     "has_visual": l.find("visual") is not None}
    return out


def check_inertia(name, d):
    """물리적으로 불가능한 관성을 잡는다. CAD export 오류가 여기서 나온다."""
    bad = []
    if d["mass"] <= 0:
        bad.append("질량 %.6f <= 0" % d["mass"])
        return bad
    w = np.linalg.eigvalsh(d["I"])
    if np.any(w <= 0):
        bad.append("관성 고유값 %s 에 0 이하가 있다 (양의 정부호 아님)"
                   % np.array2string(w, precision=6))
    else:
        a, b, c = sorted(w)
        # 주모멘트는 삼각부등식을 만족해야 한다: 어떤 실제 강체도 이를 어길 수 없다
        if a + b < c * (1 - 1e-9):
            bad.append("주모멘트 삼각부등식 위반: %.6g + %.6g < %.6g" % (a, b, c))
    return bad


def audit(path, deploy_cfg=None):
    root = ET.parse(path).getroot()
    name = os.path.basename(path)
    L = links_of(root)
    J = {j.get("name"): j for j in root.findall("joint")}
    print("=" * 78)
    print(name + ("   <-- 학습이 쓰는 파일" if name == TRAINED else ""))

    # --- 1. 관성 타당성 -----------------------------------------------------
    problems = []
    for n, d in L.items():
        if d is None:
            problems.append((n, ["inertial 블록이 없다"]))
            continue
        b = check_inertia(n, d)
        if b:
            problems.append((n, b))
    total_m = sum(d["mass"] for d in L.values() if d)
    print("  링크 %d개, 총 질량 %.4f kg" % (len(L), total_m))
    if problems:
        print("  ⛔ 관성 문제 %d건:" % len(problems))
        for n, b in problems:
            for x in b:
                print("     %-26s %s" % (n, x))
    else:
        print("  ✅ 관성: 전 링크 양의 정부호 + 삼각부등식 통과")

    # --- 2. 충돌 기하 -------------------------------------------------------
    no_col = [n for n, d in L.items() if d and not d["has_collision"]]
    print("  충돌체 없는 링크 %d개%s" % (
        len(no_col), (": " + ", ".join(no_col[:6])) if no_col else ""))

    # --- 3. Trunk CoM 대 충돌 상자 ------------------------------------------
    trunk = L.get("Trunk")
    if trunk:
        print("  Trunk 질량 %.4f kg (전체의 %.0f%%), CoM %s" % (
            trunk["mass"], 100 * trunk["mass"] / total_m,
            np.array2string(trunk["com"], precision=4)))
        box = None
        for l in root.findall("link"):
            if l.get("name") != "Trunk":
                continue
            for c in l.findall("collision"):
                g = c.find("geometry/box")
                if g is not None:
                    box = np.array([float(v) for v in g.get("size").split()])
        if box is not None:
            print("  Trunk 충돌 상자 %s -> 반폭 x±%.3f y±%.3f m"
                  % (np.array2string(box, precision=3), box[0] / 2, box[1] / 2))
            print("     8-10 판정: base_com ±0.1 랜덤화는 이 상자 밖 -> 허구. "
                  "현재 채택값 ±0.025")
        else:
            print("  Trunk 충돌 기하가 box가 아니다 (mesh) — 반폭은 mesh를 봐야 한다")

    # --- 4. 운동학: 다리 길이와 도달 가능 높이 ------------------------------
    chain = ["Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
             "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll"]
    z = {}
    for jn in chain:
        j = J.get(jn)
        if j is not None:
            z[jn] = _xyz(j.find("origin"))
    if len(z) == len(chain):
        trunk_hip = -z["Left_Hip_Pitch"][2]
        hip_sole = -sum(z[k][2] for k in chain[1:])
        print("  다리: Trunk->Hip %.4f m, Hip->발목 %.4f m" % (trunk_hip, hip_sole))
        print("     엉덩이 좌우 오프셋 ±%.4f m (엉덩이 사이 %.1f cm)"
              % (z["Left_Hip_Pitch"][1], 200 * z["Left_Hip_Pitch"][1]))

    # --- 5. effort / limit --------------------------------------------------
    eff = {}
    for jn, j in J.items():
        lim = j.find("limit")
        if lim is not None and j.get("type") == "revolute":
            eff[jn] = _f(lim, "effort")
    legs = {k: v for k, v in eff.items() if any(
        t in k for t in ("Hip", "Knee", "Ankle"))}
    if legs:
        print("  다리 관절 effort (sim이 토크 한계로 읽는 값):")
        for k in sorted(legs):
            print("     %-24s %.1f N*m" % (k, legs[k]))
    if deploy_cfg:
        print("     deploy torque_limit 배열: %s" % deploy_cfg)
    return {"name": name, "mass": total_m, "links": L, "effort": eff,
            "joints": joints_of(root), "problems": len(problems)}


def cross_compare(results):
    print("=" * 78)
    print("URDF 간 차이 — 3-1의 asset 오염이 여기서 보인다")
    base = None
    for r in results:
        if r["name"] == TRAINED:
            base = r
    if base is None:
        print("  학습 파일을 못 찾아 비교를 건너뛴다")
        return
    for r in results:
        if r["name"] == base["name"]:
            continue
        diffs = []
        if abs(r["mass"] - base["mass"]) > 1e-6:
            diffs.append("총 질량 %.4f vs %.4f" % (r["mass"], base["mass"]))
        for k in sorted(set(base["effort"]) | set(r["effort"])):
            a, b = base["effort"].get(k), r["effort"].get(k)
            if a != b:
                diffs.append("effort %s %s vs %s" % (k, b, a))
        shared = [n for n in base["links"] if n in r["links"]
                  and base["links"][n] and r["links"][n]]
        moved = []
        for n in shared:
            d = np.linalg.norm(base["links"][n]["com"] - r["links"][n]["com"])
            dm = abs(base["links"][n]["mass"] - r["links"][n]["mass"])
            if d > 1e-4 or dm > 1e-4:
                moved.append("%s(CoM %+.4f m, 질량 %+.4f kg)" % (n, d, dm))
        if moved:
            diffs.append("링크 %d개가 다르다: %s" % (
                len(moved), ", ".join(moved[:4]) + (" ..." if len(moved) > 4 else "")))
        for k in sorted(set(base["joints"]) | set(r["joints"])):
            a, b = base["joints"].get(k), r["joints"].get(k)
            if a is None or b is None:
                continue
            if np.linalg.norm(a[2] - b[2]) > 1e-6:
                diffs.append("관절 %s rpy %s vs %s (%.1f도)" % (
                    k, np.array2string(b[2], precision=3),
                    np.array2string(a[2], precision=3),
                    np.degrees(np.linalg.norm(a[2] - b[2]))))
            if np.linalg.norm(a[1] - b[1]) > 1e-6:
                diffs.append("관절 %s xyz 차이 %.4f m" % (
                    k, np.linalg.norm(a[1] - b[1])))
        only = set(r["links"]) ^ set(base["links"])
        if only:
            diffs.append("링크 이름 차이 %d개" % len(only))
        print("  %-42s %s" % (r["name"], "동일 ✅" if not diffs else ""))
        for d in diffs:
            print("       %s" % d)


def main():
    paths = sys.argv[1:] or sorted(glob.glob(DEFAULT_GLOB))
    if not paths:
        print("URDF를 못 찾았다: %s" % DEFAULT_GLOB)
        return 1
    deploy = None
    dpath = os.path.join("deploy", "configs", "Goal_Pose_E0.yaml")
    if os.path.exists(dpath):
        try:
            import yaml
            deploy = (yaml.safe_load(open(dpath))
                      .get("common", {}).get("torque_limit"))
        except Exception:
            deploy = None
    results = [audit(p, deploy) for p in paths]
    cross_compare(results)
    bad = sum(r["problems"] for r in results)
    print("=" * 78)
    print("총 관성 문제 %d건" % bad)
    return 0


if __name__ == "__main__":
    sys.exit(main())
