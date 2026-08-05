"""실기 관절 로그를 MuJoCo 운동학에 재생해 발 간격·높이를 뽑는다.

왜 이 방식인가
--------------
직접 FK를 짰다가 실패했다(2026-08-05: 몸통 pitch -34도, 발이 몸 앞 23 cm). 평행 발목
때문에 단순 직렬 체인으로 안 풀린다. 그런데 **시뮬레이터는 그 운동학을 이미 갖고 있다.**
그래서 실기 로그의 관절각을 MuJoCo `qpos`에 그대로 써넣고 `mj_forward`만 돌리면,
내가 FK를 다시 짤 필요 없이 발 위치가 나온다.

이것으로 답하는 것:
  * 실기에서 두 발이 **얼마나 가까워졌는가** (사용자 증언: "발끼리 부딪혀서 넘어져")
  * 그 값이 MuJoCo 보행과 얼마나 다른가
  * 발이 지면 아래로 파고드는가(= 접촉 판정과 실제가 어긋나는가)

⚠️ 몸통 자세는 로그의 IMU roll/pitch로 세운다. yaw는 발 간격에 영향이 없어 0으로 둔다.
⚠️ 절대 높이는 뜻이 없다(로그에 몸통 높이가 없다). **좌우 발의 상대 위치만** 읽는다.

    python3 tools/replay_real_in_mujoco.py 실기.csv [비교용_mujoco.csv]
"""

import os
import sys
import csv
import math

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco

MJCF = "resources/K1/K1_serial.xml"
LEG_START_QPOS = 7 + 10          # free joint 7 + 앞 10관절(머리/팔)


def foot_geom_ids(model):
    """발 충돌 상자의 geom id. 시각 메시가 아니라 **충돌 상자**를 쓴다 --
    메시는 발목 하우징까지 5.2 cm를 포함해 접촉면이 아니다."""
    out = {}
    for side in ("left", "right"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "%s_foot_link" % side)
        for g in range(model.ngeom):
            if model.geom_bodyid[g] == bid and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
                out[side] = g
    return out


def corners(model, data, gid):
    """발 상자의 바닥 네 모서리를 월드 좌표로."""
    pos = data.geom_xpos[gid]
    R = data.geom_xmat[gid].reshape(3, 3)
    sx, sy, sz = model.geom_size[gid]
    pts = []
    for ex in (-sx, sx):
        for ey in (-sy, sy):
            pts.append(pos + R @ np.array([ex, ey, -sz]))
    return np.array(pts)


def analyse(path, model, data, gids, label):
    rows = list(csv.DictReader(open(path)))
    if "q0" not in rows[0]:
        print("%s: 관절 채널이 없다(옛 포맷)" % label)
        return None
    sep_min, sep_lat, ov = [], [], []
    for r in rows:
        q = [float(r["q%d" % i]) for i in range(12)]
        roll, pitch = float(r["roll"]), float(r["pitch"])
        data.qpos[:] = 0.0
        data.qpos[2] = 0.6
        cr, sr = math.cos(roll / 2), math.sin(roll / 2)
        cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
        data.qpos[3:7] = [cr * cp, sr * cp, cr * sp, -sr * sp]   # rpy(yaw=0) -> quat
        data.qpos[LEG_START_QPOS:LEG_START_QPOS + 12] = q
        mujoco.mj_forward(model, data)
        L = corners(model, data, gids["left"])
        R = corners(model, data, gids["right"])
        # 두 발 사이 최소 거리 (모서리 대 모서리)
        d = min(np.linalg.norm(a - b) for a in L for b in R)
        sep_min.append(d)
        # 좌우(y) 간격만: 왼발 최소 y - 오른발 최대 y
        sep_lat.append(L[:, 1].min() - R[:, 1].max())
    a = np.array(sep_min) * 100
    b = np.array(sep_lat) * 100
    print("== %s (%d 표본) ==" % (label, len(rows)))
    print("  두 발 최소거리(cm)   p1 %5.2f  p10 %5.2f  median %5.2f" % (
        np.percentile(a, 1), np.percentile(a, 10), np.median(a)))
    print("  좌우 간격(cm)        p1 %5.2f  p10 %5.2f  median %5.2f" % (
        np.percentile(b, 1), np.percentile(b, 10), np.median(b)))
    print("  좌우 간격 < 0 (겹침) %.1f%%   < 1 cm %.1f%%" % (
        100 * (b < 0).mean(), 100 * (b < 1).mean()))
    return dict(sep_min=a, sep_lat=b)


def main():
    if len(sys.argv) < 2:
        print(__doc__); return 1
    model = mujoco.MjModel.from_xml_path(MJCF)
    data = mujoco.MjData(model)
    gids = foot_geom_ids(model)
    print("발 충돌 상자 geom: %s" % gids)
    print("발 반폭 y = %.4f m -> 두 발이 붙으면 좌우 간격 0" % model.geom_size[gids["left"]][1])
    print()
    out = []
    for i, p in enumerate(sys.argv[1:]):
        out.append(analyse(p, model, data, gids, os.path.basename(p)))
        print()
    if len(out) >= 2 and out[0] and out[1]:
        r, m = out[0], out[1]
        print("== 실기 / MuJoCo 비 ==")
        print("  좌우 간격 p10: %.2f / %.2f cm" % (
            np.percentile(r["sep_lat"], 10), np.percentile(m["sep_lat"], 10)))
        print("  겹침 비율:     %.1f%% / %.1f%%" % (
            100 * (r["sep_lat"] < 0).mean(), 100 * (m["sep_lat"] < 0).mean()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
