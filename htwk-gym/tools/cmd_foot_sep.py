"""명령 발간격 대 실측 발간격 — 다리 교차를 정책이 **지령**하는가, 추종이 실패하는가.

분석 세션 제안(2026-08-09). GPU 0, 새 롤아웃 0. 이미 있는 로그만 읽는다.

  명령 목표 = default_dof_pos, 다리만 += action_scale * action
              (deploy/utils/policy_goal_pose.py:287-288 확인)

두 각도 집합을 각각 **같은 FK** 에 넣어 좌우 발 중심의 부호 있는 측방 간격을 낸다.
몸통을 단위자세에 고정하므로 간격은 관절각만의 함수다.

  - 명령 쪽에 **이미 교차가 있으면** -> armature 는 원인이 아니다. 정책이 교차를
    지령하고 있고 armature 는 다리가 거기까지 가느냐만 바꾼다. 고칠 곳은 물리가
    아니라 보상/학습이다.
  - 명령은 안 교차하는데 **실측만 교차하면** -> 추종 실패이고 armature 가설이 산다.

    python tools/cmd_foot_sep.py ../realdata/2026-08-09_t_walk_i3b_run2.csv \
        logs/mujoco/mjc_request_C/C0_base.csv logs/mujoco/mjc_request_C/C6_armzero.csv
"""

import os
import csv
import sys

import numpy as np
import yaml
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MJCF = os.path.join(ROOT, "resources", "K1", "K1_serial_realmass.xml")
CFG = os.path.join(ROOT, "deploy", "configs", "Goal_Pose_E0.yaml")
LEG0 = 10          # 다리 첫 관절 인덱스 (22관절 배열 기준)


def foot_sep(model, data, legs, nj):
    """다리 12각을 넣고 좌우 발 중심의 부호 있는 측방 간격(m). 양수=정상, 음수=교차."""
    data.qpos[:] = 0.0
    data.qpos[3] = 1.0                       # 단위 쿼터니언 -> 몸통 프레임 = 월드
    data.qpos[7 + LEG0:7 + LEG0 + 12] = legs
    mujoco.mj_forward(model, data)
    lb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot_link")
    rb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot_link")
    return float(data.xpos[lb][1] - data.xpos[rb][1])


def stats(v):
    a = np.array(v)
    return dict(n=a.size, min=float(a.min()), p1=float(np.percentile(a, 1)),
                median=float(np.median(a)), mean=float(a.mean()),
                neg=float((a < 0).mean()), below7=float((a < 0.07).mean()))


def main():
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit(__doc__)
    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    default_q = np.array(cfg["common"]["default_qpos"], dtype=np.float64)
    scale = float(cfg["policy"]["control"]["action_scale"])
    nj = default_q.size
    model = mujoco.MjModel.from_xml_path(MJCF)
    data = mujoco.MjData(model)
    print("action_scale %.3f, 다리 default %s" % (scale, np.round(default_q[LEG0:LEG0 + 6], 3)))
    print()
    for p in paths:
        rows = list(csv.DictReader(open(p, newline="", encoding="utf-8")))
        need = ["q%d" % i for i in range(12)] + ["act%d" % i for i in range(12)]
        if not rows or not set(need) <= set(rows[0]):
            print("%-40s 필요한 q/act 열이 없다 -- 건너뜀" % os.path.basename(p))
            continue
        meas, cmd = [], []
        for r in rows:
            q = np.array([float(r["q%d" % i]) for i in range(12)])
            a = np.array([float(r["act%d" % i]) for i in range(12)])
            meas.append(foot_sep(model, data, q, nj))
            cmd.append(foot_sep(model, data, default_q[LEG0:LEG0 + 12] + scale * a, nj))
        sm, sc = stats(meas), stats(cmd)
        print("%s  (%d행)" % (os.path.basename(p), sm["n"]))
        for lab, s in (("실측 q", sm), ("명령 target", sc)):
            print("   %-12s min %+.4f  p1 %+.4f  median %+.4f  음수 %5.1f%%  <7cm %5.1f%%"
                  % (lab, s["min"], s["p1"], s["median"], 100 * s["neg"], 100 * s["below7"]))
        v = "명령에 이미 교차" if sc["min"] < 0 else "명령은 교차 없음"
        print("   => %s (명령 min %+.4f m)" % (v, sc["min"]))
        print()
    print("판정: 명령 쪽에 교차가 있으면 armature 가 원인이 아니라 정책이 교차를 지령한다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
