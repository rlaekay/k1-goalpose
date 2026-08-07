"""개루프 hold 를 MuJoCo 에서 재현한다 — 실기 `--hold-diag` 의 시뮬 대응물.

실기에서 관측된 것(2026-08-08, 사용자 직접):

    사용자가 발목을 눌러 잡고 있을 때   tilt 4.9 deg
    손을 뗐을 때                        tilt 13.8 deg
    같은 구간의 관절 추종 오차          0.6 deg   <- 서보는 명령을 따르고 있다

관절이 명령대로 가 있는데 몸통만 9 deg 더 갔다 = **명령한 자세가 곧게 못 선다.**

`b`(CUSTOM 진입)부터 `r`(보행 시작)까지는 **균형을 닫는 주체가 없다** -- 정책이
균형 제어기인데 아직 안 돌고, 남는 것은 관절 위치 서보뿐이다. 그래서 그 자세는
개루프로 서 있을 수 있어야 하고, 평면에서 발이 평평하고 몸통이 수직이려면

    hip_pitch + knee_pitch + ankle_pitch = 0

  A) prepare 자세  -0.10 + 0.20 + (-0.10) =  0.00
  B) RL 자세       -0.20 + 0.40 + (-0.25) = -0.05  (= 2.87 deg 어긋남)

가설 H: B 자세에서 CoM 수평 위치가 지지다각형 앞경계를 넘거나, 안이더라도 남은
여유를 발목 pitch 토크로 못 버틴다.

  확증: 자유 조건에서 tilt 가 10-15 deg 로 가고, 발 구속 조건에서 4-5 deg 로 내려온다.
        그리고 CoP 가 발끝으로 붙거나 발목 토크가 한계에 붙는다.
  기각: CoM 이 지지다각형 중앙 부근이고 필요 토크가 여유롭다 -> 13.8 deg 는 다른 원인.

⛔ 지지다각형의 전제: **실제로 충돌하는 geom 이 무엇인가.** MuJoCo 에 직접 물어서
(contype/conaffinity) 찍고 시작한다. 상세 mesh 는 `contype=0` 이라 닿지 않고 box 만
닿는다 -- 이것을 확인하지 않고 재면 경계가 틀린다.

정책을 쓰지 않는다. PD 만으로 자세를 붙잡고 무슨 일이 나는지 본다.

    python tools/hold_diag_mujoco.py --duration 10
    python tools/hold_diag_mujoco.py --duration 10 --weld-feet
"""

import os
import sys
import json
import argparse

import numpy as np
import yaml
import mujoco

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPLOY_CFG = os.path.join(ROOT, "deploy", "configs", "Goal_Pose_E0.yaml")


def tilt_deg(quat_wxyz):
    """arccos(-g_body_z). 배포의 tilt 와 같은 정의."""
    w, x, y, z = quat_wxyz
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    g_body = R.T @ np.array([0.0, 0.0, -1.0])
    return float(np.degrees(np.arccos(np.clip(-g_body[2], -1.0, 1.0))))


def foot_contact_geoms(model):
    """실제로 충돌 가능한 발 geom 만 (contype/conaffinity 둘 다 0 이면 시각 전용)."""
    out = {}
    for side in ("left", "right"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "%s_foot_link" % side)
        gs = []
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != bid:
                continue
            live = bool(model.geom_contype[g]) or bool(model.geom_conaffinity[g])
            gs.append((g, int(model.geom_type[g]), live,
                       model.geom_pos[g].copy(), model.geom_size[g].copy()))
        out[side] = gs
    return out


def weld_xml(path):
    """발 두 링크를 world 에 용접한 모델을 문자열로 만든다.

    자산 파일을 건드리지 않는다 -- 공유 파일을 조용히 바꾸면 다른 실행이 오염된다.
    """
    with open(path, "r", encoding="utf-8") as f:
        src = f.read()
    eq = ('<equality>'
          '<weld body1="left_foot_link"/>'
          '<weld body1="right_foot_link"/>'
          '</equality>')
    assert "</mujoco>" in src
    return src.replace("</mujoco>", eq + "</mujoco>")


def settle_height(model, data, q0, nj):
    """발이 지면에 막 닿는 높이로 몸통을 내린다. 낙하 충격을 빼기 위해서다."""
    data.qpos[:] = 0.0
    data.qpos[2] = 1.0
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:7 + nj] = q0[:nj]
    mujoco.mj_forward(model, data)
    lo = np.inf
    for side in ("left", "right"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "%s_foot_link" % side)
        for g in range(model.ngeom):
            if model.geom_bodyid[g] != bid:
                continue
            if not (model.geom_contype[g] or model.geom_conaffinity[g]):
                continue
            # box 는 geom frame 에서 반높이만큼 더 내려간다
            half_z = model.geom_size[g][2] if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else 0.0
            lo = min(lo, float(data.geom_xpos[g][2] - half_z))
    return 1.0 - lo


def run(model, q_hold, kp, kd, lim, nj, duration, dt_log=0.02):
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    data.qpos[2] = settle_height(model, data, q_hold, nj)
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:7 + nj] = q_hold[:nj]
    mujoco.mj_forward(model, data)

    steps = int(duration / model.opt.timestep)
    every = max(1, int(dt_log / model.opt.timestep))
    rows = []
    ank_p = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
             for n in ("Left_Ankle_Pitch", "Right_Ankle_Pitch")]
    ank_idx = [int(model.jnt_dofadr[j]) - 6 for j in ank_p if j >= 0]

    for i in range(steps):
        q = data.qpos[7:7 + nj]
        dq = data.qvel[6:6 + nj]
        tau = np.clip(kp * (q_hold[:nj] - q) - kd * dq, -lim, lim)
        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        if i % every:
            continue
        # 발 접촉만 골라 CoP 와 수직력
        fx = fz = 0.0
        cop_num = 0.0
        for c in range(data.ncon):
            con = data.contact[c]
            bodies = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                        model.geom_bodyid[g]) or "" for g in (con.geom1, con.geom2)]
            if not any("foot" in b for b in bodies):
                continue
            f = np.zeros(6)
            mujoco.mj_contactForce(model, data, c, f)
            fn = float(f[0])            # 접촉 법선 성분
            fz += fn
            cop_num += fn * float(con.pos[0])
        rows.append({
            "t": i * model.opt.timestep,
            "tilt_deg": tilt_deg(data.qpos[3:7]),
            "trunk_x": float(data.qpos[0]),
            "trunk_z": float(data.qpos[2]),
            "track_err_deg": float(np.degrees(np.abs(q_hold[:nj] - q).max())),
            "fz_N": fz,
            "cop_x": (cop_num / fz) if fz > 1e-6 else float("nan"),
            "ank_tau": [float(tau[k]) for k in ank_idx],
        })
    return rows


def summarize(name, rows, lim_ank):
    t = np.array([r["t"] for r in rows])
    tl = np.array([r["tilt_deg"] for r in rows])
    cop = np.array([r["cop_x"] for r in rows])
    at = np.array([max(abs(x) for x in r["ank_tau"]) if r["ank_tau"] else np.nan for r in rows])
    fin = t >= t[-1] - 2.0                     # 마지막 2 초 = 수렴값
    out = {
        "name": name,
        "tilt_t1": float(np.interp(1.0, t, tl)),
        "tilt_final_med": float(np.nanmedian(tl[fin])),
        "tilt_max": float(np.nanmax(tl)),
        "track_err_max_deg": float(max(r["track_err_deg"] for r in rows)),
        "cop_x_final_med": float(np.nanmedian(cop[fin])),
        "ankle_tau_final_med": float(np.nanmedian(at[fin])),
        "ankle_tau_max": float(np.nanmax(at)),
        "ankle_limit": lim_ank,
        "fz_final_med": float(np.nanmedian([r["fz_N"] for r in rows if r["t"] >= t[-1] - 2.0])),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--xml", default=os.path.join(ROOT, "resources", "K1", "K1_serial_realmass.xml"))
    ap.add_argument("--weld-feet", action="store_true",
                    help="발 링크를 world 에 용접한다. 사용자가 발목을 눌러 잡은 조건.")
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "mujoco", "hold_diag.json"))
    a = ap.parse_args()

    cfg = yaml.safe_load(open(DEPLOY_CFG, encoding="utf-8"))
    src = weld_xml(a.xml) if a.weld_feet else None
    model = (mujoco.MjModel.from_xml_string(src) if src
             else mujoco.MjModel.from_xml_path(a.xml))
    nj = model.nu

    # ⛔ 지지다각형의 전제부터 찍는다.
    print("발 geom (contype/conaffinity 로 실제 충돌 여부 판정):")
    for side, gs in foot_contact_geoms(model).items():
        for g, t, live, pos, size in gs:
            tn = {mujoco.mjtGeom.mjGEOM_BOX: "box", mujoco.mjtGeom.mjGEOM_MESH: "mesh"}.get(t, str(t))
            mark = "닿음" if live else "시각전용"
            span = ("x[%+.4f, %+.4f]" % (pos[0] - size[0], pos[0] + size[0])
                    if tn == "box" else "-")
            print("  %-5s geom%-3d %-5s %-8s pos=%s %s" % (side, g, tn, mark, np.round(pos, 4), span))

    prep = cfg["prepare"]
    kp = np.array(prep["stiffness"], dtype=float)[:nj]
    kd = np.array(prep["damping"], dtype=float)[:nj]
    lim = np.array(cfg["common"]["torque_limit"], dtype=float)[:nj]
    lim_ank = float(lim[14])          # Ankle_Pitch

    poses = {
        "A_prepare(sum=0)": np.array(prep["default_qpos"], dtype=float),
        "B_rl(sum=-0.05)": np.array(cfg["common"]["default_qpos"], dtype=float),
    }
    results = []
    for name, q in poses.items():
        rows = run(model, q, kp, kd, lim, nj, a.duration)
        s = summarize(name, rows, lim_ank)
        s["weld_feet"] = bool(a.weld_feet)
        results.append(s)
        print("\n[%s]%s" % (name, "  (발 용접)" if a.weld_feet else ""))
        print("  tilt  1s %.2f deg | 최종 median %.2f | 최대 %.2f"
              % (s["tilt_t1"], s["tilt_final_med"], s["tilt_max"]))
        print("  관절 추종오차 최대 %.2f deg" % s["track_err_max_deg"])
        print("  CoP x 최종 median %.4f m | 수직력 %.1f N" % (s["cop_x_final_med"], s["fz_final_med"]))
        print("  발목 pitch 토크 최종 %.1f / 최대 %.1f (한계 %.0f) N*m"
              % (s["ankle_tau_final_med"], s["ankle_tau_max"], lim_ank))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"xml": a.xml, "weld_feet": bool(a.weld_feet),
                   "duration_s": a.duration, "results": results}, f, indent=2)
    print("\n결과: %s" % a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
