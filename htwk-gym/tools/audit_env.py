"""Dump what the simulator ACTUALLY does, not what the yaml says it does.

Every expensive bug in this project has been a gap between the two:

  * disturbance was submitted on 1 of 10 decimation substeps, so the delivered
    impulse was a tenth of the configured one, and eval reported it 10x high
  * the first disturbance event was drawn from the whole interval, so no env
    could be pushed before 8 s -- 27% of every episode was silently clean
  * feet_edge_pos described a foot 39% longer than the collision box PhysX
    actually pushes with, so the contact SIGNAL and the contact PHYSICS
    disagreed
  * the grid curriculum promoted on a quantity the leash had already bounded,
    so it could not fail and saturated in about a minute

None of those are visible by reading a config file. This builds the real task,
steps it, and reports measured runtime values beside the configured ones, so a
mismatch is a line of output rather than a week.

Usage:
    python tools/audit_env.py --config sweeps/I1b_force.yaml
    python tools/audit_env.py --config envs/K1/Goal_Pose_V7.yaml --steps 600
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET

import isaacgym  # noqa: F401  (must precede torch)
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs import *  # noqa: F401,F403  (registers task classes)
from utils.runner import get_task_class

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = 78


def head(t):
    print("\n" + "=" * W + "\n" + t + "\n" + "=" * W)


def row(label, configured, measured, note=""):
    flag = ""
    if configured is not None and measured is not None and str(configured) != str(measured):
        flag = "  <<< 불일치"
    print("  %-30s cfg %-16s 실측 %-16s%s%s"
          % (label, configured, measured, note, flag))


def foot_geometry(cfg):
    """feet_edge_pos vs the collision box vs the visual mesh, in one place."""
    head("발 형상 — 접촉 신호 / 물리 / 실제 메시")
    edge = cfg["asset"]["feet_edge_pos"]
    xs = [p[0] for p in edge]
    ys = [p[1] for p in edge]
    print("  feet_edge_pos (접촉 신호)   길이 %.4f  폭 %.4f  바닥 z %.4f"
          % (max(xs) - min(xs), max(ys) - min(ys), edge[0][2]))
    urdf = os.path.join(ROOT, cfg["asset"]["file"])
    try:
        r = ET.parse(urdf).getroot()
    except Exception as e:
        print("  URDF를 못 읽었다: %r" % (e,))
        return
    for link in r.iter("link"):
        if link.get("name") != "left_foot_link":
            continue
        for c in link.findall("collision"):
            g = c.find("geometry")
            box = g.find("box") if g is not None else None
            o = c.find("origin")
            if box is None:
                continue
            sx, sy, _sz = [float(v) for v in box.get("size").split()]
            ox, _oy, oz = [float(v) for v in (o.get("xyz", "0 0 0").split() if o is not None
                                              else ["0", "0", "0"])]
            print("  collision box (PhysX)      길이 %.4f  폭 %.4f  바닥 z %.4f"
                  % (sx, sy, oz - float(box.get("size").split()[2]) / 2))
            print("     -> 신호가 물리보다 길이 %+.1f%%  폭 %+.1f%%"
                  % (100 * ((max(xs) - min(xs)) / sx - 1),
                     100 * ((max(ys) - min(ys)) / sy - 1)))
            print("     (신호가 더 크면 공중의 발을 '접지'로 보고한다)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--task", default=None, help="default: config's basic.task")
    ap.add_argument("--num_envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--sim_device", default="cuda:0")
    ap.add_argument("--rl_device", default="cuda:0")
    a = ap.parse_args()

    with open(a.config, encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    task = a.task or cfg["basic"]["task"]
    cfg["basic"].update(task=task, headless=True,
                        sim_device=a.sim_device, rl_device=a.rl_device)
    cfg["env"]["num_envs"] = a.num_envs
    cfg["viewer"]["record_video"] = False
    torch.manual_seed(0)
    np.random.seed(0)

    print("config : %s" % a.config)
    print("task   : %s" % task)
    print("asset  : %s" % cfg["asset"]["file"])

    foot_geometry(cfg)

    env = get_task_class(task.split("/")[-1])(cfg)
    env.reset()

    head("인터페이스 — warm start 호환성")
    row("num_obs", cfg["env"].get("num_observations"), env.num_obs)
    row("num_actions", cfg["env"].get("num_actions"), env.num_actions)
    row("num_dofs", None, env.num_dofs)
    row("control dt", None, "%.4f s (%.0f Hz)" % (env.dt, 1.0 / env.dt))
    row("decimation", cfg["control"]["decimation"], cfg["control"]["decimation"])

    head("PD 게인 — 실기에서 조정 가능한 값과 맞는가")
    print("  cfg stiffness: %s" % cfg["control"]["stiffness"])
    print("  cfg damping  : %s" % cfg["control"]["damping"])
    # Per joint GROUP. Pooling Hip/Knee (100) with Ankle (50) makes min/median
    # meaningless -- the first version of this reported a ratio of 0.487 and
    # looked like a randomisation bug when it was just 50/100.
    ds = env.dof_stiffness.float()
    dd = env.dof_damping.float()
    r = (cfg["randomization"].get("dof_stiffness") or {}).get("range")
    print("  랜덤화 범위 %s (stiffness)" % (r,))
    names = getattr(env, "dof_names", None) or []
    groups = {}
    for i, nm in enumerate(names):
        for g in cfg["control"]["stiffness"]:
            if g in nm:
                groups.setdefault(g, []).append(i)
                break
    if not groups:
        print("  dof_names를 못 얻었다 — 전체 범위만: stiffness %.2f~%.2f, damping %.2f~%.2f"
              % (ds.min(), ds.max(), dd.min(), dd.max()))
    for g, idx in sorted(groups.items()):
        sg = ds[:, idx]
        dg = dd[:, idx]
        nom = float(cfg["control"]["stiffness"][g])
        print("  %-6s stiffness cfg %-6.1f 실측 %.2f~%.2f (비 %.3f~%.3f)   damping cfg %-5.1f 실측 %.2f~%.2f"
              % (g, nom, sg.min(), sg.max(), float(sg.min()) / nom, float(sg.max()) / nom,
                 float(cfg["control"]["damping"][g]), dg.min(), dg.max()))

    head("도메인 랜덤화 — 설정이 실제로 표본에 반영됐는가")
    # base_mass / friction are applied to rigid-body and shape properties at
    # actor creation, not held as a per-env tensor, so the sampled multiplier is
    # what to look at -- base_mass_scaled columns are [com.x, com.y, com.z, mass].
    for key, attr in (("joint_encoder_bias", "joint_encoder_bias"),
                      ("joint_target_offset", "joint_target_offset")):
        rng = (cfg["randomization"].get(key) or {}).get("range")
        t = getattr(env, attr, None)
        if t is None:
            row(key, rng, "런타임 텐서 없음")
        else:
            zero = "  (전부 0 = 이 arm에서 꺼짐)" if float(t.abs().max()) == 0 else ""
            row(key, rng, "[%+.4f, %+.4f]" % (float(t.min()), float(t.max())), zero)
    bm = getattr(env, "base_mass_scaled", None)
    if bm is not None and bm.numel():
        print("  %-30s cfg %-16s 실측 mass배율 %.3f~%.3f, CoM %+.3f~%+.3f m"
              % ("base_mass / base_com",
                 (cfg["randomization"].get("base_mass") or {}).get("range"),
                 float(bm[:, 3].min()), float(bm[:, 3].max()),
                 float(bm[:, :3].min()), float(bm[:, :3].max())))
    fr = getattr(env, "friction_coeffs", None)
    print("  %-30s cfg %-16s 실측 %s"
          % ("friction", (cfg["randomization"].get("friction") or {}).get("range"),
             ("%.3f~%.3f" % (float(fr.min()), float(fr.max()))) if fr is not None
             else "actor 생성 시 shape 속성에 적용 (텐서로 보관 안 함)"))

    head("외란 — 실제로 몇 스텝에, 얼마나 큰 힘이 걸리는가")
    d = cfg["randomization"].get("disturbance") or {}
    print("  enabled=%s  interval_s=%s" % (d.get("enabled", False), d.get("interval_s")))
    if d.get("enabled", False):
        print("  collision force_n=%s duration_s=%s"
              % ((d.get("collision") or {}).get("force_n"),
                 (d.get("collision") or {}).get("duration_s")))
        print("  support   force_n=%s duration_s=%s"
              % ((d.get("support") or {}).get("force_n"),
                 (d.get("support") or {}).get("duration_s")))
        nxt = getattr(env, "dist_next", None)
        if nxt is not None:
            print("  첫 이벤트까지 남은 스텝: min %d  median %d  max %d"
                  % (int(nxt.min()), int(nxt.median()), int(nxt.max())))
            print("     (min이 크면 에피소드 앞부분이 통째로 무외란이다)")

    act = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    active, fmax, tmax, hits = 0, 0.0, 0.0, 0
    for _ in range(a.steps):
        env.step(act)
        pf = getattr(env, "pushing_forces", None)
        if pf is not None:
            mag = pf.norm(dim=-1)
            n = int((mag > 1e-6).any(dim=-1).sum())
            if n:
                active += 1
                hits += n
                fmax = max(fmax, float(mag.max()))
            pt = getattr(env, "pushing_torques", None)
            if pt is not None:
                tmax = max(tmax, float(pt.norm(dim=-1).max()))
    print("  %d 스텝 중 외란 활성 %d 스텝 (%.0f%%), 연 env-히트 %d"
          % (a.steps, active, 100.0 * active / max(a.steps, 1), hits))
    print("  실측 최대 힘 %.1f N,  최대 토크 %.2f N*m" % (fmax, tmax))
    if d.get("enabled", False) and active == 0:
        print("  !!! enabled인데 한 번도 안 걸렸다 — 위상 초기화나 타이머를 볼 것")

    head("gait / 케이던스")
    cc = cfg["commands"].get("cadence_coupling") or {}
    print("  cadence_coupling enabled=%s %s" % (cc.get("enabled", False),
                                                {k: v for k, v in cc.items() if k != "enabled"}))
    gf = env.gait_frequency.float()
    live = gf > 1e-8
    print("  cfg gait_frequency %s" % (cfg["commands"]["gait_frequency"],))
    if int(live.sum()):
        g = gf[live]
        print("  실측 (정지 env 제외) min %.2f  median %.2f  max %.2f Hz"
              % (g.min(), g.median(), g.max()))
        print("  -> 보속 %.1f~%.1f 보/초, 1.5 m/s에 필요한 보폭 %.3f~%.3f m"
              % (2 * float(g.min()), 2 * float(g.max()),
                 1.5 / (2 * float(g.max())), 1.5 / (2 * float(g.min()))))
    print("  정지(클럭 0) env: %d / %d" % (int((~live).sum()), env.num_envs))

    head("보상 — 0이 아닌 항만")
    for k, v in sorted(cfg["rewards"]["scales"].items()):
        if float(v) != 0.0:
            print("  %-24s %s" % (k, v))
    for k in ("base_height_target", "goal_reach_radius", "constellation_weight",
              "constellation_radius", "stop_speed_threshold"):
        if k in cfg["rewards"]:
            print("  [param] %-17s %s" % (k, cfg["rewards"][k]))
    print("  init_state.pos z = %s   <-- base_height_target와 다르면 보상이 스폰을 끌어내린다"
          % cfg["init_state"]["pos"][2])

    head("goal 관측 노이즈 — 씻기는 것 / 안 씻기는 것")
    n = cfg.get("noise", {})
    for k in ("goal_pos", "goal_heading", "goal_pos_bias", "goal_heading_bias"):
        if k in n:
            print("  %-20s %s %s" % (k, n[k].get("range"),
                                     "(구간 고정 = 시간평균으로 안 씻김)" if "bias" in k else ""))
    print("  goal_obs_hold_steps  %s" % n.get("goal_obs_hold_steps"))
    fl = n.get("goal_bt_flicker") or {}
    print("  goal_bt_flicker      prob/step %s  radius %s m" % (fl.get("prob_per_step"),
                                                                fl.get("radius_m")))
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
