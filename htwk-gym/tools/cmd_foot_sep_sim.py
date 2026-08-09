#!/usr/bin/env python3
"""Isaac 롤아웃의 **명령 발간격**을 잰다. GPU 0 -- `--dump_actions` npz 만 읽는다.

    python tools/cmd_foot_sep_sim.py logs/eval_rounds/v2/N3_pathcross/seed*.walk/actions.npz \
        --json logs/eval_rounds/v2/N3_pathcross/cmdsep.json

---- 왜 -----------------------------------------------------------------------

분석 세션 요청(2026-08-09). MJC 의 명령-FK 검정이 실기 로그에서 찾은 것:
**교차가 있는 모든 칸에서 명령이 실측보다 깊다**(실기 붕괴 명령 −0.138 대 실측
−0.065). 즉 물리는 교차를 만드는 게 아니라 **완화**한다 -- 다리가 지령된 교차
자세까지 못 간다. ⇒ 정책을 재려면 **명령** 쪽을 봐야 한다. 실측만 보면 물리가
가려 준 만큼 레버 효과가 과소평가된다.

`tools/cmd_foot_sep.py` 가 실기·MuJoCo **CSV** 로 같은 일을 한다. 이 파일은 같은
FK 를 **Isaac eval 의 npz** 에 건다 -- FK 는 그쪽에서 import 하므로 **구현이 하나**다.
(두 벌로 두면 언젠가 한쪽만 고쳐지고, 이 저장소는 그 실패를 기하에서 네 번 겪었다.)

⛔ 종점 규칙 (분석 세션 사전등록):
  * `min` 을 쓰지 마라 -- 1 시드 극값이다. MJC 가 그것으로 헛짚었다.
  * **p1 + 음수체류율**을 쓴다. 분석 단위는 **시드**다(한 롤아웃 안의 스텝은
    유사반복이다). 시드별 값을 내고 `tools/stat_compare.py` 가 Wilcoxon 을 건다.
  * 정확도 cm 로 판정하지 마라(C48).

⛔ 표본에서 빼는 것: `done` 스텝과 리셋 직후 `--reset-guard-s`(기본 0.25 s, eval
   의 `reset_guard_s` 와 같은 값). 리셋 직후 자세는 정책이 지령한 것이 아니라
   스폰 자세이고, 그것을 넣으면 낙상이 잦은 arm 일수록 "명령이 좋아 보인다" --
   생존 편향과 정확히 같은 모양의 인공물이 명령 축에도 생긴다.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cmd_foot_sep as CFS          # noqa: E402  -- FK 는 이 한 벌만 쓴다

import mujoco                       # noqa: E402

# `CFS.foot_sep` 가 qpos 에 그대로 밀어넣는 순서. MJCF 힌지 순서와 같아야 한다.
LEG_ORDER = [
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw",
    "Left_Knee_Pitch", "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw",
    "Right_Knee_Pitch", "Right_Ankle_Pitch", "Right_Ankle_Roll",
]


def leg_index_map(dof_names):
    """Isaac dof 이름 -> `LEG_ORDER` 위치. 이름으로 맵한다(순서를 가정하지 않는다)."""
    names = [str(n) for n in dof_names]
    idx = []
    for want in LEG_ORDER:
        if want not in names:
            raise SystemExit(
                "dof_names 에 {} 가 없다. 이 npz 는 다른 로봇 레이아웃이다:\n  {}"
                .format(want, names))
        idx.append(names.index(want))
    return np.asarray(idx, dtype=int)


def verify_mjcf_order(model):
    """MJCF 힌지 순서가 `LEG_ORDER` 와 같은지 확인한다.

    `CFS.foot_sep` 는 `qpos[7+LEG0 : 7+LEG0+12] = legs` 로 **연속·순서 가정**을
    하고 있다. 자산이 바뀌면 조용히 틀린 다리에 각을 넣게 되므로 여기서 깬다.
    """
    adr = []
    for name in LEG_ORDER:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise SystemExit("MJCF 에 관절 {} 가 없다: {}".format(name, CFS.MJCF))
        adr.append(int(model.jnt_qposadr[jid]))
    want = list(range(7 + CFS.LEG0, 7 + CFS.LEG0 + 12))
    if adr != want:
        raise SystemExit(
            "MJCF 다리 qpos 주소가 cmd_foot_sep 의 가정과 다르다.\n"
            "  기대 {}\n  실제 {}\n"
            "⛔ `CFS.foot_sep` 의 슬라이스 대입이 틀린 관절에 각을 넣는다. "
            "자산이 바뀌었으면 LEG0 과 이 목록을 같이 고쳐라.".format(want, adr))


def live_mask(done, guard_steps):
    """`done` 스텝과 그 직후 guard 스텝을 뺀 불리언 마스크 [T, k]."""
    done = np.asarray(done, dtype=bool)
    T, k = done.shape
    keep = np.ones((T, k), dtype=bool)
    keep[done] = False
    if guard_steps > 0:
        for g in range(1, guard_steps + 1):
            keep[g:][done[:-g]] = False        # g >= 1 이므로 done[:-g] 는 항상 유효
    return keep


def before_first_fall(fell, guard_steps):
    """각 env 의 **첫 낙상 이전** 구간만 남기는 마스크 [T, k].

    왜 필요한가(분석 세션, 2026-08-09): 직립 필터(tilt<15°)로는 부족하다.
    **tilt 5~15° 는 이미 넘어지는 중의 허둥댐이고 거기서 다리가 엉킨다.**
    그대로 두면 **낙상이 잦은 arm 일수록 발간격이 나빠 보이는 것이 낙상의 결과**이고,
    레버 효과와 낙상률이 교락된다. `N1` 대 `N3` 은 낙상률이 다를 것이 예상되므로
    (그게 종점 4다) 이 교락이 주 종점에 직격한다.

    ⛔ `done` 이 아니라 `fell` 로 자른다. `done` 은 **에피소드 타임아웃도 포함**해서,
    그걸로 자르면 도착을 못 하는 arm 일수록 더 많이 잘린다 = 새 편향을 만든다.
    """
    fell = np.asarray(fell, dtype=bool)
    T, k = fell.shape
    idx = np.where(fell.any(axis=0), fell.argmax(axis=0), T)   # 낙상 없으면 T(=전부)
    keep = np.arange(T)[:, None] < idx[None, :]
    if guard_steps > 0:                     # 최초 스폰 직후도 정책이 지령한 자세가 아니다
        keep[:guard_steps] = False
    return keep


def stats_of(v):
    a = np.asarray(v, dtype=float)
    if a.size == 0:
        return None
    return dict(
        n=int(a.size),
        p1=float(np.percentile(a, 1)),
        p5=float(np.percentile(a, 5)),
        median=float(np.median(a)),
        neg_share=float((a < 0.0).mean()),
        # ⛔ 이 0.07 을 `rewards.feet_min_gap` 0.07 과 같은 양으로 읽지 마라.
        # 여기 간격은 **발 body 원점 사이 거리**이고(영자세 +0.1924 m, 실측 확인),
        # 보상 쪽은 `|feet_y_offset + feet_distance_ref|` 라 기준이 다르다.
        # 숫자가 우연히 같아서 더 위험하다. 이 열은 `cmd_foot_sep.py` 의 `below7`
        # 과 같은 정의이므로 MJC 의 실기/MuJoCo 값과는 그대로 비교된다.
        below_7cm=float((a < 0.07).mean()),
    )


def process(path, model, data, guard_steps):
    z = np.load(path, allow_pickle=True)
    idx = leg_index_map(z["dof_names"])
    scale = float(z["action_scale"])
    # ⛔ `env.default_dof_pos` 는 Isaac 에서 **(1, num_dofs)** 다(env 축으로 브로드캐스트
    # 하려고). 그대로 `[None, None, :]` 하면 (1,1,1,D) 가 되어 (T,K,D) 와 4차원으로
    # 조용히 브로드캐스트되고, 그 뒤 인덱싱이 **엉뚱한 축**을 집는다. 모양 검사가
    # 없었으면 숫자가 나왔을 것이고 그 숫자는 전부 쓰레기였을 것이다.
    default = np.asarray(z["default_dof_pos"], dtype=np.float64)
    if default.ndim == 2:
        if default.shape[0] != 1 and not np.allclose(default, default[0:1]):
            raise SystemExit("default_dof_pos 의 env 행이 서로 다르다 {} -- "
                             "명령 재구성이 env 마다 달라야 한다.".format(default.shape))
        default = default[0]
    default = default.reshape(-1)
    if default.size != np.asarray(z["actions"]).shape[-1]:
        raise SystemExit("default_dof_pos {} 와 actions 마지막 축 {} 가 안 맞는다".format(
            default.size, np.asarray(z["actions"]).shape[-1]))
    actions = np.asarray(z["actions"], dtype=np.float64)      # [T, k, nA] 클립 적용됨
    dof_pos = np.asarray(z["dof_pos"], dtype=np.float64)      # [T, k, nJ]
    done = np.asarray(z["done"])

    clip = z["clip_actions"] if "clip_actions" in z.files else None
    if clip is None:
        print("  ⚠️ {}: clip_actions 가 npz 에 없다 -- 클립 **전** 액션일 수 있다. "
              "그러면 명령 진폭이 과대평가된다(v7 clip=1.0 은 상시 걸린다). "
              "이 덤프는 인용하지 마라.".format(os.path.basename(path)))

    # 명령 목표 = default_dof_pos + action_scale * action  (goal_pose.py:926)
    # `joint_target_offset` 은 일부러 뺀다 -- 그것은 랜덤화가 더한 영점 오차이지
    # 정책이 지령한 값이 아니다. 여기서 재려는 것은 **정책의 의도**다.
    q_cmd_full = default[None, None, :] + scale * actions
    q_cmd = q_cmd_full[:, :, idx]
    q_meas = dof_pos[:, :, idx]

    keep = live_mask(done, guard_steps)
    if keep.shape != q_cmd.shape[:2]:
        raise SystemExit("done {} 과 actions {} 의 모양이 안 맞는다".format(
            keep.shape, q_cmd.shape[:2]))

    # 두 판을 **나란히** 낸다. 방향이 같으면 레버 효과가 낙상과 독립이라 결론 확정이고,
    # 크게 달라지면 전 구간 판은 낙상의 그림자였다는 뜻이다(분석 세션 판정 규약).
    views = {"all": keep}
    if "fell" in z.files:
        views["pre_fall"] = before_first_fall(z["fell"], guard_steps)
    else:
        print("  ⚠️ npz 에 `fell` 이 없다(옛 덤프) -- **첫낙상 이전 판을 못 낸다.** "
              "주 종점이 낙상률과 교락된 채로 남는다. 새 덤프로 다시 뽑아야 한다.")

    def sep_pair(mask):
        ci, mi = q_cmd[mask], q_meas[mask]
        sc = [CFS.foot_sep(model, data, ci[j], 22) for j in range(ci.shape[0])]
        sm = [CFS.foot_sep(model, data, mi[j], 22) for j in range(mi.shape[0])]
        return dict(cmd=stats_of(sc), meas=stats_of(sm),
                    kept=int(mask.sum()), total=int(mask.size))

    return {name: sep_pair(m) for name, m in views.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+", help="--dump_actions 로 만든 .npz (glob 가능)")
    ap.add_argument("--reset-guard-s", type=float, default=0.25,
                    help="리셋 직후 이만큼을 뺀다. eval 의 evaluation.reset_guard_s 와 맞춰라")
    ap.add_argument("--json", default=None, help="시드별 값을 여기에 쓴다(stat_compare 용)")
    args = ap.parse_args()

    paths = []
    for p in args.npz:
        paths.extend(sorted(glob.glob(p)) or [p])
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        sys.exit("npz 가 하나도 없다.")

    model = mujoco.MjModel.from_xml_path(CFS.MJCF)
    data = mujoco.MjData(model)
    verify_mjcf_order(model)

    print("=" * 100)
    print("명령 발간격 대 실측 발간격 (Isaac 롤아웃)   자산 {}".format(
        os.path.basename(CFS.MJCF)))
    print("=" * 100)
    print("⛔ `min` 은 안 찍는다 -- 1 시드 극값이라 판정에 쓰면 안 된다. p1 과 음수체류율로 읽어라.")
    print()

    rows = {}
    for p in paths:
        dt_guess = 0.02
        try:
            dt_guess = float(np.load(p, allow_pickle=True)["dt"])
        except Exception:
            pass
        guard = int(round(args.reset_guard_s / max(dt_guess, 1e-6)))
        r = process(p, model, data, guard)
        rows[p] = r
        label = os.path.basename(os.path.dirname(p)) or os.path.basename(p)
        print(label)
        for view, vlabel in (("all", "전 구간"), ("pre_fall", "**첫낙상 이전**")):
            v = r.get(view)
            if v is None:
                continue
            print("  [{}]  표본 {:,}/{:,} 스텝·env (guard {} 스텝 제외)".format(
                vlabel, v["kept"], v["total"], guard))
            for tag, s in (("명령 target", v["cmd"]), ("실측 q", v["meas"])):
                if s is None:
                    print("     {:<12} 표본 없음".format(tag))
                    continue
                print("     {:<12} p1 {:+.4f}  p5 {:+.4f}  median {:+.4f}  "
                      "**음수 {:5.2f}%**  <7cm {:5.2f}%".format(
                          tag, s["p1"], s["p5"], s["median"],
                          100 * s["neg_share"], 100 * s["below_7cm"]))
            if v["cmd"] and v["meas"]:
                d = v["cmd"]["p1"] - v["meas"]["p1"]
                print("     => 명령 p1 이 실측보다 {:+.4f} m {} (음수면 **물리가 교차를 가려 준다**)"
                      .format(d, "깊다" if d < 0 else "얕다"))
        print()

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({os.path.abspath(k): v for k, v in rows.items()}, f,
                      ensure_ascii=False, indent=2)
        print("시드별 값 -> {}".format(args.json))
        print("다음: tools/stat_compare.py 로 **시드 단위** Wilcoxon 을 걸어라. "
              "한 롤아웃 안의 스텝을 n 으로 쓰면 없는 검정력을 만든다.")
        print()
        print("⛔ 판정 규약(분석 세션): **두 판을 나란히 읽는다.**")
        print("   * 전 구간과 첫낙상 이전에서 방향이 **같다** ⇒ 레버 효과가 낙상과 독립. 확정.")
        print("   * 방향/크기가 **크게 다르다**  ⇒ 전 구간 판은 낙상의 그림자였다.")
        print("     **첫낙상 이전 판이 정답**이다 -- tilt 5~15° 는 이미 넘어지는 중이고")
        print("     거기서 다리가 엉키므로, 낙상이 잦은 arm 일수록 발간격이 나빠 보인다.")
        print("⛔ `p1` 은 **역치량이 아니다**(MJC 사전등록 기각: p1 +0.017 인데 3시드 전부 낙상).")
        print("   주 종점은 **음수체류율**이고 p1 은 '레버가 마진을 얼마나 움직였나' 로만 읽어라.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
