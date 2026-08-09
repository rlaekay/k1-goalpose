"""매개자 검정 둘 — 수집된 npz 만 읽는다. 새 롤아웃 0, GPU 0.

① LIP 측방 발산 (분석 세션 사전등록)
   `p1` 이 죽은 뒤 남은 후보다. 원래 H1 의 기전은 발 기하가 아니라 **측방 발산**이었다:
   반스텝이 길어지면 e^{t/tau} 로 측방 오차가 더 자란 채 다음 발을 딛는다.

     예측: 발산 인자 ∝ exp((T/2)/tau), tau = 0.214 s
           24.0 -> 26.14 ms 는 반스텝 0.250 -> 0.272 s ⇒ **+10.6 %**
     기각: 관측 증가가 예측의 **절반 미만**이거나 **부호가 반대**

   ⭐ 이 검정의 가치는 **자유모수 없이 크기까지 사전 고정**된다는 것이다. `p1` 은
   "낮으면 나쁘다" 라는 방향만 있어서 낙상의 그림자와 구별이 안 됐다.

   ⚠️ 측정은 **첫 낙상 이전 구간만** 쓴다. 낙상 중의 측방 운동을 넣으면 오늘 `p1` 에서
   겪은 것을 그대로 반복한다.

② 낙상 직전 명령 서명 (배포 세션 사전등록)
   오염은 **관측에만** 걸렸는데도 넘어진다 ⇒ 잘못된 발목 상태를 본 정책이 전신 명령을
   잘못 내는가? 낙상 −0.5~0 s 대 같은 팔의 −4~−2 s 를 채널별로 비교한다.

     판정: 오염 팔에서만 **힙** 명령 서명이 크면 "전신 오명령" 확증.
           두 팔 서명이 같으면 기전은 명령 밖(관측 지연 등)이다.

    python tools/mediator_test.py logs/mujoco/mediator
"""

import os
import sys
import glob

import numpy as np

TAU = 0.214
LEG = ["HipP", "HipR", "HipY", "Knee", "AnkP", "AnkR"] * 2
HIP = [0, 1, 2, 6, 7, 8]
ANK = [4, 5, 10, 11]


def load(path):
    z = np.load(path, allow_pickle=True)
    cols = [str(c) for c in z["state_cols"]]
    s = z["state"]
    if s.size == 0:
        return None
    d = {c: s[:, i] for i, c in enumerate(cols)}
    d["_fall_t"] = z["fall_t"]
    d["_tilt"] = z["tilt"]
    return d


def lateral(d):
    """몸통 yaw 프레임의 측방 위치·속도."""
    c, s = np.cos(-d["yaw"]), np.sin(-d["yaw"])
    y = s * d["px"] + c * d["py"]
    vy = s * d["vx"] + c * d["vy"]
    return y, vy


def divergence(d):
    """착지(지지 1->2)마다 측방 발산 지표. 첫 낙상 이전 구간만."""
    t = d["t"]
    end = d["_fall_t"][0] if d["_fall_t"].size else t[-1] + 1.0
    m = t < end
    if m.sum() < 200:
        return None
    y, vy = lateral(d)
    y, vy, sup, tt = y[m], vy[m], d["support"][m], t[m]
    land = np.where((sup[1:] >= 2) & (sup[:-1] < 2))[0] + 1
    if land.size < 6:
        return None
    # 착지 시점의 측방 불안정 모드 |xi| = |y + vy*tau| (반스텝 평균을 뺀 편차)
    xi = y + vy * TAU
    out = {"n_land": int(land.size), "window_s": float(tt[-1] - tt[0])}
    seg_pk, seg_xi = [], []
    for a, b in zip(land[:-1], land[1:]):
        if b - a < 3:
            continue
        yy = y[a:b]
        seg_pk.append(float(np.abs(yy - yy.mean()).max()))   # 반스텝 내 측방 진폭
        seg_xi.append(float(abs(xi[b] - xi[a - 1] if a else xi[b])))
    if len(seg_pk) < 5:
        return None
    out["lat_amp_median"] = float(np.median(seg_pk))
    out["vy_land_p90"] = float(np.percentile(np.abs(vy[land]), 90))
    out["xi_step_median"] = float(np.median(seg_xi))
    out["half_step_s"] = float(np.median(np.diff(tt[land])))
    return out


def cmd_signature(d, pre=(-0.5, 0.0), ctrl=(-4.0, -2.0)):
    """낙상 직전 대 대조 창의 채널별 명령 변화량 RMS 와 명령-실측 오차."""
    t, ft = d["t"], d["_fall_t"]
    if ft.size == 0:
        return None
    cmd = np.stack([d["cmd%d" % i] for i in range(12)], 1)
    q = np.stack([d["q%d" % i] for i in range(12)], 1)
    dcmd = np.abs(np.diff(cmd, axis=0))
    err = np.abs(cmd - q)
    res = {}
    for lab, (a, b) in (("pre", pre), ("ctrl", ctrl)):
        m = np.zeros(t.size, bool)
        for f in ft:
            m |= (t >= f + a) & (t < f + b)
        if m.sum() < 30:
            continue
        md = m[:-1] & m[1:]
        res[lab] = {
            "n": int(m.sum()),
            "dcmd_hip": float(np.sqrt((dcmd[md][:, HIP] ** 2).mean())),
            "dcmd_ank": float(np.sqrt((dcmd[md][:, ANK] ** 2).mean())),
            "err_hip": float(err[m][:, HIP].mean()),
            "err_ank": float(err[m][:, ANK].mean()),
        }
    return res or None


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "logs/mujoco/mediator"
    print("=" * 96)
    print("① LIP 측방 발산 — 예측: 반스텝 T/2 에 대해 exp((T/2)/%.3f)" % TAU)
    print("%-16s %8s %10s %11s %12s %11s" % ("셀", "착지수", "반스텝s", "측방진폭m", "|vy|착지p90", "예측대비"))
    base = None
    for p in sorted(glob.glob(os.path.join(root, "clean_*_series.npz"))):
        d = load(p)
        if d is None:
            continue
        r = divergence(d)
        if r is None:
            print("%-16s (표본 부족)" % os.path.basename(p)[6:-11])
            continue
        if base is None:
            base = r
            pred = "기준"
        else:
            e = np.exp((r["half_step_s"] - base["half_step_s"]) / TAU)
            obs = r["lat_amp_median"] / base["lat_amp_median"]
            pred = "예측 %+.1f%% / 관측 %+.1f%%" % (100 * (e - 1), 100 * (obs - 1))
        print("%-16s %8d %10.4f %11.5f %12.4f %11s"
              % (os.path.basename(p)[6:-11], r["n_land"], r["half_step_s"],
                 r["lat_amp_median"], r["vy_land_p90"], pred))
    print()
    print("=" * 96)
    print("② 낙상 직전 명령 서명 — 낙상 −0.5~0 s 대 대조 −4~−2 s")
    print("%-20s %-6s %10s %10s %10s %10s" % ("셀", "창", "Δ명령 힙", "Δ명령 발목", "오차 힙", "오차 발목"))
    for pat in ("clean_25.7_series.npz", "corr_25.0_s*_series.npz"):
        for p in sorted(glob.glob(os.path.join(root, pat))):
            d = load(p)
            if d is None:
                continue
            r = cmd_signature(d)
            if not r:
                continue
            for lab in ("ctrl", "pre"):
                if lab in r:
                    v = r[lab]
                    print("%-20s %-6s %10.5f %10.5f %10.5f %10.5f"
                          % (os.path.basename(p)[:-11], lab, v["dcmd_hip"],
                             v["dcmd_ank"], v["err_hip"], v["err_ank"]))
    print()
    print("판정 ①: 관측이 예측의 절반 미만이거나 부호 반대면 LIP 발산도 매개자가 아니다.")
    print("판정 ②: 오염 팔에서만 힙 Δ명령이 크면 '전신 오명령' 확증. 두 팔이 같으면 명령 밖.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
