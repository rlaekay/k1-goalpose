"""수동 LIP 한계주기인가, 이산 제어루프 불안정인가 — 무낙상 대역만으로 가른다.

배경: 매개자 후보를 넷 지웠다(발 간격 `p1` = 낙상의 산물 / 발끼리 충돌 = 두 팔 모두 높음 /
전신 오명령 = 두 팔 서명 동일 / LIP = 방향만). 남은 것이 이 둘이다.

## ① LIP 한계주기 — **측정된 스탠스 반폭 `a` 를 넣어야 진짜 검정이다**

위상 구동 발 놓기에서 LIP 측방 한계주기 진폭은

    amp = a * (1 - 1/cosh(T_half / (2*tau))),  tau = sqrt(l/g) = 0.2142 s

⭐ **진폭은 `a` 에 비례한다.** 주기가 길어지면 cosh 항이 커지지만(+7~15 %),
**`a` 가 같이 좁아지면 예측은 오히려 내려간다.** 그래서 `a` 를 셀마다 재서 넣는다.

⚠️ `a` 를 `p1`(1 퍼센타일 = 극값)으로 잡으면 안 된다. 한계주기의 `a` 는 **명목 스탠스
반폭**이므로 **중앙값/2** 가 맞는 양이다. 둘 다 찍어서 결론이 갈리는지 본다.

  기각: 관측 증가가 `a` 정규화 예측과 **부호가 반대**이면 수동 LIP 은 매개자가 아니다.

## ② 이산 제어루프 불안정 — 세 신호

| 신호 | 루프 불안정이면 | 아니면 |
|---|---|---|
| 정상성 | 측방 진폭이 창 안에서 **증가** | 평평 |
| 지배 주파수 | 봉우리 있고 주기↑ 에서 **낮은 쪽 이동** | 봉우리 없음/케이던스 고조파만 |
| 감쇠 | 주기↑ 에서 자기상관 포락이 **느리게 감쇠** | 불변 |

⛔ **케이던스와 그 고조파를 먼저 뺀다** — 셀마다 케이던스가 다르므로(1.667→1.530 Hz)
안 빼면 케이던스 이동을 "불안정 주파수 이동" 으로 읽는다.
⛔ 주파수 주장은 **Lomb-Scargle 순열 널 + Bonferroni**(CLAUDE.md 평가표준 §5).

    python tools/lip_vs_loop.py logs/mujoco/lipnoise
"""

import os
import sys
import glob
import math

import numpy as np

TAU = 0.2142
NPERM = 200


def load(p):
    z = np.load(p, allow_pickle=True)
    c = [str(x) for x in z["state_cols"]]
    s = z["state"]
    if s.size == 0:
        return None
    d = {n: s[:, i] for i, n in enumerate(c)}
    d["_fall_t"] = z["fall_t"]
    d["_sep"] = z["foot_sep"]
    d["_tilt"] = z["tilt"]
    return d


def lat_series(d):
    """첫 낙상 이전 · 직립 표본의 측방 위치(몸통 yaw 프레임)."""
    t = d["t"]
    end = d["_fall_t"][0] if d["_fall_t"].size else t[-1] + 1.0
    m = (t < end) & (d["_tilt"] < 15.0)
    c, s = np.cos(-d["yaw"][m]), np.sin(-d["yaw"][m])
    y = s * d["px"][m] + c * d["py"][m]
    return t[m], y, d["_sep"][m]


def detrend_cadence(t, y, cad, nharm=4):
    """케이던스와 고조파를 최소자승으로 뺀다. 추세도 같이 뺀다."""
    cols = [np.ones_like(t), t - t.mean()]
    for k in range(1, nharm + 1):
        cols += [np.cos(2 * np.pi * k * cad * t), np.sin(2 * np.pi * k * cad * t)]
    A = np.stack(cols, 1)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def lomb(t, y, freqs):
    """정규화 Lomb-Scargle 주기도(간단 구현)."""
    y = y - y.mean()
    var = y.var()
    out = np.empty(freqs.size)
    for i, f in enumerate(freqs):
        w = 2 * np.pi * f
        s2, c2 = np.sin(2 * w * t).sum(), np.cos(2 * w * t).sum()
        tau_ = 0.5 * math.atan2(s2, c2) / w
        ct, st = np.cos(w * (t - tau_)), np.sin(w * (t - tau_))
        out[i] = ((y * ct).sum() ** 2 / max((ct ** 2).sum(), 1e-12)
                  + (y * st).sum() ** 2 / max((st ** 2).sum(), 1e-12)) / (2 * var)
    return out


def analyse(paths, cad, rng):
    amps, sep_med, sep_p1, stat, peaks, pvals, damp = [], [], [], [], [], [], []
    for p in paths:
        d = load(p)
        if d is None:
            continue
        t, y, sep = lat_series(d)
        if t.size < 2000:
            continue
        r = detrend_cadence(t, y, cad)
        amps.append(float(np.std(r)))
        sep_med.append(float(np.median(sep)))
        sep_p1.append(float(np.percentile(sep, 1)))
        h = t.size // 2
        stat.append(float(np.std(r[h:]) / max(np.std(r[:h]), 1e-12)))     # 후반/전반
        fr = np.linspace(0.2, 8.0, 400)
        P = lomb(t, r, fr)
        k = int(P.argmax())
        peaks.append(float(fr[k]))
        # 순열 널: 값을 섞어 같은 표본시각 위에서 최대 검정력 분포를 만든다
        mx = np.empty(NPERM)
        for j in range(NPERM):
            mx[j] = lomb(t, rng.permutation(r), fr).max()
        pvals.append(float((mx >= P[k]).mean()))
        ac = np.correlate(r - r.mean(), r - r.mean(), "full")[r.size - 1:]
        ac /= ac[0]
        below = np.where(ac < 1 / math.e)[0]
        damp.append(float(below[0] * np.median(np.diff(t))) if below.size else float("nan"))
    if not amps:
        return None
    return dict(n=len(amps), amp=np.mean(amps), amp_sd=np.std(amps),
                sep_med=np.mean(sep_med), sep_p1=np.mean(sep_p1),
                stat=np.mean(stat), peak=np.median(peaks),
                p=max(pvals) if pvals else 1.0, damp=np.nanmean(damp))


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "logs/mujoco/lipnoise"
    rng = np.random.default_rng(0)
    cells = []
    for ms in ("24.0", "25.0", "25.35"):
        ps = sorted(glob.glob(os.path.join(root, "n_%s_s*_series.npz" % ms)))
        cad = (20.0 / float(ms)) * 2.0
        r = analyse(ps, cad, rng)
        if r:
            cells.append((float(ms), cad, r))
    if not cells:
        print("자료 없음")
        return 1

    print("=" * 100)
    print("① LIP 한계주기 — 측정된 스탠스 반폭 `a` 를 넣은 예측")
    print("%-7s %8s %7s %9s %9s %11s %11s %11s"
          % ("주기ms", "케이던스", "n", "a=med/2", "관측진폭", "cosh항", "예측(a포함)", "관측"))
    b_ms, b_cad, b = cells[0]
    b_hs = 1.0 / (2 * b_cad)
    b_f = 1 - 1 / math.cosh(b_hs / (2 * TAU))
    for ms, cad, r in cells:
        hs = 1.0 / (2 * cad)
        f = 1 - 1 / math.cosh(hs / (2 * TAU))
        a = r["sep_med"] / 2
        pred = (a * f) / ((b["sep_med"] / 2) * b_f) - 1
        obs = r["amp"] / b["amp"] - 1
        print("%-7.2f %8.4f %7d %9.5f %9.5f %10.1f%% %10.1f%% %10.1f%%"
              % (ms, cad, r["n"], a, r["amp"], 100 * (f / b_f - 1), 100 * pred, 100 * obs))
    print()
    print("  ⚠️ `a` 를 p1 로 잡으면: ", end="")
    for ms, cad, r in cells:
        print("%.2f→%.4f " % (ms, r["sep_p1"]), end="")
    print()

    print()
    print("=" * 100)
    print("② 이산 루프 불안정 — 세 신호 (케이던스·고조파 제거 후)")
    print("%-7s %11s %13s %10s %9s %11s"
          % ("주기ms", "잔차 SD", "후반/전반", "지배Hz", "순열 p", "자기상관 1/e"))
    for ms, cad, r in cells:
        print("%-7.2f %11.5f %13.3f %10.2f %9.3f %11.3f"
              % (ms, r["amp"], r["stat"], r["peak"], r["p"], r["damp"]))
    print()
    print("판정 ①: 관측이 `a` 포함 예측과 부호 반대면 수동 LIP 은 매개자가 아니다.")
    print("판정 ②: 후반/전반 > 1(비정상) + 주기↑ 에서 지배Hz 하강 + 자기상관 1/e 증가")
    print("        ⇒ 루프 불안정. 순열 p 는 Bonferroni(셀 %d) 로 %.4f 미만이어야 유의."
          % (len(cells), 0.05 / max(len(cells), 1)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
