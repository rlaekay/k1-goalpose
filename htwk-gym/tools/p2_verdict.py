"""P2를 닫는다: 0.40초 순항이 정책의 한계인가 구간 길이의 한계인가.

측정된 것: 2.0-2.5 m 구간에서 1.3 m/s 이상을 최장 median 0.40 s 유지한다.
그 값만으로는 두 가지를 구분할 수 없다 --

  (a) 정책이 그 이상 못 버틴다            -> R4는 정책/보상을 건드려야 한다
  (b) 2.25 m 구간이 그 이상을 담을 수 없다 -> sprint 카테고리면 충분하다

가속도를 재면 갈린다. 정지에서 출발해 거리 d를 가고 다시 멈춰야 할 때(goal_reached가
|v| < 0.1을 요구한다), 1.3 m/s 이상으로 보낼 수 있는 시간의 상한은

    t_max = (d - 1.3^2/(2a) - 1.3^2/(2a_dec)) / 1.3

이다. 1.3까지 가속하는 거리와 다시 멈추는 거리를 빼고 남은 만큼을 1.3으로 달리는 것이
시간을 최대화하는 주행이기 때문이다.

가속도는 `1.0 / time_to_1p0_s`로 잡으면 안 된다 -- 그 값에는 무게 이동 같은 기동 사구간이
들어 있어 가속도가 과소평가되고, 그러면 t_max도 같이 작아져 (b) 쪽으로 결론이 기운다.
`time_to_0p5_s`가 함께 기록되므로 사구간이 빠진 구간 가속도를 쓴다:

    a = (1.0 - 0.5) / (time_to_1p0_s - time_to_0p5_s)

감속도는 측정되지 않는다. 대칭(a_dec = a)이 t_max를 가장 작게 만들어 (b)에 가장 유리한
가정이므로, 그 가정에서도 관측이 t_max에 못 미친다면 (a)를 배제할 수 없다. 제동이 가속보다
빠른 경우까지 같이 찍는다.

GPU를 쓰지 않는다. 이미 나온 segments.csv만 읽는다.

    python tools/p2_verdict.py logs/cruise_i2a
"""

import os
import sys
import csv
import glob

import numpy as np

V_TARGET = 1.3          # P2 하한
LONG_M = 2.0            # 이 아래 거리는 순항을 담을 수 없다
AT_REST_MPS = 0.30      # eval_goal_pose.py:2439와 같은 "정지 출발" 기준
NEED = ("start_dist_m", "initial_speed_mps", "time_to_0p5_s", "time_to_1p0_s",
        "cruise_1p3_s", "peak_speed_mps", "duration_s")


def load(path):
    cols = {k: [] for k in NEED}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        head = next(r, None)
        if not head or not set(NEED) <= set(head):
            return None
        idx = {k: head.index(k) for k in NEED}
        for row in r:
            for k, i in idx.items():
                cols[k].append(row[i])
    return {k: np.array(v, dtype=float) for k, v in cols.items()}


def analyse(label, d):
    long_rest = ((d["start_dist_m"] >= LONG_M)
                 & (d["initial_speed_mps"] <= AT_REST_MPS)
                 & np.isfinite(d["time_to_0p5_s"])
                 & np.isfinite(d["time_to_1p0_s"]))
    n_long = int((d["start_dist_m"] >= LONG_M).sum())
    n = int(long_rest.sum())
    print("  %s" % label)
    print("    >=%.1f m 구간 %d개 중 정지출발+1.0 m/s 도달 %d개" % (LONG_M, n_long, n))
    if n < 20:
        print("    표본 부족 — 판정 불가")
        return None
    t05, t10 = d["time_to_0p5_s"][long_rest], d["time_to_1p0_s"][long_rest]
    dist, obs = d["start_dist_m"][long_rest], d["cruise_1p3_s"][long_rest]
    dead = t05                                   # 0.5 m/s에 닿기까지 = 기동 사구간 포함
    gap = t10 - t05
    ok = gap > 1e-6
    a_lin = np.full(n, np.nan)
    a_lin[ok] = 0.5 / gap[ok]
    a_naive = 1.0 / np.maximum(t10, 1e-6)
    print("    기동: 0.5 m/s까지 %.2f s, 1.0까지 %.2f s (median)"
          % (np.median(dead), np.median(t10)))
    print("    가속도  사구간포함 1.0/t10 = %.2f m/s^2   |   사구간제외 0.5/(t10-t05) = %.2f m/s^2"
          % (np.median(a_naive), np.nanmedian(a_lin)))
    rows = []
    for name, ratio in (("대칭 a_dec=a", 1.0), ("제동 1.5배", 1.5), ("제동 2배", 2.0)):
        m = ok & np.isfinite(a_lin)
        brake = a_lin[m] * ratio
        tmax = (dist[m] - V_TARGET ** 2 / (2 * a_lin[m])
                - V_TARGET ** 2 / (2 * brake)) / V_TARGET
        tmax = np.maximum(tmax, 0.0)
        use = obs[m] / np.maximum(tmax, 1e-9)
        rows.append((name, float(np.median(tmax)), float(np.median(obs[m])),
                     float(np.median(use)), float((obs[m] >= 0.9 * tmax).mean())))
        print("    %-12s 이론최대 median %.2f s | 관측 %.2f s | 활용률 median %.0f%% | "
              "상한의 90%% 이상 달성 %.0f%%"
              % (name, rows[-1][1], rows[-1][2], 100 * rows[-1][3], 100 * rows[-1][4]))
    return rows


def main():
    pats = sys.argv[1:] or ["logs/cruise_i2a"]
    allrows, found = [], 0
    for pat in pats:
        for base in sorted(glob.glob(pat)):
            for p in sorted(glob.glob(os.path.join(base, "**", "segments.csv"),
                                      recursive=True)):
                d = load(p)
                if d is None:
                    print("  %s -> 필요한 컬럼 없음 (옛 eval)" % p)
                    continue
                found += 1
                r = analyse(os.path.relpath(os.path.dirname(p)), d)
                if r:
                    allrows.append(r)
                print()
    if not allrows:
        print("판정할 데이터가 없다.")
        return 0
    print("=" * 78)
    sym = [r[0] for r in allrows]
    use = float(np.median([s[3] for s in sym]))
    print("측정 요약: 활용률 median %.0f%%" % (100 * use))
    print()
    print("⛔ 이 활용률로 '정책이 게으르다'를 결론지으면 안 된다. 지표가 속도가 아니다.")
    print("   t_max는 1.3 m/s '이상으로 보낸 시간'의 상한인데, 그 시간은 정확히 1.3으로")
    print("   갈 때 최대가 된다. 더 빨리 가면 거리를 빨리 지나가서 오히려 줄어든다:")
    print("     2.25 m 구간, a=1.67 기준 -- 1.45 m/s -> 1.3초과 0.86 s, 도착 2.82 s")
    print("                                 1.94 m/s -> 1.3초과 0.76 s, 도착 2.72 s")
    print("   즉 활용률은 '얼마나 빠른가'가 아니라 '얼마나 1.3에 붙어 가는가'를 잰다.")
    print("   활용률을 올리라는 것은 더 느리고 균일하게 가라는 뜻이 될 수 있다.")
    print()
    print("이 도구가 실제로 말할 수 있는 것:")
    print("  - 가속도(사구간 제외)와, 그 가속도에서 구간이 담을 수 있는 순항 시간의 상한")
    print("  - 상한이 1 s 안팎이면 그 구간은 순항 여부를 판정할 만큼 길지 않다는 것")
    print("주의: 가속도는 능력이 아니라 행동이다. 최대 요구속도 0.62 m/s인 과제에서")
    print("      빨리 가속할 이유가 없다. sprint 아래에서는 상한 자체가 올라간다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
