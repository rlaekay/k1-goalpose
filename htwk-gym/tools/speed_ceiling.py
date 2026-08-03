"""천장이 로봇에 있나, 과제에 있나 — 이미 있는 eval 출력만 읽어서 가른다.

P2는 몸통속도 1.3~1.5 m/s를 요구한다. 보고된 두 수치는 서로 반대로 들린다:
구간최고 p90은 1.29 m/s(P2 하한에 사실상 닿음)인데 1 m/s 초과 체류는 3.1%뿐이다.
둘 다 참일 수 있고, 어느 쪽이 중요한지는 요약통계가 답하지 못하는 것 하나에 달렸다 --
**그 peak이 순항인가 가속 과도구간인가.**

과제는 목표를 dx ±2 m, dy ±1.5 m에서 뽑는다. 그래서 구간 거리 median이 1.38 m이고
최대가 2.50 m다. 1.38 m 홉은 가속하다 바로 감속에 들어간다. 그 peak은 **구조상**
과도구간이고, 그런 구간을 아무리 평균 내도 애초에 유지된 적 없는 순항속도는 나오지 않는다.

그래서 거리로 묶고 peak을 거리의 함수로 읽는다:

  - 가장 긴 구간에서도 peak이 **아직 오르고 있다** -> 천장은 과제 것이다.
    정책은 거리에 막힌 것이고, sprint 카테고리(R4)면 충분하다.
  - peak이 1.3 한참 아래에서 **평평하다** -> 천장은 정책 것이다. sprint를 넣으면
    수요만 만들어지고 정책이 못 따라간다. R4는 과제 분포뿐 아니라 정책을 건드려야 한다.

GPU를 쓰지 않는다. 롤아웃도 새로 돌지 않는다. 디스크에 있는 segments.csv만 읽는다.

force-ON 셀과 clean 셀은 **절대 섞지 않고** 따로 보고한다 -- 그 혼입이 round_summary를
틀리게 만들었던 결함이다(ibatch.md §8-9).

    python tools/speed_ceiling.py                    # 기본 위치 전부
    python tools/speed_ceiling.py logs/force_ab/*/   # 특정 run만
"""

import os
import sys
import csv
import glob
import json

import numpy as np

EDGES = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, float("inf")]
DEFAULT_ROOTS = ("logs/force_ab", "logs/i2b_flat")


def _pct(a, p):
    return float(np.percentile(a, p)) if len(a) else float("nan")


def load(seg_path):
    """segments.csv + 옆의 report.json. 없으면 None."""
    out_dir = os.path.dirname(seg_path)
    rep_path = os.path.join(out_dir, "report.json")
    if not os.path.exists(rep_path):
        return None
    try:
        rep = json.load(open(rep_path, encoding="utf-8"))
    except (OSError, ValueError):
        return None
    cols = {}
    with open(seg_path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        head = next(r, None)
        if not head:
            return None
        want = {"start_dist_m", "duration_s", "peak_speed_mps",
                "mean_speed_mps", "category"}
        # 순항 컬럼은 이 컬럼을 넣기 전에 돌린 eval에는 없다. 없으면 peak만 보고
        # 하고, 있으면 순항까지 본다 -- 옛 결과를 못 읽게 만들 이유가 없다.
        extra = {"cruise_1p3_s", "time_above_1p3_s"}
        idx = {k: head.index(k) for k in want | extra if k in head}
        if not want <= set(idx):
            return None
        for k in idx:
            cols[k] = []
        for row in r:
            for k, i in idx.items():
                cols[k].append(row[i])
    if not cols["start_dist_m"]:
        return None
    num = {k: np.array(v, dtype=float) for k, v in cols.items() if k != "category"}
    num["_has_cruise"] = "cruise_1p3_s" in cols
    num["category"] = np.array(cols["category"], dtype=object)
    num["_force"] = rep.get("force_profile")
    num["_terrain"] = rep.get("eval_terrain")
    num["_ckpt"] = rep.get("checkpoint", "?")
    return num


def report(label, d):
    dist, dur = d["start_dist_m"], d["duration_s"]
    peak = d["peak_speed_mps"]
    ok = np.isfinite(dist) & np.isfinite(peak) & np.isfinite(dur) & (dur > 0)
    dist, dur, peak = dist[ok], dur[ok], peak[ok]
    if not len(dist):
        print("  (유효 구간 없음)")
        return None
    req = dist / dur
    print("  %s" % label)
    print("  외력 %-8s 바닥 %-6s 구간 %d" % (d["_force"], d["_terrain"], len(dist)))
    print("  %-12s %6s %8s %8s %8s %8s %9s %9s"
          % ("거리 구간", "구간수", "요구med", "peak med", "peak p90", "peak max",
             ">=1.0 m/s", ">=1.3 m/s"))
    rows = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (dist >= lo) & (dist < hi)
        if m.sum() < 20:            # 표본이 적은 칸은 순위를 만들 수 없다
            continue
        p = peak[m]
        rows.append((lo, hi, int(m.sum()), float(np.median(req[m])),
                     float(np.median(p)), _pct(p, 90), float(p.max()),
                     float((p >= 1.0).mean()), float((p >= 1.3).mean())))
        print("  %4.1f–%-7.1f %6d %8.2f %8.2f %8.2f %8.2f %8.1f%% %8.1f%%"
              % (lo, hi if np.isfinite(hi) else 9.9, rows[-1][2], rows[-1][3],
                 rows[-1][4], rows[-1][5], rows[-1][6],
                 100 * rows[-1][7], 100 * rows[-1][8]))
    if d.get("_has_cruise"):
        cru = d["cruise_1p3_s"][ok]
        lm = dist >= 2.0
        if lm.sum() >= 20:
            print("  순항(>=2.0 m 구간 %d개): 최장연속 median %.2f s, p90 %.2f s, "
                  "0.5s 이상 %.1f%%, 1.0s 이상 %.1f%%"
                  % (int(lm.sum()), float(np.median(cru[lm])), _pct(cru[lm], 90),
                     100 * float((cru[lm] >= 0.5).mean()),
                     100 * float((cru[lm] >= 1.0).mean())))
    else:
        print("  순항: 미기록 (이 eval은 cruise_1p3_s 컬럼 이전이다 — peak만 판단 가능)")
    return rows


def verdict(rows):
    """마지막 두 칸에서 peak이 아직 오르는지 본다."""
    if len(rows) < 2:
        print("  판정: 거리 칸이 2개 미만 — 판별 불가")
        return
    (_, _, _, _, m1, p1, _, _, _) = rows[-2]
    (lo, hi, n, _, m2, p2, mx, s10, s13) = rows[-1]
    d_med, d_p90 = m2 - m1, p2 - p1
    print("  가장 긴 두 칸의 peak median %.2f -> %.2f (%+.2f), p90 %.2f -> %.2f (%+.2f)"
          % (m1, m2, d_med, p1, p2, d_p90))
    if d_med >= 0.05 or d_p90 >= 0.05:
        print("  => 아직 오르고 있다. 거리에 막혀 있다는 신호 — 천장은 과제 쪽.")
    elif m2 >= 1.3:
        print("  => 평평하지만 이미 1.3 이상이다. P2를 순간최고로 읽으면 충족.")
    else:
        print("  => 평평하고 1.3 미만이다. 정책 쪽 천장일 수 있다 — sprint만으로는 부족.")
    print("  주의: 이 과제의 최장 거리가 2.50 m다. 2.5 m에서 오르는 중이라면 "
          "천장은 아직 관측된 적이 없는 것이고, 이 데이터로는 상한을 말할 수 없다.")


def main():
    pats = sys.argv[1:] or DEFAULT_ROOTS
    seen, found = set(), 0
    for pat in pats:
        for base in sorted(glob.glob(pat)):
            for seg in sorted(glob.glob(os.path.join(base, "**", "segments.csv"),
                                        recursive=True)):
                if seg in seen:
                    continue
                seen.add(seg)
                d = load(seg)
                if d is None:
                    continue
                found += 1
                print("=" * 92)
                rows = report(os.path.relpath(os.path.dirname(seg)), d)
                if rows:
                    verdict(rows)
                print()
    if not found:
        print("segments.csv를 못 찾았다. 경로를 인자로 주면 된다: "
              "python tools/speed_ceiling.py 'logs/force_ab/*/'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
