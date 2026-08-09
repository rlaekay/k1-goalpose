#!/usr/bin/env python3
"""**arm 을 고르는 문제인가, 언제 멈출지를 고르는 문제인가.** GPU 0.

    python tools/ckpt_vs_arm_variance.py --root logs --min-duration 120

---- 왜 -----------------------------------------------------------------------

분석 세션 제안(2026-08-09). 내가 낸 관측 하나에서 나왔다:

    arm 간 (체크포인트 고정 model_6000): NH 18.9 %  대 NF 4.6 %  = 4.1배
    arm 내 (NF 하나):  best.pth(it100) 27.5 %  대 model_6000 4.6 % = 6.0배

⇒ **체크포인트가 arm 보다 크게 움직이는 것처럼 보인다.** 그런데 그 비교에는
**선택 편향**이 있다 -- `best.pth` 는 좋도록 **고른 점**이고 `model_6000` 은 **끝점**이다.
최댓값 대 끝점을 비교하면 arm 내 분산이 과대평가된다. 게다가 `best.pth` 를 고르는
`.select` 낙상 수가 낙관 편향돼 있다(28쌍 부호검정 p=0.0000).

그래서 **고르지 않은 점들로** 다시 본다: 한 run 의 **저장된 모든 체크포인트**를
시도 분모 도착률로 재집계해 그 분포의 IQR 을 내고, **iteration 을 맞춘 arm 간 차이**와 비교한다.

    arm 내 IQR > arm 간 차이  ⇒ 레버가 아니라 **"언제 멈출지"** 가 1차 변수다.
                                40+ arm 을 6000 iteration 씩 돌린 것이 곧
                                **과최적화 구간을 사는 데 GPU 를 쓴 것**이 된다.
    반대                      ⇒ arm 선택이 여전히 1차 변수다.

⛔ 같은 run 안에서도 **프로토콜이 갈리면 비교 불가**다(게이트 문턱이 프로토콜에 달려 있다).
run × 프로토콜로 묶고, 한 묶음 안에 체크포인트가 3개 이상일 때만 IQR 을 낸다.
"""
import argparse
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict


def dig(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def wilson(s, n, z=1.959963985):
    if not n:
        return (float("nan"), float("nan"))
    ph = s / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = (len(xs) - 1) * q
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return xs[lo] if lo == hi else xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs")
    ap.add_argument("--min-duration", type=float, default=120.0)
    ap.add_argument("--min-ckpts", type=int, default=3)
    args = ap.parse_args()

    # (run, protocol) -> {iteration: (성공수, 시도수)}  같은 셀을 여러 번 잰 경우 최신 하나만.
    cells = defaultdict(dict)
    for path in glob.glob(os.path.join(args.root, "**", "report.json"), recursive=True):
        try:
            with open(path) as f:
                r = json.load(f)
        except Exception:
            continue
        if (r.get("duration_s") or 0) < args.min_duration:
            continue
        if r.get("goal_pattern"):          # waypoint 만
            continue
        strict, seg = dig(r, "success_rate_strict"), dig(r, "segments_completed")
        if strict is None or not seg:
            continue
        ck = r.get("checkpoint") or ""
        run = ck.split("/nn/")[0].split("/")[-1]
        m = re.search(r"model_(\d+)", ck.split("/nn/")[-1])
        if not m:                          # best.pth 는 **고른 점**이라 뺀다 (선택 편향)
            continue
        it = int(m.group(1))
        psha = (r.get("effective_eval_protocol_sha") or "?")[:8]
        falls = int(r.get("falls") or 0)
        direct = dig(r, "success_per_attempt", "strict")
        if direct is not None:
            n_att = dig(r, "success_per_attempt", "attempts") or 0
            s_cnt = int(round(direct * n_att))
        else:
            s_cnt, n_att = int(round(float(strict) * int(seg))), int(seg) + falls
        cells[(run, psha)][it] = (s_cnt, n_att, falls)

    groups = {k: v for k, v in cells.items() if len(v) >= args.min_ckpts}
    if not groups:
        sys.exit("체크포인트 {}개 이상인 (run, 프로토콜) 묶음이 없다.".format(args.min_ckpts))

    print("=" * 96)
    print("arm 내(체크포인트) 분산   —   도착률/시도, waypoint, >= {:.0f}s".format(args.min_duration))
    print("=" * 96)
    print("⛔ `best.pth` 는 뺐다 — **좋도록 고른 점**이라 넣으면 arm 내 분산이 과대평가된다.")
    print()

    iqrs = []
    for (run, psha), d in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        its = sorted(d)
        vals = [d[i][0] / d[i][1] for i in its]
        q1, q3 = quantile(vals, 0.25), quantile(vals, 0.75)
        iqr = q3 - q1
        iqrs.append((run, psha, iqr, min(vals), max(vals), len(its)))
        print("{}   [proto {}]   체크포인트 {}개".format(run[:46], psha, len(its)))
        for i in its:
            s, n, f = d[i]
            lo, hi = wilson(s, n)
            print("    it{:<7} 도착률 {:6.1%}  [{:5.1%}, {:5.1%}]   낙상 {:>5}  시도 {}".format(
                i, s / n, lo, hi, f, n))
        print("    → IQR **{:.1f} %p**   범위 {:.1%} ~ {:.1%}  (최대/최소 {:.1f}배)".format(
            100 * iqr, min(vals), max(vals),
            (max(vals) / min(vals)) if min(vals) > 0 else float("inf")))
        print()

    print("=" * 96)
    print("판정")
    print("=" * 96)
    best = max(iqrs, key=lambda x: x[2])
    print("가장 큰 arm 내 IQR : {} [{}]  **{:.1f} %p**  (최대/최소 {:.1f}배, n={}개 체크포인트)".format(
        best[0][:34], best[1], 100 * best[2],
        (best[4] / best[3]) if best[3] > 0 else float("inf"), best[5]))
    print()
    print("이 값을 **iteration 을 맞춘 arm 간 차이**와 비교해라. 알려진 값:")
    print("    NH_zeroclock 대 NF_dwellclock, 둘 다 model_6000, proto 6400ca58")
    print("        18.9 % [17.8, 20.0]  대  4.6 % [4.0, 5.2]   ⇒ 차이 **14.3 %p** (4.1배)")
    print()
    print("읽는 법:")
    print("  * arm 내 IQR > 14.3 %p 면 → **레버가 아니라 '언제 멈출지'가 1차 변수다.**")
    print("    40+ arm 을 6000 iteration 씩 돌린 것이 과최적화 구간을 산 것이 된다.")
    print("  * arm 내 IQR < 14.3 %p 면 → arm 선택이 여전히 1차 변수다.")
    print("  * 각 체크포인트의 Wilson 폭이 ~±1 %p 라 **분포의 퍼짐은 표본오차가 아니다.**")
    print("  * ⚠️ run 마다 학습 레시피가 달라 IQR 을 arm 간에 평균 내지 마라. 최댓값으로 읽어라.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
