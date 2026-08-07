"""pareto_scan.sh 의 출력을 iteration 순 표로 만들고 파레토 프론트를 표시한다.

두 축은 방향이 반대다: 위치오차는 작을수록 좋고 보행속도는 클수록 좋다.
그래서 "무엇이 최고인가"라는 질문에는 답이 없고, **지배당하지 않는 집합**만 있다.
어떤 체크포인트 A 가 다른 B 보다 두 축 모두에서 나쁘면 A 는 볼 이유가 없다 --
그것만 걸러내도 배포 후보가 몇 개로 줄어든다.

낙상은 세 번째 축이 아니라 **문턱**으로 쓴다. 보행 낙상률이 배포 정책(0.604)의
1/100 인 0.006 을 넘으면 후보에서 뺀다 -- 속도가 아무리 좋아도 넘어지면 소용없고,
실제로 배포 정책은 속도 median 1.12 m/s 인데 60 % 구간에서 넘어졌다.

    python tools/pareto_table.py logs/pareto/<run>
"""

import os
import re
import sys
import json
import glob

FALL_LIMIT = 0.006


def g(d, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
        if cur is None:
            return None
    return cur


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    root = sys.argv[1]
    rows = {}
    for rep in glob.glob(os.path.join(root, "*", "report.json")):
        d = os.path.basename(os.path.dirname(rep))
        m = re.match(r"it(\d+)\.(accuracy|walk)$", d)
        if not m:
            continue
        it, kind = int(m.group(1)), m.group(2)
        try:
            with open(rep, encoding="utf-8") as f:
                rows.setdefault(it, {})[kind] = json.load(f)
        except (OSError, ValueError):
            continue

    if not rows:
        print("결과가 없다: {}/it*/report.json".format(root))
        return 1

    recs = []
    for it in sorted(rows):
        a, w = rows[it].get("accuracy"), rows[it].get("walk")
        if not a or not w:
            continue
        recs.append({
            "it": it,
            "err": (g(a, "pos_err_m", "median") or float("nan")) * 100.0,
            "strict": (g(a, "success_rate_strict") or 0.0) * 100.0,
            "afall": g(a, "fall_rate_per_attempt"),
            "spd": g(w, "body_speed", "median") or float("nan"),
            "over1": (g(w, "body_speed", "share_above_1p0") or 0.0) * 100.0,
            "wfall": g(w, "fall_rate_per_attempt"),
        })

    # 파레토: 낙상 문턱을 통과한 것들 중, 오차가 더 작으면서 속도도 더 빠른 다른
    # 후보가 없으면 프론트에 남는다.
    ok = [r for r in recs if (r["wfall"] is not None and r["wfall"] <= FALL_LIMIT)]
    front = []
    for r in ok:
        dominated = any(
            (o["err"] <= r["err"] and o["spd"] >= r["spd"]) and
            (o["err"] < r["err"] or o["spd"] > r["spd"])
            for o in ok)
        if not dominated:
            front.append(r["it"])

    hdr = ["iter", "오차med(cm)", "strict(%)", "정확도낙상률",
           "보행속도", ">1.0m/s(%)", "보행낙상률", ""]
    w = [len(h) for h in hdr]
    body = []
    for r in recs:
        mark = ""
        if r["it"] in front:
            mark = "★ 파레토"
        elif r["wfall"] is not None and r["wfall"] > FALL_LIMIT:
            mark = "✗ 낙상초과"
        body.append([
            str(r["it"]), "{:.2f}".format(r["err"]), "{:.1f}".format(r["strict"]),
            "-" if r["afall"] is None else "{:.4f}".format(r["afall"]),
            "{:.2f}".format(r["spd"]), "{:.1f}".format(r["over1"]),
            "-" if r["wfall"] is None else "{:.4f}".format(r["wfall"]),
            mark,
        ])
    for row in body:
        for i, c in enumerate(row):
            w[i] = max(w[i], len(c))
    print("  ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
    print("  ".join("-" * w[i] for i in range(len(hdr))))
    for row in body:
        print("  ".join(c.ljust(w[i]) for i, c in enumerate(row)))
    print()
    print("★ = 두 축에서 지배당하지 않는 체크포인트 (보행 낙상률 <= {} 통과분 중)".format(
        FALL_LIMIT))
    print("  문턱 근거: 배포 정책의 보행 낙상률이 0.604 였다. 그 1/100 이다.")
    if not front:
        print("⛔ 파레토 프론트가 비었다 -- 낙상 문턱을 통과한 체크포인트가 없다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
