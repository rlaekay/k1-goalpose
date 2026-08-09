"""셀 안에서 측방 진폭이 낙상을 가르는가 — 주기 교락 없는 매개 검정.

셀 간 비교(주기↑ → 진폭↑ · 낙상↑)로는 매개를 못 말한다. 주기가 둘 다 움직이기
때문이다. 그런데 **같은 셀 안에서 시드가 갈린다**(25.35: 낙상 0/0/4/3/2).

⇒ 같은 주기·같은 조건에서 넘어진 시드와 안 넘어진 시드의 진폭을 비교하면
주기가 상수이므로 교락이 없다.

⚠️ 진폭은 **낙상 전후 2 s 를 뺀** 구간에서만 잰다(낙상의 산물이 되지 않게).
"""
import glob
import json
import sys

import numpy as np

sys.path.insert(0, "tools")
from lip_vs_loop import load, lat_series, detrend_cadence  # noqa: E402

for ms in ("24.0", "25.0", "25.35"):
    cad = (20.0 / float(ms)) * 2.0
    rows = []
    for p in sorted(glob.glob("logs/mujoco/lipnoise/n_%s_s*_series.npz" % ms)):
        d = load(p)
        if d is None:
            continue
        t, y, _ = lat_series(d, cad)
        if t.size < 1500:
            continue
        r = detrend_cadence(t, y, cad)
        falls = len(json.load(open(p.replace("_series.npz", ".json")))["fall_t"]) \
            if False else int(json.load(open(p.replace("_series.npz", ".json")))["falls"])
        rows.append((p.split("_s")[-1][0], falls, float(np.std(r))))
    if not rows:
        continue
    fell = [a for _, f, a in rows if f > 0]
    ok = [a for _, f, a in rows if f == 0]
    print("주기 %-6s 시드별 (낙상, 진폭): %s" % (ms, ["(%s,%d,%.4f)" % r for r in rows]))
    if fell and ok:
        print("   낙상한 시드 진폭 평균 %.4f (n=%d) | 안 넘어진 시드 %.4f (n=%d) | 차 %+.1f%%"
              % (np.mean(fell), len(fell), np.mean(ok), len(ok),
                 100 * (np.mean(fell) / np.mean(ok) - 1)))
    else:
        print("   한쪽만 있어 셀 내 비교 불가")
