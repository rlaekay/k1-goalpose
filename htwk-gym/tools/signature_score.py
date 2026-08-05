"""MuJoCo 실행이 **실기 증상을 얼마나 재현했는지** 하나의 점수로 낸다.

왜 점수인가
-----------
지금까지는 레버를 하나씩 켜고 "넘어지나?"만 봤다. 그건 해상도가 낮다 -- 여섯 개
가설이 전부 "낙상 0"으로 기각됐지만, 실기는 낙상 전에 **특정한 모양**으로 무너진다.
그 모양을 벡터로 고정하면 탐색 문제가 되고, 자동으로 쓸 수 있다.

실기 기준선 (2026-08-05, 만충 배터리, 2.4초 91표본, `--log-timing`):
그리고 대조군은 MuJoCo 기본 실행이다. 둘 다 같은 48채널 포맷이라 같은 함수로 잰다.

지표 (전부 실기/MuJoCo가 **크게 다른** 것만 골랐다):
  1. Hip_Roll 궤적 폭 L/R      실기 29.8 / 45.4도   기본 6.9 / 19.1
  2. 정책 roll 출력 폭 L/R     실기 0.910 / 1.313   기본 0.272 / 0.438
  3. 발목 pitch 토크 rms L/R   실기 6.92 / 7.26     기본 10.48 / 11.87  (실기가 **낮다**)
  4. 몸통 roll 폭              실기 35.3도          기본 8.95
  5. 좌우 발 겹침 비율         실기 9.9%            기본 2.3%

점수는 각 지표의 **로그비 제곱합**이다. 0이면 완전 재현. 부호까지 맞아야 하므로
"발목 토크가 낮다"를 "높다"로 맞추면 점수가 나빠진다.

⚠️ 이 점수가 낮다고 원인이 규명된 것은 아니다. **증상을 재현하는 조건**을 찾는 것이고,
그 조건이 물리적으로 말이 되는지는 따로 따져야 한다.

    python3 tools/signature_score.py <mujoco_dump.csv> [...]
"""

import sys
import csv
import math
import os

# 실기 기준(목표). 2026-08-05 만충 배터리 2.4초 로그에서 계산됨.
TARGET = {
    "hipR_range_L": 29.8, "hipR_range_R": 45.4,
    "act_roll_L": 0.910, "act_roll_R": 1.313,
    "ankP_tau_L": 6.92, "ankP_tau_R": 7.26,
    "trunk_roll_range": 35.3,
    "foot_overlap_pct": 9.9,
    # 관절 속도. 단일 채널로는 가장 강한 판별자였다 -- 발목 roll의 dq rms가
    # 실기 7.03/8.87 대 MuJoCo 2.42/1.99 (2.9-4.5배). 그런데 모든 pitch 관절은
    # 오히려 실기가 느리다(0.42-0.76배). 속도가 roll에만 몰려 있다는 것이
    # "몸 전체가 흔들린다"와 "roll 축만 이상하다"를 가른다. dq는 관측 채널
    # (obs[30:42])이므로 정책이 이 4.5배를 직접 본다.
    "dq_ankR_L": 7.03, "dq_ankR_R": 8.87,
    "dq_hipP_L": 1.88, "dq_hipP_R": 2.10,
}
DEG = 180.0 / math.pi


def measure(path, overlap_pct=None):
    rows = list(csv.DictReader(open(path)))
    if not rows or "q1" not in rows[0]:
        return None
    def c(k): return [float(r[k]) for r in rows]
    def rng(x): return max(x) - min(x)
    def rms(x): return (sum(v * v for v in x) / len(x)) ** 0.5
    m = {
        "hipR_range_L": rng(c("q1")) * DEG,
        "hipR_range_R": rng(c("q7")) * DEG,
        "act_roll_L": rng(c("act1")),
        "act_roll_R": rng(c("act7")),
        "ankP_tau_L": rms(c("tau4")),
        "ankP_tau_R": rms(c("tau10")),
        "trunk_roll_range": rng(c("roll")) * DEG,
        "dq_ankR_L": rms(c("dq5")), "dq_ankR_R": rms(c("dq11")),
        "dq_hipP_L": rms(c("dq0")), "dq_hipP_R": rms(c("dq6")),
    }
    if overlap_pct is not None:
        m["foot_overlap_pct"] = overlap_pct
    return m


def score(m):
    """로그비 제곱합. 지표별 기여도도 같이 낸다."""
    parts = {}
    tot = 0.0
    for k, t in TARGET.items():
        if k not in m:
            continue
        v = max(m[k], 1e-6)
        d = math.log(v / t) ** 2
        parts[k] = (v, t, d)
        tot += d
    return tot, parts


def main():
    if len(sys.argv) < 2:
        print(__doc__); return 1
    print("%-34s %8s | %s" % ("실행", "점수", "가장 어긋난 지표 3개"))
    results = []
    for p in sys.argv[1:]:
        m = measure(p)
        if m is None:
            print("%-34s   (관절 채널 없음)" % os.path.basename(p)); continue
        s, parts = score(m)
        worst = sorted(parts.items(), key=lambda kv: -kv[1][2])[:3]
        results.append((s, p, parts))
        print("%-34s %8.3f | %s" % (
            os.path.basename(p)[:34], s,
            "  ".join("%s %.2f(목표%.2f)" % (k, v, t) for k, (v, t, _) in worst)))
    if results:
        results.sort()
        print()
        print("가장 잘 재현한 것: %s (점수 %.3f)" % (
            os.path.basename(results[0][1]), results[0][0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
