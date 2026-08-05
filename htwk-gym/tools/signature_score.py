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
    # 정책 출력의 주 진동수. 실기는 roll 액션이 3.30 Hz로 발진한다 -- 명령
    # 케이던스 2.0 Hz도, MuJoCo의 4 Hz(두 걸음)도 아닌 값이다. 순수 지연 tau의
    # 폐루프 한계주기가 대략 1/(4 tau)이므로 3.3 Hz는 tau ~ 75 ms를 시사한다.
    # 그리고 실기 고관절 pitch는 0.30-1.35 Hz로 명령(2.0)보다 느리다 -- 발진과
    # 보행 정지가 같이 일어난다.
    # ⛔ f_act_hipP_R(0.30 Hz)은 뺐다. 2.4초 창의 주파수 분해능이 1/2.4 = 0.42 Hz라
    # 스캔 하한(0.25) 아래는 못 가른다 -- 표류하는 신호는 무조건 하한에 몰린다.
    # 즉 "0.30 Hz 진동"이 아니라 "주기가 없는 표류"였고, 주파수로 재면 안 된다.
    # 그래서 표류는 표류로 잰다(아래 drift).
    "f_act_hipR_L": 3.30,
    # 창 전체에 걸친 선형 추세의 크기(도). 실기 R_Hip_Roll이 2.4초에 13.5도
    # 밀려나고 안 돌아온다 -- 보행의 진동이 아니라 작동점 이탈이다.
    "drift_hipR_R": 13.5,
    # 추종률: 명령 대비 실제 도달. 실기 median 0.61, MuJoCo 기본 0.93.
    "track_median": 0.61,
}
DEG = 180.0 / math.pi


def _drift(y):
    """창 전체의 선형 추세 크기(끝-시작, 최소자승). 표류는 주파수로 재면 안 된다 --
    짧은 창에서는 DFT가 스캔 하한에 몰릴 뿐이다."""
    n = len(y)
    xm = (n - 1) / 2.0
    ym = sum(y) / n
    den = sum((i - xm) ** 2 for i in range(n))
    if den <= 0:
        return 0.0
    slope = sum((i - xm) * (y[i] - ym) for i in range(n)) / den
    return abs(slope * (n - 1))


def _dom_freq(x, dt, lo=0.25, hi=12.0, step=0.05):
    """주 진동수. 표본이 적어(실기 91) FFT보다 느린 DFT 스캔이 안전하다."""
    n = len(x)
    mu = sum(x) / n
    y = [v - mu for v in x]
    best, bf = -1.0, lo
    f = lo
    while f <= hi:
        w = 2 * math.pi * f * dt
        re = sum(y[i] * math.cos(w * i) for i in range(n))
        im = sum(y[i] * math.sin(w * i) for i in range(n))
        a = math.hypot(re, im)
        if a > best:
            best, bf = a, f
        f += step
    return bf


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
    dt = float(rows[1]["t_s"]) - float(rows[0]["t_s"]) or 0.02
    m["f_act_hipR_L"] = _dom_freq(c("act1"), dt)
    m["drift_hipR_R"] = _drift(c("q7")) * DEG
    DEF = [-0.2, 0.0, 0.0, 0.4, -0.25, 0.0] * 2
    tr = []
    for i in range(12):
        t = [DEF[i] + float(r["act%d" % i]) for r in rows]
        q = [float(r["q%d" % i]) for r in rows]
        rt = max(t) - min(t)
        tr.append((max(q) - min(q)) / rt if rt > 0 else 0.0)
    m["track_median"] = sorted(tr)[6]
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
