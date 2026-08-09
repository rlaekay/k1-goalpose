#!/usr/bin/env python3
"""두 정책을 **5-시드 풀스택** 결과로 비교한다. 지표 종류마다 맞는 검정을 쓴다.

    python tools/stat_compare.py --a logs/eval_rounds/v2/NH_*/accuracy \
                                 --b logs/eval_rounds/v2/NE_*/accuracy \
                                 --label-a NH_zeroclock --label-b NE_ctrl100

---- 왜 이 도구가 있는가 -------------------------------------------------------

이 캠페인은 철회를 50건 쌓았다. 원인이 하나로 모인다: **한 번 재고 그 숫자로 결론을
말했다.** 실측으로 드러난 것 셋:

  1. 낙상 수는 **한 번 재면 재현되지 않는다.** 같은 체크포인트·같은 조건 28쌍을
     다시 재니 부호검정 p = 0.0000, 최대 불일치 1 대 37.
  2. `CLAUDE.md` 최상단의 **"낙상간격 2배"** 는 **낙상 1건 대 2건**이었다 --
     율비 0.502, 정확 95 % CI [0.0085, 9.6](폭 1,100배), **p = 1.00**.
  3. 정확도 축은 재현된다(시드 SD 1.95 %) ⇒ **상대 5.4 % 미만은 노이즈다.**

즉 **어떤 축은 한 번으로 충분하고 어떤 축은 수십 배 차이가 나야 검출된다.** 그것을
매번 손으로 따지지 말고 도구가 하게 한다.

---- 지표별로 다른 검정을 쓰는 이유 --------------------------------------------

| 지표 | 자료형 | 검정 | 왜 |
|---|---|---|---|
| 낙상 | 노출시간당 **계수** | **정확 Poisson 율비**(이항 조건부) | 정규근사는 낙상 0~2건에서 무너진다 |
| 도착률/시도 | **이항** | **Fisher 정확검정** + Wilson CI | 시도 수가 커도 성공률 극단이면 정규근사가 틀린다 |
| 위치오차 median | 연속·비정규, **시드로 짝지음** | **Wilcoxon 부호순위** + Hodges-Lehmann 이동 | 시드가 분석 단위다. 4,600 구간은 **유사반복**이라 분모로 쓰면 p 가 가짜로 작아진다 |
| 속도·순항체류 | 연속 | 〃 | 〃 |

⛔ **분석 단위는 시드다.** 한 실행 안의 수천 구간은 독립 표본이 아니다(같은 정책,
같은 물리, 같은 초기분포). 그것을 n 으로 쓰면 **없는 검정력을 만들어 낸다** --
이 저장소가 "낙상간격 2배" 로 실제로 한 실수가 그것이다.

---- 다중비교 ------------------------------------------------------------------

한 비교에서 지표를 여러 개 본다. **Holm-Bonferroni** 로 보정한 p 를 같이 찍는다
(Bonferroni 보다 검정력이 높고 FWER 을 똑같이 통제한다). 1차 엔드포인트를
사전등록했다면 그것만 보정 없이 읽고 나머지는 탐색적이라고 적어라.

---- 검정력 --------------------------------------------------------------------

**숫자를 보기 전에** 최소검출효과(MDE)를 찍는다. 낙상 1건 대 X건에서 80 % 검정력에
필요한 배수를 함께 출력하므로, "유의하지 않다" 가 **효과가 없다**인지 **볼 수 없다**인지
구분된다. 이 둘을 섞은 것이 §6-e 에서 지적된 실수다.

의존성: numpy, scipy. 없으면 정확검정 대신 근사와 경고를 낸다.
"""
import argparse
import glob
import json
import math
import os
import sys

try:
    import numpy as np
except ImportError:
    sys.exit("numpy 가 필요하다")

try:
    from scipy import stats as st
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


# ---------------------------------------------------------------- 지표 추출
def _dig(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_reports(patterns):
    """디렉터리 glob 들을 받아 report.json 을 읽는다. 시드 순으로 정렬한다."""
    out = []
    for pat in patterns:
        for d in sorted(glob.glob(pat)):
            p = os.path.join(d, "report.json") if os.path.isdir(d) else d
            if not os.path.exists(p):
                continue
            with open(p) as f:
                r = json.load(f)
            r["_path"] = p
            out.append(r)
    out.sort(key=lambda r: (r.get("seed") if r.get("seed") is not None else -1))
    return out


def exposure_s(r):
    """낙상률의 분모가 되는 **노출 시간**(구간수 x 구간길이). 프로토콜 길이에 무관하게 만든다."""
    seg = _dig(r, "segments_completed") or 0
    falls = r.get("falls") or 0
    dur = r.get("duration_s") or 0.0
    n_env = r.get("num_envs") or 0
    # 전체 시뮬 시간 = num_envs * duration. 구간 단위가 아니라 이것이 진짜 노출이다.
    return float(n_env) * float(dur) if (n_env and dur) else float(seg + falls)


METRICS = [
    # (이름, 추출, 종류)  종류: count / binom / cont
    ("낙상", lambda r: (r.get("falls") or 0, exposure_s(r)), "count"),
    ("도착률/시도", lambda r: (_dig(r, "success_per_attempt", "strict"),
                            _dig(r, "success_per_attempt", "attempts")), "binom"),
    ("위치오차 median (cm)", lambda r: _dig(r, "pos_err_m", "median", default=None), "cont"),
    ("strict (%) [생존자만]", lambda r: _dig(r, "success_rate_strict"), "cont"),
    ("body_speed median", lambda r: _dig(r, "body_speed", "median"), "cont"),
    ("순항체류 (%)", lambda r: _dig(r, "high_speed_stability", "cruise_share"), "cont"),
]


# ---------------------------------------------------------------- 검정
def poisson_rate_ratio(k1, t1, k2, t2):
    """정확 Poisson 율비 검정. k1|k1+k2 ~ Binomial(n, t1/(t1+t2)) 조건부.

    낙상이 0~2건인 구간에서 정규근사는 신뢰구간을 **1,000배 폭**으로 만들거나
    반대로 없는 유의성을 만든다. 그래서 조건부 이항 정확검정을 쓴다.
    """
    n = k1 + k2
    if n == 0:
        return dict(ratio=float("nan"), p=1.0, ci=(0.0, float("inf")), note="양쪽 다 0건")
    p0 = t1 / (t1 + t2)
    if HAVE_SCIPY:
        p = st.binomtest(k1, n, p0).pvalue
        lo_p, hi_p = st.beta.ppf(0.025, k1, n - k1 + 1) if k1 > 0 else 0.0, \
                     st.beta.ppf(0.975, k1 + 1, n - k1) if k1 < n else 1.0
    else:
        p = float("nan")
        lo_p, hi_p = 0.0, 1.0

    def _r(pp):
        if pp <= 0:
            return 0.0
        if pp >= 1:
            return float("inf")
        return (pp / (1 - pp)) * (t2 / t1)

    r1 = k1 / t1 if t1 else float("nan")
    r2 = k2 / t2 if t2 else float("nan")
    return dict(ratio=(r1 / r2 if r2 else float("inf")), p=p,
                ci=(_r(lo_p), _r(hi_p)), rate_a=r1, rate_b=r2)


def mde_fold(k_ref, power=0.80, alpha=0.05):
    """낙상 k_ref 건을 기준으로 80 % 검정력에 필요한 **배수**를 대략 준다.

    "유의하지 않다" 가 효과가 없다는 뜻인지 **볼 수 없다는 뜻인지**를 가르기 위해서다.
    """
    if not HAVE_SCIPY or k_ref <= 0:
        return float("inf")
    za, zb = st.norm.ppf(1 - alpha / 2), st.norm.ppf(power)
    # sqrt 변환 근사: |sqrt(k1)-sqrt(k2)| = (za+zb)/2
    root = math.sqrt(k_ref) + (za + zb) / 2.0
    return (root ** 2) / k_ref


def paired_shift(a, b):
    """시드로 짝지은 연속 지표. Wilcoxon 부호순위 + Hodges-Lehmann 이동."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 2:
        return dict(n=len(a), p=float("nan"), hl=float("nan"), note="시드가 2개 미만")
    d = a - b
    hl = float(np.median([(d[i] + d[j]) / 2.0
                          for i in range(len(d)) for j in range(i, len(d))]))
    if HAVE_SCIPY and len(a) >= 3 and np.any(d != 0):
        p = st.wilcoxon(a, b, zero_method="zsplit", alternative="two-sided").pvalue
    else:
        p = float("nan")
    # ⭐ **축별 노이즈 바닥을 그 자리에서 낸다** (2026-08-09, C53 재발 방지).
    # 한 숫자(위치오차 축의 4.38 %)를 모든 지표에 쓴 것이 C53 의 원인이다 --
    # 조밀한 축에서는 진짜 차이를 노이즈로 지우고(속도 4 % 를 "같다"로 통과시켰다)
    # 성긴 축에서는 노이즈를 발견으로 만든다. 5시드가 이미 있으니 계산은 공짜다.
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    sd_a = float(np.std(a, ddof=1)) if len(a) > 1 else float("nan")
    sd_b = float(np.std(b, ddof=1)) if len(b) > 1 else float("nan")
    pooled = float(np.sqrt((sd_a ** 2 + sd_b ** 2) / 2.0))
    scale = (abs(mean_a) + abs(mean_b)) / 2.0
    rel_sd = (100.0 * pooled / scale) if scale > 0 else float("nan")
    se_diff = pooled * np.sqrt(2.0 / len(a)) if len(a) else float("nan")
    line2se = (200.0 * se_diff / scale) if scale > 0 else float("nan")
    obs = (100.0 * abs(mean_a - mean_b) / scale) if scale > 0 else float("nan")
    # 양자화 경고: 값이 전부 어떤 눈금의 배수면 SD 가 반올림에 지배될 수 있다.
    quant = float("nan")
    vals = np.concatenate([a, b])
    for step in (0.01, 0.1, 1.0):
        if np.allclose(vals / step, np.round(vals / step), atol=1e-9):
            quant = step
            break
    return dict(n=len(a), p=p, hl=hl, mean_a=mean_a, mean_b=mean_b,
                sd_a=sd_a, sd_b=sd_b, rel_sd=rel_sd, line2se=line2se,
                obs_rel=obs, quant=quant, scale=scale)


def fisher_rate(s1, n1, s2, n2):
    """도착 성공 수/시도 수. Fisher 정확검정 + Wilson 구간."""
    if HAVE_SCIPY:
        p = st.fisher_exact([[s1, n1 - s1], [s2, n2 - s2]])[1]
    else:
        p = float("nan")

    def wilson(s, n):
        if n == 0:
            return (float("nan"), float("nan"))
        z = 1.959963985
        ph = s / n
        den = 1 + z * z / n
        c = (ph + z * z / (2 * n)) / den
        h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
        return (c - h, c + h)

    return dict(p=p, a=(s1 / n1 if n1 else float("nan")), b=(s2 / n2 if n2 else float("nan")),
                ci_a=wilson(s1, n1), ci_b=wilson(s2, n2))


def holm(pvals):
    """Holm-Bonferroni 보정. 입력 순서를 유지해 돌려준다."""
    idx = [i for i, p in enumerate(pvals) if p == p]  # not nan
    order = sorted(idx, key=lambda i: pvals[i])
    m, out, prev = len(order), [float("nan")] * len(pvals), 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, max(prev, (m - rank) * pvals[i]))
        out[i], prev = adj, adj
    return out


# ---------------------------------------------------------------- 출력
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True, help="A 쪽 리포트 디렉터리 glob")
    ap.add_argument("--b", nargs="+", required=True, help="B 쪽 리포트 디렉터리 glob")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--primary", default=None,
                    help="사전등록한 1차 엔드포인트 이름. 이것만 보정 없이 읽는다")
    args = ap.parse_args()

    A, B = load_reports(args.a), load_reports(args.b)
    if not A or not B:
        sys.exit("리포트를 못 찾았다: A={} B={}".format(len(A), len(B)))

    print("=" * 78)
    print("5-시드 풀스택 비교   {}  대  {}".format(args.label_a, args.label_b))
    print("=" * 78)
    for lab, S in ((args.label_a, A), (args.label_b, B)):
        seeds = [r.get("seed") for r in S]
        print("  {:<22} n={} 시드={}".format(lab, len(S), seeds))
    if len(A) < 3 or len(B) < 3:
        print("⚠️ 시드가 3개 미만이면 Wilcoxon 이 성립하지 않는다. 5개를 채워라.")
    if not HAVE_SCIPY:
        print("⚠️ scipy 가 없다 -- p 값이 nan 으로 나온다. `pip install scipy`")
    print()

    rows, pvals = [], []
    for name, fn, kind in METRICS:
        if kind == "count":
            ka = sum((fn(r)[0] or 0) for r in A)
            ta = sum(fn(r)[1] for r in A)
            kb = sum((fn(r)[0] or 0) for r in B)
            tb = sum(fn(r)[1] for r in B)
            res = poisson_rate_ratio(ka, ta, kb, tb)
            need = mde_fold(max(kb, 1))
            rows.append((name,
                         "{}건/{:.0f}s".format(ka, ta),
                         "{}건/{:.0f}s".format(kb, tb),
                         "율비 {:.3g}  CI[{:.3g}, {:.3g}]".format(
                             res["ratio"], res["ci"][0], res["ci"][1]),
                         res["p"],
                         "80%검정력에 {:.1f}배 필요".format(need)))
            pvals.append(res["p"])
        elif kind == "binom":
            sa = sum(int(round((fn(r)[0] or 0) * (fn(r)[1] or 0))) for r in A)
            na = sum((fn(r)[1] or 0) for r in A)
            sb = sum(int(round((fn(r)[0] or 0) * (fn(r)[1] or 0))) for r in B)
            nb = sum((fn(r)[1] or 0) for r in B)
            if not na or not nb:
                rows.append((name, "(열 없음)", "(열 없음)", "2026-08-09 이전 리포트", float("nan"), ""))
                pvals.append(float("nan"))
                continue
            res = fisher_rate(sa, na, sb, nb)
            rows.append((name,
                         "{:.1%} ({}/{})".format(res["a"], sa, na),
                         "{:.1%} ({}/{})".format(res["b"], sb, nb),
                         "차이 {:+.2f}%p".format(100 * (res["a"] - res["b"])),
                         res["p"], ""))
            pvals.append(res["p"])
        else:
            va = [fn(r) for r in A]
            vb = [fn(r) for r in B]
            va = [float("nan") if v is None else float(v) for v in va]
            vb = [float("nan") if v is None else float(v) for v in vb]
            mult = 100.0 if "cm" in name else 1.0
            res = paired_shift(va, vb) if len(va) == len(vb) else paired_shift(va, vb)
            note = "n={}".format(res["n"])
            # ⛔ `None == None` 은 True 라 NaN 검사 관용구(v == v)가 None 을 통과시킨다.
            # `paired_shift` 의 조기 반환(시드 2개 미만)에는 이 키들이 아예 없다.
            def _f(key):
                v = res.get(key)
                return float("nan") if v is None else float(v)
            rel, l2, ob = _f("rel_sd"), _f("line2se"), _f("obs_rel")
            if rel == rel and l2 == l2:
                verdict = ("차>판별선" if (ob == ob and ob > l2) else "차<판별선")
                note += "  [축 노이즈 시드SD {:.2f}% · 판별선(2SE) {:.2f}% · 관측차 {:.2f}% ⇒ {}]".format(
                    rel, l2, ob, verdict)
            q, sc = _f("quant"), _f("scale")
            if q == q and sc == sc and sc > 0:
                qrel = 100.0 * (q / (12 ** 0.5)) / sc
                note += "  ⚠️눈금 {:g} 배수(양자화가 SD 에 {:.2f}%p 기여)".format(q, qrel)
            rows.append((name,
                         "{:.4g} ± {:.2g}".format(res.get("mean_a", float('nan')) * mult,
                                                  res.get("sd_a", float('nan')) * mult),
                         "{:.4g} ± {:.2g}".format(res.get("mean_b", float('nan')) * mult,
                                                  res.get("sd_b", float('nan')) * mult),
                         "HL 이동 {:+.3g}".format(res["hl"] * mult),
                         res["p"], note))
            pvals.append(res["p"])

    adj = holm(pvals)
    w = 26
    print("{:<24}{:<{w}}{:<{w}}".format("지표", args.label_a[:24], args.label_b[:24], w=w))
    print("-" * 78)
    for (name, a, b, eff, p, note), pa in zip(rows, adj):
        star = ""
        if p == p:
            star = "✅유의" if pa < 0.05 else ("~" if p < 0.05 else "")
        print("{:<24}{:<{w}}{:<{w}}".format(name, a[:w - 1], b[:w - 1], w=w))
        print("    {:<40} p={:<10.3g} Holm p={:<10.3g} {} {}".format(
            eff, p, pa, star, note))
    print()
    print("읽는 법:")
    print("  * `✅유의` = Holm 보정 후에도 p<0.05. 그 외는 **차이를 말하지 마라.**")
    print("  * `~` = 보정 전에만 유의. 탐색적 관찰이지 결론이 아니다.")
    print("  * 낙상 줄의 `80%검정력에 N배 필요` 가 크면, 유의하지 않은 것은")
    print("    **효과가 없어서가 아니라 볼 수 없어서**다. 그 둘을 섞지 마라.")
    print("  * 연속 지표의 n 은 **시드 수**다. 한 실행 안의 수천 구간은 유사반복이라")
    print("    분모로 쓰면 없는 검정력을 만든다.")
    if args.primary:
        print("  * 사전등록 1차 엔드포인트: **{}** -- 이것만 보정 없이 읽고".format(args.primary))
        print("    나머지는 탐색적이라고 적어라.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
