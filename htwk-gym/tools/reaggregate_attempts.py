#!/usr/bin/env python3
"""기존 리포트를 **시도(attempt) 분모**로 다시 집계한다. GPU 0.

    python tools/reaggregate_attempts.py --root logs --min-duration 120 --top 25

---- 왜 -----------------------------------------------------------------------

`AUDIT_FINAL_20260809` §5-1 · §6-3(3). 지금까지 arm 순위를 매긴 `pos_err_m.median` 은
**생존 편향에 인과적으로 반응한다** -- 낙상한 구간은 정확도 표본에서 **빠지므로**
더 자주 넘어질수록 정확도가 좋아 보인다. 한 정책 안에서 교락 없이 증명됐다:

    NJ_armasset/model_6000, 다른 조건 전부 동일
      자기 물리      38.75 cm / 낙상     2 / 표본탈락  0.04 %
      틀린 물리      27.26 cm / 낙상 6,145 / 표본탈락 65.1 %   <- 30 % "좋아졌다"

⇒ **분모를 시도로 바꾼다.** 다행히 **새로 채점할 필요가 없다.** 옛 리포트에도
`success_rate_strict`(생존자 중 성공률) · `segments_completed` · `falls` 가 있으므로

    성공 수      = success_rate_strict x (게이트 카테고리 완주 구간수)
    시도 수      = 완주 구간수 + 낙상 수
    P(성공|시도) = 성공 수 / 시도 수

로 **소급 재집계**된다. 낙상이 분자를 늘리지 못하고 분모만 늘리므로 지표가 단조가 된다.

⚠️ 한계를 미리 적는다:
  * 낙상을 **전부 게이트 카테고리에 청구**한다(보수적 = 성공률을 낮게). 카테고리별
    낙상 분류가 옛 리포트에 없기 때문이다. 새 리포트는 `success_per_attempt` 를
    직접 들고 있고 그쪽이 정확하다 -- 있으면 그 값을 쓴다.
  * `success_rate_strict` 는 게이트 카테고리(waypoint)에만 걸리는데 `segments_completed`
    는 전체다. path 구간이 섞인 리포트는 분모가 과대평가되어 **성공률이 낮게** 나온다.
    같은 프로토콜끼리 비교하면 이 편향은 공통이라 **순위는 보존된다**. 절대값을 인용하지 마라.
  * Wilson 95 % 구간을 같이 찍는다. **구간이 겹치면 순위를 주장하지 마라.**
"""
import argparse
import glob
import json
import math
import os
import sys


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs")
    ap.add_argument("--min-duration", type=float, default=120.0,
                    help="이 길이 미만 리포트는 뺀다. 길이가 섞이면 낙상 수가 비교 불가다")
    ap.add_argument("--goal-pattern", default=None,
                    help="예: forward_hold. 생략하면 waypoint(=null)만 본다")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()

    rows, skipped = [], {"duration": 0, "pattern": 0, "no_gate": 0}
    for path in glob.glob(os.path.join(args.root, "**", "report.json"), recursive=True):
        try:
            with open(path) as f:
                r = json.load(f)
        except Exception:
            continue

        dur = r.get("duration_s")
        if dur is None or float(dur) < args.min_duration:
            skipped["duration"] += 1
            continue
        gp = r.get("goal_pattern")
        if (args.goal_pattern or None) != (gp or None):
            skipped["pattern"] += 1
            continue

        strict = dig(r, "success_rate_strict")
        seg = dig(r, "segments_completed")
        falls = r.get("falls")
        if strict is None or not seg:
            skipped["no_gate"] += 1
            continue
        falls = int(falls or 0)

        # 새 리포트는 직접 들고 있다. 그쪽이 카테고리 분류까지 맞으므로 우선한다.
        direct = dig(r, "success_per_attempt", "strict")
        if direct is not None:
            n_att = dig(r, "success_per_attempt", "attempts") or 0
            s_cnt = int(round(direct * n_att))
            source = "직접"
        else:
            s_cnt = int(round(float(strict) * int(seg)))
            n_att = int(seg) + falls
            source = "소급"

        ck = dig(r, "input_provenance", "checkpoint", "sha256") or ""
        rows.append(dict(
            path=os.path.dirname(path),
            ckpt=(r.get("checkpoint") or "?"),
            sha=(ck[:8] if ck else "?"),
            pos_cm=(dig(r, "pos_err_m", "median") or float("nan")) * 100.0,
            strict=float(strict),
            falls=falls,
            completed=int(seg),
            attempts=n_att,
            p_att=(s_cnt / n_att if n_att else float("nan")),
            ci=wilson(s_cnt, n_att),
            source=source,
            armature=str(dig(r, "effective_eval_protocol", "asset", "armature", default="?")),
        ))

    if not rows:
        sys.exit("조건에 맞는 리포트가 없다. --min-duration / --goal-pattern 을 확인해라.")

    # 옛 순위(정확도 cm) 와 새 순위(도착률/시도) 를 나란히 놓아 **순위가 얼마나 바뀌는지**
    # 보이게 한다. 그것이 이 재집계의 요점이다.
    by_pos = sorted(rows, key=lambda x: x["pos_cm"])
    pos_rank = {id(r): i + 1 for i, r in enumerate(by_pos)}
    by_att = sorted(rows, key=lambda x: -x["p_att"])

    print("=" * 108)
    print("시도(attempt) 분모 재집계   n={} 리포트   길이>={}s   goal_pattern={}".format(
        len(rows), args.min_duration, args.goal_pattern or "(waypoint)"))
    print("=" * 108)
    print("{:<4}{:<6}{:<34}{:>9}{:>8}{:>8}{:>10}{:>20}".format(
        "새", "옛", "run / 체크포인트", "오차cm", "낙상", "시도", "도착률", "95% CI"))
    print("-" * 108)
    for i, r in enumerate(by_att[:args.top], 1):
        run = r["ckpt"].split("/nn/")[0].split("/")[-1][:24]
        it = r["ckpt"].split("/nn/")[-1].replace(".pth", "")[:9]
        move = pos_rank[id(r)] - i
        arrow = "↑" if move > 0 else ("↓" if move < 0 else " ")
        print("{:<4}{:<6}{:<34}{:>9.2f}{:>8}{:>8}{:>9.1%}   [{:.1%}, {:.1%}] {}{}".format(
            i, "{}{}".format(pos_rank[id(r)], arrow), "{}/{}".format(run, it),
            r["pos_cm"], r["falls"], r["attempts"], r["p_att"],
            r["ci"][0], r["ci"][1], r["source"],
            "" if r["armature"] == "?" else " arm={}".format(r["armature"])))

    print()
    print("읽는 법:")
    print("  * `새`=도착률/시도 순위, `옛`=정확도 cm 순위. ↑ 는 시도 분모로 바꾸니 올라간 것.")
    print("  * **낙상이 분자를 못 늘리고 분모만 늘린다** -- 더 넘어지면 반드시 나빠진다.")
    print("  * `95% CI` 가 겹치면 **순위를 주장하지 마라.** 겹침이 기본값이다.")
    print("  * `소급` = 옛 리포트를 재집계한 값(낙상을 전부 게이트 쪽에 청구, 보수적).")
    print("    `직접` = 리포트가 `success_per_attempt` 를 들고 있는 것(2026-08-09 이후).")
    print("  * `arm=?` 은 채점 물리 지문이 없는 리포트다 -- **물리를 가로질러 비교하지 마라.**")
    print()
    print("건너뜀: 길이미달 {} / 패턴불일치 {} / 게이트없음 {}".format(
        skipped["duration"], skipped["pattern"], skipped["no_gate"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
