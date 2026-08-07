"""비교 주장을 하기 **전에** 양쪽 조건이 같은지 확인한다.

    python tools/claim_check.py <report.json 또는 그 디렉터리> ...

⛔ 왜 있는가. 2026-08-07 하루에 주장을 열세 번 바꿨는데 전부 같은 모양이었다:
**비교 대상 두 쪽의 조건이 같은지 확인하기 전에 결론을 말했다.**

    체크포인트가 달랐다 (best vs final)      -> "N0_ctrl 이 세 축 모두 지배"
    프로토콜 길이가 달랐다 (60 s vs 120 s)   -> "낙상 1,177 대 10 = 118배"
    표본 밀도가 달랐다 (2점 vs 8점)          -> "N0 는 어떤 체크포인트에서도 못 따라간다"
    과제가 달랐다 (도착 있음 vs 없음)         -> "속도 요구를 충족했다"

이 도구는 조건을 나란히 찍고 **다른 항목을 표시한다.** 그리고 `RETRACTIONS.md` 에서
관련된 과거 철회를 같이 꺼내 놓는다 -- 기록을 남기는 것만으로는 부족하고
**주장하는 순간에 읽히게** 해야 같은 실수가 안 반복된다.

숫자를 해석해 주지 않는다. 판단은 사람이 한다. 이 도구가 하는 일은 하나다:
**"이 두 숫자를 나란히 놓아도 되는가"에 답한다.**
"""

import os
import re
import sys
import json
import glob

# 조건을 두 갈래로 나눈다. 이 구분이 없으면 도구가 **모든 비교에 경보를 울리고**,
# 그러면 아무도 안 읽게 된다 -- 2026-08-07 첫 실사용에서 실제로 그랬다.
#
#   IDENTITY  : 비교 대상을 **가르는** 것. 달라야 정상이다(다르니까 비교한다).
#               다만 "무엇과 무엇을 비교하는지"를 눈에 보이게 찍어 준다.
#   PROTOCOL  : 비교를 **성립시키는** 것. 하나라도 다르면 나란히 못 놓는다.
#
# `config` 는 IDENTITY 다 -- `make_eval_cfg.py` 가 arm 마다 다른 파일명을 쓴다.
# 프로토콜이 정말 같은지는 `effective_eval_protocol_sha` 가 답한다. 그게 PROTOCOL 이다.
IDENTITY = [
    ("checkpoint",        lambda r: _ckpt_label(r)),
    ("run",               lambda r: (r.get("checkpoint") or "/?/").split("/nn/")[0].split("/")[-1]),
    ("config",            lambda r: os.path.basename(r.get("config") or "?")),
]
CONDITIONS = [
    # ⛔ 2026-08-08 추가. `best.pth` 는 **가변 심볼릭 링크**라 양쪽 다 "best.pth" 로
    # 찍히면서 실제로는 다른 iteration 을 가리킬 수 있다. 실제로 그랬다:
    # NC_actfilter best.pth -> model_100, NZ_zeroiid best.pth -> model_1700.
    # 그런데도 도구는 "✅ 조건이 전부 같다" 를 출력했다 -- 이 파일 맨 위 docstring 이
    # **첫 번째 실수로 적어 둔 바로 그것**(best vs final)을 도구가 못 잡고 있었다.
    # iteration 이 다르면 "arm 차이" 와 "학습량 차이" 가 교락된다. 그래서 PROTOCOL 이다.
    ("ckpt_iter",         lambda r: _ckpt_iter(r)),
    ("goal_pattern",      lambda r: r.get("goal_pattern") or "(waypoint)"),
    ("duration_s",        lambda r: r.get("duration_s")),
    ("num_envs",          lambda r: r.get("num_envs")),
    ("terrain",           lambda r: r.get("eval_terrain") or "(config)"),
    ("force_profile",     lambda r: r.get("force_profile") or "(none)"),
    ("seed",              lambda r: r.get("seed")),
    ("deterministic",     lambda r: r.get("deterministic")),
    ("protocol_sha",      lambda r: (r.get("effective_eval_protocol_sha") or "?")[:8]),
    ("env_code_sha",      lambda r: (r.get("env_code_sha") or "?")[:8]),
]

# 조건이 같아야만 의미가 있는 지표들. 어느 조건에 의존하는지 같이 적는다.
METRICS = [
    ("pos_err med (cm)",  lambda r: _g(r, "pos_err_m", "median", mult=100),
     "config, goal_pattern, ckpt_iter"),
    ("strict (%)",        lambda r: _g(r, "success_rate_strict", mult=100), "config, ckpt_iter"),
    ("falls (원시)",       lambda r: r.get("falls"), "duration_s ⚠, ckpt_iter"),
    ("낙상률/시도",        lambda r: _g(r, "fall_rate_per_attempt"), "goal_pattern ⚠, ckpt_iter"),
    ("낙상간격 (s)",       lambda r: _mtbf(r), "ckpt_iter"),
    ("표본탈락(생존편향) %", lambda r: _drop(r), "ckpt_iter"),
    ("body_speed med",    lambda r: _g(r, "body_speed", "median"), "goal_pattern, ckpt_iter"),
    ">1.0 m/s (%)",
]
METRICS = [m for m in METRICS if isinstance(m, tuple)]
METRICS.append((">1.0 m/s (%)", lambda r: _g(r, "body_speed", "share_above_1p0", mult=100),
                "goal_pattern, ckpt_iter"))
METRICS.append(("순항체류 (%)", lambda r: _g(r, "high_speed_stability",
                                            "cruise_share_of_valid", mult=100),
                "goal_pattern, ckpt_iter"))
METRICS.append(("segments", lambda r: r.get("segments_completed"), "duration_s"))


def _resolve_ckpt(r):
    """`best.pth` 는 심볼릭 링크다. 링크가 살아 있으면 실제 대상까지 따라간다.

    링크를 못 따라가는 경우(다른 기계에서 리포트만 받아 본 경우)에는 이름 그대로
    돌려준다 -- **모른다는 것을 아는 척하지 않기 위해** `?` 를 붙여 표시한다.
    """
    ck = r.get("checkpoint") or ""
    if not ck:
        return "?", None, False
    base = os.path.basename(ck)
    try:
        if os.path.islink(ck) or os.path.exists(ck):
            real = os.path.basename(os.path.realpath(ck))
            return base, real, True
    except OSError:
        pass
    return base, None, False


def _ckpt_label(r):
    """표시용. 심볼릭 링크면 `best.pth→model_1700` 처럼 실제 대상을 같이 찍는다."""
    base, real, ok = _resolve_ckpt(r)
    if ok and real and real != base:
        return "{}→{}".format(base, real.replace(".pth", ""))
    return base


def _ckpt_iter(r):
    """체크포인트의 iteration. 비교 가능성 판정에 쓰이므로 **추측하지 않는다.**

    - 해석되면 정수 iteration
    - 링크를 못 따라가면 `?<name>` -- 두 리포트가 같은 이름이어도 **같다고 못 박지 않는다**
    """
    base, real, ok = _resolve_ckpt(r)
    name = real if (ok and real) else base
    m = re.search(r"model_(\d+)", name or "")
    if m:
        return int(m.group(1))
    if not ok and base:
        # 미해석. 같은 이름이라도 **같은 것이라고 단정하지 않는다** -- 그래서 run 을
        # 붙여 서로 다른 값으로 만든다. 결과적으로 ⛔ 가 뜨고, 사람이 서버에서
        # 다시 돌려 확인하게 된다. 조용히 통과시키는 것보다 이쪽이 안전하다.
        run = (r.get("checkpoint") or "/?/").split("/nn/")[0].split("/")[-1]
        return "?{}@{}".format(base, run[:18])
    return name or "?"


def _g(r, *keys, mult=1.0):
    cur = r
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
        if cur is None:
            return None
    try:
        return float(cur) * mult
    except (TypeError, ValueError):
        return None


def _mtbf(r):
    seg = r.get("segments_completed") or 0
    fl = r.get("falls") or 0
    dur = _g(r, "feasibility", "segment_duration_s", "median")
    if not fl or not dur:
        return None
    return (seg + fl) * dur / fl


def _drop(r):
    seg = r.get("segments_completed") or 0
    fl = r.get("falls") or 0
    return 100.0 * fl / (seg + fl) if (seg + fl) else None


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return "{:.4g}".format(v)
    return str(v)


def load(p):
    if os.path.isdir(p):
        p = os.path.join(p, "report.json")
    try:
        with open(p, encoding="utf-8") as f:
            return os.path.basename(os.path.dirname(p)), json.load(f)
    except (OSError, ValueError) as exc:
        print("!! 못 읽음 {}: {}".format(p, exc))
        return None, None


def retractions_for(names, repo_root):
    """RETRACTIONS.md 에서 관련 항목을 꺼낸다. 기록은 남기는 것보다 **읽히는 것**이 어렵다."""
    path = os.path.join(repo_root, "RETRACTIONS.md")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        text = f.read()
    blocks = re.split(r"\n### ", text)
    hits = []
    keys = set()
    for n in names:
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", n):
            keys.add(tok)
    for b in blocks[1:]:
        head = b.split("\n", 1)[0]
        if any(k in b for k in keys):
            hits.append(head.strip())
    return hits


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    cols = []
    for p in sys.argv[1:]:
        name, r = load(p)
        if r is not None:
            cols.append((name, r))
    if len(cols) < 1:
        print("읽은 리포트가 없다.")
        return 1

    labels = [c[0][:26] for c in cols]
    w0 = max(16, max(len(l) for l in labels) + 2)

    print("=" * 78)
    print("무엇과 무엇을 비교하는가 (달라도 정상 — 이게 비교 대상이다)")
    print("=" * 78)
    print("{:<18}".format("") + "".join("{:<{}}".format(l, w0) for l in labels))
    for cname, fn in IDENTITY:
        vals = [fmt(fn(r)) for _, r in cols]
        print("   {:<15}".format(cname)
              + "".join("{:<{}}".format(v[:w0 - 1], w0) for v in vals))

    print()
    print("=" * 78)
    print("프로토콜 — ⛔ 가 하나라도 있으면 그 축의 숫자를 나란히 놓지 마라")
    print("=" * 78)
    print("{:<18}".format("") + "".join("{:<{}}".format(l, w0) for l in labels))
    mismatched = []
    for cname, fn in CONDITIONS:
        vals = [fmt(fn(r)) for _, r in cols]
        same = len(set(vals)) <= 1
        mark = "   " if same else "⛔ "
        if not same:
            mismatched.append(cname)
        print("{}{:<15}".format(mark, cname)
              + "".join("{:<{}}".format(v[:w0 - 1], w0) for v in vals))

    print()
    print("=" * 78)
    print("지표 (의존하는 조건)")
    print("=" * 78)
    for mname, fn, dep in METRICS:
        vals = [fmt(fn(r)) for _, r in cols]
        bad = any(d.strip(" ⚠") in mismatched for d in dep.split(","))
        mark = "⛔ " if bad else "   "
        print("{}{:<15}".format(mark, mname)
              + "".join("{:<{}}".format(v[:w0 - 1], w0) for v in vals)
              + "   [{}]".format(dep))

    print()
    if mismatched:
        print("⛔ 어긋난 조건: {}".format(", ".join(mismatched)))
        print("   ⛔ 표시된 지표는 **비교 불가**다. 조건을 맞춰 다시 재거나,")
        print("      주장에 그 차이를 명시해라. 그냥 나란히 쓰면 이 저장소가")
        print("      2026-08-07 에 열세 번 한 실수를 다시 하는 것이다.")
    else:
        print("✅ 조건이 전부 같다. 이 리포트들은 나란히 비교해도 된다.")

    hits = retractions_for(labels, repo_root)
    if hits:
        print()
        print("=" * 78)
        print("📕 RETRACTIONS.md 에서 관련된 과거 철회 — 주장 전에 읽어라")
        print("=" * 78)
        for h in hits:
            print("   " + h)
    return 0


if __name__ == "__main__":
    sys.exit(main())
