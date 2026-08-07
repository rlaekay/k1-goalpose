"""eval_round.sh 가 만든 report.json 들을 한 표로 모은다.

키 이름은 **실제 report.json 을 덤프해서** 확인한 것만 쓴다. 이 저장소에서
`results["commands"]` 라는 존재하지 않는 키를 읽어 프로브 가드가 한 번도 발동하지
않은 전례가 있다 -- 그래서 모든 접근을 .get() 으로 감싸고, 없으면 표에 "-" 를 찍는다.
조용히 0 으로 대체하지 않는다. 없는 값과 0 은 다른 것이고, 그 둘을 섞는 것이 바로
이 저장소가 잃은 시간의 큰 몫이었다.

    python tools/round_table.py <eval_round_dir>
"""

import os
import sys
import json
import glob


def load(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def g(d, *keys, default=None):
    """중첩 .get(). 어느 단계든 없으면 default."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def mtbf(r):
    """평균 무사고 시간(초). 낙상률/구간은 **프로토콜 의존값**이라 축이 다르면 비교가
    안 된다 -- 정확도 축의 구간 median 이 5.94 s, forward_hold 가 0.98 s 로 6배 다르다.
    구간 길이를 곱해 초 단위로 환산하면 프로토콜과 무관하게 읽힌다."""
    seg = g(r, "segments_completed") or 0
    fl = g(r, "falls") or 0
    dur = g(r, "feasibility", "segment_duration_s", "median")
    if not fl or not dur:
        return None
    return (seg + fl) * dur / fl


def fmt(v, mult=1.0, nd=2):
    if v is None:
        return "-"
    try:
        return ("{:." + str(nd) + "f}").format(float(v) * mult)
    except (TypeError, ValueError):
        return "-"


def table(headers, rows):
    if not rows:
        return "(행 없음)"
    w = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            w[i] = max(w[i], len(str(c)))
    out = ["  ".join(h.ljust(w[i]) for i, h in enumerate(headers)),
           "  ".join("-" * w[i] for i in range(len(headers)))]
    for r in rows:
        out.append("  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
    return "\n".join(out)


def audit_round(arms):
    """한 라운드 안의 리포트들이 **나란히 놓일 수 있는지** 먼저 검사한다.

    ⛔ 왜 여기인가 (2026-08-08):
    `eval_goal_pose.py:765` 주석은 "Cross-arm aggregation rejects a single
    missing or unequal hash" 라고 약속한다. 그 거부를 실제로 하는 코드는
    `archive/tools/compare_hbatch_results.py` 하나였고 -- **archive 안이다.**
    지금 쓰는 교차집계기인 이 파일은 `protocol_sha` 도 `env_code_sha` 도
    체크포인트 iteration 도 **한 번도 안 봤다.** 그래서 다음 두 가지가 조용히
    한 표에 실렸다:
      * lineageB 라운드: NE_ctrl100(env 8d763fb) 과 NZ_zeroiid(env 5a985b1) --
        라운드 도중에 서버에서 git pull 이 일어나면 arm 마다 python 이 새로
        뜨므로 **한 라운드가 두 커밋으로 쪼개진다.**
      * n2na 라운드: best.pth 가 arm 마다 model_100 / model_1700 / model_3300 --
        "레버 차이" 와 "학습량 17배 차이" 가 교락된 표.
    이 저장소가 2026-08-07 에 열세 번 한 실수가 전부 이 모양이다.

    ⚠️ kind 별로 본다. `walk` 은 `--goal_pattern forward_hold` 라 `commands` 가
    바뀌어 `accuracy` 와 protocol_sha 가 다른 것이 **정상**이다.
    """
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from claim_check import _ckpt_iter, _flatten
    except Exception:                                   # 도구가 없어도 표는 나와야 한다
        _ckpt_iter = _flatten = None

    kinds = {}
    for name in arms:
        for kind, r in arms[name].items():
            kinds.setdefault(kind, []).append((name, r))

    warned = False
    for kind in sorted(kinds):
        entries = kinds[kind]
        if len(entries) < 2:
            continue
        checks = [
            ("protocol_sha", lambda r: (r.get("effective_eval_protocol_sha") or "?")[:8]),
            ("env_code_sha", lambda r: (r.get("env_code_sha") or "?")[:8]),
        ]
        if _ckpt_iter is not None:
            checks.append(("ckpt_iter", _ckpt_iter))
        for label, fn in checks:
            vals = {}
            for name, r in entries:
                vals.setdefault(str(fn(r)), []).append(name)
            if len(vals) <= 1:
                continue
            # protocol_sha 는 dict 가 있으면 거동 차이인지 먼저 가른다.
            if label == "protocol_sha" and _flatten is not None:
                protos = [r.get("effective_eval_protocol") for _, r in entries]
                if all(isinstance(p, dict) and p for p in protos):
                    flat = [_flatten(p) for p in protos]
                    keys = sorted(set().union(*[set(f) for f in flat]))
                    MISS = object()
                    real = [k for k in keys
                            if len({repr(f.get(k, MISS)) for f in flat}) > 1
                            and not all(f.get(k, MISS) in (MISS, None) for f in flat)]
                    if not real:
                        continue                        # 지문만 다르고 프로토콜은 같다
            if not warned:
                print("=" * 78)
                print("⛔ 이 라운드는 나란히 놓을 수 없는 리포트를 섞고 있다")
                print("=" * 78)
                warned = True
            print("[{}] {} 가 갈렸다:".format(kind, label))
            for v in sorted(vals):
                print("    {:<12} {}".format(v, ", ".join(sorted(vals[v]))))
            if label == "env_code_sha":
                print("    ⇒ 그 커밋들이 eval 경로를 건드렸는지 **직접** 확인해라:")
                print("       git diff <A> <B> -- htwk-gym/eval_goal_pose.py htwk-gym/envs/")
                print("       (라운드 도중 서버 git pull 이 원인인 경우가 많다.)")
            elif label == "ckpt_iter":
                print("    ⇒ 학습량이 다르면 arm 차이와 교락된다. `best.pth` 는 **가변")
                print("       심볼릭 링크**라 이름이 같아도 다른 iteration 일 수 있다.")
            else:
                print("    ⇒ 실효 프로토콜이 실제로 다르다. 조건을 맞춰 다시 재라.")
    if warned:
        print()
        print("아래 표는 **그래도** 찍는다 -- 지우면 아무도 원인을 못 본다.")
        print("단 위에서 갈린 축을 넘어 순위를 매기지 마라. 쌍마다 확인:")
        print("   python tools/claim_check.py <A>/report.json <B>/report.json")
        print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    root = sys.argv[1]

    arms = {}
    for rep in sorted(glob.glob(os.path.join(root, "*", "report.json"))):
        d = os.path.basename(os.path.dirname(rep))
        if "." not in d:
            continue
        name, _, kind = d.rpartition(".")
        r = load(rep)
        if r is None:
            continue
        arms.setdefault(name, {})[kind] = r

    if not arms:
        print("report.json 을 찾지 못했다: {}/*/report.json".format(root))
        return 1

    audit_round(arms)

    # ---- 표 1: 정확도 (공통 waypoint 프로토콜) ---------------------------
    acc_rows = []
    for name in sorted(arms):
        r = arms[name].get("accuracy")
        if r is None:
            continue
        # ⛔ 생존 편향을 표에 드러낸다. eval_goal_pose.py:1750 이
        # `completed = changed & ~done` 이므로 **낙상으로 끝난 구간은 기록 자체가
        # 안 된다.** 즉 오차/strict/heading 은 전부 "안 넘어진 구간만"의 통계다.
        # 배포 정책은 시도의 36.8 %가 표본에서 빠지고도 오차 3.10 cm 로 M7(3.54,
        # 탈락 0.2 %)보다 위에 온다. 분모가 arm 마다 다른 순위표였다.
        seg = g(r, "segments_completed") or 0
        fl = g(r, "falls") or 0
        drop = (100.0 * fl / (seg + fl)) if (seg + fl) else None
        acc_rows.append([
            name,
            fmt(g(r, "pos_err_m", "median"), 100),      # cm
            fmt(g(r, "pos_err_m", "p90"), 100),
            fmt(g(r, "heading_err_deg", "median"), 1, 1),
            str(g(r, "falls", default="-")),
            fmt(drop, 1, 1),
            fmt(g(r, "success_rate_strict"), 100, 1),
            str(g(r, "segments_completed", default="-")),
            "PASS" if g(r, "all_gates_pass") else "fail",
        ])
    print("== 정확도: 공통 waypoint 프로토콜 (모든 arm 을 같은 config 로 채점) ==")
    print(table(["arm", "오차med(cm)", "오차p90(cm)", "heading(도)",
                 "낙상", "표본탈락(%)", "strict(%)", "구간수", "게이트"], acc_rows))
    print()
    print("⛔ 표본탈락(%) = 낙상으로 끝나 **오차 통계에서 빠진** 시도의 비율이다.")
    print("   이 값이 큰 arm 의 오차 숫자는 '안 넘어졌을 때만'의 값이라 다른 arm 과")
    print("   같은 자가 아니다. 오차만으로 순위를 매기지 마라.")
    print()

    # ---- 표 2: 지속 보행 (forward_hold) ----------------------------------
    # ⛔ 이 표가 이번 라운드에 새로 생긴 것이다. 지금까지의 판정은 전부 위의 표만
    # 보고 내려졌고, 위의 표는 요구 속도 median 0.12 m/s 짜리 과제라 보행을 잰 적이
    # 없다. 배포된 정책은 위 표에서 2.8 cm 였고 실기에서 세 걸음을 못 갔다.
    walk_rows = []
    for name in sorted(arms):
        r = arms[name].get("walk")
        if r is None:
            continue
        walk_rows.append([
            name,
            fmt(g(r, "body_speed", "median")),
            fmt(g(r, "body_speed", "p90")),
            fmt(g(r, "body_speed", "segment_peak_median")),
            fmt(g(r, "body_speed", "share_above_0p5"), 100, 1),
            fmt(g(r, "body_speed", "share_above_1p0"), 100, 1),
            fmt(g(r, "high_speed_stability", "cruise_share_of_valid"), 100, 2),
            fmt(mtbf(r), 1, 0),
            fmt(g(r, "fall_rate_per_attempt"), 1, 4),
            str(g(r, "duration_s", default="-")),
        ])
    print("== 지속 보행: --goal_pattern forward_hold (목표가 항상 2 m 앞, 도착 없음) ==")
    print(table(["arm", "속도med", "속도p90", "구간peak_med",
                 ">0.5m/s(%)", ">1.0m/s(%)", "순항체류(%)", "낙상간격(s)",
                 "낙상률/구간", "길이(s)"],
                walk_rows))
    print()
    print("⛔ 낙상을 **원시 개수**로 읽지 마라. 이 표에는 120 s 런과 60 s 런이 섞여 있고")
    print("   개수는 길이에 비례한다. '낙상간격(s)'이 프로토콜에 무관한 값이다 --")
    print("   구간수와 구간 길이로 환산한 평균 무사고 시간이고, 실기 요구와 직접 대조된다.")
    print()
    print("⛔ 속도med 에는 **넘어지며 나는 속도가 들어간다.** 유일한 마스크가 리셋 직후")
    print("   0.2 s 뿐이라(eval_goal_pose.py:1174), 낙상간격이 짧은 arm 은 전도 동작이")
    print("   그대로 히스토그램에 들어간다. 배포 정책이 낙상간격 1.2 s 인데 속도 1.12 로")
    print("   M7(0.87)보다 빨라 보이는 것이 그 결과다. **순항체류(%)와 같이 읽어라.**")
    print()

    # ---- 표 3: 자세/포화 -- 실기 증상과 대조하는 축 ----------------------
    ex_rows = []
    for name in sorted(arms):
        r = arms[name].get("walk") or arms[name].get("accuracy")
        if r is None:
            continue
        ex_rows.append([
            name,
            fmt(g(r, "v7_extras", "feet_width"), 100),        # cm
            fmt(g(r, "v7_extras", "torque_occupancy"), 100, 1),
            fmt(g(r, "v7_extras", "torque_saturated"), 100, 2),
            fmt(g(r, "v7_extras", "dof_pos_occupancy"), 100, 1),
            fmt(g(r, "high_speed_stability", "cruise_roll_abs_p90_deg"), 1, 1),
            fmt(g(r, "high_speed_stability", "cruise_pitch_abs_p90_deg"), 1, 1),
        ])
    print("== 자세/포화 (실기 증상 대조축: 실기는 다리가 모이며 발끼리 부딪혔다) ==")
    print(table(["arm", "발간격(cm)", "토크점유(%)", "토크포화(%)",
                 "관절점유(%)", "roll p90(도)", "pitch p90(도)"], ex_rows))
    print()
    print("주: 발간격은 좌우 발 중심거리다. 배포 정책이 7.0 cm 였고 발 폭이 7.0 cm 라")
    print("    실기에서 발끼리 부딪혔다. 10 cm 이상이 목표다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
