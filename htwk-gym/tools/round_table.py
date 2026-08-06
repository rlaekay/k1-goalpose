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

    # ---- 표 1: 정확도 (공통 waypoint 프로토콜) ---------------------------
    acc_rows = []
    for name in sorted(arms):
        r = arms[name].get("accuracy")
        if r is None:
            continue
        acc_rows.append([
            name,
            fmt(g(r, "pos_err_m", "median"), 100),      # cm
            fmt(g(r, "pos_err_m", "p90"), 100),
            fmt(g(r, "heading_err_deg", "median"), 1, 1),
            str(g(r, "falls", default="-")),
            fmt(g(r, "success_rate_strict"), 100, 1),
            str(g(r, "segments_completed", default="-")),
            "PASS" if g(r, "all_gates_pass") else "fail",
        ])
    print("== 정확도: 공통 waypoint 프로토콜 (모든 arm 을 같은 config 로 채점) ==")
    print(table(["arm", "오차med(cm)", "오차p90(cm)", "heading(도)",
                 "낙상", "strict(%)", "구간수", "게이트"], acc_rows))
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
            str(g(r, "falls", default="-")),
            fmt(g(r, "feasibility", "required_speed_median")),
        ])
    print("== 지속 보행: --goal_pattern forward_hold (목표가 항상 2 m 앞, 도착 없음) ==")
    print(table(["arm", "속도med", "속도p90", "구간peak_med",
                 ">0.5m/s(%)", ">1.0m/s(%)", "순항체류(%)", "낙상", "요구속도med"],
                walk_rows))
    print()
    print("주: 요구속도med 는 프로브가 실제로 속도를 요구했는지의 확인용이다.")
    print("    waypoint 과제에서 이 값이 0.12 였다 -- 그게 '보행을 잰 적이 없다'의 정체다.")
    print("    forward_hold 에서 이 값이 여전히 0.2 미만이면 프로브가 안 걸린 것이므로")
    print("    아래 속도 숫자를 믿지 마라.")
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
