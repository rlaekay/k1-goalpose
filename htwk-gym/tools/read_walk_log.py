"""deploy --log-timing CSV를 읽고 **로봇이 실제로 걸었는지** 판정한다.

왜 이 도구가 따로 필요한가
--------------------------
2026-08-05, 나는 이 로그의 `walking=1`과 `gait_freq=2.00`을 보고 "85초째 걷고 있다"고
보고했다. 로봇은 1초 만에 멈춰 덜덜 떨고 있었다.

`walking`은 **명령 쪽 플래그**다 -- `_update_arrival_gait`가 목표 거리 > stop_radius일
때 세운다. `fixed` 목표는 로봇 로컬 프레임의 상수라 거리가 영원히 1.0 m다. 그래서
로봇이 멈춰 있든 넘어져 있든 `walking=1`이 유지된다. `gait_freq`도 명령이다.
**명령을 행동의 증거로 읽으면 안 된다.**

그래서 이 도구는 명령 채널을 판정에 쓰지 않는다. 행동만 본다:

  * **관절 진폭** -- 보행이면 무릎/고관절이 gait 주기로 크게 흔들린다. 멈춰 있으면 0.
  * **고관절 roll 표류** -- 사용자가 본 "다리가 벌어진다"가 이 채널이다.
  * **토크** -- 덜덜 떠는 것이 포화(한계에 붙음)인지 진동(부호가 뒤집힘)인지 가른다.
  * **기울기 진폭** -- 보행이면 몸통이 주기적으로 흔들린다.

    python3 tools/read_walk_log.py e0_walk.csv
"""

import sys
import csv
import math

LEG = ["Hip_P", "Hip_R", "Hip_Y", "Knee", "Ank_P", "Ank_R"]
# URDF effort (sim이 클램프하는 값). 포화 판정 기준.
EFFORT = [30.0, 20.0, 20.0, 40.0, 20.0, 15.0] * 2


def stat(xs):
    if not xs:
        return 0.0, 0.0, 0.0
    s = sorted(xs)
    return s[0], s[len(s) // 2], s[-1]


def windows(t, vals, w=1.0):
    """1초 창마다 (최대-최소). 보행이면 크고, 정지면 0에 붙는다."""
    out, i, n = [], 0, len(t)
    while i < n:
        j = i
        while j < n and t[j] - t[i] < w:
            j += 1
        seg = vals[i:j]
        if len(seg) >= 5:
            out.append(max(seg) - min(seg))
        i = j
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
    if len(rows) < 20:
        print("표본이 %d개뿐이다." % len(rows))
        return 1
    if "q0" not in rows[0]:
        print("⛔ 옛 포맷이다 -- 관절/토크 채널이 없어 '걸었는가'를 판정할 수 없다.")
        print("   이 로그로는 통신 타이밍(low_state_age)만 읽을 수 있다.")
        return 1

    t = [float(r["t_s"]) for r in rows]
    dur = t[-1] - t[0]
    print("표본 %d, %.1f초, %.1f Hz" % (len(rows), dur, len(rows) / max(dur, 1e-9)))
    print()

    # ---- 걸었는가 ---------------------------------------------------------
    knee = [float(r["q3"]) for r in rows]          # 왼 무릎
    tilt = [float(r["tilt_deg"]) for r in rows]
    kw = windows(t, knee)
    tw = windows(t, tilt)
    print("== 1초 창 진폭 (행동 채널) ==")
    print("  왼 무릎 각    median %.4f rad   max %.4f" % (
        sorted(kw)[len(kw) // 2] if kw else 0, max(kw) if kw else 0))
    print("  몸통 기울기   median %.3f도      max %.3f" % (
        sorted(tw)[len(tw) // 2] if tw else 0, max(tw) if tw else 0))

    # 걷는 구간: 무릎 진폭이 0.05 rad(2.9도)를 넘는 창. 서 있기만 해도 미세하게
    # 움직이므로 0은 기준이 안 되고, 한 걸음의 무릎 변화는 이보다 훨씬 크다.
    moving = [a > 0.05 for a in kw]
    share = 100.0 * sum(moving) / max(len(moving), 1)
    print("  -> 무릎이 실제로 움직인 창: %.0f%% (%d/%d)" % (
        share, sum(moving), len(moving)))
    print()

    # 실제로 움직인 구간만 골라 나머지를 잰다. 정지 구간을 섞으면 통계가 거짓이 된다.
    live = [i for i, r in enumerate(rows)
            if any(abs(float(rows[min(i + k, len(rows) - 1)]["dq3"])) > 0.5
                   for k in range(3))]
    print("== 유효 구간 (실제로 관절이 움직인 표본) ==")
    if not live:
        print("  없다. 로봇은 이 로그 내내 움직이지 않았다.")
    else:
        print("  %d 표본, t=%.1f ~ %.1f초" % (
            len(live), t[live[0]], t[live[-1]]))
    print()

    # ---- 다리가 벌어지는가 -------------------------------------------------
    hr_l = [float(r["q1"]) for r in rows]          # 왼 Hip_Roll
    hr_r = [float(r["q7"]) for r in rows]          # 오른 Hip_Roll
    print("== 고관절 roll (다리 벌어짐) ==")
    for nm, v in (("왼", hr_l), ("오른", hr_r)):
        lo, md, hi = stat(v)
        print("  %-4s 시작 %+.3f -> 끝 %+.3f rad   범위 %+.3f ~ %+.3f (%.1f도 폭)" % (
            nm, v[0], v[-1], lo, hi, math.degrees(hi - lo)))
    print()

    # ---- 토크: 포화인가 진동인가 ------------------------------------------
    print("== 다리 토크 (포화 = 한계에 붙음, 진동 = 부호 뒤집힘) ==")
    print("  %-8s %8s %8s %8s   %s" % ("관절", "median", "max|.|", "한계", "포화율"))
    for i in range(12):
        v = [float(r["tau%d" % i]) for r in rows]
        av = [abs(x) for x in v]
        lim = EFFORT[i]
        sat = 100.0 * sum(a > 0.9 * lim for a in av) / len(av)
        flips = sum(1 for a, b in zip(v, v[1:]) if a * b < 0)
        name = ("L_" if i < 6 else "R_") + LEG[i % 6]
        if sat > 1.0 or max(av) > 0.5 * lim:
            print("  %-8s %8.2f %8.2f %8.1f   %5.1f%%  부호전환 %d회" % (
                name, sorted(v)[len(v) // 2], max(av), lim, sat, flips))
    print()
    print("== 통신 (정지 상태에서도 유효) ==")
    age = [float(r["low_state_age_s"]) * 1000 for r in rows]
    tick = [float(r["tick_dt_s"]) * 1000 for r in rows if float(r["tick_dt_s"]) > 0]
    lo, md, hi = stat(age)
    print("  low_state_age ms: median %.2f  max %.2f" % (md, hi))
    if tick:
        s = sorted(tick)
        print("  tick 간격 ms (설계 20): median %.1f  p99 %.1f  max %.1f" % (
            s[len(s) // 2], s[int(0.99 * (len(s) - 1))], s[-1]))
        print("  ⚠️ tick 통계는 정지 구간을 섞으면 무의미하다 -- 위 유효 구간과 같이 보라.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
