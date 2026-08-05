"""deploy의 --log-timing CSV를 읽는다. 질문은 하나다: 관측 지연이 얼마인가.

MuJoCo 감지 열화 스윕(§8-40)에서 **관측 지연만 여유가 0**이었다. IMU 잡음·바이어스·
자이로·dof_vel은 학습값의 10-17배까지 버티는데, 관측 지연은 학습이 아예 모델링하지
않고(액션 쪽 0-18 ms만 있다) 다음처럼 무너진다:

    10 ms  낙상 0        60초 완주
    20 ms  낙상  2/분    60 걸음마다
    25 ms  낙상 13/분     9.2 걸음마다
    30 ms  낙상 34/분     3.5 걸음마다   <- "세 발자국"
    35 ms  낙상 47/분     2.6 걸음마다   <- "세 발자국"
    40 ms  낙상 49/분     2.4 걸음마다

⚠️ low_state_age는 **수신측 노후도**다. 로봇 내부의 센서->SDK 구간은 안 잡히므로
이 값은 실제 관측 지연의 **하한**이다. 하한이 이미 25 ms를 넘으면 그것만으로 판정된다.

    python tools/read_timing.py e0_timing.csv
"""

import sys
import csv


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round(p / 100.0 * (len(xs) - 1)))))
    return xs[i]


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
    if not rows:
        print("빈 파일이다. 컨트롤러가 run() 루프에 도달하지 못했다는 뜻이다 --")
        print("CUSTOM 진입 전에 멈췄는지 콘솔 로그를 보라.")
        return 1

    age = [float(r["low_state_age_s"]) * 1000.0 for r in rows]
    tick = [float(r["tick_dt_s"]) * 1000.0 for r in rows if float(r["tick_dt_s"]) > 0]
    walk = sum(int(r["walking"]) for r in rows)
    dur = float(rows[-1]["t_s"]) - float(rows[0]["t_s"])

    print("표본 %d개, %.1f초 (%.1f Hz), 걷는 중 %d개" % (
        len(rows), dur, len(rows) / max(dur, 1e-9), walk))
    print()
    print("== 관측 지연 low_state_age (ms) ==")
    print("  median %.1f   p90 %.1f   p99 %.1f   max %.1f" % (
        pct(age, 50), pct(age, 90), pct(age, 99), max(age)))
    print()
    print("== 정책 tick 간격 (ms, 설계값 20.0) ==")
    if tick:
        print("  median %.1f   p90 %.1f   p99 %.1f   max %.1f" % (
            pct(tick, 50), pct(tick, 90), pct(tick, 99), max(tick)))
        print("  20 ms 초과 비율 %.1f%%" % (100.0 * sum(t > 20.0 for t in tick) / len(tick)))
    print()

    m = pct(age, 50)
    print("== 판정 ==")
    if m >= 25.0:
        print("  ⛔ median %.1f ms -- MuJoCo에서 2-9 걸음마다 넘어지는 구간이다." % m)
        print("     이것만으로 세 발자국 증상이 설명된다. 학습에 관측 지연")
        print("     랜덤화를 넣는 것이 다음 수순이다.")
    elif m >= 15.0:
        print("  ⚠️ median %.1f ms -- 무릎 근처다(20 ms에서 흔들리기 시작)." % m)
        print("     p99를 보라. 꼬리가 30 ms를 넘으면 그 순간들이 낙상을 만든다.")
    else:
        print("  ✅ median %.1f ms -- MuJoCo 기준 안전 구간이다." % m)
        print("     ⚠️ 단 이 값은 하한이다. 센서->SDK 구간이 안 잡혀 있으므로")
        print("     '관측 지연이 원인이 아니다'로 바로 읽으면 안 된다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
