"""LowState 도착 타이밍만 잰다. 관절을 잡지 않는다.

왜 이것만으로 되는가
--------------------
MuJoCo 감지 열화 스윕(§8-40)에서 **관측 지연만 여유가 0**이었다. 학습은 관측 지연을
아예 모델링하지 않고(액션 쪽 0-18 ms만 있다), 실기가 그 구간에 있는지가 질문이다:

    10 ms  낙상 0        60초 완주
    20 ms  낙상  2/분    60 걸음마다
    25 ms  낙상 13/분     9.2 걸음마다
    30 ms  낙상 34/분     3.5 걸음마다   <- "세 발자국"
    35 ms  낙상 47/분     2.6 걸음마다   <- "세 발자국"

`deploy_goal_pose.py`의 `low_state_age`는 정책이 추론하는 순간의 LowState 나이다.
그 값의 지배항은 **LowState 발행 주기와 지터**이고, 그건 CUSTOM 모드와 무관하다 --
채널만 열려 있으면 잰다. 그래서 이 프로브는 **로봇을 움직이지 않는다.** 구독만 한다.

무엇을 찍는가
-------------
* 도착 간격 (median/p99/max) -- 발행 주기와 지터
* 50 Hz로 폴링했을 때의 나이 -- deploy의 `low_state_age`가 보는 것과 같은 양.
  추론 시점은 발행과 비동기라, 나이는 평균적으로 주기의 절반 + 지터가 된다.

⚠️ 이 값은 **수신측** 노후도다. 로봇 내부의 센서->SDK 구간은 안 잡히므로 실제 관측
지연의 **하한**이다. 하한이 이미 25 ms를 넘으면 그것만으로 판정된다.

    python3 probe_lowstate_timing.py --net 127.0.0.1 --seconds 30
"""

import sys
import time
import argparse

import booster_robotics_sdk_python as B


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    i = min(len(xs) - 1, max(0, int(round(p / 100.0 * (len(xs) - 1)))))
    return xs[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="127.0.0.1")
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()

    arrivals = []

    def on_low_state(msg):
        arrivals.append(time.monotonic())

    B.ChannelFactory.Instance().Init(0, args.net)
    sub = B.B1LowStateSubscriber(on_low_state)
    sub.InitChannel()
    print("구독 시작 (%.0f초). 로봇은 움직이지 않는다." % args.seconds)

    t0 = time.monotonic()
    ages = []
    while time.monotonic() - t0 < args.seconds:
        # deploy의 추론 주기와 같은 50 Hz로 폴링해서 "그 순간의 나이"를 잰다.
        time.sleep(0.02)
        if arrivals:
            ages.append((time.monotonic() - arrivals[-1]) * 1000.0)

    sub.CloseChannel()

    if len(arrivals) < 10:
        print("⛔ LowState를 %d개만 받았다. 채널이 안 열렸거나 로봇이 상태를 안 쏜다."
              % len(arrivals))
        return 1

    gaps = [(b - a) * 1000.0 for a, b in zip(arrivals, arrivals[1:])]
    dur = arrivals[-1] - arrivals[0]
    print()
    print("LowState %d개 / %.1f초  ->  %.1f Hz" % (len(arrivals), dur, len(arrivals) / dur))
    print()
    print("== 도착 간격 (ms) ==")
    print("  median %.2f   p90 %.2f   p99 %.2f   max %.2f"
          % (pct(gaps, 50), pct(gaps, 90), pct(gaps, 99), max(gaps)))
    print()
    print("== 50 Hz 추론 시점의 나이 (ms) -- deploy의 low_state_age와 같은 양 ==")
    print("  median %.2f   p90 %.2f   p99 %.2f   max %.2f"
          % (pct(ages, 50), pct(ages, 90), pct(ages, 99), max(ages)))
    print()

    m, p99 = pct(ages, 50), pct(ages, 99)
    print("== 판정 (기준: MuJoCo 스윕 §8-40) ==")
    if m >= 25.0:
        print("  ⛔ median %.1f ms -- 2-9 걸음마다 넘어지는 구간이다." % m)
        print("     관측 지연만으로 세 발자국 증상이 설명된다.")
    elif m >= 15.0:
        print("  ⚠️ median %.1f ms -- 무릎 근처(20 ms에서 흔들리기 시작)." % m)
        print("     p99 %.1f ms -- 꼬리가 30 ms를 넘으면 그 순간들이 낙상을 만든다." % p99)
    else:
        print("  ✅ median %.1f ms (p99 %.1f) -- MuJoCo 기준 안전 구간." % (m, p99))
        print("     ⚠️ 단 이 값은 하한이다. 센서->SDK 구간이 안 잡혀 있으므로")
        print("     '관측 지연이 원인이 아니다'로 바로 읽으면 안 된다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
