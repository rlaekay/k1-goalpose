"""발목 토크로 CoM 위치를 역산한다. 로봇을 움직이지 않는다.

원리
----
발바닥이 지면에 붙어 있고 로봇이 정지해 있으면, 발목 pitch 축 둘레의 모멘트가
0이어야 한다. 몸무게가 CoM에서 아래로 걸리므로

    sum(tau_ankle_pitch)  =  M * g * (x_CoM - x_ankle)

따라서

    x_CoM - x_ankle  =  sum(tau) / (M * g)

M = 18.714 kg (URDF 총 질량), g = 9.81 -> M*g = 183.6 N.
발목 토크 1 N*m 당 CoM이 5.4 mm 앞이다.

지지다각형 (URDF 충돌 상자 0.16 x 0.07 x 0.032, 발 링크 원점 +0.014):
    발끝  x_ankle + 0.094 m
    뒤꿈치 x_ankle - 0.066 m

즉 **발목 pitch 토크 합이 17.3 N*m을 넘으면 CoM이 발끝을 넘은 것**이고, 그때 로봇은
정적으로 앞으로 넘어진다. URDF가 맞다면 기본 자세에서 이 값이 훨씬 작아야 한다.

⚠️ 사람이 잡고 있으면 그 손이 하중을 나눠 지므로 토크가 낮게 나온다. 손을 뗀
순간의 값이 진짜다. 그래서 시간열로 찍는다 -- 손을 서서히 떼면서 최대값을 본다.

⚠️ 발목은 평행 링크다. 모터가 보고하는 pitch 토크가 관절 토크인지 액추에이터
토크인지 SDK가 보장하지 않는다. 값의 부호와 크기가 물리와 안 맞으면 그것 자체가 발견이다.

    python3 probe_com_from_ankle.py --seconds 25
"""

import sys
import time
import math
import argparse

import booster_robotics_sdk_python as B

MG = 18.714 * 9.81          # N
TOE = 0.094                 # m, 발목 축에서 발끝까지
HEEL = -0.066               # m
L_ANKLE_PITCH, R_ANKLE_PITCH = 14, 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="127.0.0.1")
    ap.add_argument("--seconds", type=float, default=25.0)
    ap.add_argument("--hz", type=float, default=20.0)
    args = ap.parse_args()

    st = {}

    def cb(m):
        st["m"] = m

    B.ChannelFactory.Instance().Init(0, args.net)
    sub = B.B1LowStateSubscriber(cb)
    sub.InitChannel()
    t0 = time.monotonic()
    while "m" not in st and time.monotonic() - t0 < 5:
        time.sleep(0.05)
    if "m" not in st:
        print("LowState 못 받음")
        return 1

    print("발목 토크 1 N*m = CoM %.1f mm 앞.  발끝까지 = %.1f N*m" % (
        1000.0 / MG, TOE * MG))
    print("잡고 있는 손을 **천천히** 떼면서 보십시오. 위험하면 바로 다시 잡으십시오.")
    print()
    print("  %6s %9s %9s %9s   %s" % ("t(s)", "tauL", "tauR", "CoM(mm)", "위치"))

    rows = []
    t0 = time.monotonic()
    nxt = t0
    while time.monotonic() - t0 < args.seconds:
        nxt += 1.0 / args.hz
        time.sleep(max(0.0, nxt - time.monotonic()))
        m = st.get("m")
        if m is None:
            continue
        ms = m.motor_state_serial
        # LowState가 가끔 짧은 motor 리스트로 온다. 첫 실행이 여기서 IndexError로
        # 죽었다 -- 그것도 손을 떼기 전에. 계측이 측정을 못 하고 죽는 것이 가장 나쁘다.
        if len(ms) <= R_ANKLE_PITCH:
            continue
        tl = float(ms[L_ANKLE_PITCH].tau_est)
        tr = float(ms[R_ANKLE_PITCH].tau_est)
        tot = tl + tr
        x = tot / MG
        r, p, _ = m.imu_state.rpy
        tilt = math.degrees(abs(p))
        rows.append((time.monotonic() - t0, tl, tr, x, tilt))
        where = ("⛔ 발끝 넘음" if x > TOE else
                 "발끝까지 %+.0f mm" % (1000 * (TOE - x)))
        print("  %6.1f %9.2f %9.2f %9.1f   %s  (pitch %.1f도)" % (
            rows[-1][0], tl, tr, 1000 * x, where, tilt))

    sub.CloseChannel()
    if not rows:
        return 1

    xs = [r[3] for r in rows]
    print()
    print("== 요약 ==")
    print("  CoM-발목 오프셋: median %+.1f mm   max %+.1f mm" % (
        1000 * sorted(xs)[len(xs) // 2], 1000 * max(xs)))
    print("  발끝 %+.1f mm / 뒤꿈치 %+.1f mm" % (1000 * TOE, 1000 * HEEL))
    if max(xs) > TOE:
        print("  ⛔ CoM이 발끝을 넘는 순간이 있었다 -- 정적으로 앞으로 넘어진다.")
        print("     기본 자세가 이 로봇에서 정적 평형이 아니라는 뜻이고,")
        print("     sim은 정지 자세를 지탱해 본 적이 없어 이 결함을 드러낼 수 없었다.")
    else:
        print("  URDF 기준 지지다각형 안이다. 손을 완전히 떼지 못했을 수 있으니")
        print("     max 값과 pitch 각을 같이 보라.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
