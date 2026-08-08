"""관절 각도를 **읽기만** 한다. 손으로 밀면서 실시간으로 값을 본다.

왜 별도 도구인가
----------------
`dump_robot_layout.py` 는 3초 기다렸다 **한 번 찍고 끝나는 스냅샷**이다. "발목을 손으로
스톱까지 밀고 그때 값을 읽는다" 는 측정에는 맞지 않는다 -- 미는 순간과 찍는 순간이
맞아떨어져야 하기 때문이다. 이 도구는 연속으로 읽으면서 **구간 최소/최대**를 남긴다.

무엇을 가르려는 측정인가
------------------------
실기 로그의 발목 roll 이 **±0.55 rad** 인데 URDF/MJCF 전 자산의 하드 스톱은 **±0.345**
(벤더 공개 사양 Ankle R ±20° = ±0.349 와도 일치)다. 두 가지가 가능하다:

  (a) 실기 ROM 이 진짜 넓다        -> 우리 자산이 틀렸다
  (b) `motor_state_serial` 이 URDF 관절각과 **다른 양**이다
      -> `obs[23]`/`obs[29]` 가 학습과 다른 것을 싣고 있다는 뜻이라 훨씬 심각하다

손으로 스톱까지 밀었을 때:
  * `|serial|` 이 0.345 근처에서 멈춘다  -> (b). 관측 정의가 다르다
  * `|serial|` 이 0.55 근처까지 간다     -> (a). 자산이 틀렸다
  * 그 사이에서 멈춘다                   -> 그 값이 참 한계다

⭐ `serial` 과 `parallel` 을 **나란히** 찍는다. 둘의 비가 일정하면 그 배율이 곧
변환 게인이고, (b) 의 정체가 그 자리에서 드러난다.

안전
----
**읽기 전용이다.** `B1LowStateSubscriber` 하나만 만든다 -- publisher 도,
`ChangeMode` 도, `LowCmd` 도 없다. 이 프로세스는 로봇에 아무것도 명령하지 않으므로
로봇을 **DAMPING(힘 빠진 상태)** 에 두고 손으로 움직이면 된다. Ctrl-C 로 언제든
끊어도 로봇 상태가 바뀌지 않는다.

사용
----
    # 발목 roll (기본): 왼발 15, 오른발 21
    python3 tools/read_joint_live.py

    # 다른 관절 / 더 오래
    python3 tools/read_joint_live.py --idx 14,15,20,21 --seconds 120
"""

import argparse
import sys
import time

import booster_robotics_sdk_python as sdk

# 이 K1 의 22관절 배치. 다리는 10 부터, 관절당 순서는
# Hip_Pitch / Hip_Roll / Hip_Yaw / Knee_Pitch / Ankle_Pitch / Ankle_Roll.
NAMES = {
    10: "L_Hip_Pitch", 11: "L_Hip_Roll", 12: "L_Hip_Yaw",
    13: "L_Knee_Pitch", 14: "L_Ankle_Pitch", 15: "L_Ankle_Roll",
    16: "R_Hip_Pitch", 17: "R_Hip_Roll", 18: "R_Hip_Yaw",
    19: "R_Knee_Pitch", 20: "R_Ankle_Pitch", 21: "R_Ankle_Roll",
}
# 참고선. URDF 하드 스톱 = 벤더 공개 사양과 일치하는 값.
URDF_LIMIT = {
    14: (-0.87, 0.345), 15: (-0.345, 0.345),
    20: (-0.87, 0.345), 21: (-0.345, 0.345),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--net", default="127.0.0.1")
    ap.add_argument("--idx", default="15,21",
                    help="볼 관절 인덱스, 쉼표 구분 (기본: 발목 roll 좌우)")
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--hz", type=float, default=5.0, help="화면 갱신률")
    args = ap.parse_args()

    idx = [int(x) for x in args.idx.split(",") if x.strip()]

    sdk.ChannelFactory.Instance().Init(0, args.net)

    box = {}

    def on_low(m):
        # 콜백 안에서 값을 복사한다 -- 메시지 객체는 콜백 동안만 유효하다.
        try:
            ser = [float(j.q) for j in m.motor_state_serial]
            par = [float(j.q) for j in m.motor_state_parallel]
        except Exception:
            return
        if ser:
            box["serial"], box["parallel"], box["n"] = ser, par, box.get("n", 0) + 1

    sub = sdk.B1LowStateSubscriber(on_low)      # ⛔ 구독만. 발행 없음.
    sub.InitChannel()

    deadline = time.time() + 3.0
    while "serial" not in box and time.time() < deadline:
        time.sleep(0.05)
    if "serial" not in box:
        print("low_state 가 3초 안에 안 온다 -- 로봇 전원과 모션 스택을 확인해라.",
              file=sys.stderr)
        return 1

    lo = {i: float("inf") for i in idx}
    hi = {i: float("-inf") for i in idx}
    t0 = time.time()
    n0 = box.get("n", 0)
    print("읽기 전용. Ctrl-C 로 끊어도 로봇 상태는 안 바뀐다.")
    print("발목을 **손으로** 양쪽 끝까지 천천히 밀어라. 최대/최소가 갱신된다.\n")
    try:
        while time.time() - t0 < args.seconds:
            time.sleep(1.0 / max(args.hz, 0.5))
            ser, par = box["serial"], box["parallel"]
            line = []
            for i in idx:
                if i >= len(ser):
                    continue
                v = ser[i]
                lo[i] = min(lo[i], v)
                hi[i] = max(hi[i], v)
                p = par[i] if i < len(par) else float("nan")
                ratio = (p / v) if abs(v) > 1e-3 else float("nan")
                lim = URDF_LIMIT.get(i)
                mark = ""
                if lim and (v < lim[0] - 1e-6 or v > lim[1] + 1e-6):
                    mark = " ⛔초과"
                line.append("%-14s ser %+.4f  par %+.4f  par/ser %+.3f  [%.4f, %.4f]%s"
                            % (NAMES.get(i, "idx%d" % i), v, p, ratio, lo[i], hi[i], mark))
            print("  " + "\n  ".join(line))
            print("  --- (%.0f s, low_state %d 개)" % (time.time() - t0, box.get("n", 0) - n0))
    except KeyboardInterrupt:
        pass

    print("\n================ 결과 ================")
    for i in idx:
        if lo[i] == float("inf"):
            continue
        lim = URDF_LIMIT.get(i)
        s = "  %-14s 관측 범위 [%+.4f, %+.4f]  폭 %.4f rad (%.1f도)" % (
            NAMES.get(i, "idx%d" % i), lo[i], hi[i], hi[i] - lo[i],
            (hi[i] - lo[i]) * 57.29578)
        if lim:
            s += "\n                 URDF 한계 [%+.4f, %+.4f]" % lim
            worst = max(abs(lo[i]), abs(hi[i]))
            s += "  -> |최대| %.4f = URDF 의 %.2f 배" % (worst, worst / max(abs(lim[1]), 1e-9))
        print(s)
    print("""
읽는 법 (발목 roll, URDF 한계 ±0.345):
  |최대| 이 0.345 ± 0.02  -> (b) 확증. motor_state_serial 이 URDF 관절각과 다른 양이다.
                             obs[23]/obs[29] 가 학습과 다른 것을 싣고 있다는 뜻이다.
  |최대| 이 0.55 ± 0.03    -> (a) 확증. 우리 자산의 하드 스톱이 틀렸다.
  그 사이               -> 그 값이 참 한계다. 자산을 그것으로 맞춘다.
  par/ser 이 일정한 상수 -> 그 배율이 직렬<->평행 변환 게인이다.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
