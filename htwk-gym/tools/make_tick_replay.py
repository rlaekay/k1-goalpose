"""실기 로그에서 (벽시계 틱 주기, 카운터 증분) 쌍을 뽑는다 -- 시간 팽창 재생용.

왜 이 두 개인가
---------------
`deploy/utils/timer.py` 의 `Timer` 는 벽시계가 아니라 **LowState 콜백에서만 증가하는
카운터**다(`counter * time_step`, 증가 지점은 `deploy_goal_pose.py:871` 하나뿐).
정책이 보는 시간축은 그 카운터이고, 로봇이 실제로 사는 시간은 벽시계다. 콜백이
설계 500 Hz 보다 느리게 **실행**되면 두 축이 벌어지고, 보행 위상이 벽시계 기준으로
늘어진다.

  카운터 증분 = (틱 사이 콜백 수) x time_step
  벽시계 증분 = tick_dt_s

카운터 증분은 로그에 직접 없지만 **복원된다** -- `gait_process` 는 카운터 시간의
적분이므로

  콜백 수 = (gait_process 증분) / (time_step x gait_frequency)

이 나눗셈이 정수로 떨어지는지가 이 기전의 **지문**이다. 실측(91틱): 증분 90개가
전부 0.004 의 정수배(잔차 0.0), 값 7개(0.028~0.052), 콜백 수 7~13, mean 9.989 =
`decimation` 10. 벽시계 시계였다면 연속분포에 sd = 2*sd(tick_dt) = 0.0167 이
나와야 하는데 실측은 0.00424 다.

⛔ 정수로 안 떨어지면 이 스크립트는 **멈춘다.** 그때는 기전 가정이 틀린 것이고,
재생 파일을 만들면 안 된다.

    python tools/make_tick_replay.py ../realdata/2026-08-0x_real_walk_i3b.csv \
        -o logs/mujoco/tick_replay_real.csv
"""

import os
import csv
import sys
import argparse

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="실기 --log-timing CSV")
    ap.add_argument("-o", "--out", default="logs/mujoco/tick_replay_real.csv")
    ap.add_argument("--time-step", type=float, default=0.002,
                    help="deploy config common.dt -- Timer 의 눈금")
    ap.add_argument("--gait-hz", type=float, default=None,
                    help="생략하면 로그의 gait_freq 열에서 읽는다")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="정수 판정 허용 잔차")
    args = ap.parse_args()

    with open(args.log, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit("빈 로그: %s" % args.log)

    gait = args.gait_hz
    if gait is None:
        gf = [float(r["gait_freq"]) for r in rows if r.get("gait_freq") not in (None, "")]
        gf = [v for v in gf if v > 0]
        if not gf:
            raise SystemExit("gait_freq 열이 없거나 전부 0 이다 -- --gait-hz 로 줘라")
        gait = float(np.median(gf))
    quantum = args.time_step * gait          # 콜백 1회분 위상
    print("gait_frequency %.3f Hz, time_step %.4f s -> 위상 양자 %.6f" % (gait, args.time_step, quantum))

    gp = [float(r["gait_process"]) for r in rows]
    wall = [float(r["tick_dt_s"]) for r in rows]

    pairs, bad = [], []
    for i in range(1, len(gp)):
        x = gp[i] - gp[i - 1]
        if x < -0.5:
            x += 1.0                          # 0..1 톱니 랩 복원
        if abs(x) < 1e-12:
            continue                          # 위상이 얼어 있는 틱(도착/정지)은 건너뛴다
        n = x / quantum
        if abs(n - round(n)) > args.tol:
            bad.append((i, x, n))
        # 벽시계 주기는 **그 틱의** tick_dt_s 다(행 i 가 그 간격을 보고한다)
        pairs.append((wall[i], round(n) * args.time_step))

    if bad:
        print("\n⛔ 위상 증분이 %.6f 의 정수배가 아니다 -- %d/%d 틱:" % (quantum, len(bad), len(pairs)))
        for i, x, n in bad[:10]:
            print("   행 %d: 증분 %.6f -> 콜백수 %.4f" % (i, x, n))
        print("\n카운터 시계 가정이 이 로그에서 성립하지 않는다. 재생 파일을 만들지 않는다.")
        return 1

    w = np.array([p[0] for p in pairs])
    c = np.array([p[1] for p in pairs])
    nb = c / args.time_step
    print("틱 %d개 -- 전부 정수배 ✅ (지문 확인)" % len(pairs))
    print("  콜백수      median %.1f  mean %.3f  범위 %d~%d  (decimation 설계 10)"
          % (np.median(nb), nb.mean(), nb.min(), nb.max()))
    print("  벽시계 주기 median %.5f  mean %.5f  sd %.5f s" % (np.median(w), w.mean(), w.std()))
    print("  카운터 증분 median %.5f  mean %.5f s" % (np.median(c), c.mean()))
    print("  ⭐ 시계 비 = 카운터/벽시계 = %.4f  -> 콜백 실행률 %.0f Hz (설계 %.0f)"
          % (c.sum() / w.sum(), (c.sum() / w.sum()) / args.time_step, 1.0 / args.time_step))
    print("  ⭐ 벽시계 케이던스 = %.4f Hz (명령 %.2f)" % (gait * c.sum() / w.sum(), gait))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["wall_dt_s", "counter_dt_s"])
        for a, b in pairs:
            wr.writerow(["%.6f" % a, "%.6f" % b])
    print("\n%s (%d 쌍)" % (args.out, len(pairs)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
