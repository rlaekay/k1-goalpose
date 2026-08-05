"""발목 두 축의 명령 대 실측을 잰다 -- 평행 링크 매핑이 실기에 있고 sim에 없는지.

왜
--
2026-08-05, 실기 2.4초 로그와 MuJoCo를 같은 축에 겹쳐 보니 **roll 축만** 어긋난다:

    관절 궤적 폭 (실기/MuJoCo)   HipP·Knee·HipY 0.6-1.1배,  AnkP 1.3배
                                 **L_HipR 4.3, L_AnkR 2.5, R_HipR 2.4, R_AnkR 1.9**
    정책 출력 폭                  pitch/yaw 1.0-1.3배,  **HipR 3.0-3.3, L_AnkR 3.1**
    발목 pitch 토크 rms          **실기가 sim의 0.6-0.7배** (각도는 1.3배 더 크게 움직임)

각도는 크고 토크는 작다 = 발목이 지면을 못 밀고 있다. 발목은 보행 중 몸통 roll을
잡는 주 액추에이터이므로, 그것이 약하면 몸통이 흔들리고 -> 정책이 Hip_Roll로
보상하고 -> Hip_Roll이 3배 폭주하고 -> 다리가 벌어져 부딪힌다. 증상 전체가 이어진다.

용의자는 평행 링크다:
  * `mech.parallel_mech_indexes: [14, 15, 20, 21]` = 양발 Ankle_Pitch/Roll
  * `deploy_base_walk.py:181`은 이 인덱스에 직렬-평행 토크 변환을 한다.
    `deploy_goal_pose.py:691`은 **변환 없이 위치 명령**한다(주석이 의도라고 밝힌다).
  * **어느 sim 자산에도 평행 링크가 없다** -- Isaac도 MuJoCo도 직렬 2관절이다.

이 프로브가 가르는 것: 관절 공간에서 pitch만 흔들었을 때
  (a) 진폭이 명령대로 나오는가  -> 이득 오차
  (b) roll이 같이 움직이는가    -> 교차결합 (평행 매핑이 보정 안 된 증거)

⛔ 안전
-------
CUSTOM 모드로 관절을 잡는다. **발을 지면에서 띄운 상태(로봇을 들고 있거나 거치대)로
하는 것을 권한다** -- 발목이 무부하가 되어 넘어질 수 없고, 액추에이터->관절 매핑은
무부하에서도 드러난다. 진폭은 ±3도로 작다.
`touch /tmp/ankle_abort` 로 언제든 중단하면 DAMPING으로 나간다.
"""

import os
import sys
import time
import math
import argparse

import numpy as np
import yaml

from booster_robotics_sdk_python import (
    ChannelFactory, B1LocoClient, B1LowCmdPublisher, B1LowStateSubscriber,
    LowCmd, RobotMode)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ABORT = "/tmp/ankle_abort"
L_AP, L_AR, R_AP, R_AR = 14, 15, 20, 21


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/Goal_Pose_E0.yaml")
    ap.add_argument("--net", default="127.0.0.1")
    ap.add_argument("--amp-deg", type=float, default=3.0)
    ap.add_argument("--freq", type=float, default=0.5, help="Hz")
    ap.add_argument("--cycles", type=int, default=4)
    ap.add_argument("--out", default="/tmp/ankle_map.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    n = int(cfg["common"]["joint_cnt"])
    dt = float(cfg["common"]["dt"])
    q0 = np.array(cfg["common"]["default_qpos"], dtype=np.float32)
    kp = np.array(cfg["common"]["stiffness"], dtype=np.float32)
    kd = np.array(cfg["common"]["damping"], dtype=np.float32)

    state = {"q": None}

    def on_low_state(msg):
        try:
            ms = msg.motor_state_serial
            if len(ms) >= n:
                # 콜백 안에서 복사한다. 밖에서 읽으면 SDK가 버퍼를 재사용해
                # 떠 있는 참조가 된다(2026-08-05에 segfault로 배웠다).
                state["q"] = [float(ms[i].q) for i in range(n)]
                state["tau"] = [float(ms[i].tau_est) for i in range(n)]
        except Exception:
            pass

    if os.path.exists(ABORT):
        os.remove(ABORT)
    ChannelFactory.Instance().Init(0, args.net)
    sub = B1LowStateSubscriber(on_low_state); sub.InitChannel()
    pub = B1LowCmdPublisher(); pub.InitChannel()
    client = B1LocoClient(); client.Init()
    t0 = time.time()
    while state["q"] is None and time.time() - t0 < 5:
        time.sleep(0.05)
    if state["q"] is None:
        print("LowState 없음"); os._exit(1)

    cmd = LowCmd()
    def send(q_target):
        for i in range(n):
            cmd.motor_cmd[i].q = float(q_target[i])
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].tau = 0.0
            cmd.motor_cmd[i].kp = float(kp[i])
            cmd.motor_cmd[i].kd = float(kd[i])
        pub.Write(cmd)

    # ⛔ 기본자세(서 있는 자세)로 램프하지 않는다. 엎드린 로봇에게 그것을 명령하면
    # 바닥을 밀며 요동친다. 이 시험에 필요한 것은 발목이 자유롭게 움직이는 것뿐이고,
    # 액추에이터->관절 매핑은 기구학적 성질이라 어떤 자세에서 재도 같다.
    # 그래서 **진입 시점의 측정 자세를 그대로 붙든다.**
    print("CUSTOM 진입 -- 현재 자세를 그대로 붙든다 (기본자세로 옮기지 않는다)")
    client.ChangeMode(RobotMode.kCustom)
    time.sleep(1.2)
    q0 = np.array(state["q"], dtype=np.float32)      # 기준 자세 = 지금 자세
    print("  기준 자세 발목(도): LAP %.1f LAR %.1f RAP %.1f RAR %.1f" % tuple(
        math.degrees(q0[i]) for i in (L_AP, L_AR, R_AP, R_AR)))
    hold_t0 = time.time()
    while time.time() - hold_t0 < 1.5:
        if os.path.exists(ABORT):
            print("[abort]"); break
        send(q0)
        time.sleep(dt)

    rows = []
    amp = math.radians(args.amp_deg)
    dur = args.cycles / args.freq

    def sweep(name, idx_l, idx_r):
        print("  %s 축 %.1f도 %.1f Hz x %d주기" % (name, args.amp_deg, args.freq, args.cycles))
        s0 = time.time()
        while time.time() - s0 < dur:
            if os.path.exists(ABORT):
                return False
            t = time.time() - s0
            u = amp * math.sin(2 * math.pi * args.freq * t)
            tgt = q0.copy()
            tgt[idx_l] += u
            tgt[idx_r] += u
            send(tgt)
            q = state["q"]; tau = state.get("tau") or [0.0] * n
            rows.append([name, t, u] + [q[i] for i in (L_AP, L_AR, R_AP, R_AR)]
                        + [tau[i] for i in (L_AP, L_AR, R_AP, R_AR)])
            time.sleep(dt)
        # 원위치
        for _ in range(150):
            send(q0); time.sleep(dt)
        return True

    ok = sweep("pitch", L_AP, R_AP)
    if ok:
        time.sleep(0.5)
        ok = sweep("roll", L_AR, R_AR)

    print("DAMPING 으로 나간다")
    try:
        client.ChangeMode(RobotMode.kDamping)
    except Exception as e:
        print("ChangeMode 실패:", e)

    with open(args.out, "w") as f:
        f.write("axis,t,u,qLAP,qLAR,qRAP,qRAR,tLAP,tLAR,tRAP,tRAR\n")
        for r in rows:
            f.write(",".join([str(r[0])] + ["%.6g" % v for v in r[1:]]) + "\n")
    print("샘플 %d -> %s" % (len(rows), args.out))
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
