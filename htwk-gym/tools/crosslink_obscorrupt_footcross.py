"""§8-69(발목 roll 관측 오염) 과 MJC §6-c(정책이 발 교차를 **지령**한다)를 잇는다.

두 발견이 따로 서 있다:
  * §8-69: `motor_state_serial` 발목 roll 이 오염돼 policy 관측으로 들어갔다.
           보행 run2 에서 왼발목 **44 %** 의 틱이 오염이었다.
  * MJC §6-c: 낙상 0 셀은 발끼리 충돌 0, 낙상 셀은 전부 충돌. 그리고 명령-FK 로
           보면 **정책이 교차를 지령한다**(붕괴 명령 min −0.138 m).

⇒ **묻는 것**: 오염된 관측을 먹은 틱의 **명령 발간격이 더 깊게 교차하는가?**

  그렇다 -> §8-69 는 발 교차의 **상류**이고, 배포 쪽 게이트가 교차를 줄인다.
            (그러면 보상만 켜는 것으로는 절반만 고치는 것이다)
  아니다 -> 둘은 독립이고, 교차는 MJC 말대로 **학습 쪽 보상** 문제다.
            §8-69 게이트는 여전히 옳지만 4초 붕괴의 설명은 아니다.

⛔ 이건 관측연구다. 오염 틱은 무작위 배정이 아니라 **다리가 빠를 때** 몰린다
(§8-69 T4). 그래서 다리 속도를 **짝지어** 비교한다 -- 안 그러면 "빠를 때 교차한다"를
"오염되면 교차한다"로 잘못 읽는다. 그것이 이 파일의 유일한 통계적 주의점이다.

    python tools/crosslink_obscorrupt_footcross.py ../realdata/<walk>.csv
"""
import csv
import os
import sys

import numpy as np
import yaml
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cmd_foot_sep import foot_sep, MJCF, CFG, LEG0  # noqa: E402

JUMP = 0.20
ROLL = (5, 11)          # q5 = L_Ankle_Roll, q11 = R_Ankle_Roll


def main():
    path = sys.argv[1]
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    cfg = yaml.safe_load(open(CFG, encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(MJCF)
    data = mujoco.MjData(model)
    default = np.array(cfg["common"]["default_qpos"], dtype=float)
    scale = float(cfg["policy"]["control"]["action_scale"])

    cmd, meas, legspd, jump = [], [], [], []
    for i, r in enumerate(rows):
        act = np.array([float(r["act%d" % k]) for k in range(12)])
        q = np.array([float(r["q%d" % k]) for k in range(12)])
        cmd.append(foot_sep(model, data, default[LEG0:LEG0 + 12] + scale * act,
                            model.nq))
        meas.append(foot_sep(model, data, q, model.nq))
        # 다리 전체 속도(교락 변수). 발목 roll 두 채널은 **뺀다** -- 그 채널이
        # 오염 그 자체라 짝짓기 변수에 넣으면 처치를 대조군에 섞는 것이다.
        legspd.append(float(np.mean([abs(float(r["dq%d" % k]))
                                     for k in range(12) if k not in ROLL])))
        if i == 0:
            jump.append(False)
        else:
            jump.append(any(abs(float(rows[i][
                "q%d" % k]) - float(rows[i - 1]["q%d" % k])) >= JUMP
                for k in ROLL))

    cmd, meas = np.array(cmd), np.array(meas)
    legspd, jump = np.array(legspd), np.array(jump)
    print("### %s  (%d 틱)" % (os.path.basename(path), len(rows)))
    print("오염 틱 %d (%.0f %%)" % (jump.sum(), 100 * jump.mean()))
    print("명령 발간격  median %+.4f m  min %+.4f  음수 %.1f %%"
          % (np.median(cmd), cmd.min(), 100 * (cmd < 0).mean()))
    print("실측 발간격  median %+.4f m  min %+.4f  음수 %.1f %%"
          % (np.median(meas), meas.min(), 100 * (meas < 0).mean()))

    print("\n-- 조 비교 (짝짓기 없음) --")
    for lag in (0, 1, 2):
        a = cmd[lag:][jump[:len(cmd) - lag]]
        b = cmd[lag:][~jump[:len(cmd) - lag]]
        if len(a) and len(b):
            print("  lag %d: 오염뒤 median %+.4f (n=%d) vs 정상뒤 %+.4f (n=%d)"
                  "  차 %+.4f m" % (lag, np.median(a), len(a), np.median(b),
                                    len(b), np.median(a) - np.median(b)))

    print("\n-- ⭐ 다리속도 짝짓기 (교락 제거) --")
    qs = np.quantile(legspd, [0, .25, .5, .75, 1.0])
    for lo, hi in zip(qs[:-1], qs[1:]):
        m = (legspd >= lo) & (legspd <= hi)
        a, b = cmd[m & jump], cmd[m & ~jump]
        if len(a) >= 3 and len(b) >= 3:
            print("  다리속도 [%.2f, %.2f]  오염 %+.4f (n=%d) vs 정상 %+.4f (n=%d)"
                  "  차 %+.4f" % (lo, hi, np.median(a), len(a), np.median(b),
                                  len(b), np.median(a) - np.median(b)))
        else:
            print("  다리속도 [%.2f, %.2f]  표본 부족 (오염 %d / 정상 %d)"
                  % (lo, hi, len(a), len(b)))


if __name__ == "__main__":
    main()
