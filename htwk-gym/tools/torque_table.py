"""관절별 토크 점유율 표. 판정 기준은 `tools/queue_torqueaudit.sh` 에 사전 고정돼 있다.

    python tools/torque_table.py logs/eval_rounds/torqueaudit

⛔ 이 도구는 `torque_by_joint`(롤아웃 전체 누산)만 읽는다.
`torque_occupancy` / `torque_saturated` 는 `_v7_extras` 호출 시점의 **한 프레임**이고
eval 은 그것을 롤아웃 끝에 한 번 읽는다 -- RETRACTIONS C11 이 발 간격으로 이미 당한
결함이라 여기서는 아예 안 읽는다.
"""

import os
import sys
import json
import glob

# 사전 고정된 판정 문턱. 여기 있는 값을 결과를 보고 바꾸지 마라 -- 그게 이 파일에
# 문턱을 적어 두는 이유다.
RIVAL_LIMIT_NM = 20.0     # armsdown URDF 의 Hip_Roll effort (배포 계보가 학습한 값)
DEPLOY_LIMIT_NM = 30.0    # deploy/configs/Goal_Pose_E0.yaml 의 Hip_Roll
WATCH = ("HipR", "HipP", "KneeP", "AnkleP", "AnkleR", "HipY")


def load(p):
    if os.path.isdir(p):
        p = os.path.join(p, "report.json")
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        print("!! 못 읽음 {}: {}".format(p, exc))
        return None


def main(root):
    reps = sorted(glob.glob(os.path.join(root, "*", "report.json")))
    if not reps:
        print("리포트가 없다: {}".format(root))
        return 1
    print("\n=== 관절별 토크 점유율 (롤아웃 전체 누산) ===\n")
    verdicts = []
    for rp in reps:
        r = load(rp)
        if not r:
            continue
        tj = (r.get("v7_extras") or {}).get("torque_by_joint")
        name = os.path.basename(os.path.dirname(rp))
        if not tj:
            print("{}: torque_by_joint 없음 -- 누산기가 없는 커밋에서 채점됐다\n".format(name))
            continue
        steps = (r.get("v7_extras") or {}).get("torque_steps_accumulated")
        print("## {}   (누산 {} 스텝, {}s)".format(name, steps, r.get("duration_s")))
        print("   {:<12} {:>6} {:>9} {:>9} {:>9} {:>9}".format(
            "관절", "한계", "평균N·m", "평균점유", "최대점유", "포화비"))
        # 왼다리만 찍는다 -- 좌우가 같은 값이면 표가 두 배로 길어질 뿐이다.
        for jn in sorted(tj):
            if not jn.startswith("Left"):
                continue
            d = tj[jn]
            print("   {:<12} {:>6.1f} {:>9.3f} {:>9.4f} {:>9.4f} {:>9.5f}".format(
                jn.replace("Left_", "L."), d["limit"], d["mean_Nm"],
                d["mean_occ"], d["peak_occ"], d["sat_share"]))
        hr = tj.get("Left_HipR") or tj.get("Left_Hip_R")
        if hr:
            lim = hr["limit"]
            # 사전 고정된 판정: 배포 계보(20)와 배포 config(30)를 넘는가.
            over_rival = hr["peak_occ"] * lim > RIVAL_LIMIT_NM
            over_dep = hr["peak_occ"] * lim > DEPLOY_LIMIT_NM
            verdicts.append((name, lim, hr["mean_Nm"], hr["peak_occ"] * lim,
                             over_rival, over_dep))
        print()

    if verdicts:
        print("=== 판정: Hip_Roll 이 배포가 안 주는 토크를 쓰는가 ===\n")
        print("   {:<22} {:>6} {:>9} {:>9}  {}".format(
            "run", "한계", "평균N·m", "최대N·m", "판정"))
        for name, lim, mean_nm, peak_nm, over_rival, over_dep in verdicts:
            if over_dep:
                v = "⛔ 배포 config 30 N·m 도 넘는다"
            elif over_rival:
                v = "⛔ 배포 계보(armsdown 20)를 넘는다"
            else:
                v = "✅ 20 N·m 안에서 논다 -- 이 축은 원인이 아니다"
            print("   {:<22} {:>6.1f} {:>9.3f} {:>9.3f}  {}".format(
                name[:22], lim, mean_nm, peak_nm, v))
        print("\n⚠️ 기각이어도 URDF 세 값(35/20/30)이 다른 것 자체는 고쳐야 한다.")
        print("   기각은 '지금 이 증상의 원인이 아니다' 이지 '문제가 없다' 가 아니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "logs/eval_rounds/torqueaudit"))
