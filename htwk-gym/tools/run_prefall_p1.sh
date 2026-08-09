#!/usr/bin/env bash
# `p1` 이 조건의 성질인가, 낙상의 산물인가 — 분석 세션이 건 판별.
#
# 직립 필터(tilt<15°)로는 부족하다. tilt 5~15° 는 **이미 넘어지는 중의 허둥댐**이고
# 거기서 다리가 엉킨다. ⇒ 낙상이 많은 칸일수록 `p1` 이 낮은 것이 **낙상의 결과**일 수 있다.
#
# 사전등록:
#   - 낙상에서 떨어진 표본만으로 재계산해도 **조건별 `p1` 순서가 유지**되면
#     ⇒ `p1` 은 조건의 성질이다. "청정 팔의 기하학적 기전" 이 산다.
#   - 차이가 **사라지거나 크게 줄면** ⇒ `p1` 은 낙상의 산물이고 그 해석도 죽는다.
#
# ⚠️ 이 셀들을 이미 네 번 다시 돌렸다. 이번에는 **원시 시계열(npz)** 을 남기므로
# 같은 자료에 대한 다음 질문은 재실행 없이 답한다.
set -u

ROOT=/mnt/DATA/workspace/ws_eungkyu/k1-goalpose/htwk-gym
cd "$ROOT" || exit 1
P=${POLICY:-logs/K1/K1/Goal_Pose_V7/2026-08-04-09-48-36_I3b_stance10/nn/model_200.pt}
B="--policy $P --real-asset --goal-hold --duration 120 --armature-preset vendor --clock-mode counter"
O=logs/mujoco/prefall
mkdir -p "$O"

# 청정 팔은 결정론적이라 시드 1개면 충분하다(정수까지 같다는 것을 확인했다).
for MS in 25.0 25.7 26.14 27.0; do
    python play_mujoco_goalpose.py $B --seed 0 --period-ms "$MS" \
        --out "$O/clean_${MS}.json" || exit 1
done
# 오염 팔은 rng 를 쓰므로 진짜 반복이다.
for MS in 25.0 25.7 26.14; do
    for S in 0 1 2; do
        python play_mujoco_goalpose.py $B --seed "$S" --period-ms "$MS" \
            --corrupt-lankr 0.44 --corrupt-rankr 0.21 \
            --out "$O/corr_${MS}_s${S}.json" || exit 1
    done
done
# 대조: 설계 주기, 낙상 0 -> 낙상 오염이 원리적으로 없는 칸
python play_mujoco_goalpose.py --policy "$P" --real-asset --goal-hold --duration 120 \
    --armature-preset vendor --period-ms 20.0 --clock-mode sim --seed 0 \
    --out "$O/ctrl20.json"

echo "=== p1: 분모를 바꿔 가며 ==="
python - "$O" <<'PY'
import json, glob, os, sys, statistics as st

def load(pat):
    return sorted(glob.glob(os.path.join(sys.argv[1], pat)))

def cell(lab, ps):
    if not ps:
        return
    def m(fn):
        vs = [fn(json.load(open(p))) for p in ps]
        vs = [v for v in vs if v is not None]
        return st.mean(vs) if vs else float("nan")
    f = [json.load(open(p))["falls"] for p in ps]
    print("%-16s 낙상 %-10s | 전체 %+.4f | 직립 %+.4f | 첫낙상前 %+.4f | 1s초과 %+.4f | 2s초과 %+.4f"
          % (lab, "/".join(map(str, f)),
             m(lambda r: r.get("foot_sep_m", {}).get("p1")),
             m(lambda r: r.get("foot_sep_upright_m", {}).get("p1")),
             m(lambda r: r.get("foot_sep_prefall_m", {}).get("before_first_fall", {}).get("p1")),
             m(lambda r: r.get("foot_sep_prefall_m", {}).get("gap_gt_1s", {}).get("p1")),
             m(lambda r: r.get("foot_sep_prefall_m", {}).get("gap_gt_2s", {}).get("p1"))))

cell("대조 20.0", load("ctrl20.json"))
for ms in ("25.0", "25.7", "26.14", "27.0"):
    cell("청정 %s" % ms, load("clean_%s.json" % ms))
for ms in ("25.0", "25.7", "26.14"):
    cell("오염 %s" % ms, load("corr_%s_s*.json" % ms))
print()
print("판정: 낙상에서 떨어진 분모로도 조건별 순서가 유지되면 p1 은 조건의 성질이다.")
print("      차이가 사라지면 p1 은 낙상의 산물이고 '기하학적 기전' 해석이 죽는다.")
PY
