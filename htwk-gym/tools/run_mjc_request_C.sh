#!/usr/bin/env bash
# 배포 세션 요청 셀 C0~C5 (MJC_TEST_REQUEST_20260809.md §3). 한 셀 한 레버.
#
# 판정 사전등록(문서 §4): 재현 = 측방 capture 단일지지 위반 >= 50 % **그리고**
# 발목 roll |dq| p90 >= 10 rad/s.
#
# ⚠️ 내가 짚는 두 가지 (결과와 함께 보고한다)
#
# 1. **분모가 정의돼 있지 않다.** 실기 로그에는 접촉 센서가 없는데 표적이
#    "단일지지 표본 기준 68 %" 다. 무엇으로 단일지지를 갈랐는지 문서에 없다.
#    그래서 MuJoCo 쪽은 분모를 **전부** 낸다(단일지지/전체 × 직립필터).
#
# 2. ⛔ **낙상이 capture 위반율을 오염시킨다.** 문서는 "낙상 수 비교 금지, 비교
#    축은 capture 위반율" 이라고 했는데, MuJoCo 에서 낙상하는 셀은 넘어지는 동안
#    capture 를 자동으로 위반한다. 실기는 **잡아준 상태라 낙상이 0** 이다.
#    그러므로 위반율을 그대로 비교하면 "낙상 수 비교"를 이름만 바꿔서 하는 것이다.
#    -> `viol_*_upright`(tilt<15도 표본만)를 **1차 지표로 읽는다.** 그것이 실기가
#    있던 상태다. 원시 위반율도 같이 내되 그것만으로 판정하지 않는다.
#
# C2 는 문서가 "액션 1차 지연 35 ms (또는 EMA fc 6.57 Hz @185 Hz)" 로 둘을 허용했다.
# 둘 다 돌린다 -- C2 는 순수 전달지연(사전등록 문구 그대로), C2b 는 실제 배포
# 코드 경로(필터를 185 Hz 로 돌린 것)다. 같은 가설의 두 구현이 갈리면 그 자체가 정보다.
#
#   bash tools/run_mjc_request_C.sh
set -u

ARM=${ARM:-vendor}
OUT=logs/mujoco/mjc_request_C
mkdir -p "$OUT"
DUR=${DUR:-120}
POLICY=${POLICY:-logs/K1/K1/Goal_Pose_V7/2026-08-04-09-48-36_I3b_stance10/nn/model_200.pt}
# ⛔ armature 는 셀 레버가 아니라 **바닥 물리**다. C3 만 그것을 흔들므로 나머지는
# 고정해야 한다. 기본을 vendor 로 두는 이유: 실기 로봇에 벤더 로터 관성이 있다.
# C3 는 문서 지정대로 "발목만 0.0565, 나머지 0"(=학습 조건 + 발목) 이라 별도다.
COMMON="--policy $POLICY --real-asset --goal-hold --duration $DUR --seed 0"

run() {
    local name=$1; shift
    echo "=============== $name ==============="
    python play_mujoco_goalpose.py $COMMON "$@" \
        --out "$OUT/$name.json" --video "$OUT/$name.mp4" --dump-csv "$OUT/$name.csv" 2>&1 | tail -12
}

# C0  대조: 올바른 타이밍(50 Hz tick, 2.0 Hz 시계, 필터 없음)
run C0_base        --armature-preset $ARM --period-ms 20.0  --clock-mode sim
# C1  시계 0.79 배 -> 25.32 ms 에서 nominal 20 ms 만 전진 = gait 1.58 Hz
run C1_clock079    --armature-preset $ARM --period-ms 25.32 --clock-mode counter
# C2  액션 전달지연 35 ms (사전등록 문구 그대로)
run C2_actlag35    --armature-preset $ARM --period-ms 20.0  --clock-mode sim --act-lag-ms 35
# C2b 같은 가설의 배포 코드 경로: EMA 를 185 Hz 로 (fc ~6 Hz)
run C2b_ema185     --armature-preset $ARM --period-ms 20.0  --clock-mode sim \
                   --deploy-filter --filter-hz 185
# C3  발목만 벤더 armature, 힙·무릎 0 (= 학습 조건 + 발목)
run C3_ankarm      --armature-preset ankle --period-ms 20.0 --clock-mode sim
# C4  C1 + C2 (= 실측 배포 조건)
run C4_clock_lag   --armature-preset $ARM --period-ms 25.32 --clock-mode counter --act-lag-ms 35
# C5  C0 + 몸통 지속 측방 외력 20 N
run C5_force20     --armature-preset $ARM --period-ms 20.0  --clock-mode sim \
                   --body-force-n 20 --body-force-dir lateral

echo
echo "=============== 요약 ==============="
python - "$OUT" <<'PY'
import json, glob, os, sys
print("%-16s %6s %8s | %9s %9s %9s | %7s %7s | %8s %8s"
      % ("셀", "낙상", "직립%", "cap단일", "cap단일直", "cap전체直",
         "AnkR L", "AnkR R", "발간격min", "Lhipδ"))
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    r = json.load(open(p)); c = r.get("capture", {})
    a = r.get("ankle_roll_dq", {}); s = r.get("foot_sep_m", {})
    d = r.get("l_hip_pitch_drift", {})
    def pc(v): return "-" if v is None else "%.1f%%" % (100 * v)
    print("%-16s %6d %8s | %9s %9s %9s | %7s %7s | %8s %8s"
          % (os.path.basename(p)[:-5], r["falls"], pc(c.get("share_upright")),
             pc(c.get("viol_4p4_single")), pc(c.get("viol_4p4_single_upright")),
             pc(c.get("viol_4p4_all_upright")),
             a.get("L_p90"), a.get("R_p90"), s.get("min"), d.get("delta")))
print()
print("실기 표적: cap 단일지지 위반 68 %, 발목 roll |dq| p90 22.2(L), L_hip 드리프트 -0.19 rad")
print("사전등록 재현 기준: cap 단일지지 위반 >= 50 % AND 발목 roll |dq| p90 >= 10")
PY
