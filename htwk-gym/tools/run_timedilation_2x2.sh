#!/usr/bin/env bash
# 시간 팽창 2x2 (+ 지터 재생 2칸) -- MuJoCo, 배포 정책 그대로, 실물 질량.
#
# 무엇을 가르는가
# ---------------
# 실기 `Timer` 는 벽시계가 아니라 LowState 콜백 카운터다. 콜백 실행이 설계
# 500 Hz 보다 느리면 정책이 보는 시간축이 늘어나고, 보행이 벽시계 기준으로
# 느려진다(지령 2.0 -> 실측 1.51~1.58 Hz).
#
# 그런데 **두 축이 교락돼 있다**: 팽창이 있는 실행은 제어 주기도 같이 느리다.
#   - "제어가 39 Hz 로 느려서 나쁘다"
#   - "시계가 21~24 % 늘어져서 나쁘다"
# 2x2 가 이것을 가른다. B 와 D 는 **제어율이 같고 시계만 다르다.**
#
#   셀   제어주기      시계        벽시계 케이던스
#   A    20.00 ms     sim         2.00 Hz   기준선
#   B    26.14 ms     counter     1.53 Hz   붕괴 로그 재현
#   C    20.00 ms     counter     2.00 Hz   **널 셀** -- A 와 같아야 한다
#   D    26.14 ms     sim         2.00 Hz   수정(제어만 느림)
#   E1   재생(붕괴)   counter     변동      지터 포함
#   E2   재생(성공)   counter     변동      지터 포함
#
# ⛔ 사전등록
#   - C 가 A 와 다르면 **플래그 구현이 틀린 것**이다. 결과를 버리고 코드를 고친다.
#   - B 는 무너지는데 D 는 멀쩡하면 원인은 **시계**다.
#   - B 와 D 가 같이 무너지면 원인은 시계가 아니라 **제어율**이고,
#     `Timer` 를 고쳐도 소용없다.
#   - ⭐ **실기는 21 % 팽창에서 9걸음 낙상 0 을 갔다(ibatch §8-66).**
#     B 가 크게 무너지면 그것은 시계 가설의 확증이 아니라 **MuJoCo 가 이 결함의
#     비용을 과장한다**는 뜻이다 -- 실기가 양성 대조다.
#
# 26.14 ms 는 붕괴 로그의 tick_dt **mean** 이다. median(25.24)이 아니다 --
# 위상은 적분량이라 케이던스는 평균율로 정해지고, median 은 오른쪽 꼬리를 버려
# 케이던스를 과대평가한다(1.585 대 실측 1.512).
#
#   bash tools/run_timedilation_2x2.sh
set -u

# ⛔ armature 는 이 실험의 **교락 변수**다. 학습 세션 측정: 채점 물리만 armature 로
# 바꿔도 낙상간격이 1.5 s 와 3,740 s 로 갈린다. 셀 사이에서는 고정되므로 B 대 D 의
# **대조**는 어느 값에서도 유효하지만, "평균 낙상간격 3.1 s" 같은 **크기**는 그렇지
# 않다. 그리고 MJCF 기본(asset)은 벤더 기준 가장 큰 값인 무릎이 0 이다.
# 실기를 재현하려면 vendor 다 -- 로봇에는 벤더 로터 관성이 들어 있다.
# ⚠️ 게인은 **우리 것을 그대로 둔다.** 로봇이 우리 게인으로 돈다. --vendor-gains 는
# 다른 로봇을 재는 것이므로 이 스크립트에서는 켜지 않는다.
ARM=${ARM:-asset}
OUT=logs/mujoco/timedilation_$ARM
mkdir -p "$OUT"
DUR=${DUR:-120}
# 배포 계보 그대로. 필터 실험(§8-45/§8-47)과 **같은 체크포인트**라 나란히 놓을 수 있다.
# `deploy/models/goal_pose_i3b.pt` 는 이것을 export 한 것이고 서버에는 없다.
POLICY=${POLICY:-logs/K1/K1/Goal_Pose_V7/2026-08-04-09-48-36_I3b_stance10/nn/model_200.pt}
COMMON="--policy $POLICY --real-asset --goal-hold --duration $DUR --seed 0 --armature-preset $ARM"

# 지터 재생 파일을 실기 로그에서 만든다(지문 검증이 그 안에 있다).
python tools/make_tick_replay.py ../realdata/2026-08-0x_real_walk_i3b.csv \
    -o "$OUT/replay_collapse.csv"      || exit 1
python tools/make_tick_replay.py ../realdata/2026-08-09_t_walk_i3b_run2.csv \
    -o "$OUT/replay_success.csv"       || exit 1

run() {   # run <이름> <추가인자...>
    local name=$1; shift
    echo "=============== $name ==============="
    python play_mujoco_goalpose.py $COMMON "$@" \
        --out    "$OUT/$name.json" \
        --video  "$OUT/$name.mp4" \
        --dump-csv "$OUT/$name.csv" 2>&1 | tail -40
}

run A_base                                    --period-ms 20.00 --clock-mode sim
run B_dilated                                 --period-ms 26.14 --clock-mode counter
run C_null                                    --period-ms 20.00 --clock-mode counter
run D_fixed                                   --period-ms 26.14 --clock-mode sim
run E1_replay_collapse --clock-mode counter   --tick-replay "$OUT/replay_collapse.csv"
run E2_replay_success  --clock-mode counter   --tick-replay "$OUT/replay_success.csv"

echo
echo "=============== 요약 ==============="
python - "$OUT" <<'PY'
import json, glob, os, sys
rows = []
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    r = json.load(open(p))
    g = r.get("gait", {})
    rows.append((os.path.basename(p)[:-5], r.get("period_ms"), g.get("clock_mode"),
                 g.get("cadence_hz"), r.get("falls"), r.get("falls_per_min"),
                 g.get("incr_median"), g.get("incr_unique")))
print("%-22s %8s %8s %9s %6s %9s %10s %6s"
      % ("셀", "주기ms", "시계", "케이던스", "낙상", "낙상/분", "증분med", "고유"))
for x in rows:
    print("%-22s %8s %8s %9s %6s %9s %10s %6s" % tuple("" if v is None else v for v in x))
print()
print("실기 붕괴  : 케이던스 1.5116 Hz, 증분 median 0.040, 고유 7값, tilt max 22.6도 -> 붕괴")
print("실기 성공  : 케이던스 1.5813 Hz, 증분 median 0.040, 고유 8값, tilt max  8.3도 -> 9걸음 낙상 0")
PY
