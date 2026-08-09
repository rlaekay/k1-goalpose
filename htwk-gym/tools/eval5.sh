#!/usr/bin/env bash
# 한 체크포인트를 **5개 독립 시드 × 두 축**으로 채점한다. 이것이 판정의 기본 단위다.
#
#   bash tools/eval5.sh <run_dir> [<라벨>] [<추가 eval 인자...>]
#
# 예) 자기 물리에서 재려면 config 를 미리 만들어 EVAL_CFG 로 넘긴다:
#   EVAL_CFG=/tmp/nj_own.yaml bash tools/eval5.sh logs/.../NJ_armasset NJ_own
#
# ---- 왜 5개인가 ---------------------------------------------------------------
#
# 한 번 재고 결론을 말한 것이 이 캠페인의 철회 50건의 공통 원인이다. 실측:
#   * 낙상 수는 같은 조건 재측정에서 **1 대 37** 까지 갈렸다(28쌍 부호검정 p=0.0000)
#   * `CLAUDE.md` 최상단 "낙상간격 2배" 는 **1건 대 2건**, p=1.00 이었다
#   * 반면 정확도 축은 재현된다(시드 SD 1.95 %)
# ⇒ **축마다 재현성이 다르다.** 시드 5개면 축별 산포를 직접 보고, 그 위에서
#   `tools/stat_compare.py` 가 지표에 맞는 검정을 건다.
#
# 5는 임의가 아니라 **비용과 검정력의 절충**이다: Wilcoxon 부호순위는 n=5 에서
# 양측 최소 p 가 0.0625 라 단독으로는 p<0.05 를 못 만든다. 그래서 연속 지표는
# **효과크기(Hodges-Lehmann)와 산포**로 읽고, 계수 지표(낙상)는 5회를 **합쳐**
# 노출시간 분모로 정확 Poisson 검정을 건다 -- 그쪽은 n 이 아니라 노출이 검정력이다.
# n 을 더 키우는 것이 필요하면 그때 10 으로 올린다(도구는 그대로 쓴다).
#
# ⛔ 시드만 바꾼다. 다른 것을 같이 바꾸면 그건 5-시드 반복이 아니라 다른 실험이다.
set -e

RUN="${1:?사용법: eval5.sh <run_dir> [라벨] [추가인자...]}"
LABEL="${2:-$(basename "$RUN")}"
shift 2 2>/dev/null || shift 1
EXTRA="$@"

CKPT="${CKPT:-$RUN/nn/model_6000.pth}"
[ -e "$CKPT" ] || { echo "⛔ 체크포인트가 없다: $CKPT"; exit 1; }

OUT="logs/eval_rounds/v2/$LABEL"
mkdir -p "$OUT"

# 채점 config. EVAL_CFG 로 미리 만든 것을 주면 그걸 쓰고(레버를 켠 채점 등),
# 없으면 공통 프로토콜에 관측 인터페이스만 이식한다.
if [ -n "$EVAL_CFG" ]; then
    CFG="$EVAL_CFG"
    echo "채점 config(외부 지정): $CFG"
else
    CFG="/tmp/eval5_$LABEL.yaml"
    python -u tools/make_eval_cfg.py --common sweeps/N0_ctrl.yaml --run "$RUN" --out "$CFG"
fi
cp "$CFG" "$OUT/protocol.cfg.yaml"
echo "체크포인트: $(readlink -f "$CKPT")"

run_one () {          # $1 = seed, $2 = axis(accuracy|walk)
    local seed="$1" axis="$2" dest="$OUT/seed${1}.${2}" pattern="" dump=""
    [ "$axis" = "walk" ] && pattern="--goal_pattern forward_hold"
    # ⛔ 덤프 경로는 **런마다** 달라야 한다. `$EXTRA` 에 --dump_actions 를 넣으면
    # 시드 10개가 같은 파일을 덮어써서 마지막 하나만 남는다 -- 그러면 5-시드
    # 프로토콜이 명령 축에서만 조용히 1-시드로 무너진다.
    [ -n "$DUMP_ACTIONS" ] && dump="--dump_actions $dest/actions.npz --dump_actions_envs ${DUMP_ACTIONS_ENVS:-16}"
    if [ -f "$dest/report.json" ]; then
        echo "  건너뜀(이미 있음): $dest"; return 0
    fi
    mkdir -p "$dest"
    set +e
    python -u eval_goal_pose.py --task K1/Goal_Pose_V7 --config "$CFG" \
        --checkpoint "$CKPT" --terrain plane --duration_s 120 --seed "$seed" $pattern \
        $dump $EXTRA --sim_device cuda:$GPU_INDEX --rl_device cuda:$GPU_INDEX \
        --out "$dest" > "$dest.log" 2>&1
    local rc=$?
    set -e
    # ⛔ 종료코드로 판정하지 않는다. `eval_goal_pose` 는 리포트를 다 쓴 뒤 종료
    # 과정에서 segfault(rc139) 하는 경우가 있고, 그걸 실패로 세면 워커가 같은
    # 채점을 한 번 더 태운다(2026-08-09 에 렌더 두 건에서 실제로 그랬다).
    if [ ! -f "$dest/report.json" ]; then
        echo "  ⛔ 실패 seed=$seed axis=$axis (rc=$rc) -- 로그: $dest.log"; return 1
    fi
    [ "$rc" -ne 0 ] && echo "  ⚠️ rc=$rc 이지만 리포트가 남았다(종료 segfault). 성공으로 친다."
    echo "  ok seed=$seed axis=$axis"
}

FAIL=0
for seed in 0 1 2 3 4; do
    for axis in accuracy walk; do
        run_one "$seed" "$axis" || FAIL=$((FAIL + 1))
    done
done

echo
echo "완료: $OUT   (실패 $FAIL 건)"
echo "다음: python tools/stat_compare.py --a '$OUT/seed*.accuracy' --b '<대조군>/seed*.accuracy' \\"
echo "        --label-a $LABEL --label-b <대조군>"
[ "$FAIL" -gt 0 ] && exit 1
exit 0
