#!/usr/bin/env bash
# 큐를 **처음부터 다시 만든다**. warm start 를 iteration 번호로 못 박는다.
#
#   bash tools/rebuild_queue.sh          # 무엇을 지우고 무엇을 넣을지 보여주기만 한다
#   bash tools/rebuild_queue.sh --yes    # 실제로 적용
#
# ⛔ 왜: 2026-08-07 감사에서 큐/plan 의 작업 스크립트 10개 중 **7개가 `best.pth` 를
# warm start 로 쓰고 있었다.** 그건 `select_best_checkpoint.py --link_best` 가 평가할
# 때마다 다시 만드는 **심볼릭 링크**이고, 실측하니 `N1_path/nn/best.pth` 는
# **model_100** 을 가리킨다(정확도 승자 = 붕괴 **전** 지점).
#
# 결과가 두 가지였다:
#   1. 주석은 "N1 의 최종 정책에서 출발한다"고 말하는데 실제로는 100 iteration
#      짜리에서 출발했다 -- "N1 대 이 arm" 이 한 레버 비교가 아니었다.
#   2. 출발점이 **가변**이라, 평가를 한 번 더 돌리면 다음 arm 의 warm start 가
#      조용히 바뀐다. 재현 불가능한 실험이다.
#
# ---- 계보를 둘로 나누고, 각각에 대조군을 둔다 -------------------------------
#
# 계보 A -- `I3b_stance10/model_200` (배포된 정책). 대조군은 `N1_path` 다.
#   N1_path 와 한 레버 차이인 arm 이 여기 온다: ND_dwell, N3_pathcross.
#
# 계보 B -- `N1_path/model_100` (붕괴 전). 이미 NA_histzero / NZ_zeroiid /
#   NC_actfilter 세 개가 이 계보로 돌아버렸다(위 사고 때문). 되돌리는 것보다
#   **그 계보를 완성하는 편이 싸다** -- 셋을 버리지 않아도 되고, 텐서보드상
#   goal_reached 가 N1(1.03)의 1.7배(1.78)라 붕괴하지 않을 가능성이 보인다.
#   ⭐ 다만 **대조군이 없다.** 그게 NE_ctrl100 이다: 같은 출발점, 레버 없음, 6000.
#   그것 없이는 "NA 가 안 무너졌다"가 레버 덕인지 출발점 덕인지 못 가른다.

cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"
APPLY=""
[ "${1:-}" = "--yes" ] && APPLY=1

STANCE10="logs/K1/K1/Goal_Pose_V7/2026-08-04-09-48-36_I3b_stance10/nn/model_200.pth"
N1RUN=$(MIN_CKPT=10 bash tools/pick_run.sh N1_path) || exit 1
N1_100="$N1RUN/nn/model_100.pth"

for f in "$STANCE10" "$N1_100"; do
    [ -e "$ROOT/$f" ] || { echo "⛔ warm start 원본이 없다: $f"; exit 1; }
done
echo "계보 A warm start: $STANCE10"
echo "계보 B warm start: $N1_100"
echo

# 유지할 것 -- 이미 올바르게 쓰여 있다(체크포인트가 명시적이거나 warm start 가 없다)
KEEP="004-eval_N0final_acc.sh 006-pareto_fine_N2_pathgrid.sh"

echo "== 삭제 대상 (warm start 가 best.pth = 가변 심볼릭 링크) =="
for f in "$ROOT"/queue/gpu0/*.sh "$ROOT"/queue/gpu1/*.sh \
         "$ROOT"/queue/plan/gpu0/*.sh "$ROOT"/queue/plan/gpu1/*.sh; do
    [ -e "$f" ] || continue
    b=$(basename "$f")
    case " $KEEP " in *" $b "*) continue ;; esac
    if grep -q 'best\.pth' "$f" || [ "$b" = "900-N1_long.sh" ]; then
        why="best.pth"
        [ "$b" = "900-N1_long.sh" ] && why="ls -1td 최신 선택 + 파레토가 900~6000 평평함을 보임(가치 소멸)"
        echo "  ${f#$ROOT/queue/}   ($why)"
        [ -n "$APPLY" ] && rm -f "$f"
    fi
done
echo

# emit <arm> <gpu> <prio> <lineage A|B> <new_obs> <new_priv> <live|plan>
emit() {
    local arm=$1 gpu=$2 prio=$3 lin=$4 nobs=$5 npriv=$6 where=$7
    local ck; [ "$lin" = "A" ] && ck="$STANCE10" || ck="$N1_100"
    local dir="$ROOT/queue/gpu$gpu"; [ "$where" = "plan" ] && dir="$ROOT/queue/plan/gpu$gpu"
    local out="$dir/$prio-$arm.sh"
    local surgery=""
    if [ "$nobs" != "54" ] || [ "$npriv" != "14" ]; then
        surgery="WARM=/tmp/${arm}_warm.pth
python -u tools/expand_checkpoint.py --src \"\$CK\" --dst \"\$WARM\" \\
    --old-obs 54 --new-obs $nobs --old-priv 14 --new-priv $npriv --verify || exit 1"
    fi
    [ -n "$APPLY" ] && { mkdir -p "$dir"; cat > "$out" <<OUTER
#!/usr/bin/env bash
# $arm  (계보 $lin, obs 54->$nobs, priv 14->$npriv)
#
# warm start 를 **iteration 번호로 못 박는다.** best.pth 는 평가할 때마다 다시
# 만들어지는 심볼릭 링크라 출발점이 조용히 바뀐다(2026-08-07 감사, ibatch §8-49).
CK="$ck"
[ -e "\$CK" ] || { echo "warm start 원본이 없다: \$CK"; exit 1; }
echo "warm start: \$CK"
WARM="\$CK"
python -u tools/make_v7_arms.py --only $arm --max_iterations 6000 > /dev/null || exit 1
$surgery
python -u train_v7.py --task=K1/Goal_Pose_V7 --config sweeps/$arm.yaml \\
    --headless True --checkpoint "\$WARM" \\
    --num_envs 4096 --max_iterations 6000 \\
    --sim_device cuda:\$GPU_INDEX --rl_device cuda:\$GPU_INDEX
OUTER
    chmod +x "$out"; }
    printf '  %-34s 계보 %s  obs->%-4s priv->%-3s %s\n' \
        "${out#$ROOT/queue/}" "$lin" "$nobs" "$npriv" "$where"
}

echo "== 새로 넣을 것 =="
# ⭐ 계보 B 의 대조군이 최우선이다 -- 이미 끝난 arm 셋(NA/NZ/NC)의 해석이 여기 달렸다.
emit NE_ctrl100    0 010 B  54 14 live
# 관측 축에서 정보량이 가장 큰 것(이력). 계보 B.
emit N4_hist       1 020 B 270 14 live
# 계보 A: N1_path 와 한 레버 차이. 파레토 절벽에 대한 직접 처방.
emit ND_dwell      0 030 A  54 14 live
emit N7_critic     1 040 B  54 29 plan
emit N9_zerostruct 0 050 B  54 14 plan
emit NB_zerocritic 1 060 B  54 29 plan
emit N5_tau        0 070 B  66 14 plan
emit N6_foot       1 080 B  56 14 plan
emit N3_pathcross  0 090 A  54 14 plan

echo
if [ -n "$APPLY" ]; then
    echo "live gpu0: $(ls -1 "$ROOT/queue/gpu0" 2>/dev/null | tr '\n' ' ')"
    echo "live gpu1: $(ls -1 "$ROOT/queue/gpu1" 2>/dev/null | tr '\n' ' ')"
    echo "plan gpu0: $(ls -1 "$ROOT/queue/plan/gpu0" 2>/dev/null | tr '\n' ' ')"
    echo "plan gpu1: $(ls -1 "$ROOT/queue/plan/gpu1" 2>/dev/null | tr '\n' ' ')"
else
    echo "미리보기다. 적용하려면 --yes."
fi
