#!/usr/bin/env bash
# 관측 확장 arm(N4-N7)을 autopilot 의 plan/ 에 넣는다.
#
# 왜 별도 스크립트인가: 이 작업 스크립트들은 런타임에 체크포인트 수술을 하고
# `$GPU_INDEX` 를 그대로 남겨야 해서 인용이 3중으로 겹친다. ssh 한 줄에 넣으면
# 어느 층에서 확장되는지가 사람 눈으로 안 보이고, 실제로 한 번 깨졌다.
# 저장소에 두고 서버에서 실행하면 그 문제가 통째로 사라진다.
#
#   bash tools/stage_obs_arms.sh
#
# plan/ 에 있다는 것은 "GPU 가 30분 놀면 그때 승격된다"는 뜻이다. 지금 큐에
# 직접 넣지 않는 이유는 N1_path 의 결과를 보고 나서 순서를 바꿀 여지를 남기기
# 위해서다 -- 그 판단이 안 오면 autopilot 이 알아서 집는다.

cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"
mkdir -p "$ROOT/queue/plan/gpu0" "$ROOT/queue/plan/gpu1"

# 인자로 준 arm 은 plan/ 이 아니라 **live 큐**에 바로 들어간다.
#   bash tools/stage_obs_arms.sh NA_histzero
LIVE=" $* "

# stage <arm> <new_obs> <new_priv> <gpu> <prio>
# new_obs == 54 이고 new_priv == 14 이면 관측 폭이 안 바뀌므로 수술을 건너뛴다.
stage() {
    local arm=$1 nobs=$2 npriv=$3 gpu=$4 prio=$5
    local dir="$ROOT/queue/plan/gpu$gpu"
    case "$LIVE" in *" $arm "*) dir="$ROOT/queue/gpu$gpu" ;; esac
    mkdir -p "$dir"
    local out="$dir/$prio-$arm.sh"
    cat > "$out" <<OUTER
#!/usr/bin/env bash
# $arm -- N1_path 위에 레버 하나. warm start 는 N1 의 최종 정책에 관측 수술을 해서 쓴다.
#
# 수술(tools/expand_checkpoint.py)은 첫 층만 넓히고 새 열을 0 으로 채우므로
# **대수적 항등**이다 -- float64 검증에서 actor 최대차 0.0, critic 2.8e-14.
#
# ⛔ 2026-08-07 정정: 여기가 `best.pth` 였는데 그건 **심볼릭 링크**이고
# `select_best_checkpoint.py --link_best` 가 평가할 때마다 다시 만든다. 그리고
# 실측하니 N1_path 의 best.pth 는 **model_100** 을 가리킨다 -- 정확도로 뽑은
# 승자라 붕괴 **전** 지점이다. 즉 주석이 "N1 의 최종 정책에서 출발한다"고 말하는
# 동안 코드는 100 iteration 짜리에서 출발하고 있었고, 이미 돌아버린 arm 들
# (NA_histzero / NZ_zeroiid / NC_actfilter)이 그 계보다.
#
# 두 가지가 문제였다:
#   1. 주석과 코드가 다르다 -> "N1 대 이 arm" 이 한 레버 비교가 아니게 된다.
#   2. 출발점이 **가변**이다 -- 평가를 한 번 더 돌리면 앞으로 나갈 arm 의
#      warm start 가 조용히 바뀐다. 재현 불가능한 실험이 된다.
#
# 그래서 **iteration 번호로 못 박는다.** WARM_IT 를 주면 그 체크포인트를 쓰고,
# 안 주면 마지막 것을 쓴다. 어느 쪽이든 로그에 실제 파일명이 찍힌다.
D=\$(MIN_CKPT=10 bash tools/pick_run.sh N1_path) || exit 0
if [ -n "\${WARM_IT:-}" ]; then
    CK="\$D/nn/model_\${WARM_IT}.pth"
else
    CK=\$(ls -1v "\$D"/nn/model_*.pth 2>/dev/null | tail -1)
fi
if [ ! -e "\$CK" ]; then echo "warm start 체크포인트가 없다: \$CK"; exit 0; fi
echo "warm start 원본: \$CK"
WARM="\$CK"
python -u tools/make_v7_arms.py --only $arm --max_iterations 6000 > /dev/null || exit 1
if [ "$nobs" != "54" ] || [ "$npriv" != "14" ]; then
    WARM=/tmp/${arm}_warm.pth
    python -u tools/expand_checkpoint.py --src "\$CK" --dst "\$WARM" \\
        --old-obs 54 --new-obs $nobs --old-priv 14 --new-priv $npriv --verify || exit 1
fi
python -u train_v7.py --task=K1/Goal_Pose_V7 --config sweeps/$arm.yaml \\
    --headless True --checkpoint "\$WARM" \\
    --num_envs 4096 --max_iterations 6000 \\
    --sim_device cuda:\$GPU_INDEX --rl_device cuda:\$GPU_INDEX
OUTER
    chmod +x "$out"
    printf '  %-38s obs 54->%-4s priv 14->%s\n' "${out#$ROOT/queue/}" "$nobs" "$npriv"
}

echo "arm 배치 (인자로 준 것은 live 큐, 나머지는 plan/):"
# 우선순위 근거: 정보량이 큰 것부터. 이력은 진짜 새 정보이자 보행 RL 배포의
# 사실상 표준이고, 비평자 확장은 배포되는 정책 입력이 한 글자도 안 바뀌어서
# 이득이 나오면 sim2real 위험 없이 그냥 가져올 수 있다.
#
# NA_histzero 는 사용자가 직접 요청한 상호작용 셀(이력 + 영점)이라 030 으로 앞에 둔다.
stage NA_histzero 270 14 1 030
stage N4_hist     270 14 1 800
stage N7_critic    54 29 1 810
stage NZ_zeroiid   54 14 1 830
stage N5_tau       66 14 0 810
stage N6_foot      56 14 0 820
stage N9_zerostruct 54 14 0 830

echo
echo "live gpu0: $(ls -1 "$ROOT/queue/gpu0" 2>/dev/null | tr '\n' ' ')"
echo "live gpu1: $(ls -1 "$ROOT/queue/gpu1" 2>/dev/null | tr '\n' ' ')"
echo "plan gpu0: $(ls -1 "$ROOT/queue/plan/gpu0" | tr '\n' ' ')"
echo "plan gpu1: $(ls -1 "$ROOT/queue/plan/gpu1" | tr '\n' ' ')"
