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

# stage <arm> <new_obs> <new_priv> <gpu> <prio>
stage() {
    local arm=$1 nobs=$2 npriv=$3 gpu=$4 prio=$5
    local out="$ROOT/queue/plan/gpu$gpu/$prio-$arm.sh"
    cat > "$out" <<OUTER
#!/usr/bin/env bash
# $arm -- N1_path 위에 레버 하나. warm start 는 N1 의 최종 정책에 관측 수술을 해서 쓴다.
#
# 수술(tools/expand_checkpoint.py)은 첫 층만 넓히고 새 열을 0 으로 채우므로
# **대수적 항등**이다 -- float64 검증에서 actor 최대차 0.0, critic 2.8e-14.
# 그래서 이 arm 은 N1 과 완전히 같은 정책에서 출발하고, 'N1 대 $arm' 이 정확히
# "관측을 더하니 좋아지는가" 가 된다. 처음부터 학습하면 그 비교가 관측이 아니라
# 학습 예산을 재게 된다 -- 이 저장소가 반복해서 낸 오류가 그 부류다.
D=\$(ls -1td logs/K1/K1/Goal_Pose_V7/*_N1_path 2>/dev/null | head -1)
if [ -z "\$D" ]; then echo "N1_path run dir 이 없다 -- 건너뛴다."; exit 0; fi
CK="\$D/nn/best.pth"
[ -e "\$CK" ] || CK=\$(ls -1v "\$D"/nn/model_*.pth 2>/dev/null | tail -1)
if [ -z "\$CK" ]; then echo "N1_path 체크포인트가 없다 -- 건너뛴다."; exit 0; fi
echo "warm start 원본: \$CK"
python -u tools/make_v7_arms.py --only $arm --max_iterations 6000 > /dev/null || exit 1
python -u tools/expand_checkpoint.py --src "\$CK" --dst /tmp/${arm}_warm.pth \\
    --old-obs 54 --new-obs $nobs --old-priv 14 --new-priv $npriv --verify || exit 1
python -u train_v7.py --task=K1/Goal_Pose_V7 --config sweeps/$arm.yaml \\
    --headless True --checkpoint /tmp/${arm}_warm.pth \\
    --num_envs 4096 --max_iterations 6000 \\
    --sim_device cuda:\$GPU_INDEX --rl_device cuda:\$GPU_INDEX
OUTER
    chmod +x "$out"
    echo "  gpu$gpu/$prio-$arm.sh  (obs 54->$nobs, priv 14->$npriv)"
}

echo "관측 arm 을 plan/ 에 넣는다:"
# 우선순위 근거: 정보량이 큰 것부터. 이력은 진짜 새 정보이자 보행 RL 배포의
# 사실상 표준이고, 비평자 확장은 배포되는 정책 입력이 한 글자도 안 바뀌어서
# 이득이 나오면 sim2real 위험 없이 그냥 가져올 수 있다.
stage N4_hist   270 14 1 800
stage N7_critic  54 17 1 810
stage N5_tau     66 14 0 810
stage N6_foot    56 14 0 820

echo
echo "plan/gpu0: $(ls -1 "$ROOT/queue/plan/gpu0" | tr '\n' ' ')"
echo "plan/gpu1: $(ls -1 "$ROOT/queue/plan/gpu1" | tr '\n' ' ')"
