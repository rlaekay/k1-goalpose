#!/usr/bin/env bash
# N2_pathgrid 와 NA_histzero 의 채점을 미리 큐에 걸어 둔다 (사용자 요청:
# "N2랑 NA 끝나면 두 축 다 보고해").
#
# 둘은 끝나는 시각이 3시간쯤 차이 난다(N2 ~11:00, NA ~14:00). 그래서 채점도 각자
# 자기 학습 바로 뒤에 걸되, **같은 출력 디렉터리**(EVAL_OUT)에 쌓게 한다 --
# round_table.py 는 디렉터리 하나를 읽으므로 그래야 마지막에 한 표로 비교된다.
#
# 각 채점은 pick_run.sh 로 run 을 고른다. 시간순 최신을 쓰면 스모크 디렉터리를
# 집는다(2026-08-07 실제 사고, ibatch §8-47).

cd "$(dirname "$0")/.." || exit 1
ROOT="$PWD"
OUT="logs/eval_rounds/n2na"

mk() {   # mk <arm> <gpu> <prio>
    local arm=$1 gpu=$2 prio=$3
    local f="$ROOT/queue/gpu$gpu/$prio-eval_$arm.sh"
    cat > "$f" <<OUTER
#!/usr/bin/env bash
# $arm 채점. 정확도(공통 waypoint) + 지속 보행(forward_hold) 두 축.
D=\$(MIN_CKPT=10 bash tools/pick_run.sh $arm) || exit 0
echo "채점 대상: \$D"
EVAL_OUT="$ROOT/$OUT" bash tools/eval_round.sh "\$D"
OUTER
    chmod +x "$f"
    echo "  queue/gpu$gpu/$prio-eval_$arm.sh"
}

# N2 는 gpu0 에서 곧 끝난다. 025 로 두어 030-N8_pathdelay 보다 먼저 돌게 한다 --
# 3.7시간짜리 학습 뒤로 미루면 결과를 보는 시점이 그만큼 늦어진다.
mk N2_pathgrid 0 025
# NA 는 gpu1 에서 ~14:00 에 끝난다. 그 뒤에 바로.
mk NA_histzero 1 040

echo
echo "live gpu0: $(ls -1 "$ROOT/queue/gpu0" | tr '\n' ' ')"
echo "live gpu1: $(ls -1 "$ROOT/queue/gpu1" | tr '\n' ' ')"
echo "두 채점 모두 -> $OUT (한 표로 읽으려면 그 디렉터리를 round_table.py 에 준다)"
