#!/bin/bash
# 서버 로그를 이 맥으로 가져온다. **맥에서** 실행할 것 (서버에서 실행하면 conda의
# OpenSSL 충돌로 죽는다). 긴 rsync 한 줄을 붙여넣다 깨지는 일이 반복돼 스크립트로 굳혔다.
#
#   bash fetch.sh              리포트/로그/설정만 (기본, 가볍다)
#   bash fetch.sh --ckpt       체크포인트(.pth)까지 (수십 MB씩)
#   bash fetch.sh --video      영상(mp4)까지. logs/ 밖의 shared_eval_videos/ 도 받는다.
#
# 영상은 원래 이 스크립트로 받을 수 없었다: train_and_eval.sh가 mp4를 logs/ 가 아닌
# shared_eval_videos/ 에 떨어뜨리는데 SRC가 logs/ 뿐이었고 mp4가 --include에도 없었다.
# 급하면 받지 말고 대시보드에서 바로 보면 된다 (모니터가 /video 로 스트리밍한다).
set -euo pipefail
HOST="${HOST:-user@165.246.193.194}"
PORT="${PORT:-6666}"
SRC="${SRC:-/mnt/DATA/workspace/ws_eungkyu/k1-goalpose/htwk-gym/logs/}"
DST="${DST:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/server-dump/logs/}"

INC=(--include='*/' --include='*.json' --include='*.md' --include='*.jsonl'
     --include='*.log' --include='*.yaml' --include='*.csv')
[ "${1:-}" = "--ckpt" ] && INC+=(--include='*.pth')
[ "${1:-}" = "--video" ] && INC+=(--include='*.mp4')

mkdir -p "$DST"
echo "서버 $HOST:$PORT"
echo "  $SRC"
echo "  -> $DST"
rsync -avz --partial -e "ssh -p $PORT" \
      "${INC[@]}" --exclude='*' "$HOST:$SRC" "$DST"

if [ "${1:-}" = "--video" ]; then
  VSRC="${VSRC:-/mnt/DATA/workspace/ws_eungkyu/k1-goalpose/htwk-gym/shared_eval_videos/}"
  VDST="${VDST:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/server-dump/shared_eval_videos/}"
  mkdir -p "$VDST"
  echo
  echo "  $VSRC -> $VDST"
  rsync -avz --partial -e "ssh -p $PORT" "$HOST:$VSRC" "$VDST"
fi
echo
echo "완료. 최근 리포트:"
find "$DST" -name 'report.json' -newermt '-3 hours' 2>/dev/null | tail -5
