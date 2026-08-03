#!/bin/bash
# 서버 로그를 이 맥으로 가져온다. **맥에서** 실행할 것 (서버에서 실행하면 conda의
# OpenSSL 충돌로 죽는다). 긴 rsync 한 줄을 붙여넣다 깨지는 일이 반복돼 스크립트로 굳혔다.
#
#   bash fetch.sh              리포트/로그/설정만 (기본, 가볍다)
#   bash fetch.sh --ckpt       체크포인트(.pth)까지 (수십 MB씩)
set -euo pipefail
HOST="${HOST:-user@165.246.193.194}"
PORT="${PORT:-6666}"
SRC="${SRC:-/mnt/DATA/workspace/ws_eungkyu/k1-goalpose/htwk-gym/logs/}"
DST="${DST:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/server-dump/logs/}"

INC=(--include='*/' --include='*.json' --include='*.md' --include='*.jsonl'
     --include='*.log' --include='*.yaml' --include='*.csv')
[ "${1:-}" = "--ckpt" ] && INC+=(--include='*.pth')

mkdir -p "$DST"
echo "서버 $HOST:$PORT"
echo "  $SRC"
echo "  -> $DST"
rsync -avz --partial --info=progress2 -e "ssh -p $PORT" \
      "${INC[@]}" --exclude='*' "$HOST:$SRC" "$DST"
echo
echo "완료. 최근 리포트:"
find "$DST" -name 'report.json' -newermt '-3 hours' 2>/dev/null | tail -5
