#!/bin/bash
# Pull finished eval results from the training server. RUN THIS ON THE MAC.
#
#   bash tools/fetch_results.sh                 # everything new
#   bash tools/fetch_results.sh E1_path         # just runs matching a pattern
#   SERVER=192.168.0.42 bash tools/fetch_results.sh
#   PORT=6666 SERVER=user@1.2.3.4 bash tools/fetch_results.sh  # non-default ssh port
#   REPORTS_ONLY=1 bash tools/fetch_results.sh  # skip the mp4s (fast)
#
# Videos, reports and checkpoints are gitignored on purpose -- they never travel
# over git push/pull, so this is the only way they reach the Mac.
#
# rsync, not scp: it resumes a half-copied mp4 instead of restarting it, skips
# files already fetched, and can be re-run any number of times while runs finish
# one by one. The videos are 10-13 MB each, so re-copying everything each time
# adds up.

set -euo pipefail

SERVER="${SERVER:-user@user-ESC4000A-E12}"
PORT="${PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-/mnt/DATA/workspace/ws_eungkyu/k1-goalpose/htwk-gym/shared_eval_videos}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_DIR="${LOCAL_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)/K1_walk/v7}"
PATTERN="${1:-}"

mkdir -p "$LOCAL_DIR"

FILTER=(--include='*/' --include='report*.md' --include='report*.json'
        --include='segments.csv' --include='selection.md' --include='BEST_CHECKPOINT')
if [ "${REPORTS_ONLY:-0}" != "1" ]; then
  FILTER+=(--include='*.mp4')
fi
FILTER+=(--exclude='*')

SRC="$SERVER:$REMOTE_DIR/"
[ -n "$PATTERN" ] && SRC="$SERVER:$REMOTE_DIR/*${PATTERN}*/"

echo "=== $SERVER (port $PORT)"
echo "    $REMOTE_DIR"
echo " -> $LOCAL_DIR"
[ -n "$PATTERN" ] && echo "    (필터: *${PATTERN}*)"
echo ""

# shellcheck disable=SC2086
rsync -avP --prune-empty-dirs -e "ssh -p $PORT" "${FILTER[@]}" $SRC "$LOCAL_DIR/"

echo ""
echo "=== 받은 리포트 ==="
find "$LOCAL_DIR" -name 'report.md' -newermt '1 day ago' | sort

echo ""
echo "=== 게이트 요약 ==="
python3 - "$LOCAL_DIR" <<'PY'
import glob, json, os, sys
root = sys.argv[1]
paths = sorted(glob.glob(os.path.join(root, "**", "report.json"), recursive=True))
if not paths:
    print("  (아직 report.json 없음)")
    raise SystemExit
hdr = "  {:<28} {:>9} {:>9} {:>8} {:>7} {:>9} {:>9}"
print(hdr.format("run", "pos_med", "pos_p90", "head", "falls", "성공률", "속도p99"))
print("  " + "-" * 78)
for p in paths:
    try:
        r = json.load(open(p, encoding="utf-8"))
    except Exception:
        continue
    if r.get("mode", "").startswith("stress"):
        continue
    name = os.path.basename(os.path.dirname(p))[:28]
    def g(*ks, default=None):
        n = r
        for k in ks:
            if not isinstance(n, dict) or k not in n:
                return default
            n = n[k]
        return n
    pos_med = g("pos_err_m", "median")
    print(hdr.format(
        name,
        "{:.1f}cm".format(pos_med * 100) if pos_med is not None else "—",
        "{:.1f}cm".format(g("pos_err_m", "p90") * 100) if g("pos_err_m", "p90") is not None else "—",
        "{:.1f}°".format(g("heading_err_deg", "median")) if g("heading_err_deg", "median") is not None else "—",
        str(g("falls", default="—")),
        "{:.0f}%".format(g("success_rate_strict") * 100) if g("success_rate_strict") is not None else "—",
        "{:.2f}".format(g("body_speed", "p99")) if g("body_speed", "p99") is not None else "—"))
PY
