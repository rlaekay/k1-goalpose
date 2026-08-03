#!/bin/bash
# Evaluate one checkpoint under all three conditions and print them side by side.
#
#   bash tools/eval_suite.sh <task> <checkpoint> [gpu]
#
# Example:
#   bash tools/eval_suite.sh K1/Goal_Pose_V7 logs/.../nn/model_9000.pth 0
#   bash tools/eval_suite.sh K1/Goal_Pose logs/.../armB.../nn/model_11500.pth 1
#
#   clean      외란 OFF  -- the gate numbers, comparable with armA-D
#   perturbed  외란 ON   -- the gap vs clean IS the robustness measure
#   jitter     목표를 50 Hz로 ±3 m 재추첨 -- 낙상/발산만 측정
#
# All three matter together: a policy can pass clean and fall over the moment a
# behaviour tree starts flip-flopping, and clean alone would never show it.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

if [ $# -lt 2 ]; then
  echo "usage: bash tools/eval_suite.sh <task> <checkpoint> [gpu]" >&2
  exit 1
fi

TASK=$1
CKPT=$2
GPU="${3:-0}"
DUR="${DUR:-120}"
VIDEO_S="${VIDEO_S:-60}"
DEV="cuda:$GPU"

[ -f "$CKPT" ] || { echo "!!! 체크포인트 없음: $CKPT" >&2; exit 1; }

RUN_DIR=$(dirname "$(dirname "$(readlink -f "$CKPT")")")
TS=$(date +%Y%m%d-%H%M%S)
OUT="$REPO_ROOT/shared_eval_videos/suite_$(basename "$RUN_DIR")_$TS"
mkdir -p "$OUT"

echo "=== [1/3] clean (외란 OFF) — 게이트 판정 ==="
python eval_goal_pose.py --task "$TASK" --checkpoint "$CKPT" \
  --sim_device "$DEV" --rl_device "$DEV" --duration_s "$DUR" \
  --record_video --record_video_s "$VIDEO_S" --out "$OUT/clean"

echo ""
echo "=== [2/3] perturbed (외란 ON) — 강건성 ==="
python eval_goal_pose.py --task "$TASK" --checkpoint "$CKPT" \
  --sim_device "$DEV" --rl_device "$DEV" --duration_s "$DUR" \
  --keep_perturbations --record_video --record_video_s "$VIDEO_S" \
  --out "$OUT/perturbed"

echo ""
echo "=== [3/3] jitter stress (목표 50Hz 재추첨) — 낙상/발산 ==="
python eval_goal_pose.py --task "$TASK" --checkpoint "$CKPT" \
  --sim_device "$DEV" --rl_device "$DEV" --duration_s 60 \
  --stress jitter --record_video --record_video_s 30 \
  --out "$OUT/stress_jitter"

echo ""
echo "================= 요약 ================="
python - "$OUT" <<'PY'
import json, os, sys
out = sys.argv[1]

def load(name):
    p = os.path.join(out, name, "report.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None

clean, pert, stress = load("clean"), load("perturbed"), load("stress_jitter")

def row(label, fn):
    vals = []
    for r in (clean, pert):
        try:
            vals.append(fn(r)) if r else vals.append("—")
        except Exception:
            vals.append("—")
    print("  {:<26} {:>12} {:>12}".format(label, *vals))

print("\n  {:<26} {:>12} {:>12}".format("", "clean", "perturbed"))
print("  " + "-" * 52)
row("위치오차 median",  lambda r: "{:.1f} cm".format(r["pos_err_m"]["median"] * 100))
row("위치오차 p90",     lambda r: "{:.1f} cm".format(r["pos_err_m"]["p90"] * 100))
row("heading median",   lambda r: "{:.1f}°".format(r["heading_err_deg"]["median"]))
row("낙상",             lambda r: "{}회".format(r["falls"]))
row("엄격 성공률",      lambda r: "{:.1f}%".format(r["success_rate_strict"] * 100))
row("몸통속도 median",  lambda r: "{:.2f} m/s".format(r["body_speed"]["median"]))
row("몸통속도 p99",     lambda r: "{:.2f} m/s".format(r["body_speed"]["p99"]))
row("1.0 m/s 초과 시간", lambda r: "{:.1f}%".format(r["body_speed"]["share_above_1p0"] * 100))

if clean and pert:
    try:
        d = (pert["pos_err_m"]["median"] - clean["pos_err_m"]["median"]) * 100
        df = pert["falls"] - clean["falls"]
        print("\n  강건성 비용: 위치오차 {:+.1f} cm, 낙상 {:+d}회".format(d, df))
    except Exception:
        pass

if stress:
    print("\n  jitter stress (게이트 아님 — 생존/발산만):")
    print("    낙상률          {:.2f} 회/env·분".format(stress["falls_per_env_minute"]))
    print("    직립 유지       {:.1f}%".format(stress["upright_share"] * 100))
    print("    각속도 p90      {:.2f} rad/s   (정상 보행 1~2)".format(stress["body_angvel"]["p90"]))
PY

echo ""
echo "리포트/영상: $OUT"
