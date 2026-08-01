#!/bin/bash
# M-cell short factorial: disturbance / joint DR / mirror loss, 4 cells, 2 GPUs.
#
#   nohup bash tools/run_mcells.sh > mcells.log 2>&1 &
#   tail -f mcells.log
#
# WHY IT IS SHORT
# ---------------
# The H batch trained 4 arms x 12000 iterations (~10 h each) and its selector
# picked model_0 -- the untrained warm start -- for every one. The four
# selection tables show waypoint position median going 7.3 -> ~10.5-13.2 cm by
# iteration 100 and saturating near 38-42 cm, while falls fall from 29 to 0-15.
# The verdict was already legible at iteration 100. These cells stop at 200.
#
#   ~12000 iters/arm  ->  ~10 h/arm, 4 arms serialised over 2 GPUs  ~= 20 h
#   ~200 iters/cell   ->  ~20 min/cell, 4 cells in ONE wave        ~= 25 min
#
# GPU PLAN
# --------
# 2 cells per GPU, all 4 concurrent in a single wave. Two-per-card is already
# proven on this hardware: the G batch ran G1+G2 on cuda:0 and G3+G4 on cuda:1
# at the same 4096 envs. Running all four together also means every cell sees
# the same wall-clock conditions on the shared server, so a slow neighbour
# cannot masquerade as a treatment effect.
#
#   cuda:0 -> M0_control  M2_jointdr
#   cuda:1 -> M1_force    M3_mirror
#
# M0 and M1 are the pair most likely to diverge, so they sit on different cards;
# if one card is throttled, the control and the disturbance cell are not both
# on it.
#
# GATE
# ----
# Every cell is smoke-gated on its own config first. Cells that pass launch;
# cells that fail do NOT block the others and get their own log under
# logs/mcells/smoke_failures/. That is the fix for the G batch, where an
# all-or-nothing launcher held two passing arms hostage to one failing one.
#
# Environment overrides:
#   CKPT=...       G1 warm start (must be the same file for every cell)
#   CELLS="..."    subset, default all four
#   SESSION=m      tmux session name

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

SESSION="${SESSION:-m}"
CELLS="${CELLS:-M0_control M1_force M2_jointdr M3_mirror}"
CKPT="${CKPT:-logs/K1/K1/Goal_Pose_V7/2026-07-28-17-02-35_G1_speed/nn/model_10700.pth}"
ENVS="${ENVS:-4096}"
FAILDIR="logs/mcells/smoke_failures"
mkdir -p "$FAILDIR"

say() { echo "[$(date +%H:%M:%S)] $*"; }

say "=== M-cell factorial: 외란 / joint DR / mirror loss ==="
say "warm start: $CKPT"

if [ ! -f "$CKPT" ]; then
  say "!!! warm start 없음: $CKPT"
  say "    확인: ls logs/K1/K1/Goal_Pose_V7/*G1_speed*/nn/model_*.pth | tail"
  exit 1
fi

# The paired design only holds if every cell starts from the SAME bytes.
CKPT_SHA=$(sha256sum "$CKPT" | cut -d' ' -f1)
say "warm start sha256: ${CKPT_SHA:0:16}"

# ---- 0) configs ------------------------------------------------------------
say "=== config 생성 ==="
if ! python tools/make_mcell_configs.py --checkpoint "$CKPT"; then
  say "!!! config 생성 실패 — 중단"
  exit 1
fi

# ---- 1) 정적 검사 ----------------------------------------------------------
say "=== 이름/import 검사 ==="
if ! python tools/check_names.py > /tmp/mcell_names.txt 2>&1; then
  if grep -v "import \*" /tmp/mcell_names.txt | grep -qE "UNDEFINED|IMPORT|FORMAT"; then
    say "!!! 정적 검사 실패 — 중단"
    grep -E "UNDEFINED|IMPORT|FORMAT" /tmp/mcell_names.txt
    exit 1
  fi
fi
say "정적 검사 통과"

# ---- 2) 셀별 스모크 --------------------------------------------------------
# Round-robin the smoke across both cards too: a serial smoke on one card was
# what left GPU 1 idle for 30 minutes during the re-eval.
say "=== 스모크 (셀별, 실패해도 나머지는 계속) ==="
PASSED=""
i=0
for cell in $CELLS; do
  gpu=$((i % 2))
  log="$FAILDIR/${cell}.log"
  say "--- 스모크: $cell (cuda:$gpu)"
  if python tools/smoke_hbatch.py --config "sweeps/mcells/${cell}.yaml" \
       --checkpoint "$CKPT" --sim_device "cuda:$gpu" --rl_device "cuda:$gpu" \
       > "$log" 2>&1; then
    say "    ✅ $cell 통과"
    rm -f "$log"
    PASSED="$PASSED $cell"
  else
    say "    ❌ $cell 실패 — 로그: $log"
    tail -20 "$log" | sed 's/^/        /'
  fi
  i=$((i + 1))
done

PASSED="${PASSED# }"
if [ -z "$PASSED" ]; then
  say "!!! 통과한 셀이 없다. 아무것도 학습시키지 않는다."
  exit 1
fi
say "통과: $PASSED"

# ---- 3) tmux 실행 ----------------------------------------------------------
if tmux has-session -t "$SESSION" 2>/dev/null; then
  say "!!! tmux 세션 '$SESSION'이 이미 있다: tmux kill-session -t $SESSION"
  exit 1
fi
if [ -z "${CONDA_PREFIX:-}" ]; then
  say "!!! conda 환경 없음. 'conda activate k1goalpose' 후 재실행."
  exit 1
fi

# tmux windows inherit the tmux SERVER's environment, not this shell's, and that
# server predates the conda activation -- without this prelude isaacgym cannot
# find libpython3.8.so.1.0.
CONDA_BASE="$(conda info --base 2>/dev/null || echo "${CONDA_PREFIX%/envs/*}")"
ENV_NAME="${CONDA_DEFAULT_ENV:-$(basename "$CONDA_PREFIX")}"
PRELUDE="source '$CONDA_BASE/etc/profile.d/conda.sh' && conda activate '$ENV_NAME' && export LD_LIBRARY_PATH='$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}' &&"

launch() {  # cell gpu
  local cell=$1 gpu=$2
  # `python -u`: output behind `tee` is block-buffered otherwise, which made a
  # healthy run look stalled for minutes during the v7 batch.
  # TRAIN ONLY. train_and_eval_hbatch.sh runs seven evaluations (clean, force,
  # jitter, combined, lateral, reverse, video) per run; across 4 cells x 5
  # checkpoints that is 140 evaluations, i.e. the diagnostic would cost far more
  # than the training it is diagnosing. compare_mcells.py instead runs ONE
  # paired protocol over the checkpoints that matter, on both cards, and
  # evaluates the shared model_0 exactly once.
  local cmd="$PRELUDE cd $REPO_ROOT && python -u train_hbatch.py \
--task=K1/Goal_Pose_HBatch --config sweeps/mcells/${cell}.yaml --headless True \
--checkpoint $CKPT --num_envs $ENVS --max_iterations 200 \
--sim_device cuda:$gpu --rl_device cuda:$gpu 2>&1 | tee logs/mcells/${cell}.train.log"
  if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-window -t "$SESSION" -n "$cell" "$cmd; exec bash"
  else
    tmux new-session -d -s "$SESSION" -n "$cell" "$cmd; exec bash"
  fi
  say "  launched $cell on cuda:$gpu"
}

say "=== 실행 (2 GPU, 카드당 2셀 동시) ==="
for cell in $PASSED; do
  case "$cell" in
    M0_control)  launch "$cell" 0 ;;
    M2_jointdr)  launch "$cell" 0 ;;
    M1_force)    launch "$cell" 1 ;;
    M3_mirror)   launch "$cell" 1 ;;
    *)           launch "$cell" 0 ;;
  esac
done

# ---- 4) 기동 확인 ----------------------------------------------------------
say "기동 확인 (120s)..."
sleep 120
DEAD=""
for cell in $PASSED; do
  pid=$(tmux list-panes -t "$SESSION:$cell" -F '#{pane_pid}' 2>/dev/null | head -1)
  if [ -n "$pid" ] && pstree -p "$pid" 2>/dev/null | grep -q python; then
    say "  ✅ $cell 살아 있음"
  else
    say "  ❌ $cell 죽었음"
    DEAD="$DEAD $cell"
    tmux capture-pane -p -t "$SESSION:$cell" 2>/dev/null | tail -25 \
      > "logs/mcells/training_failures_${cell}.log"
  fi
done

say ""
say "=== 요약 ==="
say "실행 중: ${PASSED}"
[ -n "$DEAD" ] && say "죽음:$DEAD (로그: logs/mcells/training_failures_*.log)"
say "진행:   tmux attach -t $SESSION"
say "완료 후: python tools/compare_mcells.py"
say ""
say "예상 소요: 200 iter x 4셀, 카드당 2셀 동시 -> 약 25-40분"
