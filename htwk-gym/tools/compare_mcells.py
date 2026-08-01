"""Evaluate and compare the M-cell factorial: 외란 / joint DR / mirror loss.

Runs ONE paired protocol per (cell, checkpoint) instead of the seven-mode H
suite, on both GPUs, and evaluates the shared `model_0` exactly once because it
is byte-identical across cells by construction.

    cells x checkpoints x modes = 4 x 3 x 2  + 2 (shared model_0)  = 26 runs
    (the seven-mode suite over the same grid would be 140)

WHAT IT ANSWERS
---------------
Each cell is M0 plus exactly one lever, so the paired difference
`cell(iter) - M0(iter)` IS that lever's effect, measured against a control that
shares the seed, the warm start and the protocol. The H batch could not do this:
H1 moved mirror loss, mirror augmentation, encoder bias, target offset and
init-q sigma together.

The primary readout is deliberately the DEGRADATION SLOPE, not the endpoint.
Every H arm degraded monotonically from iteration 0 (7.3 -> ~10.5-13.2 cm by
100), so "which cell is best at 200" is a weaker question than "which lever
makes the slope worse, and by how much". A lever that is merely neutral on the
slope is cheap and can be kept; one that steepens it is what broke the H batch.

    python tools/compare_mcells.py                 # evaluate + report
    python tools/compare_mcells.py --report_only   # re-render from cached json
"""

import argparse
import json
import os
import subprocess
import sys
import time


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

CELLS = ["M0_control", "M1_force", "M2_jointdr", "M3_mirror"]
ITERS = [0, 50, 100, 200]
# clean isolates the task cost of each lever; combined (goal jitter + external
# force) isolates whether the lever bought any robustness for that cost. E2 and
# G2 both looked "safe" on clean because they had stopped moving, so neither
# number means anything without the other.
MODES = {
    "clean": [],
    "combined": ["--stress", "jitter", "--keep_perturbations"],
}
DURATION_S = 40
NUM_ENVS = 256
OUT = os.path.join(ROOT, "logs", "mcells", "compare")

# Every eval must be the same exam for every cell, or a treatment effect and a
# harder test are indistinguishable. The cells' own training configs differ, so
# the evaluation deliberately uses ONE config for all of them.
EVAL_CONFIG = os.path.join("sweeps", "mcells", "M0_control.yaml")


def ckpt_path(cell, it):
    run = latest_run(cell)
    return None if run is None else os.path.join(run, "nn", "model_{}.pth".format(it))


def latest_run(cell):
    base = os.path.join(ROOT, "logs", "K1", "K1", "Goal_Pose_HBatch")
    if not os.path.isdir(base):
        return None
    hits = sorted(d for d in os.listdir(base) if cell in d)
    return os.path.join(base, hits[-1]) if hits else None


def run_eval(cell, it, mode, gpu):
    tag = "{}_{}_{}".format(cell, it, mode)
    outdir = os.path.join(OUT, tag)
    report = os.path.join(outdir, "report.json")
    if os.path.exists(report):
        return report
    ck = ckpt_path(cell, it)
    if ck is None or not os.path.exists(ck):
        print("  MISSING {} -> {}".format(tag, ck))
        return None
    os.makedirs(outdir, exist_ok=True)
    cmd = ["python", "-u", "eval_goal_pose.py",
           "--task", "K1/Goal_Pose_HBatch",
           "--config", EVAL_CONFIG,
           "--checkpoint", ck,
           "--num_envs", str(NUM_ENVS),
           "--duration_s", str(DURATION_S),
           "--seed", "0",
           "--sim_device", "cuda:{}".format(gpu),
           "--rl_device", "cuda:{}".format(gpu),
           "--exploratory",   # these are diagnostics, not gate evaluations
           "--out", outdir] + MODES[mode]
    log = os.path.join(outdir, "eval.log")
    with open(log, "w") as fh:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0 or not os.path.exists(report):
        # Silent eval failure is exactly how four v7 videos went missing; say so.
        print("  FAILED  {} (exit {}) -> {}".format(tag, proc.returncode, log))
        return None
    return report


def jobs():
    """(cell, iter, mode) grid, with model_0 evaluated once and reused."""
    out = []
    for mode in MODES:
        out.append(("M0_control", 0, mode))       # shared warm start
        for cell in CELLS:
            for it in ITERS:
                if it == 0:
                    continue
                out.append((cell, it, mode))
    return out


def load(cell, it, mode):
    # model_0 is byte-identical across cells (same warm start, load_optimizer_
    # state False, no training) so all cells read the one measurement.
    src = "M0_control" if it == 0 else cell
    p = os.path.join(OUT, "{}_{}_{}".format(src, it, mode), "report.json")
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def g(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def fmt(v, n=2):
    return "-" if v is None else "{:.{}f}".format(v, n)


def report():
    lines = []
    W = lines.append
    W("# M-cell 결과 — 외란 / joint DR / mirror loss\n")
    W("각 셀은 **M0_control + 레버 1개**다. 따라서 `셀 − M0`가 그 레버의 효과다.\n")

    for mode in MODES:
        W("\n## {} 프로토콜 ({} s x {} env, seed 0, 공통 config)\n".format(
            mode, DURATION_S, NUM_ENVS))
        W("| 지표 | iter | " + " | ".join(CELLS) + " |")
        W("|---|---:|" + "---:|" * len(CELLS))
        metrics = [
            ("위치 median (cm)", ("pos_err_m", "median"), 100.0, 2),
            ("위치 p90 (cm)", ("pos_err_m", "p90"), 100.0, 2),
            ("heading median (°)", ("heading_err_deg", "median"), 1.0, 2),
            ("strict success (%)", ("success_rate_strict",), 100.0, 2),
            ("낙상", ("falls",), 1.0, 0),
        ]
        for label, path, scale, nd in metrics:
            for it in ITERS:
                row = []
                for cell in CELLS:
                    d = load(cell, it, mode)
                    v = g(d, *path) if d else None
                    row.append(fmt(v * scale, nd) if v is not None else "-")
                W("| {} | {} | {} |".format(label if it == ITERS[0] else "", it,
                                            " | ".join(row)))

    # ---- the actual verdict --------------------------------------------
    W("\n## 레버별 판정 (clean, M0 대비 차이)\n")
    W("| 레버 | Δ위치 med @100 | Δ위치 med @200 | Δ낙상 @200 | 판정 |")
    W("|---|---:|---:|---:|---|")
    base = {it: load("M0_control", it, "clean") for it in ITERS}
    LEVER = {"M1_force": "외란(시나리오)", "M2_jointdr": "joint DR",
             "M3_mirror": "mirror loss"}
    for cell in CELLS[1:]:
        d100, d200 = load(cell, 100, "clean"), load(cell, 200, "clean")
        b100, b200 = base.get(100), base.get(200)
        def delta(a, b, path, scale=100.0):
            va, vb = g(a, *path) if a else None, g(b, *path) if b else None
            return None if va is None or vb is None else (va - vb) * scale
        dp100 = delta(d100, b100, ("pos_err_m", "median"))
        dp200 = delta(d200, b200, ("pos_err_m", "median"))
        df200 = delta(d200, b200, ("falls",), 1.0)
        if dp200 is None:
            verdict = "데이터 없음"
        elif dp200 <= 0.5:
            verdict = "✅ 비용 없음 — 채택 가능"
        elif dp200 <= 2.0:
            verdict = "⚠️ 경미한 비용 — dose 조정 후 재검"
        else:
            verdict = "❌ 이 레버가 열화를 만든다"
        W("| {} | {} | {} | {} | {} |".format(
            LEVER[cell], fmt(dp100), fmt(dp200), fmt(df200, 0), verdict))

    W("\n> 판정 기준: H 배치에서 iteration 200까지의 열화가 H0 +5.1 cm, H1 +7.4 cm,")
    W("> H2 +6.2 cm였다. 한 레버가 그 열화의 대부분을 설명하면 그 레버가 범인이고,")
    W("> M0 자체가 크게 열화하면 범인은 레버가 아니라 **fine-tune 설정 자체**다.")
    W("> M0의 절대 열화가 먼저 읽어야 할 숫자다.\n")

    m0_200 = g(base.get(200), "pos_err_m", "median")
    m0_0 = g(base.get(0), "pos_err_m", "median")
    if m0_0 is not None and m0_200 is not None:
        W("**M0 자체 열화: {:.2f} → {:.2f} cm ({:+.2f} cm)**".format(
            m0_0 * 100, m0_200 * 100, (m0_200 - m0_0) * 100))
        if (m0_200 - m0_0) * 100 > 3.0:
            W("\n> ⛔ **레버를 하나도 켜지 않은 대조군이 이미 열화한다.** 원인은 외란·"
              "joint DR·mirror가 아니라 fine-tune 설정(LR, KL, warm start 정합성)이다. "
              "레버별 차이를 해석하기 전에 이것부터 고쳐야 한다.")
        else:
            W("\n> ✅ 대조군은 안정적이다. 따라서 위 레버별 차이를 그대로 해석할 수 있다.")

    path = os.path.join(OUT, "mcell-report.md")
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("\n=> {}".format(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report_only", action="store_true")
    args = ap.parse_args()
    if not args.report_only:
        os.makedirs(OUT, exist_ok=True)
        todo = jobs()
        print("평가 {}건 (2 GPU 라운드로빈)".format(len(todo)))
        t0 = time.time()
        # Round-robin across both cards; a serial sweep is what left GPU 1 idle
        # for 30 minutes during the v7 re-eval.
        procs = []
        for i, (cell, it, mode) in enumerate(todo):
            gpu = i % 2
            print("[{}/{}] {} iter{} {} -> cuda:{}".format(
                i + 1, len(todo), cell, it, mode, gpu))
            run_eval(cell, it, mode, gpu)
        print("총 {:.1f}분".format((time.time() - t0) / 60.0))
    report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
