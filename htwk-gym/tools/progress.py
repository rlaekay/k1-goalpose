"""Print iteration, pace and ETA for every running training run.

Checkpoint mtimes are the honest progress signal: they are written straight to
disk, so unlike console output they are never delayed by Python's stdout
buffering, and unlike nvidia-smi they say WHICH iteration a run is on.

A falling pace is the early-warning sign to watch for: if a policy starts
falling more, episodes reset more often and s/iter climbs. A run whose pace
matches its own first few hundred iterations is training stably.

Usage:
    python tools/progress.py
    python tools/progress.py --task Goal_Pose_V7 --watch 60
"""

import argparse
import glob
import os
import re
import time


def runs_for(task, root="logs"):
    pattern = os.path.join(root, "**", task, "*", "nn")
    return sorted({os.path.dirname(p) for p in glob.glob(pattern, recursive=True)})


def scan(run_dir):
    ckpts = []
    for p in glob.glob(os.path.join(run_dir, "nn", "model_*.pth")):
        m = re.search(r"model_(\d+)\.pth$", p)
        if m:
            ckpts.append((int(m.group(1)), os.path.getmtime(p)))
    if len(ckpts) < 2:
        return None
    ckpts.sort()
    (i0, t0), (i1, t1) = ckpts[0], ckpts[-1]
    # Pace over the whole run so far, and over the most recent stretch, so a
    # slowdown that started recently is visible instead of averaged away.
    span = max(len(ckpts) // 4, 1)
    (ir, tr) = ckpts[-1 - span] if len(ckpts) > span else ckpts[0]
    return {
        "iter": i1,
        "last_write": t1,
        "overall": (t1 - t0) / max(i1 - i0, 1),
        "recent": (t1 - tr) / max(i1 - ir, 1),
    }


def max_iters(run_dir, default=12000):
    cfg = os.path.join(run_dir, "config.yaml")
    if os.path.exists(cfg):
        with open(cfg, "r", encoding="utf-8") as f:
            for line in f:
                m = re.match(r"\s*max_iterations:\s*(\d+)", line)
                if m:
                    return int(m.group(1))
    return default


def hms(seconds):
    seconds = int(max(seconds, 0))
    return "{}h {:02d}m".format(seconds // 3600, (seconds % 3600) // 60)


def phase(run_dir, s, total, stale, shared_dir):
    """Which of the three stages a run is in.

    A finished run keeps its checkpoints, so "no new checkpoint" alone cannot
    tell a completed run apart from a hung one -- iteration count has to decide.
    After training, train_and_eval.sh still runs best-checkpoint search, video
    and the stress eval, which occupy the GPU for another 30-60 min. That gap is
    exactly when it is tempting (and wrong) to launch the next arm.
    """
    name = os.path.basename(run_dir)
    if s["iter"] < total:
        if stale > 20 * s["recent"] + 600:
            return "stalled", "⚠️ {:.0f}분째 저장 없음".format(stale / 60)
        return "training", ""
    if shared_dir and glob.glob(os.path.join(shared_dir, name + "_*")):
        return "done", "✅ 전부 완료 (GPU 비었음)"
    return "evaluating", "⏳ 학습완료, 평가/영상 진행중 (GPU 아직 사용중)"


def report(task, shared_dir=None):
    now = time.time()
    rows = []
    for run_dir in runs_for(task):
        s = scan(run_dir)
        if not s:
            continue
        total = max_iters(run_dir)
        eta = max(total - s["iter"], 0) * s["recent"]
        stale = now - s["last_write"]
        ph, note = phase(run_dir, s, total, stale, shared_dir)
        rows.append((os.path.basename(run_dir), s, total, eta, ph, note))

    if not rows:
        print("run을 찾지 못했습니다 (task={}).".format(task))
        print("확인할 것: htwk-gym 디렉토리에서 실행했는지, logs/ 아래에 해당 task가 있는지.")
        print("  ls -d logs/*/*/*/  |  python tools/progress.py --task <이름>")
        return

    print("{:<34} {:>13} {:>8} {:>8} {:>8} {:>8}".format(
        "run", "iteration", "s/iter", "최근", "남은", "완료예상"))
    print("-" * 84)
    for name, s, total, eta, ph, note in sorted(rows, key=lambda r: (r[4] != "training", r[3])):
        eta_str = "—" if ph != "training" else time.strftime("%H:%M", time.localtime(now + eta))
        print("{:<34} {:>6}/{:<6} {:>8.2f} {:>8.2f} {:>8} {:>8}".format(
            name[:34], s["iter"], total, s["overall"], s["recent"],
            hms(eta) if ph == "training" else "—", eta_str))
        if note:
            print(" " * 36 + note)

    print("")
    n_done = sum(1 for r in rows if r[4] == "done")
    n_eval = sum(1 for r in rows if r[4] == "evaluating")
    if n_done:
        print("✅ {}개 arm이 완전히 끝났습니다 — GPU 자리가 났으면 E3를 띄우세요:".format(n_done))
        print("     git pull && GPU=0 bash tools/run_e3.sh")
        print("   먼저 실제로 빠졌는지 확인: nvidia-smi --query-compute-apps=pid,used_memory --format=csv")
    if n_eval:
        print("⏳ {}개 arm은 학습만 끝나고 평가/영상 중입니다 — GPU를 아직 씁니다. 기다리세요.".format(n_eval))
    print("s/iter = 전체 평균, 최근 = 마지막 1/4 구간. 최근이 전체보다 뚜렷이 크면")
    print("낙상이 늘어 리셋이 잦아진 것이므로 해당 arm을 들여다볼 것.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Goal_Pose_V7")
    ap.add_argument("--watch", type=float, default=0, help="refresh every N seconds")
    ap.add_argument("--shared_dir", default="shared_eval_videos",
                    help="where train_and_eval.sh drops finished results; used to tell "
                         "'training done, still evaluating' from 'GPU actually free'")
    args = ap.parse_args()
    while True:
        if args.watch:
            os.system("clear")
            print(time.strftime("%Y-%m-%d %H:%M:%S"))
        report(args.task, args.shared_dir)
        if not args.watch:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
