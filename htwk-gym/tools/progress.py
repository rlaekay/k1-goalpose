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


def report(task):
    now = time.time()
    rows = []
    for run_dir in runs_for(task):
        s = scan(run_dir)
        if not s:
            continue
        total = max_iters(run_dir)
        left = total - s["iter"]
        eta = left * s["recent"]
        stale = now - s["last_write"]
        rows.append((os.path.basename(run_dir), s, total, left, eta, stale))

    if not rows:
        print("진행 중인 run을 찾지 못했습니다 (task={})".format(task))
        return

    print("{:<34} {:>13} {:>9} {:>9} {:>9} {:>9}".format(
        "run", "iteration", "s/iter", "최근", "남은", "완료예상"))
    print("-" * 88)
    for name, s, total, left, eta, stale in sorted(rows, key=lambda r: r[4]):
        # Checkpoints land every save_interval iterations; nothing for several
        # intervals means the run died or moved on to its eval phase.
        flag = ""
        if stale > 20 * s["recent"] + 600:
            flag = "  ⚠️ {:.0f}분째 저장 없음".format(stale / 60)
        print("{:<34} {:>6}/{:<6} {:>9.2f} {:>9.2f} {:>9} {:>9}{}".format(
            name[:34], s["iter"], total, s["overall"], s["recent"],
            hms(eta), time.strftime("%H:%M", time.localtime(now + eta)), flag))
    print("")
    print("s/iter = 전체 평균, 최근 = 마지막 1/4 구간. 최근이 전체보다 뚜렷이 크면")
    print("낙상이 늘어 리셋이 잦아진 것이므로 해당 arm을 들여다볼 것.")
    print("학습 종료 후에도 최적 체크포인트 탐색 + 영상 + stress 평가가 30~60분 더 GPU를 씁니다.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Goal_Pose_V7")
    ap.add_argument("--watch", type=float, default=0, help="refresh every N seconds")
    args = ap.parse_args()
    while True:
        if args.watch:
            os.system("clear")
            print(time.strftime("%Y-%m-%d %H:%M:%S"))
        report(args.task)
        if not args.watch:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
