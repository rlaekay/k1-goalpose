"""One command that says how a round went, and why if it did not.

Every previous round ended with the same loop: ssh in, hunt for the run dir,
open selection.json, notice evaluation produced nothing, hunt for the log that
says why. This prints all of it at once so a round can be judged from a single
paste.

Reads only what already exists on disk -- watch_eval.jsonl, any report.json,
selection.json, and the tail of any failure log -- so it works whether the round
finished, is mid-flight, or died.

Usage:
    python tools/round_summary.py                       # newest run of every arm
    python tools/round_summary.py --arms I0a_repro I0b_foot
    python tools/round_summary.py --ref 0.0272 --ref-falls 2
"""

import argparse
import glob
import json
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def newest_runs(arms):
    """Newest run dir per arm name (dirs are timestamp-prefixed by Recorder)."""
    out = {}
    for p in glob.glob(os.path.join(ROOT, "logs", "**", "nn"), recursive=True):
        run = os.path.dirname(p)
        name = os.path.basename(run)
        m = re.match(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}_(.+)$", name)
        # Runs from before Recorder tagged dirs with a description have no arm
        # name at all; showing the bare timestamp is honest but noisy, so they
        # are only included when explicitly asked for.
        arm = m.group(1) if m else None
        if arm is None:
            if not arms:
                continue
            arm = name
        if arms and arm not in arms:
            continue
        if arm not in out or name > os.path.basename(out[arm]):
            out[arm] = run
    return out


def last_watch(run):
    p = os.path.join(run, "watch_eval.jsonl")
    rows = []
    if os.path.exists(p):
        for line in open(p, encoding="utf-8", errors="replace"):
            line = line.strip()
            if line.endswith("}"):
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
    return rows


def reports(run):
    out = []
    for p in glob.glob(os.path.join(run, "**", "report.json"), recursive=True):
        try:
            d = json.load(open(p))
        except (ValueError, OSError):
            continue
        out.append((os.path.getmtime(p), d))
    out.sort()
    return [d for _, d in out]


def failures(run, arm):
    """Anything on disk that explains a missing result."""
    notes = []
    for p in glob.glob(os.path.join(run, "watch_eval", "*", "eval_failed.log")):
        notes.append((p, open(p, encoding="utf-8", errors="replace").read()[-600:]))
    wl = os.path.join(ROOT, "logs", "watch_eval", arm + ".log")
    if os.path.exists(wl):
        txt = open(wl, encoding="utf-8", errors="replace").read()
        if "Traceback" in txt or "never appeared" in txt or "need --run" in txt:
            notes.append((wl, txt[-800:]))
    return notes


def cm(v):
    return "-" if v is None else "%.2f" % (v * 100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="*", default=None)
    ap.add_argument("--ref", type=float, default=0.0272, help="reference position median [m]")
    ap.add_argument("--ref-falls", dest="ref_falls", type=int, default=2)
    a = ap.parse_args()

    runs = newest_runs(set(a.arms) if a.arms else None)
    if not runs:
        print("logs/ 아래에 run이 없다. htwk-gym 에서 실행할 것.")
        return 1

    print("=== 라운드 요약  %s ===" % time.strftime("%Y-%m-%d %H:%M:%S"))
    print("기준: 위치 median %.2f cm / 낙상 %d  (E0@6200)\n" % (a.ref * 100, a.ref_falls))
    hdr = "%-32s %-13s %8s %9s %8s %7s %7s %8s"
    print(hdr % ("arm", "시작", "iter", "pos med cm", "p90 cm", "hd °", "falls", "strict %"))
    print("-" * 100)

    trouble = []
    for arm in sorted(runs):
        run = runs[arm]
        started = os.path.basename(run)[:16].replace("_", " ")
        w = last_watch(run)
        reps = reports(run)
        src = None
        clean = []
        if reps:
            # A force-ON report has the same shape as a clean one but a very
            # different fall count, so picking "the newest report" silently mixed
            # test conditions -- I1a_base showed 17 falls, which is exactly its
            # held-out force median, next to clean numbers from other arms.
            clean = [d for d in reps if not d.get("force_profile")]
            d = (clean or reps)[-1]
            src = dict(pos=(d.get("pos_err_m") or {}).get("median"),
                       p90=(d.get("pos_err_m") or {}).get("p90"),
                       hd=(d.get("heading_err_deg") or {}).get("median"),
                       falls=d.get("falls"), strict=d.get("success_rate_strict"),
                       it=os.path.basename(d.get("checkpoint", "")))
        elif w:
            r = w[-1]
            src = dict(pos=r.get("pos_median"), p90=r.get("pos_p90"),
                       hd=r.get("heading_median"), falls=r.get("falls"),
                       strict=r.get("strict"), it=str(r.get("it")))
        n_ck = len(glob.glob(os.path.join(run, "nn", "model_*.pth")))
        if src is None:
            print(hdr % (arm[:32], started, "ck %d" % n_ck, "-", "-", "-", "-", "-") + "  << 결과 없음")
            trouble.append((arm, run))
            continue
        it = re.sub(r"^model_|\.pth$", "", str(src["it"]))
        mode = ""
        if reps and reps[-1].get("force_profile") and not clean:
            mode = "  << 외력 리포트만 있음 (clean과 비교 불가)"
        print(hdr % (arm[:32], started, it[:8], cm(src["pos"]), cm(src["p90"]),
                     "-" if src["hd"] is None else "%.1f" % src["hd"],
                     "-" if src["falls"] is None else str(src["falls"]),
                     "-" if src["strict"] is None else "%.1f" % (src["strict"] * 100))
              + mode)
        stops = [r for r in w if r.get("stop")]
        if stops:
            print("%14s   STOP: %s" % ("", stops[-1]["why"]))

    if trouble:
        print("\n=== 결과가 없는 arm의 원인 ===")
        for arm, run in trouble:
            notes = failures(run, arm)
            if not notes:
                print("\n[%s] 실패 로그 없음. 평가가 아직 시작되지 않았을 수 있다." % arm)
                print("   run: %s" % os.path.relpath(run, ROOT))
                wl = os.path.join("logs", "watch_eval", arm + ".log")
                print("   확인: tail -30 %s" % wl)
            for p, txt in notes:
                print("\n[%s] %s" % (arm, os.path.relpath(p, ROOT)))
                for ln in txt.strip().split("\n")[-12:]:
                    print("   " + ln)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
