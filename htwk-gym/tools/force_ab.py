"""Score two or more arms on ONE held-out force test, and compare.

This is the question the project has asked three times and never answered. E2,
G2 and I1b were each built to buy robustness; every one of them was then scored
with disturbance switched off, so all three produced accuracy numbers and zero
evidence about the thing they were built for. I1b's own report says it outright:
"force events: 0, this report cannot be used as evidence of collision
robustness".

--keep_perturbations does not fix that, because it retains each arm's OWN
disturbance config: the arm that trained without disturbance is scored with
none. Every policy graded on its own homework. eval_goal_pose --force_profile
heldout installs one profile for everybody, harder than any of them trained on,
and this drives it across arms and seeds.

Because fall counts do not reproduce (I1c/model_95 gave 3 and 18 under an
identical protocol), every cell is run over several seeds and reported as a
range. A difference smaller than the spread is not a difference.

Usage:
    python tools/force_ab.py \\
      --arm I1a_base logs/.../I1a_base/nn/model_200.pth sweeps/I1a_base.yaml \\
      --arm I1b_force logs/.../I1b_force/nn/model_175.pth sweeps/I1b_force.yaml \\
      --seeds 0 1 2
"""

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def one(name, ckpt, cfg, task, seed, envs, secs, device, out_root, force):
    out = os.path.join(out_root, name, ("force" if force else "clean") + "_seed%d" % seed)
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "-u", "eval_goal_pose.py",
           "--task", task, "--config", cfg, "--checkpoint", ckpt,
           "--num_envs", str(envs), "--duration_s", str(secs), "--seed", str(seed),
           "--sim_device", device, "--rl_device", device, "--out", out]
    if force:
        cmd += ["--force_profile", "heldout"]
    # capture_output swallows everything until the child exits, and one eval is
    # about 14 minutes -- so the tool looked hung. Stream to a log and print a
    # heartbeat instead: silence for a quarter of an hour is indistinguishable
    # from a crash, which is the same mistake as piping a slow run through tail.
    t0 = time.time()
    log = os.path.join(out, "eval.log")
    print("    %-12s %-5s seed %d  시작 … (~14분, 로그 %s)"
          % (name, "force" if force else "clean", seed, log), flush=True)
    with open(log, "w") as lf:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, text=True)
        beat = 0
        while p.poll() is None:
            time.sleep(30)
            beat += 30
            if beat % 180 == 0:
                print("      … %d분 경과" % (beat // 60), flush=True)
    r = subprocess.CompletedProcess(cmd, p.returncode)
    reps = glob.glob(os.path.join(out, "**", "report.json"), recursive=True)
    if not reps:
        with open(os.path.join(out, "eval_failed.log"), "w") as f:
            tail = ""
            try:
                tail = open(log, encoding="utf-8", errors="replace").read()[-4000:]
            except OSError:
                pass
            f.write("rc=%s\n--- eval.log tail ---\n%s" % (r.returncode, tail))
        print("    %-12s %-5s seed %d  실패" % (name, "force" if force else "clean", seed),
              flush=True)
        return None
    d = json.load(open(reps[0]))
    # The audit lives under disturbance_eval; reading a flat "force_events" would
    # silently give None and the run would look unaudited.
    de = d.get("disturbance_eval") or {}
    d["_events"] = de.get("events")
    d["_falls_during_force"] = de.get("falls_during_force")
    d["_active_share"] = de.get("active_share")
    fe = d["_events"]
    print("    %-12s %-5s seed %d  pos %.4f  falls %-3s  force_events %-5s  (%.0fs)"
          % (name, "force" if force else "clean", seed,
             d["pos_err_m"]["median"], d.get("falls"),
             fe if fe is not None else "?", time.time() - t0), flush=True)
    return d


def rng(vals):
    v = [x for x in vals if x is not None]
    if not v:
        return None, None, None
    return statistics.median(v), min(v), max(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs=3, action="append", metavar=("NAME", "CKPT", "CFG"),
                    required=True, help="repeatable: name, checkpoint, config")
    ap.add_argument("--task", default="K1/Goal_Pose_V7")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--num_envs", type=int, default=256)
    ap.add_argument("--duration_s", type=float, default=120.0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--skip-clean", dest="skip_clean", action="store_true",
                    help="force test only (clean numbers already known)")
    ap.add_argument("--skip-force", dest="skip_force", action="store_true",
                    help="clean 셀만 돈다 — 이미 끝난 force 셀에 clean을 채워 넣을 때")
    ap.add_argument("--out_run",
                    help="새 타임스탬프를 만들지 않고 이 run 디렉터리에 이어 쓴다. "
                         "force 셀이 이미 있는 run에 clean을 합쳐 2x2를 닫을 때 쓴다 "
                         "— 따로 돌리면 다른 폴더로 가서 clean/force 비교가 안 된다.")
    ap.add_argument("--out", default=os.path.join(ROOT, "logs", "force_ab"))
    a = ap.parse_args()

    if a.skip_clean and a.skip_force:
        ap.error("--skip-clean과 --skip-force를 같이 주면 돌 셀이 없다")
    modes = [m for m in ("clean", "force")
             if not (m == "clean" and a.skip_clean)
             and not (m == "force" and a.skip_force)]
    out_root = a.out_run or os.path.join(a.out, time.strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(out_root, exist_ok=True)
    n_cells = len(a.arm) * len(a.seeds) * len(modes)
    print("held-out force: interval 4-8 s, collision 50-120 N, support 4-10 N")
    print("seeds %s, %d envs x %.0f s" % (a.seeds, a.num_envs, a.duration_s))
    print("총 %d회 평가, 회당 약 14분 -> 예상 %.1f시간\n" % (n_cells, n_cells * 14 / 60.0))

    res = {}
    for name, ckpt, cfg in a.arm:
        print("  [%s]" % name, flush=True)
        res[name] = {"clean": [], "force": []}
        for mode in modes:
            for s in a.seeds:
                d = one(name, ckpt, cfg, a.task, s, a.num_envs, a.duration_s,
                        a.device, out_root, mode == "force")
                if d:
                    res[name][mode].append(d)

    print("\n" + "=" * 88)
    print("공통 held-out 외력 시험 결과   (범위보다 작은 차이는 차이가 아니다)")
    print("=" * 88)
    hdr = "%-13s %-6s %-22s %-22s %-14s %s"
    print(hdr % ("arm", "mode", "위치 median cm", "낙상", "낙상/1000", "force events"))
    print("-" * 88)
    for name in res:
        for mode in ("clean", "force"):
            rs = res[name][mode]
            if not rs:
                continue
            pm, plo, phi = rng([r["pos_err_m"]["median"] * 100 for r in rs])
            fm, flo, fhi = rng([r.get("falls") for r in rs])
            rm, _, _ = rng([1000 * (r.get("fall_rate_per_attempt") or 0) for r in rs])
            ev, _, _ = rng([r.get("_events") for r in rs])
            fdf, _, _ = rng([r.get("_falls_during_force") for r in rs])
            print(hdr % (name, mode,
                         "%.2f  [%.2f, %.2f]" % (pm, plo, phi),
                         "%.0f  [%.0f, %.0f]" % (fm, flo, fhi),
                         "%.2f" % (rm or 0),
                         "%s%s" % (int(ev) if ev is not None else "미기록",
                                    ("  (외력 중 낙상 %d)" % int(fdf)) if fdf else "")))
    print()
    for name in res:
        rs = res[name]["force"]
        if rs and all((r.get("_events") or 0) == 0 for r in rs):
            print("  !!! %s: force event 0회 — 이 결과는 강건성 근거가 아니다." % name)
    print("  결과: %s" % out_root)
    if a.out_run:
        print("  (이어쓰기) 이 run의 clean/force 전체 표: "
              "python tools/review_round.py\n")
    else:
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
