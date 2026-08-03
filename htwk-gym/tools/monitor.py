"""Headless training monitor: web dashboard + terminal fallback.

The server has no display and nobody watches its console, so until now the only
way to know whether a run was training, finished, evaluating or dead was to ssh
in and read checkpoint mtimes by hand -- and once the terminal scrolled away,
the reward history was gone with it.

This collects everything into flat JSON under `monitor/data/`, which a
single-file HTML page renders. Two properties matter more than features:

  * stdlib only. No flask, no tensorboard, no CDN. The dashboard has to come up
    on a box with no internet and whatever python the conda env happens to have.
  * append-only, poll-based. The collector never holds a lock and never talks to
    a training process, so it cannot stall or crash one. If the collector dies,
    training does not notice.

Scalars come from `<run>/scalars.jsonl`, which utils/recorder.py now writes
alongside its SummaryWriter. Runs from before that existed have no jsonl; they
still show progress/phase/eval, just no reward curve.

Usage:
    python tools/monitor.py --serve                 # collect + serve on :8420
    python tools/monitor.py --serve --port 9000 --interval 20
    python tools/monitor.py --tui                   # terminal, no browser
    python tools/monitor.py --once                  # one collection pass
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

# Defaults to the repo the script lives in, which is what running it from
# htwk-gym gives you. --root exists so the collector can be pointed at a copy of
# logs/ (e.g. rsynced to a laptop) and so this file is testable without a GPU.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = DATA = ""


def set_root(path):
    global ROOT, OUT, DATA
    ROOT = os.path.abspath(path)
    OUT = os.path.join(ROOT, "monitor")
    DATA = os.path.join(OUT, "data")


set_root(ROOT)
STALE_S = 900.0          # no new checkpoint for this long while training -> stalled


# ---------------------------------------------------------------- collection

def run_dirs(root="logs"):
    """Every directory that has an nn/ subdir, i.e. every training run."""
    base = os.path.join(ROOT, root)
    return sorted({os.path.dirname(p)
                   for p in glob.glob(os.path.join(base, "**", "nn"), recursive=True)})


def checkpoints(run):
    out = []
    for p in glob.glob(os.path.join(run, "nn", "model_*.pth")):
        m = re.search(r"model_(\d+)\.pth$", p)
        if m:
            out.append((int(m.group(1)), os.path.getmtime(p), os.path.getsize(p)))
    return sorted(out)


def read_cfg(run):
    """max_iterations and description without a yaml dependency.

    The config is dumped by yaml.dump so the two keys we need sit on their own
    line as `key: value`. Parsing them with a regex avoids making the monitor
    depend on PyYAML being importable, which is exactly the kind of dependency
    that stops a dashboard from starting on a fresh box.
    """
    p = os.path.join(run, "config.yaml")
    cfg = {"max_iterations": None, "description": "", "task": "", "checkpoint": ""}
    try:
        txt = open(p, encoding="utf-8", errors="replace").read()
    except OSError:
        return cfg
    for key, cast in (("max_iterations", int), ("description", str),
                      ("task", str), ("checkpoint", str)):
        m = re.search(r"^\s*%s:\s*(.+?)\s*$" % key, txt, re.M)
        if m:
            v = m.group(1).strip().strip("'\"")
            try:
                cfg[key] = cast(v)
            except ValueError:
                pass
    return cfg


def scalars(run, keep=400):
    """Downsampled scalar series from scalars.jsonl.

    Kept to `keep` points per tag so a 12k-iteration run stays a small JSON --
    the browser has to re-fetch this every poll, and the reward curve's shape is
    what matters, not its every sample.
    """
    p = os.path.join(run, "scalars.jsonl")
    if not os.path.exists(p):
        return {}
    series = {}
    try:
        with open(p, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or not line.endswith("}"):
                    continue          # a torn trailing write; skip, do not fail
                try:
                    d = json.loads(line)
                    series.setdefault(d["tag"], []).append((d["it"], d["v"]))
                except (ValueError, KeyError):
                    continue
    except OSError:
        return {}
    out = {}
    for tag, pts in series.items():
        pts.sort()
        if len(pts) > keep:
            step = len(pts) / float(keep)
            pts = [pts[min(len(pts) - 1, int(i * step))] for i in range(keep)]
        out[tag] = pts
    return out


def eval_results(run, shared="shared_eval_videos"):
    """Any evaluation output that names this run, with its gate verdict."""
    name = os.path.basename(run)
    found = []
    for base in (os.path.join(ROOT, shared), ROOT):
        for p in glob.glob(os.path.join(base, "**", "report.json"), recursive=True):
            try:
                d = json.load(open(p))
            except (ValueError, OSError):
                continue
            if name not in (d.get("checkpoint", "") + d.get("config", "")):
                continue
            found.append({
                "path": os.path.relpath(p, ROOT),
                "date": d.get("date", ""),
                "checkpoint": os.path.basename(d.get("checkpoint", "")),
                "pos_median": (d.get("pos_err_m") or {}).get("median"),
                "pos_p90": (d.get("pos_err_m") or {}).get("p90"),
                "heading_median": (d.get("heading_err_deg") or {}).get("median"),
                "falls": d.get("falls"),
                "strict": d.get("success_rate_strict"),
                "gates_pass": d.get("all_gates_pass"),
            })
    found.sort(key=lambda r: r["date"])
    return found


def phase(run, cks, cfg, evals):
    """training / stalled / done / evaluating / evaluated / empty.

    Checkpoint mtimes are the honest signal (see tools/progress.py): console
    output is buffered and nvidia-smi cannot say which iteration a run is on.
    """
    if not cks:
        return "empty"
    last_it, last_t = cks[-1][0], cks[-1][1]
    age = time.time() - last_t
    total = cfg.get("max_iterations")
    finished = bool(total) and last_it >= int(total)
    if evals:
        return "evaluated" if finished or age > STALE_S else "evaluating"
    if finished:
        return "done"
    return "stalled" if age > STALE_S else "training"


def pace(cks, window=8):
    """Seconds per iteration over the last few checkpoints, and ETA."""
    if len(cks) < 2:
        return None, None
    sel = cks[-window:] if len(cks) > window else cks
    d_it = sel[-1][0] - sel[0][0]
    d_t = sel[-1][1] - sel[0][1]
    if d_it <= 0 or d_t <= 0:
        return None, None
    return d_t / d_it, None


def batch_of(desc, run):
    """Group runs into batches so the page can file them by version.

    Descriptions look like E0_armB_armsdown / G1_speed / H2_... / I0a_repro, so
    the leading letter+digit is the batch. Anything that does not match lands in
    'other' rather than being dropped -- a run you cannot see is worse than a
    run in the wrong bucket.
    """
    m = re.match(r"([A-Za-z])(\d)", desc or "")
    if m:
        return m.group(1).upper()
    m = re.search(r"_([A-Za-z])(\d)[A-Za-z_]", os.path.basename(run))
    return m.group(1).upper() if m else "other"


def collect():
    runs = []
    for run in run_dirs():
        cks = checkpoints(run)
        cfg = read_cfg(run)
        ev = eval_results(run)
        spi, _ = pace(cks)
        last_it = cks[-1][0] if cks else 0
        total = cfg.get("max_iterations") or 0
        eta = (total - last_it) * spi if (spi and total > last_it) else None
        desc = cfg.get("description") or os.path.basename(run)
        runs.append({
            "run": os.path.relpath(run, ROOT),
            "name": os.path.basename(run),
            "desc": desc,
            "batch": batch_of(desc, run),
            "task": cfg.get("task", ""),
            "warm_start": os.path.basename(cfg.get("checkpoint") or ""),
            "phase": phase(run, cks, cfg, ev),
            "iter": last_it,
            "max_iter": total,
            "n_ckpt": len(cks),
            "s_per_iter": spi,
            "eta_s": eta,
            "last_ckpt_age_s": (time.time() - cks[-1][1]) if cks else None,
            "started": cks[0][1] if cks else None,
            "checkpoints": [c[0] for c in cks],
            "best": _best(run),
            "evals": ev,
            "scalars": scalars(run),
        })
    runs.sort(key=lambda r: (r["batch"], r["desc"]))
    return runs


def _best(run):
    p = os.path.join(run, "BEST_CHECKPOINT")
    if os.path.exists(p):
        try:
            return open(p).read().strip().split("\n")[0]
        except OSError:
            pass
    return ""


def write(runs):
    os.makedirs(DATA, exist_ok=True)
    index = []
    for r in runs:
        slug = re.sub(r"[^A-Za-z0-9_.-]", "_", r["name"])
        with open(os.path.join(DATA, slug + ".json"), "w") as f:
            json.dump(r, f)
        light = {k: v for k, v in r.items() if k not in ("scalars", "evals", "checkpoints")}
        light["slug"] = slug
        light["n_evals"] = len(r["evals"])
        index.append(light)
    with open(os.path.join(DATA, "index.json"), "w") as f:
        json.dump({"generated": time.time(), "runs": index}, f)
    page = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitor_page.html")
    if os.path.exists(page):
        shutil.copyfile(page, os.path.join(OUT, "index.html"))
    return len(index)


# ------------------------------------------------------------------ terminal

def hms(s):
    if s is None:
        return "-"
    s = int(s)
    return "%d:%02d:%02d" % (s // 3600, (s % 3600) // 60, s % 60)


def tui(runs):
    icon = {"training": "\033[32m▶\033[0m", "stalled": "\033[31m■\033[0m",
            "done": "\033[36m✓\033[0m", "evaluating": "\033[33m◐\033[0m",
            "evaluated": "\033[35m★\033[0m", "empty": "\033[90m·\033[0m"}
    print("\033[2J\033[H", end="")
    print("K1 학습 모니터  %s\n" % time.strftime("%Y-%m-%d %H:%M:%S"))
    if not runs:
        print("  logs/ 아래에 run이 없다. htwk-gym 디렉터리에서 실행했는지 확인할 것.")
        return
    cur = None
    hdr = "  %-1s %-26s %9s %8s %8s %9s  %s"
    for r in runs:
        if r["batch"] != cur:
            cur = r["batch"]
            print("\033[1m[%s 배치]\033[0m" % cur)
            print(hdr % ("", "run", "iter", "s/iter", "ETA", "상태", "최근 평가"))
        ev = r["evals"][-1] if r["evals"] else None
        note = ""
        if ev:
            note = "%s  pos %s cm  falls %s" % (
                ev["checkpoint"],
                ("%.1f" % (ev["pos_median"] * 100)) if ev["pos_median"] is not None else "-",
                ev["falls"])
        print(hdr % (icon.get(r["phase"], "?"), r["desc"][:26],
                     "%d/%s" % (r["iter"], r["max_iter"] or "?"),
                     ("%.2f" % r["s_per_iter"]) if r["s_per_iter"] else "-",
                     hms(r["eta_s"]), r["phase"], note))
    print("\n  ▶ 학습중  ■ 멈춤(%d분+)  ✓ 학습완료  ◐ 평가중  ★ 평가완료" % int(STALE_S / 60))


# -------------------------------------------------------------------- server

def serve(port, interval):
    import threading
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    os.makedirs(DATA, exist_ok=True)

    class H(SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=OUT, **k)

        def end_headers(self):
            # The page polls the same filenames forever; without this a browser
            # will happily serve a five-minute-old index.json from cache and the
            # dashboard silently stops updating.
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def log_message(self, *a):
            pass

    def loop():
        while True:
            try:
                n = write(collect())
                print("[%s] %d runs" % (time.strftime("%H:%M:%S"), n), flush=True)
            except Exception as e:                       # never let the poller die
                print("collect failed: %r" % (e,), flush=True)
            time.sleep(interval)

    threading.Thread(target=loop, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", port), H)
    print("모니터: http://<서버IP>:%d/   (LAN에서 접속. Ctrl-C로 종료)" % port, flush=True)
    print("SSH 터널: ssh -L %d:localhost:%d <user>@<host> -p <port>  ->  http://localhost:%d/"
          % (port, port, port), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true", help="collect + serve the web dashboard")
    ap.add_argument("--tui", action="store_true", help="terminal view, no browser needed")
    ap.add_argument("--once", action="store_true", help="one collection pass, then exit")
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--root", default=None, help="repo root holding logs/ (default: this repo)")
    a = ap.parse_args()
    if a.root:
        set_root(a.root)
    if a.serve:
        serve(a.port, a.interval)
    elif a.tui:
        try:
            while True:
                tui(collect())
                time.sleep(a.interval)
        except KeyboardInterrupt:
            pass
    else:
        runs = collect()
        n = write(runs)
        tui(runs)
        print("\n  monitor/data 에 %d개 기록. 웹으로 보려면: python tools/monitor.py --serve" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
