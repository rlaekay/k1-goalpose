"""Pick the best checkpoint of a training run without evaluating all of them.

Training saves a checkpoint every `runner.save_interval` iterations (~200 files
for a 20k-iteration run) and the LAST one is not necessarily the BEST one -- the
reward curve can keep creeping up while the actual task metrics regress. But a
full-protocol eval is ~9 minutes, so scoring every checkpoint would take a day.

This uses successive halving (the Hyperband inner loop): evaluate many candidates
cheaply, keep only the survivors, and spend the long rollouts on the few that are
still in contention.

    stage 1:  12 candidates x 20 s   -> keep 5
    stage 2:   5 candidates x 60 s   -> keep 2
    stage 3:   2 candidates x 120 s  -> winner (full MASTERPLAN protocol)

Three things make it cheap and fair:
  - the sim environment is built ONCE and the policy weights are swapped in place,
    so 12 candidates cost one Isaac Gym startup, not twelve;
  - every candidate is re-seeded identically, so they all face the same sequence
    of goal deltas and resample times (a paired comparison -- differences are the
    policy, not the draw);
  - scoring and metrics come from eval_goal_pose.py itself, so "best" here means
    exactly what the MASTERPLAN gates mean, not a second opinion.

Ranking is lexicographic: gate-passing candidates first, then by worst normalized
gate ratio (how far the weakest gate is from passing), then by the mean ratio.
95% bootstrap CIs are reported so a "win" inside the noise is called out as a tie.

Usage:
    python tools/select_best_checkpoint.py \
        --run_dir logs/K1/K1/Goal_Pose/2026-07-24-17-21-10_armA_continue \
        --sim_device cuda:1 --rl_device cuda:1

    # compare several runs head to head (same env config required)
    python tools/select_best_checkpoint.py --run_dir logs/.../armA --run_dir logs/.../armB
"""

import os
import re
import math
import sys
import glob
import json
import time
import random
import argparse
import subprocess
import copy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import isaacgym  # noqa: F401  (must be imported before torch)

import eval_goal_pose as ev

import torch
import numpy as np
import yaml


def discover_checkpoints(run_dir):
    """[(iteration, path), ...] sorted by iteration."""
    found = []
    for path in glob.glob(os.path.join(run_dir, "nn", "model_*.pth")):
        m = re.search(r"model_(\d+)\.pth$", os.path.basename(path))
        if m:
            found.append((int(m.group(1)), path))
    return sorted(found)


def pick_candidates(items, tail_frac, max_candidates, include=()):
    """Evenly sample the converged tail of training, always keeping the last
    checkpoint (the one you would otherwise have shipped) plus any --include.

    tail_frac only exists to fit the candidate count into the budget, so a run
    that already has fewer checkpoints than the budget keeps all of them."""
    if not items:
        return []
    if len(items) <= max_candidates:
        return sorted(items)
    iters = [i for i, _ in items]
    lo, hi = min(iters), max(iters)
    cutoff = hi - (hi - lo) * tail_frac
    pool = [(i, p) for i, p in items if i >= cutoff]
    if len(pool) > max_candidates:
        idx = sorted(set(np.linspace(0, len(pool) - 1, max_candidates).round().astype(int).tolist()))
        pool = [pool[k] for k in idx]
    chosen = dict(pool)
    chosen[items[-1][0]] = items[-1][1]
    for want in include:
        for i, p in items:
            if i == want:
                chosen[i] = p
    return sorted(chosen.items())


def env_signature(cfg):
    """The parts of a config that change what the policy is even controlling.
    Two runs that disagree here cannot be compared inside one environment."""
    return (
        cfg.get("sim", {}).get("dt"),
        cfg.get("control", {}).get("decimation"),
        cfg.get("env", {}).get("num_observations"),
        cfg.get("env", {}).get("num_actions"),
    )


# Falls are reported but do not decide selection. A 60 s screening stage
# completes ~2300 segments, so ONE fall is a rate of 0.43/1000 whose 95% Poisson
# interval is [0.01, 2.4] per 1000 -- it contains zero. Ranking on it ranks on
# noise, and it decided a real round: I2a_dr's iterations 150, 175 and 200 each
# recorded exactly one fall, failed `max_falls: 0`, and were demoted below
# iteration 25 by the lexicographic key, handing the run to a checkpoint 25 steps
# from its warm start. The same count is non-reproducible besides -- one
# checkpoint returned 3 falls and 18 under an identical protocol.
_SELECTION_IGNORED_GATES = ("falls",)


def task_error_m(results, cfg):
    """What training actually minimises, in metres. Lower is better.

    constellation is d^2 + 2 r^2 (1 - cos th), which for small th is d^2 + r^2 th^2,
    so with constellation_radius r = 1.0 m ONE DEGREE of heading is worth 1.75 cm
    of position. The selector ranked on the worst gate ratio instead, and since
    every candidate's worst gate was pos_median (limit 5 cm, while heading's limit
    is 10 deg), that made it a pure position ranking. It therefore preferred
    2.2 cm / 3.7 deg over 2.6 cm / 1.8 deg -- and by the objective the policy was
    trained on, the second is 2.8x better.

    The gates and the reward disagree about the exchange rate by 3.5x (gates:
    10 deg ~ 5 cm, so 1 deg = 0.5 cm; reward: 1 deg = 1.75 cm). Selection follows
    the reward, because that is the quantity the weights were moved to minimise.
    """
    d = results["pos_err_m"]["median"]
    th = math.radians(results["heading_err_deg"]["median"])
    r = float(((cfg or {}).get("rewards") or {}).get("constellation_radius", 1.0))
    return float(math.hypot(d, r * th))


def score(results):
    """(worst gate ratio, mean gate ratio, per-gate ratios). Lower is better; <=1
    on every gate means passing."""
    ratios = ev.gate_ratios(results)
    values = [v for k, v in ratios.items() if k not in _SELECTION_IGNORED_GATES]
    # HBatch has a separate G1-preservation + robustness gate set.  Conjoining
    # it with the legacy 5 cm / max_falls=0 flag makes documented H candidates
    # structurally unable to pass even when every H ratio is <= 1.
    legacy_gate_required = not bool(results.get("hbatch_gates"))
    # results["all_gates_pass"] includes max_falls, so reading it here would put
    # the retired gate straight back into the decision through the side door.
    legacy_pass = all(v.get("pass") for k, v in (results.get("gates") or {}).items()
                      if k not in _SELECTION_IGNORED_GATES)
    return (float(max(values)), float(np.mean(values)), ratios,
            (not legacy_gate_required or legacy_pass)
            and bool(values)
            and all(np.isfinite(v) and v <= 1.0 for v in values))


def rank_key(entry):
    # task_error first: it is the trained objective, and it weighs heading against
    # position the way the reward does. worst_ratio stays as the tie-break and
    # still enforces "no gate is blown out" through selection_gates_pass.
    return (not entry["selection_gates_pass"],
            entry.get("task_error", entry["worst_ratio"]),
            entry["worst_ratio"], entry["mean_ratio"])


def restore_paired_protocol(env, initial_task_state):
    """Remove cross-candidate state before rollout() calls env.reset()."""
    if initial_task_state is not None and hasattr(env, "load_checkpoint_state"):
        env.load_checkpoint_state(initial_task_state)
    for name in ("actions", "last_actions", "last_dof_vel", "torques"):
        value = getattr(env, name, None)
        if isinstance(value, torch.Tensor):
            value.zero_()
    if hasattr(env, "gait_process"):
        env.gait_process.zero_()
    if hasattr(env, "is_path_env"):
        # Prevent the manual full reset from reporting the previous candidate's
        # final path segment into the restored grid statistics.
        env.is_path_env.zero_()
    for name in ("track_ok_steps", "track_steps"):
        value = getattr(env, name, None)
        if isinstance(value, torch.Tensor):
            value.zero_()
    for name in ("goal_segment_id", "dist_event_serial"):
        value = getattr(env, name, None)
        if isinstance(value, torch.Tensor):
            value.zero_()
    env.common_step_counter = 0
    if hasattr(env, "_path_advanced_at"):
        env._path_advanced_at = -1
    if hasattr(env, "_last_speed_adjust"):
        env._last_speed_adjust = 0


def evaluate_candidate(env, model, checkpoint, duration_s, cfg, config_path, task,
                       num_envs, seed, authoritative, progress_every,
                       initial_task_state=None):
    """One paired evaluation: identical seed, identical env, swapped weights."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    restore_paired_protocol(env, initial_task_state)
    device = cfg["basic"]["rl_device"]
    ev.load_policy(checkpoint, env, device, model=model, verbose=False)
    roll = ev.rollout(env, model, int(duration_s / env.dt), device,
                      progress_every=progress_every, progress_prefix="      ")
    results = ev.summarize(
        roll, cfg, num_envs, duration_s, env.dt, checkpoint, config_path, task,
        exploratory=not authoritative, seed=seed)
    return results, roll


def evaluate_hbatch_robustness(env, model, cfg, initial_task_state,
                               disturbance_cfg, duration_s, seed):
    """Paired short combined force+jitter screen for final H candidates."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    restore_paired_protocol(env, initial_task_state)

    commands = cfg["commands"]
    saved = {
        "resampling_time_s": copy.deepcopy(commands["resampling_time_s"]),
        "goal_dx": copy.deepcopy(commands["goal_dx"]),
        "goal_dy": copy.deepcopy(commands["goal_dy"]),
        "goal_categories": copy.deepcopy(commands.get("goal_categories")),
        "goal_mode_mixture": copy.deepcopy(commands.get("goal_mode_mixture")),
        "disturbance": copy.deepcopy(cfg["randomization"].get("disturbance")),
        "path_share": getattr(env, "path_share", None),
    }
    try:
        commands["resampling_time_s"] = [env.dt, 2.0 * env.dt]
        commands["goal_dx"] = [-3.0, 3.0]
        commands["goal_dy"] = [-3.0, 3.0]
        if isinstance(commands.get("goal_categories"), dict):
            commands["goal_categories"]["enabled"] = False
        if isinstance(commands.get("goal_mode_mixture"), dict):
            commands["goal_mode_mixture"] = {"waypoint": 1.0, "path": 0.0}
        if hasattr(env, "path_share"):
            env.path_share = 0.0
        active_disturbance = copy.deepcopy(disturbance_cfg)
        active_disturbance["enabled"] = True
        active_disturbance["ramp_steps"] = 1
        cfg["randomization"]["disturbance"] = active_disturbance
        roll = ev.rollout(
            env, model, int(duration_s / env.dt), cfg["basic"]["rl_device"],
            progress_every=0, stress="jitter")
    finally:
        commands["resampling_time_s"] = saved["resampling_time_s"]
        commands["goal_dx"] = saved["goal_dx"]
        commands["goal_dy"] = saved["goal_dy"]
        if saved["goal_categories"] is None:
            commands.pop("goal_categories", None)
        else:
            commands["goal_categories"] = saved["goal_categories"]
        if saved["goal_mode_mixture"] is None:
            commands.pop("goal_mode_mixture", None)
        else:
            commands["goal_mode_mixture"] = saved["goal_mode_mixture"]
        cfg["randomization"]["disturbance"] = saved["disturbance"]
        if saved["path_share"] is not None:
            env.path_share = saved["path_share"]

    env_minutes = roll["env_minutes"]
    ang_p90 = ev._hist_pct(
        roll["angvel_hist"], roll["angvel_hist_max"], 90)
    disturbance = roll.get("disturbance_eval") or {}
    return {
        "duration_s": duration_s,
        "falls_per_env_min": roll["falls"] / max(env_minutes, 1.0e-9),
        "body_angvel_p90": ang_p90,
        "force_events": int(disturbance.get("events", 0)),
        "force_survival_5s": (disturbance.get("overall") or {}).get(
            "survival_5s", float("nan")),
    }


def render_selection(sel):
    md = []
    md.append("# 체크포인트 선택 결과 — {}".format(sel["date"]))
    md.append("")
    md.append("- 후보 {}개 (전체 체크포인트 {}개 중 상위 {:.0f}% 구간에서 균등 추출)".format(
        len(sel["candidates"]), sel["total_checkpoints"], sel["tail_frac"] * 100))
    md.append("- 단계: {}".format(" → ".join(
        "{:.0f}s×{}개".format(s["duration_s"], s["n_candidates"]) for s in sel["stages"])))
    md.append("- 조건: {} envs, seed {} 고정(모든 후보가 동일한 목표 시퀀스를 받음), "
              "clean screening".format(sel["num_envs"], sel["seed"]))
    if sel.get("hbatch_robust_s", 0) > 0:
        md.append("- HBatch 후반 후보에는 {:.0f}s paired combined force+jitter screen을 추가해 "
                  "clean-only winner 편향을 막았다.".format(sel["hbatch_robust_s"]))
    md.append("- 총 벽시계 {:.1f}분 (전 후보 full-protocol 평가 대비 약 {:.1f}배 절약)".format(
        sel["wall_s"] / 60.0, sel["speedup_vs_bruteforce"]))
    md.append("")
    md.append("## 최종 선택")
    md.append("")
    best = sel["best"]
    md.append("```")
    md.append(best["checkpoint"])
    md.append("```")
    md.append("")
    md.append("- iteration {} / 위치 median {:.1f} cm / p90 {:.1f} cm / heading {:.1f}° / 낙상 {}회 "
              "/ **과제오차 {:.1f} cm**".format(
                  best["iteration"], best["pos_median"] * 100, best["pos_p90"] * 100,
                  best["heading_median"], best["falls"], best.get("task_error", float("nan")) * 100))
    md.append("- 체크포인트 선택 screen: {} (direction/force-recovery/video/cross-arm은 "
              "후속 full suite에서 별도 판정)".format(
                  "✅ 통과" if best["selection_gates_pass"] else "❌ 미통과 있음"))
    if sel.get("tie_note"):
        md.append("- ⚠️ {}".format(sel["tie_note"]))
    if best["iteration"] != sel["final_iteration"]:
        md.append("- 마지막 체크포인트(iteration {})가 아니라 iteration {}가 선택되었다. "
                  "순위는 **과제오차**(= 학습 목적함수 sqrt(d² + r²θ²), 1° ≈ 1.75 cm)로 매긴다. "
                  "위치 median만 보면 학습이 나빠진 것처럼 보이지만 heading을 함께 보면 "
                  "반대일 수 있으므로, 두 열을 같이 읽을 것.".format(
                      sel["final_iteration"], best["iteration"]))
    md.append("")

    for si, stage in enumerate(sel["stages"]):
        md.append("## 단계 {} — {:.0f}s × {} 후보".format(si + 1, stage["duration_s"], stage["n_candidates"]))
        md.append("")
        rows = []
        for rank, e in enumerate(stage["entries"], 1):
            rows.append([
                rank, e["iteration"], e["segments"],
                "{:.1f}".format(e["pos_median"] * 100),
                "{:.1f}".format(e["pos_p90"] * 100),
                "{:.1f}".format(e["heading_median"]),
                "{:.1f}".format(e.get("task_error", float("nan")) * 100),
                e["falls"],
                "{:.2f}".format(e["worst_ratio"]),
                "✅" if e["selection_gates_pass"] else "—",
                "진출" if e["advanced"] else "",
            ])
        md += ev._table(
            ["#", "iter", "구간수", "위치med(cm)", "위치p90(cm)", "heading(°)", "과제오차(cm)",
             "낙상(순위제외)", "worst비", "게이트", ""],
            rows)
        md.append("")

    md.append("## 읽는 법")
    md.append("")
    md.append("- `worst비` = 가장 나쁜 게이트의 기준 대비 배수. 1.0 이하면 그 게이트는 통과. "
              "예: 위치 median 18.5cm / 기준 5cm = 3.7.")
    md.append("- 짧은 단계는 구간 표본이 적어 순위 노이즈가 크다. 그래서 상위 후보만 긴 단계로 넘긴다.")
    md.append("- 선택된 체크포인트의 상세 진단(실패 모드 분해, 유형별, 실현가능성)은 같은 폴더의 "
              "`report.md`에 있다.")
    return "\n".join(md)


def main():
    started = time.perf_counter()
    p = argparse.ArgumentParser(description="Successive-halving search for the best checkpoint of a run.")
    p.add_argument("--run_dir", action="append", required=True,
                   help="training run dir (repeatable to compare runs head to head)")
    p.add_argument("--task", default="K1/Goal_Pose")
    p.add_argument("--config", help="evaluation yaml (default: envs/<task>.yaml)")
    p.add_argument("--num_envs", type=int, help="override evaluation.num_envs")
    p.add_argument("--stages", default="20,60,120", help="rollout seconds per stage, increasing")
    p.add_argument("--keep", default="5,2", help="survivors after each stage (len == stages-1)")
    p.add_argument("--max_candidates", type=int, default=12)
    p.add_argument("--max_iteration", type=int, default=None,
                   help="이 iteration을 넘는 체크포인트는 후보에서 제외한다. 라운드가 "
                        "200 iter 설계인데 한 arm만 더 오래 돌았을 때, 그 arm의 tail만 "
                        "뽑히면 비교가 '레버'가 아니라 '추가 학습'을 재게 된다.")
    p.add_argument("--fast", action="store_true",
                   help="20 s 한 단계로 끝낸다(--stages 20 --keep '' 와 같다). "
                        "arm 사이를 가릴 때 쓴다 -- 아래 주석의 해상도 한계를 보라.")
    p.add_argument("--tail_frac", type=float, default=0.6,
                   help="후보를 예산에 맞추기 위한 것일 뿐이다. 기본값 0.6이 "
                        "'이른 체크포인트는 최적일 리 없다'는 뜻은 아니다 -- "
                        "L1_long의 봉우리는 1250이었고 2542가 아니었다. "
                        "체크포인트 수가 max_candidates 이하면 애초에 물지 않는다.")
    p.add_argument("--include", default="", help="comma-separated iterations to force into the candidate set")
    p.add_argument("--terrain", choices=["as_trained", "plane"], default=None,
                   help="지형만 덮어쓴다. 기본은 arm의 자기 config -- 그래서 I2b_terrain은 "
                        "자기 trimesh 위에서 채점됐다. 평지에서 비교하려면 plane. "
                        "--config로 다른 yaml을 통째로 주는 것과 다르다: 그쪽은 지형뿐 "
                        "아니라 goal_mode_mixture 같은 과제 자체를 바꿔버린다.")
    p.add_argument("--sim_device")
    p.add_argument("--rl_device")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--progress_every", type=int, default=1000)
    p.add_argument(
        "--hbatch_robust_s", type=float, default=20.0,
        help="paired combined force+jitter screen for final HBatch candidates")
    p.add_argument("--record_video", action="store_true",
                   help="after selecting, re-run eval_goal_pose.py on the winner to record an mp4")
    p.add_argument("--record_video_s", type=float, default=120.0)
    p.add_argument("--link_best", action="store_true", help="also write <run_dir>/nn/best.pth as a symlink")
    p.add_argument("--out", help="output dir (default: <first run_dir>/eval/select_<timestamp>)")
    args = p.parse_args()

    # --fast: 20 s 한 단계. 근거는 추정이 아니라 두 번의 실측이다 --
    # 2026-08-05의 L1peak/L3peak 두 실행에서 20 s 단계의 1위가 120 s 단계의 1위와
    # **둘 다 일치**했다(L1 it 1250, L3 it 500). 60 s와 120 s 단계는 240+240초의
    # 롤아웃을 더 쓰고 승자를 바꾸지 못했다.
    #
    # ⛔ 해상도 한계 -- 이걸 모르고 쓰면 안 된다. 같은 체크포인트의 과제오차가
    # 20 s와 120 s 사이에서 0.2-0.3 cm 움직인다(L1 @1250: 3.8 -> 3.9 -> 3.7).
    # 따라서 20 s가 가릴 수 있는 것은 **약 0.4 cm 이상 벌어진 차이**뿐이다.
    #   * arm 사이 비교는 대개 그보다 크다 (phase-free 5.1 vs 위상기반 3.8) -> --fast 적합
    #   * 한 arm 안에서 체크포인트 고르기는 0.1-0.3 cm 다툼이라 못 가린다.
    #     그런데 그 구간에서는 **어느 것을 골라도 같다**는 뜻이기도 하다.
    #     고원 안에서 아무거나 집는 것과 3단계를 다 도는 것의 결과가 같다.
    if args.fast:
        if args.stages != "20,60,120" or args.keep != "5,2":
            p.error("--fast는 --stages/--keep과 같이 쓸 수 없다 (그 둘을 정하는 것이 --fast다)")
        args.stages, args.keep = "20", ""
    stages = [float(s) for s in args.stages.split(",") if s.strip()]
    keep = [int(k) for k in args.keep.split(",") if k.strip()]
    if not stages:
        p.error("--stages must list at least one rollout duration")
    if len(keep) != len(stages) - 1:
        p.error("--keep must have exactly one fewer entry than --stages "
                "(got {} keep values for {} stages)".format(len(keep), len(stages)))
    if sorted(stages) != stages:
        p.error("--stages must be increasing (cheap screening first)")
    include = [int(i) for i in args.include.split(",") if i.strip()]

    # Default to the RUN'S OWN config, not envs/<task>.yaml.
    #
    # The v7 batch (2026-07-27) was silently evaluated wrong because of this:
    # every arm fell back to envs/K1/Goal_Pose_V7.yaml, which has path mode at
    # 50% and the full perception-noise stack. E0 and E2 were trained with path
    # OFF, so 44-46% of their evaluation segments were a task they had never
    # seen once -- and 90 of E0's 93 falls and 123 of E2's 125 falls were in
    # exactly those segments. Their headline gates measured the harness, not the
    # policy. Comparing across arms stayed fair (all four got the identical
    # task), but comparing any of them to armA-D did not, and neither did
    # reading a single arm's number as "how good is this policy at its job".
    #
    # A single --config still overrides, for the deliberate case of scoring
    # several runs on one common task.
    if args.config:
        config_path = args.config
    elif len(args.run_dir) == 1 and os.path.exists(os.path.join(args.run_dir[0], "config.yaml")):
        config_path = os.path.join(args.run_dir[0], "config.yaml")
        print("config: using the run's own {} (pass --config to override)".format(config_path))
    else:
        config_path = os.path.join("envs", "{}.yaml".format(args.task))
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    # The ground is part of the exam, not part of the policy -- the same rule
    # eval_goal_pose.py:648 already applies. Overriding ONLY terrain keeps the
    # task identical; swapping the whole yaml does not.
    if args.terrain:
        os.environ["EVAL_TERRAIN"] = args.terrain
        if args.terrain == "plane":
            cfg.setdefault("terrain", {})["type"] = "plane"
        print("terrain: {} (config {} 그대로, 지형만 덮어씀)".format(
            args.terrain, os.path.basename(config_path)))
    eval_cfg = cfg.setdefault("evaluation", {})
    hbatch_profile = bool(eval_cfg.get("hbatch_gates"))
    num_envs = args.num_envs or eval_cfg.get("num_envs", 256)
    full_duration = eval_cfg.get("duration_s", 120.0)

    # ---- candidates ------------------------------------------------------
    candidates = []
    total_found = 0
    for run_dir in args.run_dir:
        items = discover_checkpoints(run_dir)
        if not items:
            print("WARNING: no model_*.pth under {}/nn — skipped".format(run_dir))
            continue
        total_found += len(items)
        run_cfg_path = os.path.join(run_dir, "config.yaml")
        if os.path.exists(run_cfg_path):
            with open(run_cfg_path, "r", encoding="utf-8") as f:
                run_cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
            if env_signature(run_cfg) != env_signature(cfg):
                print("WARNING: {} was trained with a different sim dt / decimation / obs-action size "
                      "than {} ({} vs {}). Evaluating it in this env measures the wrong thing — "
                      "give it its own --config.".format(
                          run_dir, config_path, env_signature(run_cfg), env_signature(cfg)))
        if args.max_iteration is not None:
            kept = [(i, p_) for i, p_ in items if i <= args.max_iteration]
            if len(kept) != len(items):
                print("  {}: iteration <= {} 로 {} -> {} 개".format(
                    os.path.basename(run_dir.rstrip("/")), args.max_iteration,
                    len(items), len(kept)))
            items = kept
            if not items:
                raise SystemExit("--max_iteration {} 아래로 남는 체크포인트가 없다".format(
                    args.max_iteration))
        for it, path in pick_candidates(items, args.tail_frac, args.max_candidates, include):
            candidates.append({"iteration": it, "checkpoint": path, "run_dir": run_dir})
    if not candidates:
        raise SystemExit("no checkpoints to evaluate")

    planned = len(candidates) * stages[0] + sum(k * d for k, d in zip(keep, stages[1:]))
    if hbatch_profile and args.hbatch_robust_s > 0.0:
        stage_counts = [len(candidates)] + keep
        planned += args.hbatch_robust_s * sum(stage_counts[-2:])
    bruteforce = len(candidates) * full_duration
    print("candidates: {} (from {} checkpoints across {} run(s))".format(
        len(candidates), total_found, len(args.run_dir)))
    print("plan: {}".format(" -> ".join(
        "{:.0f}s x{}".format(d, n) for d, n in zip(stages, [len(candidates)] + keep))))
    print("simulated rollout budget: {:.0f} s vs {:.0f} s if every candidate got the full protocol "
          "({:.1f}x cheaper)".format(planned, bruteforce, bruteforce / max(planned, 1e-9)))

    # ---- one environment, reused for every candidate ---------------------
    ev.prepare_cfg(cfg, args.task, num_envs, args.sim_device, args.rl_device)
    # prepare_cfg replaces arm-specific training disturbances with HBatch's
    # shared held-out exam profile.  Reuse that exact distribution for the
    # paired combined screen so checkpoint selection and final cross-arm force
    # reports do not test different policies at different difficulty levels.
    common_eval_disturbance_cfg = copy.deepcopy(
        cfg.get("randomization", {}).get("disturbance") or {})
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = ev.build_env(cfg, args.task)
    device = cfg["basic"]["rl_device"]
    model = ev.load_policy(candidates[0]["checkpoint"], env, device)
    initial_task_state = (env.get_checkpoint_state()
                          if hasattr(env, "get_checkpoint_state") else None)

    # ---- successive halving ----------------------------------------------
    alive = candidates
    stage_records = []
    best_results, best_roll, winner, runner_up = None, None, None, None
    for si, duration_s in enumerate(stages):
        authoritative = abs(duration_s - full_duration) < 1e-6
        print("\n=== stage {}/{}: {:.0f}s x {} candidates ===".format(
            si + 1, len(stages), duration_s, len(alive)))
        entries = []
        for ci, cand in enumerate(alive, 1):
            print("  [{}/{}] iteration {} — {}".format(ci, len(alive), cand["iteration"], cand["checkpoint"]))
            results, roll = evaluate_candidate(
                env, model, cand["checkpoint"], duration_s, cfg, config_path, args.task,
                num_envs, args.seed, authoritative, args.progress_every,
                initial_task_state=initial_task_state)
            worst, mean, ratios, selection_gates_pass = score(results)
            robust_screen = None
            if (hbatch_profile and si >= max(0, len(stages) - 2)
                    and args.hbatch_robust_s > 0.0):
                robust_screen = evaluate_hbatch_robustness(
                    env, model, cfg, initial_task_state,
                    common_eval_disturbance_cfg, args.hbatch_robust_s,
                    args.seed + 100003)
                gates = eval_cfg.get("hbatch_gates") or {}

                def robust_upper(value, limit):
                    try:
                        value, limit = float(value), float(limit)
                    except (TypeError, ValueError):
                        return float("inf")
                    return (value / max(limit, 1.0e-9)
                            if np.isfinite(value) and np.isfinite(limit)
                            else float("inf"))

                def robust_lower(value, limit):
                    try:
                        value, limit = float(value), float(limit)
                    except (TypeError, ValueError):
                        return float("inf")
                    return (limit / max(value, 1.0e-9)
                            if np.isfinite(value) and np.isfinite(limit)
                            else float("inf"))

                ratios.update({
                    "h_combined_falls": robust_upper(
                        robust_screen["falls_per_env_min"],
                        gates.get("combined_falls_per_env_min_max", 0.50)),
                    "h_combined_angvel": robust_upper(
                        robust_screen["body_angvel_p90"],
                        gates.get("combined_body_angvel_p90_max", 3.0)),
                    "h_combined_force_events": (
                        0.0 if robust_screen["force_events"] > 0 else float("inf")),
                    "h_combined_force_survival": robust_lower(
                        robust_screen["force_survival_5s"],
                        gates.get("force_survival_5s_min", 0.98)),
                })
                values = list(ratios.values())
                worst, mean = float(max(values)), float(np.mean(values))
                selection_gates_pass = (
                    selection_gates_pass
                    and all(np.isfinite(v) and v <= 1.0 for v in values))
            entry = dict(cand)
            entry.update({
                "segments": results["segments_completed"],
                "pos_median": results["pos_err_m"]["median"],
                "pos_p90": results["pos_err_m"]["p90"],
                "heading_median": results["heading_err_deg"]["median"],
                "falls": results["falls"],
                "success_rate_strict": results["success_rate_strict"],
                "selection_gates_pass": selection_gates_pass,
                "task_error": task_error_m(results, cfg),
                "worst_ratio": worst,
                "mean_ratio": mean,
                "gate_ratios": ratios,
                "hbatch_robust_screen": robust_screen,
                "ci95_pos_median": results["ci95"]["pos_median"],
                "advanced": False,
                "_results": results,
                "_roll": roll,
            })
            entries.append(entry)
            print("    -> 위치 median {:.1f} cm, p90 {:.1f} cm, heading {:.1f}°, 낙상 {}, "
                  "과제오차 {:.1f} cm, worst비 {:.2f}".format(
                      entry["pos_median"] * 100, entry["pos_p90"] * 100, entry["heading_median"],
                      entry["falls"], entry["task_error"] * 100, worst))
            if robust_screen is not None:
                print("       combined screen: falls {:.3f}/env-min, |omega| p90 {:.2f}, "
                      "force events {}, 5s survival {:.1%}".format(
                          robust_screen["falls_per_env_min"],
                          robust_screen["body_angvel_p90"],
                          robust_screen["force_events"],
                          robust_screen["force_survival_5s"]))

        entries.sort(key=rank_key)
        n_keep = keep[si] if si < len(keep) else 1
        for e in entries[:n_keep]:
            e["advanced"] = si < len(stages) - 1
        stage_records.append({
            "duration_s": duration_s,
            "n_candidates": len(entries),
            "authoritative": authoritative,
            "entries": [{k: v for k, v in e.items() if not k.startswith("_")} for e in entries],
        })
        if si == len(stages) - 1:
            best_results = entries[0]["_results"]
            best_roll = entries[0]["_roll"]
            winner = entries[0]
            runner_up = entries[1] if len(entries) > 1 else None
        else:
            alive = [{"iteration": e["iteration"], "checkpoint": e["checkpoint"], "run_dir": e["run_dir"]}
                     for e in entries[:n_keep]]

    # ---- tie detection ----------------------------------------------------
    tie_note = None
    if runner_up is not None:
        lo_w, hi_w = winner["ci95_pos_median"]
        lo_r, hi_r = runner_up["ci95_pos_median"]
        if np.isfinite(lo_w) and np.isfinite(lo_r) and lo_w <= hi_r and lo_r <= hi_w:
            tie_note = ("1위(iter {})와 2위(iter {})의 위치오차 95% CI가 겹친다 — 통계적으로 동률이다. "
                        "낙상 수({} vs {})와 학습 안정성으로 판단할 것.".format(
                            winner["iteration"], runner_up["iteration"], winner["falls"], runner_up["falls"]))

    wall_s = time.perf_counter() - started
    out_dir = args.out or os.path.join(args.run_dir[0], "eval",
                                       "select_" + time.strftime("%Y-%m-%d-%H-%M-%S"))
    os.makedirs(out_dir, exist_ok=True)

    selection = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_dirs": args.run_dir,
        "config": config_path,
        "task": args.task,
        "num_envs": num_envs,
        "seed": args.seed,
        "hbatch_robust_s": args.hbatch_robust_s if hbatch_profile else 0.0,
        "tail_frac": args.tail_frac,
        "total_checkpoints": total_found,
        "candidates": [{"iteration": c["iteration"], "checkpoint": c["checkpoint"]} for c in candidates],
        "stages": stage_records,
        "final_iteration": max(c["iteration"] for c in candidates),
        "best": {k: v for k, v in winner.items() if not k.startswith("_")},
        "tie_note": tie_note,
        "wall_s": wall_s,
        "speedup_vs_bruteforce": bruteforce / max(planned, 1e-9),
    }
    selection_md = render_selection(selection)

    with open(os.path.join(out_dir, "selection.json"), "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2, ensure_ascii=False, default=float)
    with open(os.path.join(out_dir, "selection.md"), "w", encoding="utf-8") as f:
        f.write(selection_md + "\n")
    with open(os.path.join(out_dir, "BEST_CHECKPOINT"), "w", encoding="utf-8") as f:
        f.write(winner["checkpoint"] + "\n")

    # the winner's full diagnostic report, from the final full-protocol rollout
    ev.write_outputs(out_dir, best_results, best_roll, ev.render_report(best_results))

    if args.link_best:
        link = os.path.join(winner["run_dir"], "nn", "best.pth")
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(os.path.abspath(winner["checkpoint"]), link)
        print("symlinked: {} -> {}".format(link, winner["checkpoint"]))

    print("")
    print(selection_md)
    print("")
    print("selection saved to: {}".format(out_dir))

    if args.record_video:
        print("\nrecording video for the winner...")
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        video_dir = os.path.abspath(os.path.join(out_dir, "winner_video"))
        proc = subprocess.run([
            sys.executable, "eval_goal_pose.py",
            "--task", args.task,
            "--config", config_path,
            "--checkpoint", winner["checkpoint"],
            "--sim_device", cfg["basic"]["sim_device"],
            "--rl_device", cfg["basic"]["rl_device"],
            "--out", video_dir,
            "--record_video", "--record_video_s", str(args.record_video_s),
        ], check=False, cwd=repo_root, capture_output=True, text=True,
           encoding="utf-8", errors="replace")
        # This used to be check=False with the output discarded, so when the
        # 2026-07-27 v7 batch produced no mp4 for any of its four runs there was
        # nothing anywhere saying why. A failure here must not kill the
        # selection (the numbers above are already valid and expensive), but it
        # must be impossible to miss.
        mp4 = os.path.join(video_dir, "rollout_env0.mp4")
        if proc.returncode != 0 or not os.path.exists(mp4):
            print("!" * 70)
            print("VIDEO FAILED (exit {}) — selection results above are still valid.".format(proc.returncode))
            for stream, name in ((proc.stderr, "stderr"), (proc.stdout, "stdout")):
                tail = [ln for ln in (stream or "").strip().splitlines() if ln.strip()][-15:]
                if tail:
                    print("--- {} (last {} lines) ---".format(name, len(tail)))
                    print("\n".join(tail))
            print("!" * 70)
        else:
            print("video: {}".format(mp4))


if __name__ == "__main__":
    main()
