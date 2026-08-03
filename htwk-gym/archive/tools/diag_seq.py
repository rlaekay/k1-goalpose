"""Why does is_seq_env collapse from ~50% to almost nothing over a smoke run?

G4_smoothturn's smoke measured share=0.50 configured but 1/256 envs sequential
by the end of a 300-step run, with only 1 goal ever banked. is_seq_env is only
ever written in GoalPoseV8._reset_idx (a fresh 50/50 draw for whichever env_ids
are being reset), so a collapse toward zero over time means one of two things:
either sequential envs are resetting (falling/timing out) far more often than
non-sequential ones -- each reset an independent coin flip, so a fast-churning
subpopulation converges toward the unlucky tail purely by attrition -- or
something about the goal generation makes them nearly unreachable, which would
show up as very few `_reached()` hits before every reset.

This borrows diag_reset.py's monkeypatch trick (snapshot _check_termination's
booleans before _reset_idx can touch state) and adds is_seq_env / seq_idx /
goal_dist to the snapshot, so it can report, PER STEP:

    how many envs are currently sequential
    how many sequential envs reset this step, and via which clause
    goal_dist / seq_idx at the moment of that reset

Usage:
    python tools/diag_seq.py --config sweeps/G4_smoothturn.yaml --task K1/Goal_Pose_V8 \\
        --checkpoint <path> --sim_device cuda:0 --rl_device cuda:0
"""

import argparse
import os
import sys

import isaacgym  # noqa: F401  (must precede torch)
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs import *  # noqa: F401,F403  (registers task classes)
from utils.model import ActorCritic
from utils.runner import get_task_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="envs/K1/Goal_Pose_V8.yaml")
    ap.add_argument("--task", default="K1/Goal_Pose_V8")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--num_envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--report_every", type=int, default=20)
    ap.add_argument("--sim_device", default="cuda:0")
    ap.add_argument("--rl_device", default="cuda:0")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    cfg["basic"]["task"] = args.task
    cfg["basic"]["headless"] = True
    cfg["basic"]["sim_device"] = args.sim_device
    cfg["basic"]["rl_device"] = args.rl_device
    cfg["env"]["num_envs"] = args.num_envs
    cfg["viewer"]["record_video"] = False
    torch.manual_seed(0)
    np.random.seed(0)

    env = get_task_class(args.task.split("/")[-1])(cfg)
    dev = env.device

    policy = None
    if args.checkpoint and os.path.exists(args.checkpoint):
        rl_dev = cfg["basic"]["rl_device"]
        model = ActorCritic(env.num_actions, env.num_obs, env.num_privileged_obs).to(rl_dev)
        sd = torch.load(args.checkpoint, map_location=rl_dev, weights_only=True)
        model.load_state_dict(sd["model"], strict=False)
        model.eval()
        policy = model
        print("driving with policy: {}".format(args.checkpoint))
    else:
        print("driving with ZERO actions (no checkpoint)")

    st = cfg["commands"].get("smooth_turn", {}) or {}
    print("\nst_on (config)  : {}".format(st.get("enabled", False)))
    print("st_share        : {}".format(st.get("share", 0.0)))
    print("st_level (init) : {}".format(st.get("init_level", 0.0)))
    print("dtheta_max*c    : {}  (c=level -- 0 at init_level=0 means straight hops only)".format(
        float(st.get("dtheta_max", np.pi)) * float(st.get("init_level", 0.0))))
    print("terminate       : vel^2 > {}, height < {}".format(
        cfg["rewards"]["terminate_vel"], cfg["rewards"]["terminate_height"]))

    orig_check = env._check_termination
    diag = {}

    def hooked_check():
        v2 = env.root_states[:, 7:13].square().sum(dim=-1)
        h = env.base_pos[:, 2] - env.terrain.terrain_heights(env.base_pos)
        c = torch.zeros_like(v2, dtype=torch.bool)
        if len(env.termination_contact_indices) > 0:
            c = torch.any(torch.norm(
                env.contact_forces[:, env.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        diag["c"] = c
        diag["vb"] = v2 > cfg["rewards"]["terminate_vel"]
        diag["hb"] = h < cfg["rewards"]["terminate_height"]
        diag["was_seq"] = env.is_seq_env.clone()
        diag["seq_idx"] = env.seq_idx.clone()
        diag["goal_dist"] = env.goal_dist.clone()
        orig_check()

    env._check_termination = hooked_check

    obs, _ = env.reset()
    n_seq0 = int(env.is_seq_env.sum().item())
    print("\nis_seq_env right after reset(): {}/{} ({:.0%})".format(
        n_seq0, env.num_envs, n_seq0 / env.num_envs))

    seq_reset_clause = {"contact": 0, "velocity": 0, "height": 0, "timeout": 0}
    seq_reset_seqidx = []
    seq_reset_goaldist = []
    bank_events = 0

    for t in range(args.steps):
        if policy is not None:
            with torch.no_grad():
                act = policy.act(obs.to(cfg["basic"]["rl_device"])).loc.to(dev)
        else:
            act = torch.zeros(env.num_envs, env.num_actions, device=dev)
        seq_idx_before = env.seq_idx.clone()
        was_seq_before_step = env.is_seq_env.clone()
        obs, _, _, _ = env.step(act)

        # Bank events: seq_idx increased for an env that was (and, since a bank
        # doesn't reset it, still is) sequential.
        bank_events += int(((env.seq_idx > seq_idx_before) & was_seq_before_step
                            & env.is_seq_env).sum().item())

        was_seq = diag["was_seq"]
        c, vb, hb = diag["c"], diag["vb"], diag["hb"]
        term = c | vb | hb | env.time_out_buf
        seq_term = was_seq & term
        if bool(seq_term.any()):
            for name, clause in (("contact", c), ("velocity", vb), ("height", hb),
                                 ("timeout", env.time_out_buf & ~(c | vb | hb))):
                seq_reset_clause[name] += int((seq_term & clause).sum().item())
            seq_reset_seqidx += diag["seq_idx"][seq_term].cpu().tolist()
            seq_reset_goaldist += diag["goal_dist"][seq_term].cpu().tolist()

        if (t + 1) % args.report_every == 0:
            n = int(env.is_seq_env.sum().item())
            print("  step {:>4}  is_seq_env {:>4}/{}  ({:>4.0%})  banked so far: {}".format(
                t + 1, n, env.num_envs, n / env.num_envs, bank_events))

    print("\n--- sequential 환경이 리셋될 때 어느 조건이 걸렸는가 (총 {}건) ---".format(
        sum(seq_reset_clause.values())))
    for name, n in seq_reset_clause.items():
        print("  {:<9} {:>6}".format(name, n))
    if seq_reset_seqidx:
        arr = np.array(seq_reset_seqidx)
        print("\n  리셋 시점 seq_idx 분포 (0=첫 목표도 못 감): {}".format(
            {int(i): int((arr == i).sum()) for i in sorted(set(arr.tolist()))}))
    if seq_reset_goaldist:
        arr = np.array(seq_reset_goaldist)
        print("  리셋 시점 goal_dist: p50 {:.2f}  p90 {:.2f}  max {:.2f} m".format(
            np.percentile(arr, 50), np.percentile(arr, 90), arr.max()))

    print("\n총 goal 은행(banked) 횟수: {}".format(bank_events))


if __name__ == "__main__":
    main()
