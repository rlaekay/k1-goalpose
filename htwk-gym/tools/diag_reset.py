"""Why is this env resetting? Break check_termination down into its clauses.

smoke_v7 can tell you that goal_segment_id ticks 0.68 times per env-step, which
means the envs are resetting constantly, but not WHICH of the three termination
clauses is firing or what is driving it. G3_full hit exactly that: 52519 resets
in 300 steps, with no way to tell a falling policy from an asset that explodes
on contact at spawn.

check_termination ORs three things together:

    contact  any termination_contact body over 1.0 N
    velocity |root twist|^2 > terminate_vel      <- an exploding asset lands here
    height   base height below terminate_height  <- a genuine fall lands here

step() calls _check_termination() -- which reads root_states/base_pos/contact_forces
to decide reset_buf -- and THEN calls _reset_idx(reset_buf.nonzero()), which
teleports exactly those envs back to spawn. Reading root_states/base_pos again
after env.step() returns therefore sees the FRESH spawn state for every env that
terminated, not the state that triggered the termination -- the velocity and
height clauses silently under-report to zero this way (confirmed: 444,180 N on
left_hand_link on step 1, with velocity/height both reporting 0/51200).

So this monkeypatches _check_termination to snapshot the three clause booleans
the instant they are computed, before _reset_idx can touch anything. The
booleans themselves are fresh tensors from a `>` comparison, not views into
root_states, so they survive the reset that follows in the same step().

Usage:
    python tools/diag_reset.py --config sweeps/G3_full.yaml --task K1/Goal_Pose_V7 \\
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
    ap.add_argument("--config", default="envs/K1/Goal_Pose_V7.yaml")
    ap.add_argument("--task", default="K1/Goal_Pose_V7")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--num_envs", type=int, default=256)
    ap.add_argument("--steps", type=int, default=200)
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
        print("driving with ZERO actions (no checkpoint) -- resets will be inflated")

    body_names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])

    print("\nasset      : {}".format(cfg["asset"]["file"]))
    print("num_dofs   : {}".format(env.num_dofs))
    print("self_colls : {} (0 = enabled)".format(cfg["asset"].get("self_collisions")))
    print("terminate  : vel^2 > {}, height < {}, contact bodies = {}".format(
        cfg["rewards"]["terminate_vel"], cfg["rewards"]["terminate_height"],
        cfg["rewards"]["terminate_contacts_on"]))

    orig_check = env._check_termination
    diag = {"c": None, "vb": None, "hb": None, "v2": None, "h": None}

    def hooked_check():
        # Replicate goal_pose._check_termination's three clauses on the SAME
        # pre-reset tensors it reads, and stash them, then defer to the real
        # implementation so behavior (including any subclass override) is
        # unchanged.
        v2 = env.root_states[:, 7:13].square().sum(dim=-1)
        h = env.base_pos[:, 2] - env.terrain.terrain_heights(env.base_pos)
        c = torch.zeros_like(v2, dtype=torch.bool)
        if len(env.termination_contact_indices) > 0:
            c = torch.any(torch.norm(
                env.contact_forces[:, env.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        diag["c"] = c
        diag["vb"] = v2 > cfg["rewards"]["terminate_vel"]
        diag["hb"] = h < cfg["rewards"]["terminate_height"]
        diag["v2"], diag["h"] = v2, h
        orig_check()

    env._check_termination = hooked_check

    obs, _ = env.reset()
    n_contact = n_vel = n_height = n_timeout = 0
    first_step_resets = 0
    worst_force = torch.zeros(env.contact_forces.shape[1], device=dev)
    vel_at_reset, h_at_reset = [], []

    for t in range(args.steps):
        if policy is not None:
            with torch.no_grad():
                act = policy.act(obs.to(cfg["basic"]["rl_device"])).loc.to(dev)
        else:
            act = torch.zeros(env.num_envs, env.num_actions, device=dev)
        # worst_force must be sampled BEFORE step(), which is when contact_forces
        # holds the value that triggered this step's termination decision -- by
        # the time step() returns, envs that got reset already show fresh (near
        # zero) contact forces for the next physics tick.
        worst_force = torch.maximum(worst_force,
                                    torch.norm(env.contact_forces, dim=-1).max(dim=0).values)
        obs, _, _, _ = env.step(act)

        c, vb, hb, v2, h = diag["c"], diag["vb"], diag["hb"], diag["v2"], diag["h"]
        n_contact += int(c.sum()); n_vel += int(vb.sum()); n_height += int(hb.sum())
        n_timeout += int(env.time_out_buf.sum())
        if t == 0:
            first_step_resets = int((c | vb | hb).sum())
        if bool(vb.any()):
            vel_at_reset.append(float(v2[vb].max()))
        if bool(hb.any()):
            h_at_reset.append(float(h[hb].min()))

    tot = args.steps * env.num_envs
    print("\n--- 종료 조건별 발생 (env-step {} 중) ---".format(tot))
    for label, n in (("contact", n_contact), ("velocity", n_vel),
                     ("height", n_height), ("timeout", n_timeout)):
        print("  {:<9} {:>8}  ({:.1%})".format(label, n, n / float(tot)))
    print("  첫 스텝에 이미 종료 조건 만족: {}/{} envs".format(first_step_resets, env.num_envs))
    if vel_at_reset:
        print("  속도 종료 시 |twist|^2 최대 {:.1f} (임계 {})".format(
            max(vel_at_reset), cfg["rewards"]["terminate_vel"]))
    if h_at_reset:
        print("  높이 종료 시 최저 {:.3f} m (임계 {})".format(
            min(h_at_reset), cfg["rewards"]["terminate_height"]))

    print("\n--- 강체별 최대 접촉력 [N] (상위 12) ---")
    order = torch.argsort(worst_force, descending=True)[:12]
    for i in order.tolist():
        name = body_names[i] if body_names and i < len(body_names) else "body_{}".format(i)
        print("  {:<22} {:>9.1f}".format(name, float(worst_force[i])))


if __name__ == "__main__":
    main()
